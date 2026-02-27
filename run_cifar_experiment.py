import argparse
import datetime
import os
from types import SimpleNamespace

import torch

from fisher_edl.cifar_data import build_cifar10_loaders
from fisher_edl.cifar_model import build_cifar_model
from fisher_edl.train_cifar import CifarTrainConfig, train_cifar
from run_cifar_eval import evaluate_cifar


def parse_args():
    parser = argparse.ArgumentParser(description="Fisher-EDL CIFAR experiments")
    parser.add_argument("--method", choices=["ce", "mcdropout", "edl", "fisher"], default="fisher")
    parser.add_argument("--edl-preset", choices=["vanilla", "strong"], default="vanilla")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--anneal-kl", action="store_true", help="Enable KL annealing for EDL")
    parser.add_argument("--anneal-epochs", type=int, default=10)

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--data_root", dest="data_root", type=str)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num_workers", dest="num_workers", type=int)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val_split", dest="val_split", type=float)

    parser.add_argument("--id-dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--ood-datasets", type=str, default="svhn,cifar100")
    parser.add_argument("--metrics", type=str, default="acc,nll,brier,ece,auroc,aupr,fpr95,aurc")
    parser.add_argument("--score-type", choices=["uncertainty", "energy", "msp"], default="uncertainty")
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--skip-missing-ood", action="store_true")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=1)

    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run_dir", dest="run_dir", type=str)
    parser.add_argument("--backbone", choices=["simple", "resnet18"], default="simple")
    parser.add_argument("--dropout-p", type=float, default=0.0)

    parser.add_argument("--mc-dropout-passes", type=int, default=10)
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--eval-out-csv", type=str, default=None)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb_project", dest="wandb_project", type=str)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb_name", dest="wandb_name", type=str)

    # Fisher ablations
    parser.add_argument("--info-type", choices=["fisher", "evidence", "entropy"], default="fisher")
    parser.add_argument("--gate-type", choices=["exp", "inverse", "sigmoid"], default="exp")
    parser.add_argument("--detach-weight", action="store_true")
    parser.add_argument("--objective", choices=["risk_plus_kl", "kl_only"], default="risk_plus_kl")

    return parser.parse_args()


def _write_eval_rows(path, rows):
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = v
            writer.writerow(out)


def _method_dropout_p(method, dropout_p):
    if method == "mcdropout":
        return dropout_p if dropout_p > 0 else 0.2
    return dropout_p


def main():
    args = parse_args()

    if args.id_dataset != "cifar10":
        raise ValueError("This script currently supports only --id-dataset=cifar10")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("runs", exist_ok=True)

    if args.run_dir:
        root_run_dir = args.run_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_run_dir = os.path.join("runs", f"cifar_{args.method}_{stamp}")

    if args.wandb_name:
        base_wandb_name = args.wandb_name
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_wandb_name = f"{args.method}_lr{args.lr}_b{args.beta}_g{args.gamma}_{stamp}"

    all_eval_rows = []

    for idx in range(args.num_seeds):
        seed = args.seed + idx
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader, val_loader, _, _ = build_cifar10_loaders(
            batch_size=args.batch_size,
            data_root=args.data_root,
            num_workers=args.num_workers,
            val_split=args.val_split,
            seed=seed,
        )

        method = args.method
        if method == "edl" and args.edl_preset == "strong":
            args.backbone = "resnet18"
            if not args.anneal_kl:
                args.anneal_kl = True
                args.anneal_epochs = 10
            args.optimizer = "adamw"
            if args.scheduler == "none":
                args.scheduler = "cosine"
            if args.warmup_epochs == 0:
                args.warmup_epochs = 5
            if args.weight_decay == 0.0:
                args.weight_decay = 5e-4

        dropout_p = _method_dropout_p(method, args.dropout_p)
        model = build_cifar_model(backbone=args.backbone, num_classes=10, dropout_p=dropout_p)

        if args.num_seeds > 1:
            run_dir = os.path.join(root_run_dir, f"seed_{seed}")
            wandb_name = f"{base_wandb_name}_seed{seed}"
        else:
            run_dir = root_run_dir
            wandb_name = base_wandb_name

        cfg = CifarTrainConfig(
            method=method,
            epochs=args.epochs,
            lr=args.lr,
            beta=args.beta,
            gamma=args.gamma,
            anneal_kl=args.anneal_kl,
            anneal_epochs=args.anneal_epochs,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            device=device,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_name,
            info_type=args.info_type,
            gate_type=args.gate_type,
            detach_weight=args.detach_weight,
            objective=args.objective,
        )

        print(f"Running {method} on {device} seed={seed}. logs -> {run_dir}")
        train_result = train_cifar(
            model,
            train_loader,
            val_loader,
            num_classes=10,
            cfg=cfg,
            run_dir=run_dir,
        )

        if args.eval_after_train:
            eval_args = SimpleNamespace(
                ckpt=train_result["best_ckpt"],
                ensemble_ckpts=None,
                method=method,
                backbone="auto",
                batch_size=args.batch_size,
                data_root=args.data_root,
                num_workers=args.num_workers,
                seed=seed,
                id_dataset=args.id_dataset,
                ood_datasets=args.ood_datasets,
                score_type=args.score_type,
                calibration=args.calibration,
                mc_dropout_passes=args.mc_dropout_passes if method == "mcdropout" else 1,
                skip_missing_ood=args.skip_missing_ood,
                out_csv=None,
                hist_out=None,
                hist_png=None,
                reliability_out=None,
                wandb=False,
                wandb_project=args.wandb_project,
                wandb_name=None,
            )
            eval_result = evaluate_cifar(eval_args)
            for row in eval_result["rows"]:
                row["seed"] = seed
                all_eval_rows.append(row)

    if args.eval_after_train and all_eval_rows:
        if args.eval_out_csv:
            eval_out_csv = args.eval_out_csv
        else:
            eval_out_csv = os.path.join(root_run_dir, "eval_metrics.csv")
        _write_eval_rows(eval_out_csv, all_eval_rows)
        print(f"Saved evaluation rows -> {eval_out_csv}")


if __name__ == "__main__":
    main()
