import argparse
import csv
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from fisher_edl.cifar_data import one_hot, ood_group, parse_ood_datasets
from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss
from fisher_edl.metrics import (
    compute_aupr,
    compute_auroc,
    compute_fpr_at_tpr,
    fit_temperature,
    ood_score_from_logits,
    summarize_id_metrics,
)
from fisher_edl.wandb_utils import import_wandb


class OODWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, -1


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet-100 benchmark for Info-EDL")
    parser.add_argument("--method", choices=["ce", "mcdropout", "edl", "fisher"], default="fisher")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--anneal-kl", action="store_true")
    parser.add_argument("--anneal-epochs", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--id-root", type=str, default=None, help="ImageNet-100 root. default: <data-root>/imagenet100")
    parser.add_argument("--ood-datasets", type=str, default="inaturalist,texture,openimage-o,imagenet-o")
    parser.add_argument("--score-type", choices=["uncertainty", "energy", "msp"], default="uncertainty")
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--mc-dropout-passes", type=int, default=10)

    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--eval-out-csv", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="info-edl")
    parser.add_argument("--wandb-name", type=str, default=None)

    parser.add_argument("--info-type", choices=["fisher", "evidence", "entropy"], default="fisher")
    parser.add_argument("--gate-type", choices=["exp", "inverse", "sigmoid"], default="exp")
    parser.add_argument("--detach-weight", action="store_true")
    parser.add_argument("--objective", choices=["risk_plus_kl", "kl_only"], default="risk_plus_kl")

    return parser.parse_args()


def _build_model(num_classes, method, dropout_p=0.2):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    if method == "mcdropout":
        model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def _get_loaders(args):
    id_root = args.id_root or os.path.join(args.data_root, "imagenet100")
    train_root = os.path.join(id_root, "train")
    val_root = os.path.join(id_root, "val")

    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        raise FileNotFoundError(
            f"ImageNet-100 folders not found. Expected: {train_root} and {val_root}"
        )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds = datasets.ImageFolder(val_root, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ood_loaders = {}
    for name in parse_ood_datasets(args.ood_datasets):
        folder = os.path.join(args.data_root, "ood", name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"OOD folder missing for {name}: {folder}. "
                "Place folder-structured images under this path."
            )
        ood_ds = datasets.ImageFolder(folder, transform=test_tf)
        ood_loaders[name] = DataLoader(
            OODWrapper(ood_ds),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, ood_loaders, len(train_ds.classes)


def _enable_dropout_only(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def _collect_logits_labels(model, loader, device, mc_dropout_passes=1):
    logits_all = []
    labels_all = []
    model.eval()
    if mc_dropout_passes > 1:
        _enable_dropout_only(model)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if mc_dropout_passes > 1:
                logits = torch.stack([model(images) for _ in range(mc_dropout_passes)], dim=0).mean(dim=0)
            else:
                logits = model(images)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

    return torch.cat(logits_all), torch.cat(labels_all)


def _train(model, train_loader, val_loader, num_classes, args, device, run_dir):
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    model.to(device)
    best_acc = -1.0
    best_path = os.path.join(run_dir, "best_val_acc.pt")
    last_path = os.path.join(run_dir, "last.pt")

    metrics_path = os.path.join(run_dir, "metrics.csv")
    wandb_run = None
    if args.wandb:
        wandb = import_wandb(".")

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "method": args.method,
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": args.beta,
                "gamma": args.gamma,
                "seed": args.seed,
                "run_dir": run_dir,
            },
        )

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_acc", "val_acc", "time_sec"])

        train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            if args.method == "edl" and args.anneal_kl:
                kl_weight = args.beta * min(1.0, epoch / max(1, args.anneal_epochs))
            else:
                kl_weight = args.beta

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)

                if args.method in {"ce", "mcdropout"}:
                    loss = F.cross_entropy(logits, labels)
                elif args.method == "edl":
                    targets = one_hot(labels, num_classes).to(device)
                    loss, _ = edl_mse_loss(logits, targets, kl_weight=kl_weight)
                else:
                    targets = one_hot(labels, num_classes).to(device)
                    loss, _ = fisher_edl_mse_loss(
                        logits,
                        targets,
                        beta=args.beta,
                        gamma=args.gamma,
                        info_type=args.info_type,
                        gate_type=args.gate_type,
                        detach_weight=args.detach_weight,
                        objective=args.objective,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_seen += labels.size(0)

            train_acc = total_correct / max(1, total_seen)

            val_logits, val_labels = _collect_logits_labels(model, val_loader, device, mc_dropout_passes=1)
            val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()

            elapsed = time.perf_counter() - epoch_start
            writer.writerow(
                [
                    epoch,
                    f"{total_loss / max(1, len(train_loader)):.6f}",
                    f"{train_acc:.6f}",
                    f"{val_acc:.6f}",
                    f"{elapsed:.6f}",
                ]
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train_loss": total_loss / max(1, len(train_loader)),
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "epoch_time_sec": elapsed,
                    }
                )

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "meta": {
                    "method": args.method,
                    "train_time_sec": time.perf_counter() - train_start,
                },
            }
            torch.save(state, last_path)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(state, best_path)

            print(
                f"Epoch {epoch:03d} | loss={total_loss / max(1, len(train_loader)):.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

    if wandb_run:
        wandb_run.finish()
    return best_path


def _evaluate(best_ckpt_path, model, val_loader, ood_loaders, args, device):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    infer_start = time.perf_counter()
    val_logits, val_labels = _collect_logits_labels(
        model,
        val_loader,
        device,
        mc_dropout_passes=args.mc_dropout_passes if args.method == "mcdropout" else 1,
    )

    temp = 1.0
    if args.calibration == "temperature":
        temp = fit_temperature(val_logits, val_labels)

    val_logits = val_logits / temp
    id_summary = summarize_id_metrics(val_logits, val_labels)
    id_scores = ood_score_from_logits(val_logits, score_type=args.score_type)

    rows = []
    for ood_name, loader in ood_loaders.items():
        ood_logits, _ = _collect_logits_labels(
            model,
            loader,
            device,
            mc_dropout_passes=args.mc_dropout_passes if args.method == "mcdropout" else 1,
        )
        ood_logits = ood_logits / temp
        ood_scores = ood_score_from_logits(ood_logits, score_type=args.score_type)

        labels = torch.cat([torch.zeros_like(id_scores), torch.ones_like(ood_scores)])
        scores = torch.cat([id_scores, ood_scores])

        rows.append(
            {
                "dataset_id": "imagenet100",
                "dataset_ood": ood_name,
                "ood_group": ood_group(ood_name),
                "method": args.method,
                "seed": args.seed,
                "score_type": args.score_type,
                "calibration": args.calibration,
                "temperature": temp,
                "acc": id_summary["acc"],
                "nll": id_summary["nll"],
                "brier": id_summary["brier"],
                "ece": id_summary["ece"],
                "classwise_ece": id_summary["classwise_ece"],
                "miscls_auroc": id_summary["miscls_auroc"],
                "aurc": id_summary["aurc"],
                "auroc": compute_auroc(scores, labels),
                "aupr": compute_aupr(scores, labels),
                "fpr95": compute_fpr_at_tpr(scores, labels),
                "train_time": ckpt.get("meta", {}).get("train_time_sec", float("nan")),
                "infer_time": float("nan"),
                "ckpt": best_ckpt_path,
            }
        )

    infer_time = time.perf_counter() - infer_start
    for row in rows:
        row["infer_time"] = infer_time
    return rows


def _write_rows(path, rows):
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run_dir:
        run_dir = args.run_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"imagenet100_{args.method}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader, ood_loaders, num_classes = _get_loaders(args)
    model = _build_model(num_classes=num_classes, method=args.method)

    best_path = _train(model, train_loader, val_loader, num_classes, args, device, run_dir)
    rows = _evaluate(best_path, model, val_loader, ood_loaders, args, device)

    if args.eval_out_csv:
        out_csv = args.eval_out_csv
    else:
        out_csv = os.path.join(run_dir, "eval_metrics.csv")
    _write_rows(out_csv, rows)

    for row in rows:
        print(
            f"ImageNet100 vs {row['dataset_ood']} ({row['ood_group']}): "
            f"AUROC={row['auroc']:.4f} AUPR={row['aupr']:.4f} FPR95={row['fpr95']:.4f} "
            f"Acc={row['acc']:.4f} ECE={row['ece']:.4f}"
        )
    print(f"Saved eval rows -> {out_csv}")


if __name__ == "__main__":
    main()
