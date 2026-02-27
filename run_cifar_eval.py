import argparse
import csv
import os
import pickle
import time

import torch

from fisher_edl.cifar_data import (
    build_cifar10_loaders,
    build_ood_loaders,
    ood_group,
    parse_ood_datasets,
)
from fisher_edl.cifar_model import build_cifar_model
from fisher_edl.metrics import (
    compute_aupr,
    compute_auroc,
    compute_fpr_at_tpr,
    fit_temperature,
    ood_score_from_logits,
    reliability_bins,
    summarize_id_metrics,
)
from fisher_edl.wandb_loader import import_wandb
from fisher_edl.wandb_utils import import_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved CIFAR model checkpoints")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ensemble-ckpts", type=str, default=None)
    parser.add_argument("--method", type=str, default="auto")
    parser.add_argument("--backbone", choices=["simple", "resnet18", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--id-dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--ood-datasets", type=str, default="svhn,cifar100")
    parser.add_argument("--score-type", choices=["uncertainty", "energy", "msp"], default="uncertainty")
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--mc-dropout-passes", type=int, default=1)
    parser.add_argument("--skip-missing-ood", action="store_true")

    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--hist-out", type=str, default=None)
    parser.add_argument("--hist-png", type=str, default=None)
    parser.add_argument("--reliability-out", type=str, default=None)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


class _PickleModule:
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("fisher_edl"):
                return type("Dummy", (), {})
            return super().find_class(module, name)

    dump = pickle.dump
    dumps = pickle.dumps
    load = pickle.load
    loads = pickle.loads


def _safe_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        pass
    except ModuleNotFoundError:
        pass
    except pickle.UnpicklingError:
        pass
    except RuntimeError:
        pass
    return torch.load(path, map_location=device, pickle_module=_PickleModule)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def _extract_method(ckpt, method_arg):
    if method_arg and method_arg != "auto":
        return method_arg
    if isinstance(ckpt, dict):
        cfg = ckpt.get("config")
        if cfg is not None and hasattr(cfg, "method"):
            return str(getattr(cfg, "method"))
        meta = ckpt.get("meta", {})
        if isinstance(meta, dict) and "method" in meta:
            return str(meta["method"])
    return "unknown"


def _extract_train_time(ckpt):
    if isinstance(ckpt, dict):
        meta = ckpt.get("meta", {})
        if isinstance(meta, dict) and "train_time_sec" in meta:
            try:
                return float(meta["train_time_sec"])
            except Exception:
                return float("nan")
    return float("nan")


def _infer_num_classes(state_dict):
    candidate_keys = [
        "fc.weight",
        "fc.1.weight",
        "classifier.weight",
        "classifier.1.weight",
    ]
    for key in candidate_keys:
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])
    for key, value in state_dict.items():
        if key.endswith("weight") and value.ndim == 2:
            return int(value.shape[0])
    raise RuntimeError("Failed to infer num_classes from state_dict")


def _infer_dropout_p(state_dict):
    # Sequential heads store fc.1.weight and fc.0 is dropout.
    if any(k.startswith("fc.1.") for k in state_dict.keys()):
        return 0.2
    if any(k.startswith("classifier.1.") for k in state_dict.keys()):
        return 0.2
    return 0.0


def _enable_dropout_only(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
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
                mc_logits = []
                for _ in range(mc_dropout_passes):
                    mc_logits.append(model(images))
                logits = torch.stack(mc_logits, dim=0).mean(dim=0)
            else:
                logits = model(images)

            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

    return torch.cat(logits_all), torch.cat(labels_all)


def _collect_ensemble_logits_labels(models, loader, device):
    logits_all = []
    labels_all = []
    for model in models:
        model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            member_logits = []
            for model in models:
                member_logits.append(model(images))
            logits = torch.stack(member_logits, dim=0).mean(dim=0)

            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

    return torch.cat(logits_all), torch.cat(labels_all)


def _build_id_loaders(args, require_val=False):
    if args.id_dataset != "cifar10":
        raise ValueError(f"Unsupported id dataset: {args.id_dataset}")

    val_split = 0.1 if require_val else 0.0
    _, val_loader, test_loader, test_transform = build_cifar10_loaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        val_split=val_split,
        seed=args.seed,
    )
    return val_loader, test_loader, test_transform


def _load_models_from_ckpts(ckpt_paths, backbone, device):
    models = []
    ckpts = []

    for path in ckpt_paths:
        ckpt = _safe_load_checkpoint(path, device)
        state_dict = _extract_state_dict(ckpt)

        bb = backbone
        if bb == "auto":
            if any(k.startswith("layer1.") for k in state_dict.keys()):
                bb = "resnet18"
            else:
                bb = "simple"

        num_classes = _infer_num_classes(state_dict)
        dropout_p = _infer_dropout_p(state_dict)
        model = build_cifar_model(backbone=bb, num_classes=num_classes, dropout_p=dropout_p)
        model.load_state_dict(state_dict)
        model.to(device)

        models.append(model)
        ckpts.append(ckpt)

    return models, ckpts


def evaluate_cifar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ensemble_ckpts:
        ckpt_paths = [x.strip() for x in args.ensemble_ckpts.split(",") if x.strip()]
    elif args.ckpt:
        ckpt_paths = [args.ckpt]
    else:
        raise ValueError("Either --ckpt or --ensemble-ckpts must be provided")

    val_loader, test_loader, test_transform = _build_id_loaders(
        args, require_val=args.calibration == "temperature"
    )
    ood_names = parse_ood_datasets(args.ood_datasets)
    ood_loaders = build_ood_loaders(
        ood_names,
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        transform=test_transform,
        skip_missing=args.skip_missing_ood,
    )

    models, ckpts = _load_models_from_ckpts(ckpt_paths, args.backbone, device)
    method = _extract_method(ckpts[0], args.method)
    train_time = _extract_train_time(ckpts[0])

    infer_start = time.perf_counter()
    if len(models) == 1:
        model = models[0]
        test_logits, test_labels = _collect_logits_labels(
            model,
            test_loader,
            device,
            mc_dropout_passes=max(1, int(args.mc_dropout_passes)),
        )
    else:
        test_logits, test_labels = _collect_ensemble_logits_labels(models, test_loader, device)

    temp = 1.0
    if args.calibration == "temperature":
        if val_loader is None:
            raise RuntimeError("Temperature calibration requested but val_loader is unavailable")
        if len(models) == 1:
            val_logits, val_labels = _collect_logits_labels(models[0], val_loader, device, mc_dropout_passes=1)
        else:
            val_logits, val_labels = _collect_ensemble_logits_labels(models, val_loader, device)
        temp = fit_temperature(val_logits, val_labels)

    test_logits = test_logits / temp
    id_summary = summarize_id_metrics(test_logits, test_labels, n_bins=15)

    id_scores = ood_score_from_logits(test_logits, score_type=args.score_type)

    rows = []
    hist_data = {}

    if not ood_names:
        rows.append(
            {
                "dataset_id": args.id_dataset,
                "dataset_ood": "id_only",
                "ood_group": "id",
                "method": method,
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
                "auroc": float("nan"),
                "aupr": float("nan"),
                "fpr95": float("nan"),
                "train_time": train_time,
                "infer_time": float("nan"),
                "ckpt": ";".join(ckpt_paths),
            }
        )

    for name, loader in ood_loaders.items():
        if len(models) == 1:
            ood_logits, _ = _collect_logits_labels(
                models[0],
                loader,
                device,
                mc_dropout_passes=max(1, int(args.mc_dropout_passes)),
            )
        else:
            ood_logits, _ = _collect_ensemble_logits_labels(models, loader, device)

        ood_logits = ood_logits / temp
        ood_scores = ood_score_from_logits(ood_logits, score_type=args.score_type)

        labels = torch.cat([torch.zeros_like(id_scores), torch.ones_like(ood_scores)])
        scores = torch.cat([id_scores, ood_scores])

        auroc = compute_auroc(scores, labels)
        aupr = compute_aupr(scores, labels)
        fpr95 = compute_fpr_at_tpr(scores, labels, tpr_target=0.95)

        rows.append(
            {
                "dataset_id": args.id_dataset,
                "dataset_ood": name,
                "ood_group": ood_group(name),
                "method": method,
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
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95,
                "train_time": train_time,
                "infer_time": float("nan"),
                "ckpt": ";".join(ckpt_paths),
            }
        )

        hist_data[name] = {
            "id_scores": id_scores,
            "ood_scores": ood_scores,
        }

    infer_time = time.perf_counter() - infer_start
    for row in rows:
        row["infer_time"] = infer_time

    reliability = reliability_bins(torch.softmax(test_logits, dim=1), test_labels, n_bins=15)

    return {
        "rows": rows,
        "reliability": reliability,
        "hist_data": hist_data,
        "id_scores": id_scores,
    }


def _write_rows_csv(path, rows):
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


def _write_reliability_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_hist_csv(path, hist_data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "bin_left",
                "bin_right",
                "id_count",
                "ood_count",
                "score_min",
                "score_max",
            ]
        )
        for dataset, payload in hist_data.items():
            min_score = min(payload["id_scores"].min().item(), payload["ood_scores"].min().item())
            max_score = max(payload["id_scores"].max().item(), payload["ood_scores"].max().item())
            if abs(max_score - min_score) < 1e-12:
                max_score = min_score + 1.0
            bins = torch.linspace(min_score, max_score, steps=51)
            id_hist = torch.histc(payload["id_scores"], bins=50, min=min_score, max=max_score)
            ood_hist = torch.histc(payload["ood_scores"], bins=50, min=min_score, max=max_score)
            for i in range(50):
                writer.writerow(
                    [
                        dataset,
                        f"{bins[i].item():.6f}",
                        f"{bins[i + 1].item():.6f}",
                        f"{id_hist[i].item():.0f}",
                        f"{ood_hist[i].item():.0f}",
                        f"{min_score:.6f}",
                        f"{max_score:.6f}",
                    ]
                )


def _write_hist_png(path, hist_data):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    num = len(hist_data)
    if num == 0:
        return

    fig, axes = plt.subplots(num, 1, figsize=(7, 3 * num), squeeze=False)
    for idx, (dataset, payload) in enumerate(hist_data.items()):
        ax = axes[idx][0]
        min_score = min(payload["id_scores"].min().item(), payload["ood_scores"].min().item())
        max_score = max(payload["id_scores"].max().item(), payload["ood_scores"].max().item())
        if abs(max_score - min_score) < 1e-12:
            max_score = min_score + 1.0
        ax.hist(
            payload["id_scores"].numpy(),
            bins=50,
            range=(min_score, max_score),
            alpha=0.6,
            label="ID",
        )
        ax.hist(
            payload["ood_scores"].numpy(),
            bins=50,
            range=(min_score, max_score),
            alpha=0.6,
            label=dataset,
        )
        ax.set_title(f"ID vs {dataset}")
        ax.set_xlabel("OOD score")
        ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    result = evaluate_cifar(args)
    rows = result["rows"]

    for row in rows:
        ood_name = row["dataset_ood"]
        if ood_name == "id_only":
            print(
                f"ID={row['dataset_id']} method={row['method']} acc={row['acc']:.4f} "
                f"nll={row['nll']:.4f} ece={row['ece']:.4f} aurc={row['aurc']:.4f}"
            )
        else:
            print(
                f"ID={row['dataset_id']} OOD={ood_name} ({row['ood_group']}) "
                f"AUROC={row['auroc']:.4f} AUPR={row['aupr']:.4f} FPR95={row['fpr95']:.4f} "
                f"Acc={row['acc']:.4f} ECE={row['ece']:.4f}"
            )

    if args.out_csv and rows:
        _write_rows_csv(args.out_csv, rows)

    if args.reliability_out:
        _write_reliability_csv(args.reliability_out, result["reliability"])

    if args.hist_out:
        _write_hist_csv(args.hist_out, result["hist_data"])

    if args.hist_png:
        _write_hist_png(args.hist_png, result["hist_data"])

    if args.wandb:
        wandb = import_wandb(".")

        run = wandb.init(
            project=args.wandb_project or "fisher-edl",
            name=args.wandb_name,
            config={
                "ckpt": args.ckpt,
                "ensemble_ckpts": args.ensemble_ckpts,
                "backbone": args.backbone,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "id_dataset": args.id_dataset,
                "ood_datasets": args.ood_datasets,
                "score_type": args.score_type,
                "calibration": args.calibration,
                "mc_dropout_passes": args.mc_dropout_passes,
            },
        )
        for row in rows:
            prefix = f"eval/{row['dataset_ood']}"
            run.log(
                {
                    f"{prefix}/acc": row["acc"],
                    f"{prefix}/nll": row["nll"],
                    f"{prefix}/ece": row["ece"],
                    f"{prefix}/auroc": row["auroc"],
                    f"{prefix}/aupr": row["aupr"],
                    f"{prefix}/fpr95": row["fpr95"],
                    f"{prefix}/aurc": row["aurc"],
                }
            )
        run.finish()


if __name__ == "__main__":
    main()
