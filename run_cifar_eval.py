import argparse
import csv
import os
import pickle
import torch

from fisher_edl.cifar_data import (
    build_cifar10_loaders,
    build_cifar100_loader,
    build_svhn_loader,
)
from fisher_edl.cifar_model import build_cifar_model
from fisher_edl.metrics import compute_auroc, compute_fpr_at_tpr, uncertainty_from_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved CIFAR model checkpoints")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", choices=["simple", "resnet18", "auto"], default="simple")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--hist-out", type=str, default=None)
    parser.add_argument("--hist-png", type=str, default=None)
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
        ckpt = torch.load(path, map_location=device, weights_only=True)
        return ckpt
    except TypeError:
        pass
    except ModuleNotFoundError:
        pass
    return torch.load(path, map_location=device, pickle_module=_PickleModule)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def _collect_uncertainties(model, loader, device):
    model.eval()
    uncertainties = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            uncertainties.append(uncertainty_from_logits(logits))
            if labels.min().item() >= 0:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    acc = correct / max(1, total)
    return acc, torch.cat(uncertainties)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _, test_loader, test_transform = build_cifar10_loaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        val_split=0.0,
        seed=args.seed,
    )

    svhn_loader = build_svhn_loader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        transform=test_transform,
    )
    cifar100_loader = build_cifar100_loader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        transform=test_transform,
    )

    ckpt = _safe_load_checkpoint(args.ckpt, device)
    state_dict = _extract_state_dict(ckpt)

    backbone = args.backbone
    if backbone == "auto":
        if any(k.startswith("layer1.") for k in state_dict.keys()):
            backbone = "resnet18"
        else:
            backbone = "simple"

    model = build_cifar_model(backbone=backbone, num_classes=10)
    model.load_state_dict(state_dict)
    model.to(device)

    test_acc, test_unc = _collect_uncertainties(model, test_loader, device)
    svhn_acc, svhn_unc = _collect_uncertainties(model, svhn_loader, device)
    cifar100_acc, cifar100_unc = _collect_uncertainties(model, cifar100_loader, device)

    test_labels = torch.zeros_like(test_unc)
    svhn_labels = torch.ones_like(svhn_unc)
    cifar100_labels = torch.ones_like(cifar100_unc)

    svhn_auroc_test = compute_auroc(torch.cat([test_unc, svhn_unc]), torch.cat([test_labels, svhn_labels]))
    cifar100_auroc_test = compute_auroc(torch.cat([test_unc, cifar100_unc]), torch.cat([test_labels, cifar100_labels]))
    svhn_fpr95 = compute_fpr_at_tpr(torch.cat([test_unc, svhn_unc]), torch.cat([test_labels, svhn_labels]))
    cifar100_fpr95 = compute_fpr_at_tpr(torch.cat([test_unc, cifar100_unc]), torch.cat([test_labels, cifar100_labels]))

    results = {
        "ckpt": args.ckpt,
        "test_acc": test_acc,
        "test_uncertainty": test_unc.mean().item(),
        "svhn_uncertainty": svhn_unc.mean().item(),
        "cifar100_uncertainty": cifar100_unc.mean().item(),
        "svhn_auroc_test": svhn_auroc_test,
        "cifar100_auroc_test": cifar100_auroc_test,
        "svhn_fpr95_test": svhn_fpr95,
        "cifar100_fpr95_test": cifar100_fpr95,
    }

    print(f"Checkpoint: {args.ckpt}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test unc: {test_unc.mean().item():.4f}")
    print(f"SVHN unc: {svhn_unc.mean().item():.4f} | CIFAR100 unc: {cifar100_unc.mean().item():.4f}")
    print(f"SVHN AUROC (test): {svhn_auroc_test:.4f}")
    print(f"CIFAR100 AUROC (test): {cifar100_auroc_test:.4f}")
    print(f"SVHN FPR@95TPR (test): {svhn_fpr95:.4f}")
    print(f"CIFAR100 FPR@95TPR (test): {cifar100_fpr95:.4f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(list(results.keys()))
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in results.values()])

    if args.hist_out:
        os.makedirs(os.path.dirname(args.hist_out) or ".", exist_ok=True)
        bins = torch.linspace(0.0, 1.0, steps=51)
        id_hist = torch.histc(test_unc, bins=50, min=0.0, max=1.0)
        svhn_hist = torch.histc(svhn_unc, bins=50, min=0.0, max=1.0)
        cifar100_hist = torch.histc(cifar100_unc, bins=50, min=0.0, max=1.0)
        with open(args.hist_out, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["bin_left", "bin_right", "id_count", "svhn_count", "cifar100_count"])
            for i in range(50):
                writer.writerow(
                    [
                        f"{bins[i].item():.6f}",
                        f"{bins[i + 1].item():.6f}",
                        f"{id_hist[i].item():.0f}",
                        f"{svhn_hist[i].item():.0f}",
                        f"{cifar100_hist[i].item():.0f}",
                    ]
                )

    if args.hist_png:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(args.hist_png) or ".", exist_ok=True)
        id_vals = test_unc.cpu().numpy()
        svhn_vals = svhn_unc.cpu().numpy()
        cifar100_vals = cifar100_unc.cpu().numpy()
        plt.figure(figsize=(7, 4))
        plt.hist(id_vals, bins=50, range=(0.0, 1.0), alpha=0.6, label="ID (CIFAR-10)")
        plt.hist(svhn_vals, bins=50, range=(0.0, 1.0), alpha=0.6, label="SVHN")
        plt.hist(cifar100_vals, bins=50, range=(0.0, 1.0), alpha=0.6, label="CIFAR-100")
        plt.xlabel("Uncertainty")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.hist_png, dpi=150)
        plt.close()

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project or "fisher-edl",
            name=args.wandb_name,
            config={
                "ckpt": args.ckpt,
                "backbone": args.backbone,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
        )
        run.log(results)
        run.finish()


if __name__ == "__main__":
    main()
