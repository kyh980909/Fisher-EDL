import argparse
import csv
import os
import torch

from fisher_edl.cifar_data import (
    build_cifar10_loaders,
    build_cifar100_loader,
    build_svhn_loader,
)
from fisher_edl.cifar_model import build_cifar_model
from fisher_edl.metrics import compute_auroc, uncertainty_from_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved CIFAR model checkpoints")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", choices=["simple", "resnet18"], default="simple")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


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

    model = build_cifar_model(backbone=args.backbone, num_classes=10)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_acc, test_unc = _collect_uncertainties(model, test_loader, device)
    svhn_acc, svhn_unc = _collect_uncertainties(model, svhn_loader, device)
    cifar100_acc, cifar100_unc = _collect_uncertainties(model, cifar100_loader, device)

    test_labels = torch.zeros_like(test_unc)
    svhn_labels = torch.ones_like(svhn_unc)
    cifar100_labels = torch.ones_like(cifar100_unc)

    svhn_auroc_test = compute_auroc(torch.cat([test_unc, svhn_unc]), torch.cat([test_labels, svhn_labels]))
    cifar100_auroc_test = compute_auroc(torch.cat([test_unc, cifar100_unc]), torch.cat([test_labels, cifar100_labels]))

    results = {
        "ckpt": args.ckpt,
        "test_acc": test_acc,
        "test_uncertainty": test_unc.mean().item(),
        "svhn_uncertainty": svhn_unc.mean().item(),
        "cifar100_uncertainty": cifar100_unc.mean().item(),
        "svhn_auroc_test": svhn_auroc_test,
        "cifar100_auroc_test": cifar100_auroc_test,
    }

    print(f"Checkpoint: {args.ckpt}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test unc: {test_unc.mean().item():.4f}")
    print(f"SVHN unc: {svhn_unc.mean().item():.4f} | CIFAR100 unc: {cifar100_unc.mean().item():.4f}")
    print(f"SVHN AUROC (test): {svhn_auroc_test:.4f}")
    print(f"CIFAR100 AUROC (test): {cifar100_auroc_test:.4f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(list(results.keys()))
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in results.values()])

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
