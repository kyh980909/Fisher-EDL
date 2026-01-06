import argparse
import datetime
import os
import torch

from fisher_edl.cifar_data import (
    build_cifar10_loaders,
    build_svhn_loader,
    build_cifar100_loader,
)
from fisher_edl.cifar_model import build_cifar_model
from fisher_edl.train_cifar import CifarTrainConfig, train_cifar


def parse_args():
    parser = argparse.ArgumentParser(description="Fisher-EDL CIFAR experiments")
    parser.add_argument("--method", choices=["edl", "fisher"], default="fisher")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--data_root", dest="data_root", type=str)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num_workers", dest="num_workers", type=int)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val_split", dest="val_split", type=float)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run_dir", dest="run_dir", type=str)
    parser.add_argument("--backbone", choices=["simple", "resnet18"], default="simple")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb_project", dest="wandb_project", type=str)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb_name", dest="wandb_name", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, test_transform = build_cifar10_loaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        val_split=args.val_split,
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

    if args.run_dir:
        run_dir = args.run_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"cifar_{args.method}_{stamp}")

    if args.wandb_name:
        wandb_name = args.wandb_name
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_name = f"{args.method}_lr{args.lr}_b{args.beta}_g{args.gamma}_{stamp}"

    cfg = CifarTrainConfig(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_name,
    )

    print(f"Running {args.method} on {device}. logs -> {run_dir}")
    if val_loader is None:
        val_loader = test_loader

    train_cifar(
        model,
        train_loader,
        val_loader,
        test_loader,
        svhn_loader,
        cifar100_loader,
        num_classes=10,
        cfg=cfg,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
