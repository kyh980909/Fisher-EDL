import argparse
import datetime
import os
import torch

from fisher_edl.data import build_loaders
from fisher_edl.model import MLP
from fisher_edl.train import TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Fisher-EDL toy experiments")
    parser.add_argument("--method", choices=["edl", "fisher"], default="fisher")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight or base weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Fisher weight decay")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--ood-train", type=int, default=0)
    parser.add_argument("--ood-val", type=int, default=200)
    parser.add_argument("--hard-overlap", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, val_loader = build_loaders(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        ood_train=args.ood_train,
        ood_val=args.ood_val,
        hard_overlap=args.hard_overlap,
        seed=args.seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_dim=64, num_classes=args.num_classes)

    if args.run_dir:
        run_dir = args.run_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"{args.method}_{stamp}")

    cfg = TrainConfig(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        device=device,
    )

    print(f"Running {args.method} on {device}. logs -> {run_dir}")
    train(model, train_loader, val_loader, args.num_classes, cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
