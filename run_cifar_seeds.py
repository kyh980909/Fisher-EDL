import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDL/FEDL across multiple seeds")
    parser.add_argument("--method", choices=["edl", "fisher", "both"], default="both")
    parser.add_argument("--seeds", type=str, default="1234,2345,3456")
    parser.add_argument("--backbone", choices=["simple", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--run-dir", type=str, default="runs/seed_sweep")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


def _parse_ints(csv_str):
    return [int(x.strip()) for x in csv_str.split(",") if x.strip()]


def main():
    args = parse_args()
    seeds = _parse_ints(args.seeds)
    methods = ["edl", "fisher"] if args.method == "both" else [args.method]

    os.makedirs(args.run_dir, exist_ok=True)
    py = sys.executable

    for method in methods:
        for seed in seeds:
            run_dir = os.path.join(args.run_dir, f"{method}_seed{seed}")
            cmd = [
                py,
                "run_cifar_experiment.py",
                "--method",
                method,
                "--beta",
                "1.0",
                "--gamma",
                "1.0",
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--data-root",
                args.data_root,
                "--num-workers",
                str(args.num_workers),
                "--val-split",
                str(args.val_split),
                "--backbone",
                args.backbone,
                "--run-dir",
                run_dir,
            ]
            if args.wandb:
                cmd.extend([
                    "--wandb",
                    "--wandb-project",
                    args.wandb_project or "fisher-edl",
                    "--wandb-name",
                    f"{method}_seed{seed}",
                ])
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()