import argparse
import csv
import glob
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoints by val_acc")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--method", choices=["edl", "fisher", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backbone", choices=["simple", "resnet18", "auto"], default=None)
    parser.add_argument("--auto-backbone", action="store_true", help="Infer backbone from checkpoint")
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args()


def _best_val_from_metrics(metrics_path):
    best_val = None
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_acc = row.get("val_acc")
            if val_acc is None:
                continue
            try:
                val_acc = float(val_acc)
            except ValueError:
                continue
            if best_val is None or val_acc > best_val:
                best_val = val_acc
    return best_val


def main():
    args = parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    entries = []

    for run_dir in glob.glob(os.path.join(runs_dir, "cifar_*")):
        if not os.path.isdir(run_dir):
            continue
        if args.method != "all" and args.method not in os.path.basename(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.csv")
        ckpt_path = os.path.join(run_dir, "best_val_acc.pt")
        if not os.path.exists(metrics_path) or not os.path.exists(ckpt_path):
            continue
        best_val = _best_val_from_metrics(metrics_path)
        if best_val is None:
            continue
        entries.append((best_val, run_dir, ckpt_path))

    entries.sort(key=lambda x: x[0], reverse=True)
    selected = entries[: args.top_k]
    if not selected:
        print("No runs found.")
        return

    for best_val, run_dir, ckpt_path in selected:
        backbone = args.backbone
        if backbone is None and args.auto_backbone:
            backbone = "auto"
        if backbone is None:
            backbone = "simple"

        print(f"Evaluating {ckpt_path} (best_val={best_val:.4f}, backbone={backbone})")
        cmd = [
            "python",
            "run_cifar_eval.py",
            "--ckpt",
            ckpt_path,
            "--backbone",
            backbone,
        ]
        if args.out_csv:
            base = os.path.splitext(os.path.basename(ckpt_path))[0]
            out_csv = os.path.join(args.out_csv, f"{os.path.basename(run_dir)}_{base}.csv")
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            cmd.extend(["--out-csv", out_csv])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
