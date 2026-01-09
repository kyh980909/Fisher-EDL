import argparse
import glob
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all best_val_acc checkpoints in runs/")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--out-dir", type=str, default="results/all_eval")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ckpts = glob.glob(os.path.join(runs_dir, "cifar_*", "best_val_acc.pt"))
    ckpts.sort()
    print(f"found {len(ckpts)} checkpoints")

    for i, ckpt in enumerate(ckpts, 1):
        run_dir = os.path.basename(os.path.dirname(ckpt))
        out_csv = os.path.join(out_dir, f"{run_dir}_best_val_acc.csv")
        if args.skip_existing and os.path.exists(out_csv):
            print(f"[{i}/{len(ckpts)}] skip {out_csv}")
            continue
        cmd = [
            args.python,
            "run_cifar_eval.py",
            "--ckpt",
            ckpt,
            "--backbone",
            "auto",
            "--num-workers",
            str(args.num_workers),
            "--out-csv",
            out_csv,
        ]
        print(f"[{i}/{len(ckpts)}] " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=os.path.dirname(runs_dir))


if __name__ == "__main__":
    main()
