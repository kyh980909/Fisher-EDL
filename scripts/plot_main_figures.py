import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot main paper figures from benchmark CSVs")
    parser.add_argument("--eval-csv", type=str, required=True)
    parser.add_argument("--train-metrics-glob", type=str, default="runs/**/metrics.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    return parser.parse_args()


def _read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean(xs):
    vals = [x for x in xs if not math.isnan(x)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def plot_main_auroc(rows, out_path):
    grouped = defaultdict(list)
    for r in rows:
        ds = r.get("dataset_ood", "")
        if ds == "id_only":
            continue
        key = (r.get("method", ""), ds)
        grouped[key].append(_f(r.get("auroc", "nan")))

    methods = sorted({k[0] for k in grouped.keys()})
    datasets = sorted({k[1] for k in grouped.keys()})
    if not methods or not datasets:
        return

    width = 0.8 / max(1, len(methods))
    x = list(range(len(datasets)))

    plt.figure(figsize=(max(8, len(datasets) * 1.3), 4.5))
    for m_idx, method in enumerate(methods):
        vals = [_mean(grouped.get((method, ds), [])) for ds in datasets]
        xpos = [v + (m_idx - (len(methods) - 1) / 2.0) * width for v in x]
        plt.bar(xpos, vals, width=width, label=method)

    plt.xticks(x, datasets, rotation=25, ha="right")
    plt.ylabel("AUROC")
    plt.ylim(0.0, 1.0)
    plt.title("OOD AUROC by Dataset and Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_near_far(rows, out_path):
    grouped = defaultdict(list)
    for r in rows:
        g = r.get("ood_group", "")
        if g not in {"near", "far"}:
            continue
        grouped[(r.get("method", ""), g)].append(_f(r.get("auroc", "nan")))

    methods = sorted({k[0] for k in grouped.keys()})
    groups = ["near", "far"]
    if not methods:
        return

    width = 0.35
    x = list(range(len(groups)))

    plt.figure(figsize=(7, 4))
    for m_idx, method in enumerate(methods):
        vals = [_mean(grouped.get((method, g), [])) for g in groups]
        xpos = [v + (m_idx - (len(methods) - 1) / 2.0) * (width / max(1, len(methods) / 2.0)) for v in x]
        plt.bar(xpos, vals, width=width / max(1, len(methods) / 2.0), label=method)

    plt.xticks(x, groups)
    plt.ylabel("AUROC")
    plt.ylim(0.0, 1.0)
    plt.title("Near vs Far OOD AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _read_config_txt(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def plot_fisher_heatmap(rows, out_path):
    # Uses ckpt -> run_dir/config.txt to recover beta/gamma.
    bucket = defaultdict(list)
    betas = set()
    gammas = set()

    for r in rows:
        if r.get("method") != "fisher":
            continue
        ckpt = r.get("ckpt", "")
        if not ckpt:
            continue
        run_dir = os.path.dirname(ckpt.split(";")[0])
        cfg = _read_config_txt(os.path.join(run_dir, "config.txt"))
        beta = _f(cfg.get("beta", "nan"))
        gamma = _f(cfg.get("gamma", "nan"))
        auroc = _f(r.get("auroc", "nan"))
        if math.isnan(beta) or math.isnan(gamma) or math.isnan(auroc):
            continue
        betas.add(beta)
        gammas.add(gamma)
        bucket[(beta, gamma)].append(auroc)

    if not bucket:
        return

    betas = sorted(betas)
    gammas = sorted(gammas)
    matrix = [[_mean(bucket.get((b, g), [])) for g in gammas] for b in betas]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="AUROC")
    plt.xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    plt.yticks(range(len(betas)), [f"{b:g}" for b in betas])
    plt.xlabel("gamma")
    plt.ylabel("beta")
    plt.title("Fisher Hyperparameter Heatmap (AUROC)")

    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            val = matrix[i][j]
            if not math.isnan(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dynamics(train_metrics_glob, out_path):
    paths = sorted(glob.glob(train_metrics_glob, recursive=True))
    if not paths:
        return

    target = None
    for path in paths:
        # Prefer fisher runs because they contain lambda/info columns.
        if "cifar_fisher" in path or "fisher" in path:
            target = path
            break
    if target is None:
        target = paths[0]

    rows = _read_csv(target)
    if not rows:
        return

    epochs = [_f(r.get("epoch", "nan")) for r in rows]
    lambdas = [_f(r.get("weight", "nan")) for r in rows]
    infos = [_f(r.get("info", "nan")) for r in rows]
    unc = [_f(r.get("val_uncertainty", "nan")) for r in rows]
    grad = [_f(r.get("grad_norm", "nan")) for r in rows]

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, lambdas)
    plt.title("Lambda Mean")
    plt.xlabel("epoch")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, infos)
    plt.title("Info Mean")
    plt.xlabel("epoch")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, unc)
    plt.title("Val Uncertainty")
    plt.xlabel("epoch")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, grad)
    plt.title("Gradient Norm")
    plt.xlabel("epoch")

    plt.suptitle(f"Training Dynamics ({target})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = _read_csv(args.eval_csv)
    if not rows:
        print(f"No rows in {args.eval_csv}")
        return

    plot_main_auroc(rows, os.path.join(args.out_dir, "main_auroc.png"))
    plot_near_far(rows, os.path.join(args.out_dir, "near_far_auroc.png"))
    plot_fisher_heatmap(rows, os.path.join(args.out_dir, "fisher_heatmap_auroc.png"))
    plot_dynamics(args.train_metrics_glob, os.path.join(args.out_dir, "training_dynamics.png"))

    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
