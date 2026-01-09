import argparse
import csv
import json
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize W&B runs and flag collapsed EDL runs")
    parser.add_argument("--wandb-dir", type=str, default="wandb")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.9)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    return parser.parse_args()


def _load_config(cfg_path):
    cfg = {}
    if not os.path.exists(cfg_path):
        return cfg
    current = None
    with open(cfg_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0].strip()
                current = key
            else:
                if current and "value:" in line:
                    val = line.split("value:", 1)[1].strip()
                    cfg[current] = val
    return cfg


def main():
    args = parse_args()
    base = os.path.abspath(args.wandb_dir)
    records = []

    for path in glob.glob(os.path.join(base, "run-*", "files", "wandb-summary.json")):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, "r") as f:
            summary = json.load(f)
        cfg_path = os.path.join(os.path.dirname(path), "config.yaml")
        cfg = _load_config(cfg_path)
        records.append((run_id, cfg, summary))

    rows = []
    for run_id, cfg, summary in records:
        method = cfg.get("method")
        beta = cfg.get("beta")
        gamma = cfg.get("gamma")
        train_unc = summary.get("Uncertainty/Train_Mean")
        val_acc = summary.get("val_acc")
        train_acc = summary.get("train_acc")
        loss = summary.get("Loss/Total")
        collapsed = False
        if method == "edl" and train_unc is not None and train_unc >= args.uncertainty_threshold:
            collapsed = True

        rows.append({
            "run_id": run_id,
            "method": method,
            "beta": beta,
            "gamma": gamma,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_uncertainty": train_unc,
            "loss_total": loss,
            "collapsed": collapsed,
        })

    rows.sort(key=lambda r: (r["method"] or "", r["val_acc"] or -1), reverse=True)

    collapsed_rows = [r for r in rows if r["collapsed"]]
    print(f"Runs scanned: {len(rows)}")
    print(f"Collapsed EDL runs (threshold={args.uncertainty_threshold}): {len(collapsed_rows)}")
    for r in collapsed_rows[:10]:
        print(
            f"{r['run_id']} method={r['method']} beta={r['beta']} gamma={r['gamma']} "
            f"val_acc={r['val_acc']} train_unc={r['train_uncertainty']}"
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write("| run_id | method | beta | gamma | train_acc | val_acc | train_uncertainty | loss_total | collapsed |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for r in rows:
                f.write(
                    f"| {r['run_id']} | {r['method']} | {r['beta']} | {r['gamma']} | {r['train_acc']} | {r['val_acc']} | "
                    f"{r['train_uncertainty']} | {r['loss_total']} | {r['collapsed']} |\n"
                )


if __name__ == "__main__":
    main()
