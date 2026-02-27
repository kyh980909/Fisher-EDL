import argparse
import csv
import math
import os
import pathlib
import sys
from collections import defaultdict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fisher_edl.metrics import compute_ci95


NUMERIC_FIELDS = [
    "acc",
    "nll",
    "brier",
    "ece",
    "classwise_ece",
    "miscls_auroc",
    "aurc",
    "auroc",
    "aupr",
    "fpr95",
    "train_time",
    "infer_time",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create summary tables from benchmark CSVs")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated input CSV paths")
    parser.add_argument("--out-csv", type=str, default="results/tables_summary.csv")
    parser.add_argument("--out-md", type=str, default="results/tables_summary.md")
    return parser.parse_args()


def _float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _valid(values):
    return [v for v in values if not math.isnan(v)]


def _mean(values):
    vals = _valid(values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _std(values):
    vals = _valid(values)
    n = len(vals)
    if n <= 1:
        return 0.0 if n == 1 else float("nan")
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return var ** 0.5


def _load_rows(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] missing input: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset_id", ""),
            row.get("dataset_ood", ""),
            row.get("ood_group", ""),
            row.get("method", ""),
            row.get("score_type", ""),
            row.get("calibration", ""),
        )
        grouped[key].append(row)

    out_rows = []
    for key, group_rows in grouped.items():
        dataset_id, dataset_ood, group, method, score_type, calibration = key
        out = {
            "dataset_id": dataset_id,
            "dataset_ood": dataset_ood,
            "ood_group": group,
            "method": method,
            "score_type": score_type,
            "calibration": calibration,
            "n_runs": len(group_rows),
        }
        for field in NUMERIC_FIELDS:
            values = [_float_or_nan(r.get(field, "nan")) for r in group_rows]
            out[f"{field}_mean"] = _mean(values)
            out[f"{field}_std"] = _std(values)
            out[f"{field}_ci95"] = compute_ci95(values)
        out_rows.append(out)

    out_rows.sort(key=lambda r: (r["dataset_id"], r["dataset_ood"], r["method"]))
    return out_rows


def _aggregate_by_ood_group(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        key = (row["dataset_id"], row["ood_group"], row["method"], row["score_type"], row["calibration"])
        grouped[key].append(row)

    out_rows = []
    for key, rows in grouped.items():
        dataset_id, group, method, score_type, calibration = key
        out = {
            "dataset_id": dataset_id,
            "dataset_ood": f"{group}_aggregate",
            "ood_group": group,
            "method": method,
            "score_type": score_type,
            "calibration": calibration,
            "n_runs": sum(int(r["n_runs"]) for r in rows),
        }
        for field in NUMERIC_FIELDS:
            means = [float(r[f"{field}_mean"]) for r in rows if not math.isnan(float(r[f"{field}_mean"]))]
            out[f"{field}_mean"] = _mean(means)
            out[f"{field}_std"] = _std(means)
            out[f"{field}_ci95"] = compute_ci95(means)
        out_rows.append(out)
    out_rows.sort(key=lambda r: (r["dataset_id"], r["ood_group"], r["method"]))
    return out_rows


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = v
            writer.writerow(out)


def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4f}"
    return str(x)


def _write_md(path, summary_rows, group_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write("## Per-dataset OOD\n\n")
        f.write("| dataset_id | dataset_ood | group | method | acc | ece | auroc | aupr | fpr95 | n |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in summary_rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        row["dataset_id"],
                        row["dataset_ood"],
                        row["ood_group"],
                        row["method"],
                        _fmt(row["acc_mean"]),
                        _fmt(row["ece_mean"]),
                        _fmt(row["auroc_mean"]),
                        _fmt(row["aupr_mean"]),
                        _fmt(row["fpr95_mean"]),
                        str(row["n_runs"]),
                    ]
                )
                + " |\n"
            )

        f.write("\n## Near/Far Aggregate\n\n")
        f.write("| dataset_id | ood_group | method | auroc | aupr | fpr95 | n |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in group_rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        row["dataset_id"],
                        row["ood_group"],
                        row["method"],
                        _fmt(row["auroc_mean"]),
                        _fmt(row["aupr_mean"]),
                        _fmt(row["fpr95_mean"]),
                        str(row["n_runs"]),
                    ]
                )
                + " |\n"
            )


def main():
    args = parse_args()
    input_paths = [x.strip() for x in args.inputs.split(",") if x.strip()]
    rows = _load_rows(input_paths)
    if not rows:
        print("No rows loaded. Check --inputs paths.")
        return

    summary_rows = _summarize(rows)
    group_rows = _aggregate_by_ood_group(summary_rows)

    _write_csv(args.out_csv, summary_rows + group_rows)
    _write_md(args.out_md, summary_rows, group_rows)

    print(f"Wrote CSV summary -> {args.out_csv}")
    print(f"Wrote Markdown summary -> {args.out_md}")


if __name__ == "__main__":
    main()
