import argparse
import csv
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Run paired statistical tests on benchmark results")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--ref-method", type=str, default="fisher")
    parser.add_argument("--metric", type=str, default="auroc")
    parser.add_argument("--out-md", type=str, default="results/stat_tests.md")
    return parser.parse_args()


def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _paired_samples(rows, ref_method, metric):
    by_key = defaultdict(dict)
    methods = set()

    for row in rows:
        dataset_id = row.get("dataset_id", "")
        dataset_ood = row.get("dataset_ood", "")
        seed = row.get("seed", "")
        method = row.get("method", "")
        val = _f(row.get(metric, "nan"))
        if math.isnan(val):
            continue
        key = (dataset_id, dataset_ood, seed)
        by_key[key][method] = val
        methods.add(method)

    methods = sorted(methods)
    out = {}
    for method in methods:
        if method == ref_method:
            continue
        pairs = defaultdict(lambda: ([], []))
        for (dataset_id, dataset_ood, _seed), method_vals in by_key.items():
            if ref_method not in method_vals or method not in method_vals:
                continue
            a, b = pairs[(dataset_id, dataset_ood)]
            a.append(method_vals[ref_method])
            b.append(method_vals[method])
        out[method] = pairs
    return out


def _paired_ttest(a, b):
    # Basic paired t-test without scipy fallback.
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        return 0.0
    std = var ** 0.5
    t = mean / (std / (n ** 0.5))
    return t


def _try_scipy_tests(a, b):
    try:
        from scipy.stats import ttest_rel, wilcoxon

        t_res = ttest_rel(a, b, nan_policy="omit")
        try:
            w_res = wilcoxon(a, b)
            w_p = float(w_res.pvalue)
        except Exception:
            w_p = float("nan")
        return float(t_res.statistic), float(t_res.pvalue), w_p
    except Exception:
        return _paired_ttest(a, b), float("nan"), float("nan")


def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.6f}"
    return str(x)


def main():
    args = parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    paired = _paired_samples(rows, args.ref_method, args.metric)

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# Paired Statistical Tests ({args.metric})\n\n")
        f.write(f"Reference method: `{args.ref_method}`\n\n")
        f.write("| dataset_id | dataset_ood | compare_method | n | mean_ref | mean_cmp | mean_diff | t_stat | t_pvalue | wilcoxon_p |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")

        for method, per_dataset in sorted(paired.items()):
            for (dataset_id, dataset_ood), (ref_vals, cmp_vals) in sorted(per_dataset.items()):
                if len(ref_vals) == 0:
                    continue
                t_stat, t_p, w_p = _try_scipy_tests(ref_vals, cmp_vals)
                mean_ref = sum(ref_vals) / len(ref_vals)
                mean_cmp = sum(cmp_vals) / len(cmp_vals)
                mean_diff = mean_ref - mean_cmp
                f.write(
                    "| "
                    + " | ".join(
                        [
                            dataset_id,
                            dataset_ood,
                            method,
                            str(len(ref_vals)),
                            _fmt(mean_ref),
                            _fmt(mean_cmp),
                            _fmt(mean_diff),
                            _fmt(t_stat),
                            _fmt(t_p),
                            _fmt(w_p),
                        ]
                    )
                    + " |\n"
                )

    print(f"Saved statistical test report -> {args.out_md}")


if __name__ == "__main__":
    main()
