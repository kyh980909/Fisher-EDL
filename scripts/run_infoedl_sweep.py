import argparse
import csv
import itertools
import os
import random
import subprocess
import sys
from collections import defaultdict

DEFAULTS = {
    "beta": 1.0,
    "gamma": 1.0,
    "info-type": "fisher",
    "gate-type": "exp",
    "detach-weight": False,
    "objective": "risk_plus_kl",
    "backbone": "resnet18",
}


def _split_csv(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _safe_slug(parts):
    text = "_".join(parts)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in text)


def _load_yaml(path):
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for --space-yaml. Install with `pip install pyyaml`."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML format: {path}")
    return data


def _space_values(space, key, default_value):
    params = space.get("parameters", {})
    spec = params.get(key)
    if spec is None:
        spec = params.get(key.replace("-", "_"))
    if spec is None:
        return [default_value]
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        if "value" in spec:
            return [spec["value"]]
    return [default_value]


def parse_args():
    p = argparse.ArgumentParser(description="Info-EDL hyperparameter sweep runner (local)")
    p.add_argument("--space-yaml", type=str, default="sweep_grid.yaml")
    p.add_argument("--mode", choices=["grid", "random"], default="grid")
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=1234)
    p.add_argument(
        "--tunable",
        type=str,
        default="beta,gamma,info-type,gate-type,detach-weight,objective,backbone",
    )

    p.add_argument("--method", choices=["fisher"], default="fisher")
    p.add_argument("--seeds", type=str, default="1234,2345,3456")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ood-datasets", type=str, default="svhn,cifar100")
    p.add_argument("--score-type", choices=["uncertainty", "energy", "msp"], default="uncertainty")
    p.add_argument("--calibration", choices=["none", "temperature"], default="none")
    p.add_argument("--skip-missing-ood", action="store_true")

    p.add_argument("--run-root", type=str, default="runs/infoedl_sweep")
    p.add_argument("--result-root", type=str, default="results/infoedl_sweep")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="info-edl")
    return p.parse_args()


def _build_trials(args, space):
    tunable = _split_csv(args.tunable)
    values_by_key = {}
    for key in tunable:
        values = _space_values(space, key, DEFAULTS.get(key))
        if key == "detach-weight":
            values = [_to_bool(v) for v in values]
        values_by_key[key] = values

    keys = list(values_by_key.keys())
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*[values_by_key[k] for k in keys])]

    if args.mode == "random" and args.max_trials is not None and args.max_trials < len(grid):
        rng = random.Random(args.random_seed)
        rng.shuffle(grid)
        grid = grid[: args.max_trials]
    elif args.max_trials is not None:
        grid = grid[: args.max_trials]

    return grid


def _run(cmd, dry_run=False):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _merge_row_level(records, out_csv):
    if not records:
        return
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = []
    for rec in records:
        for key in rec.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


def _make_trial_summary(records, out_csv):
    if not records:
        return
    num_keys = ["acc", "nll", "brier", "ece", "classwise_ece", "miscls_auroc", "aurc", "auroc", "aupr", "fpr95"]
    group = defaultdict(list)
    for r in records:
        group[r["trial_id"]].append(r)

    rows = []
    for trial_id, items in group.items():
        head = dict(items[0])
        out = {k: head.get(k) for k in ["trial_id", "method", "beta", "gamma", "info_type", "gate_type", "detach_weight", "objective", "backbone"]}
        out["n_rows"] = len(items)
        for key in num_keys:
            vals = []
            for it in items:
                try:
                    vals.append(float(it[key]))
                except Exception:
                    pass
            out[f"{key}_mean"] = sum(vals) / len(vals) if vals else float("nan")
        rows.append(out)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    space = _load_yaml(args.space_yaml)
    trials = _build_trials(args, space)
    seeds = [int(x) for x in _split_csv(args.seeds)]

    py = sys.executable
    os.makedirs(args.run_root, exist_ok=True)
    os.makedirs(args.result_root, exist_ok=True)

    row_level_records = []
    trial_count = 0

    for idx, hp in enumerate(trials, start=1):
        full = dict(DEFAULTS)
        full.update(hp)
        trial_id = _safe_slug(
            [
                f"t{idx:03d}",
                f"b{full.get('beta')}",
                f"g{full.get('gamma')}",
                f"i{full.get('info-type')}",
                f"gt{full.get('gate-type')}",
                f"dw{int(bool(full.get('detach-weight')))}",
                f"obj{full.get('objective')}",
                f"bb{full.get('backbone')}",
            ]
        )
        trial_count += 1

        for seed in seeds:
            run_dir = os.path.join(args.run_root, trial_id, f"seed_{seed}")
            out_csv = os.path.join(args.result_root, f"{trial_id}_seed{seed}.csv")

            if args.skip_existing and os.path.exists(out_csv):
                print(f"skip existing: {out_csv}")
            else:
                cmd = [
                    py,
                    "run_cifar_experiment.py",
                    "--method",
                    args.method,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--data-root",
                    args.data_root,
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(seed),
                    "--num-seeds",
                    "1",
                    "--run-dir",
                    run_dir,
                    "--id-dataset",
                    "cifar10",
                    "--ood-datasets",
                    args.ood_datasets,
                    "--score-type",
                    args.score_type,
                    "--calibration",
                    args.calibration,
                    "--eval-after-train",
                    "--eval-out-csv",
                    out_csv,
                    "--beta",
                    str(full.get("beta")),
                    "--gamma",
                    str(full.get("gamma")),
                    "--info-type",
                    str(full.get("info-type")),
                    "--gate-type",
                    str(full.get("gate-type")),
                    "--objective",
                    str(full.get("objective")),
                    "--backbone",
                    str(full.get("backbone")),
                ]
                if bool(full.get("detach-weight")):
                    cmd.append("--detach-weight")
                if args.skip_missing_ood:
                    cmd.append("--skip-missing-ood")
                if args.wandb:
                    cmd.extend(
                        [
                            "--wandb",
                            "--wandb-project",
                            args.wandb_project,
                            "--wandb-name",
                            f"{trial_id}_seed{seed}",
                        ]
                    )
                _run(cmd, dry_run=args.dry_run)

            if not os.path.exists(out_csv):
                continue

            with open(out_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["trial_id"] = trial_id
                    row["beta"] = full.get("beta")
                    row["gamma"] = full.get("gamma")
                    row["info_type"] = full.get("info-type")
                    row["gate_type"] = full.get("gate-type")
                    row["detach_weight"] = int(bool(full.get("detach-weight")))
                    row["objective"] = full.get("objective")
                    row["backbone"] = full.get("backbone")
                    row_level_records.append(row)

    row_csv = os.path.join(args.result_root, "summary_all.csv")
    trial_csv = os.path.join(args.result_root, "summary_by_trial.csv")
    if not args.dry_run:
        _merge_row_level(row_level_records, row_csv)
        _make_trial_summary(row_level_records, trial_csv)
        print(f"trials: {trial_count}")
        print(f"row-level summary -> {row_csv}")
        print(f"trial summary -> {trial_csv}")


if __name__ == "__main__":
    main()
