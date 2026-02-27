import argparse
import csv
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark suites for Info-EDL")
    parser.add_argument("--suite", choices=["cifar", "imagenet100", "both"], default="cifar")
    parser.add_argument("--methods", type=str, default="ce,mcdropout,edl,fisher")
    parser.add_argument("--seeds", type=str, default="1234,2345,3456")

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--ood-datasets", type=str, default="svhn,cifar100,texture,lsun-crop,lsun-resize,isun,places365")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--run-root", type=str, default="runs/benchmark_suite")
    parser.add_argument("--result-root", type=str, default="results/benchmark_suite")

    parser.add_argument("--include-deep-ensemble", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=3)

    parser.add_argument("--score-type", choices=["uncertainty", "energy", "msp"], default="uncertainty")
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="info-edl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--skip-missing-ood", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _split_csv(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _run(cmd, dry_run=False):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _concat_csv(csv_paths, out_path):
    rows = []
    header = None
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                rows.append(row)
    if not rows or not header:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return True


def _run_cifar(args, py, methods, seeds):
    run_root = os.path.join(args.run_root, "cifar")
    result_root = os.path.join(args.result_root, "cifar")
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)

    csv_paths = []

    for method in methods:
        for seed in seeds:
            run_dir = os.path.join(run_root, f"{method}_seed{seed}")
            eval_csv = os.path.join(result_root, f"{method}_seed{seed}.csv")
            csv_paths.append(eval_csv)

            if args.skip_existing and os.path.exists(eval_csv):
                print(f"skip existing: {eval_csv}")
                continue

            cmd = [
                py,
                "run_cifar_experiment.py",
                "--method",
                method,
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
                eval_csv,
            ]
            if args.wandb:
                cmd.extend(
                    [
                        "--wandb",
                        "--wandb-project",
                        args.wandb_project,
                        "--wandb-name",
                        f"{method}_seed{seed}",
                    ]
                )
            if args.skip_missing_ood:
                cmd.append("--skip-missing-ood")
            _run(cmd, dry_run=args.dry_run)

    if args.include_deep_ensemble:
        for seed in seeds:
            member_ckpts = []
            for k in range(args.ensemble_size):
                member_seed = seed + k
                run_dir = os.path.join(run_root, f"deep_ensemble_seed{seed}_member{k}")
                ckpt_path = os.path.join(run_dir, "best_val_acc.pt")
                member_ckpts.append(ckpt_path)

                train_cmd = [
                    py,
                    "run_cifar_experiment.py",
                    "--method",
                    "ce",
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--data-root",
                    args.data_root,
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(member_seed),
                    "--num-seeds",
                    "1",
                    "--run-dir",
                    run_dir,
                ]
                if args.wandb:
                    train_cmd.extend(
                        [
                            "--wandb",
                            "--wandb-project",
                            args.wandb_project,
                            "--wandb-name",
                            f"deep_ensemble_member{k}_seed{member_seed}",
                        ]
                    )
                if not (args.skip_existing and os.path.exists(ckpt_path)):
                    _run(train_cmd, dry_run=args.dry_run)

            eval_csv = os.path.join(result_root, f"deep_ensemble_seed{seed}.csv")
            csv_paths.append(eval_csv)
            if args.skip_existing and os.path.exists(eval_csv):
                print(f"skip existing: {eval_csv}")
                continue

            eval_cmd = [
                py,
                "run_cifar_eval.py",
                "--ensemble-ckpts",
                ",".join(member_ckpts),
                "--method",
                "deep_ensemble",
                "--id-dataset",
                "cifar10",
                "--ood-datasets",
                args.ood_datasets,
                "--data-root",
                args.data_root,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(seed),
                "--score-type",
                args.score_type,
                "--calibration",
                args.calibration,
                "--out-csv",
                eval_csv,
            ]
            if args.wandb:
                eval_cmd.extend(
                    [
                        "--wandb",
                        "--wandb-project",
                        args.wandb_project,
                        "--wandb-name",
                        f"deep_ensemble_eval_seed{seed}",
                    ]
                )
            if args.skip_missing_ood:
                eval_cmd.append("--skip-missing-ood")
            _run(eval_cmd, dry_run=args.dry_run)

    merged = os.path.join(result_root, "summary_all.csv")
    if not args.dry_run:
        if _concat_csv(csv_paths, merged):
            print(f"merged summary -> {merged}")


def _run_imagenet100(args, py, methods, seeds):
    run_root = os.path.join(args.run_root, "imagenet100")
    result_root = os.path.join(args.result_root, "imagenet100")
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)

    csv_paths = []

    # For ImageNet-100, keep OOD defaults aligned with E2.
    ood_datasets = "inaturalist,texture,openimage-o,imagenet-o"

    for method in methods:
        for seed in seeds:
            run_dir = os.path.join(run_root, f"{method}_seed{seed}")
            eval_csv = os.path.join(result_root, f"{method}_seed{seed}.csv")
            csv_paths.append(eval_csv)

            if args.skip_existing and os.path.exists(eval_csv):
                print(f"skip existing: {eval_csv}")
                continue

            cmd = [
                py,
                "run_imagenet100_experiment.py",
                "--method",
                method,
                "--epochs",
                str(max(10, args.epochs // 2)),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--data-root",
                args.data_root,
                "--seed",
                str(seed),
                "--ood-datasets",
                ood_datasets,
                "--score-type",
                args.score_type,
                "--calibration",
                args.calibration,
                "--run-dir",
                run_dir,
                "--eval-out-csv",
                eval_csv,
            ]
            if args.wandb:
                cmd.extend(
                    [
                        "--wandb",
                        "--wandb-project",
                        args.wandb_project,
                        "--wandb-name",
                        f"imagenet100_{method}_seed{seed}",
                    ]
                )
            _run(cmd, dry_run=args.dry_run)

    merged = os.path.join(result_root, "summary_all.csv")
    if not args.dry_run:
        if _concat_csv(csv_paths, merged):
            print(f"merged summary -> {merged}")


def main():
    args = parse_args()
    py = sys.executable

    methods = _split_csv(args.methods, str)
    seeds = _split_csv(args.seeds, int)

    if args.suite in {"cifar", "both"}:
        _run_cifar(args, py, methods, seeds)

    if args.suite in {"imagenet100", "both"}:
        _run_imagenet100(args, py, methods, seeds)


if __name__ == "__main__":
    main()
