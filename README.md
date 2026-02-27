# Fisher-EDL / Info-EDL Benchmark Suite

This repository now supports journal-oriented benchmark workflows for:

- CIFAR-10 ID + multi-OOD evaluation (Near/Far split)
- ImageNet-100 ID + folder-based OOD evaluation
- CE / MC-Dropout / EDL / Fisher-EDL baselines
- Extended uncertainty metrics (NLL, Brier, ECE, classwise ECE, AUPR, AURC)

## Setup

Requires Python 3, PyTorch, torchvision, matplotlib.

## Quick Start (CIFAR)

Train and immediately evaluate Fisher-EDL on CIFAR-10:

```bash
python run_cifar_experiment.py \
  --method fisher \
  --backbone resnet18 \
  --epochs 100 \
  --ood-datasets svhn,cifar100,texture,lsun-crop,lsun-resize,isun,places365 \
  --score-type uncertainty \
  --calibration temperature \
  --eval-after-train
```

Multi-seed run:

```bash
python run_cifar_experiment.py --method fisher --num-seeds 5 --eval-after-train
```

## CIFAR Evaluation

Evaluate a checkpoint with the common CSV schema:

```bash
python run_cifar_eval.py \
  --ckpt runs/cifar_fisher_YYYYMMDD_HHMMSS/best_val_acc.pt \
  --ood-datasets svhn,cifar100,texture \
  --score-type uncertainty \
  --calibration temperature \
  --out-csv results/cifar_eval.csv
```

Evaluate Deep Ensemble checkpoints:

```bash
python run_cifar_eval.py \
  --ensemble-ckpts runs/m1/best_val_acc.pt,runs/m2/best_val_acc.pt,runs/m3/best_val_acc.pt \
  --method deep_ensemble \
  --out-csv results/deep_ensemble_eval.csv
```

## ImageNet-100 Benchmark

Expected folder layout:

- `data/imagenet100/train/<class_name>/*.jpg`
- `data/imagenet100/val/<class_name>/*.jpg`
- `data/ood/<ood_name>/<class_name>/*.jpg` for each OOD dataset

Run:

```bash
python run_imagenet100_experiment.py \
  --method fisher \
  --ood-datasets inaturalist,texture,openimage-o,imagenet-o \
  --eval-out-csv results/imagenet100_eval.csv
```

## Full Suite Automation

Run the benchmark matrix:

```bash
python run_benchmark_suite.py \
  --suite cifar \
  --methods ce,mcdropout,edl,fisher \
  --seeds 1234,2345,3456 \
  --include-deep-ensemble
```

Dry-run command preview:

```bash
python run_benchmark_suite.py --suite both --dry-run
```

## Info-EDL Hyperparameter Search

Run local sweep for Info-EDL (`method=fisher`) with automatic CSV aggregation:

```bash
python scripts/run_infoedl_sweep.py \
  --space-yaml sweep_infoedl.yaml \
  --mode grid \
  --seeds 1234,2345,3456 \
  --epochs 100 \
  --batch-size 256 \
  --ood-datasets svhn,cifar100,texture,lsun-crop,lsun-resize,isun,places365 \
  --skip-missing-ood \
  --wandb --wandb-project info-edl
```

Quick sanity check without training:

```bash
python scripts/run_infoedl_sweep.py --space-yaml sweep_infoedl.yaml --max-trials 1 --seeds 1234 --dry-run
```

Outputs:

- Row-level merged results: `results/infoedl_sweep/summary_all.csv`
- Trial-level summary: `results/infoedl_sweep/summary_by_trial.csv`

## Reporting Utilities

Create summary tables (mean/std/95% CI):

```bash
python scripts/make_tables.py \
  --inputs results/benchmark_suite/cifar/summary_all.csv \
  --out-csv results/tables_summary.csv \
  --out-md results/tables_summary.md
```

Run paired statistical tests (t-test / Wilcoxon if SciPy available):

```bash
python scripts/stat_tests.py \
  --input-csv results/benchmark_suite/cifar/summary_all.csv \
  --ref-method fisher \
  --metric auroc \
  --out-md results/stat_tests.md
```

Create figures (AUROC, near/far, heatmap, dynamics):

```bash
python scripts/plot_main_figures.py \
  --eval-csv results/benchmark_suite/cifar/summary_all.csv \
  --train-metrics-glob 'runs/**/metrics.csv' \
  --out-dir results/figures
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Legacy Toy Example

Toy 2D experiment is still available:

```bash
python run_experiment.py --method fisher --beta 1.0 --gamma 1.0
```
