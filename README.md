# Fisher-EDL Toy Experiments

This repo provides a minimal, self-contained experiment harness to test
Fisher-Weighted Evidential Loss on a synthetic 2D classification task.
It includes a baseline EDL loss and a Fisher-adaptive variant.

## CIFAR-10 / OOD Experiments

Train on CIFAR-10 (Known), evaluate Far-OOD on SVHN and Near-OOD on CIFAR-100.

```
python run_cifar_experiment.py --method edl --beta 1.0
```

```
python run_cifar_experiment.py --method fisher --beta 1.0 --gamma 1.0
```

The script logs CSV metrics to `runs/cifar_<method>_<timestamp>/metrics.csv`,
including AUROC for OOD detection (SVHN vs CIFAR-10, CIFAR-100 vs CIFAR-10)
using evidential uncertainty as the score.

Backbone selection:

```
python run_cifar_experiment.py --backbone resnet18
```

Evaluate a saved checkpoint:

```
python run_cifar_eval.py --ckpt runs/cifar_fisher_YYYYMMDD_HHMMSS/best_val_acc.pt
```

Save eval metrics to CSV:

```
python run_cifar_eval.py --ckpt runs/cifar_fisher_YYYYMMDD_HHMMSS/best_val_acc.pt --out-csv runs/eval_results.csv
```

Log eval metrics to W&B:

```
python run_cifar_eval.py --ckpt runs/cifar_fisher_YYYYMMDD_HHMMSS/best_val_acc.pt --wandb --wandb-project fisher-edl
```

Colab notebook:

- `colab_fisher_edl.ipynb`

By default, 10% of CIFAR-10 train is held out for validation and logged as
`val_acc` during training. Test accuracy is logged as `test_acc`. Control the
split with `--val-split` (set to 0 to disable).

Enable Weights & Biases logging:

```
python run_cifar_experiment.py --method fisher --wandb --wandb-project fisher-edl
```

## Setup

This code requires Python 3 and PyTorch.

## Run

Baseline EDL:

```
python run_experiment.py --method edl --beta 1.0
```

Fisher-EDL:

```
python run_experiment.py --method fisher --beta 1.0 --gamma 1.0
```

Hard-negative emphasis (more overlap):

```
python run_experiment.py --method fisher --hard-overlap 0.6
```

OOD validation points:

```
python run_experiment.py --method fisher --ood-val 400
```

Metrics are written to `runs/<method>_<timestamp>/metrics.csv`.

## Notes

- The dataset is synthetic: 2D Gaussian clusters with optional overlap
  (hard negatives) and uniform OOD points.
- The Fisher weight uses the Dirichlet trace approximation
  `sum trigamma(alpha) - K * trigamma(sum alpha)`.
- The loss uses the EDL MSE risk + Dirichlet KL to uniform prior.

## Files

- `fisher_edl/losses.py`: EDL and Fisher-EDL losses
- `fisher_edl/data.py`: Toy data generation and loaders
- `fisher_edl/model.py`: Simple MLP model
- `fisher_edl/train.py`: Training loop and CSV logging
- `run_experiment.py`: CLI entry point
- `fisher_edl/cifar_data.py`: CIFAR-10/SVHN/CIFAR-100 loaders
- `fisher_edl/cifar_model.py`: Simple CNN model for CIFAR
- `fisher_edl/metrics.py`: Uncertainty + AUROC utilities
- `fisher_edl/train_cifar.py`: CIFAR training + OOD eval loop
- `run_cifar_experiment.py`: CIFAR experiment CLI
