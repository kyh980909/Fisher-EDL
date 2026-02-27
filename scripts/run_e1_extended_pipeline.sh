#!/usr/bin/env bash
set -euo pipefail

cd /home/yongho/fedl
PY=/home/yongho/miniconda3/envs/openmmlab/bin/python

OODS="svhn,cifar100,texture,lsun-crop,lsun-resize,isun,places365"
RUN_ROOT="runs/benchmark_journal_gpu_e100_full"
RESULT_ROOT="results/benchmark_journal_gpu_e100_full"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

# Step 1: E1 full OOD benchmark (missing OOD folders are skipped with warning)
$PY run_benchmark_suite.py \
  --suite cifar \
  --methods ce,mcdropout,edl,fisher \
  --seeds 1234,2345,3456 \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 2 \
  --ood-datasets "$OODS" \
  --run-root "$RUN_ROOT" \
  --result-root "$RESULT_ROOT" \
  --wandb \
  --wandb-project info-edl \
  --skip-missing-ood

# Step 2: CE/MCDropout score-axis expansion (MSP/Energy + temp/no-temp)
EXTRA_DIR="$RESULT_ROOT/cifar_score_sweep"
mkdir -p "$EXTRA_DIR"

for method in ce mcdropout; do
  for seed in 1234 2345 3456; do
    ckpt="$RUN_ROOT/cifar/${method}_seed${seed}/best_val_acc.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "[WARN] missing checkpoint: $ckpt" >&2
      continue
    fi
    if [[ "$method" == "mcdropout" ]]; then
      mc_passes=10
    else
      mc_passes=1
    fi

    for score in msp energy; do
      for calib in none temperature; do
        out_csv="$EXTRA_DIR/${method}_seed${seed}_${score}_${calib}.csv"
        $PY run_cifar_eval.py \
          --ckpt "$ckpt" \
          --method "$method" \
          --id-dataset cifar10 \
          --ood-datasets "$OODS" \
          --data-root ./data \
          --batch-size 256 \
          --num-workers 2 \
          --seed "$seed" \
          --score-type "$score" \
          --calibration "$calib" \
          --mc-dropout-passes "$mc_passes" \
          --skip-missing-ood \
          --out-csv "$out_csv" \
          --wandb \
          --wandb-project info-edl \
          --wandb-name "${method}_seed${seed}_${score}_${calib}_eval"
      done
    done
  done
done

$PY - <<'PY'
import csv
from pathlib import Path
root = Path('results/benchmark_journal_gpu_e100_full/cifar_score_sweep')
out = root / 'summary_scores.csv'
rows = []
header = None
for p in sorted(root.glob('*.csv')):
    if p.name == out.name:
        continue
    with p.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if header is None:
            header = r.fieldnames
        for row in r:
            rows.append(row)
if header and rows:
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f'merged -> {out}')
else:
    print('no score-sweep rows merged')
PY

# Step 3: report regeneration for the expanded main benchmark
mkdir -p "$RESULT_ROOT/tables" "$RESULT_ROOT/figures" "$RESULT_ROOT/stats"
$PY scripts/make_tables.py \
  --inputs "$RESULT_ROOT/cifar/summary_all.csv" \
  --out-csv "$RESULT_ROOT/tables/main_table.csv" \
  --out-md "$RESULT_ROOT/tables/main_table.md"

$PY scripts/plot_main_figures.py \
  --eval-csv "$RESULT_ROOT/cifar/summary_all.csv" \
  --out-dir "$RESULT_ROOT/figures"

$PY scripts/stat_tests.py \
  --input-csv "$RESULT_ROOT/cifar/summary_all.csv" \
  --out-md "$RESULT_ROOT/stats/stat_tests.md"

printf '\n[DONE] Extended E1 pipeline finished.\n'
