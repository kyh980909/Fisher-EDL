#!/usr/bin/env bash
set -euo pipefail
cd /home/yongho/fedl
PY=/home/yongho/miniconda3/envs/openmmlab/bin/python

OODS="svhn,cifar100,texture,lsun-crop,lsun-resize,isun,places365"
CKPT_ROOT="runs/benchmark_journal_gpu_e100/cifar"
OUT_ROOT="results/benchmark_journal_gpu_e100_full_evalonly"
MAIN_OUT="$OUT_ROOT/cifar"
SCORE_OUT="$OUT_ROOT/cifar_score_sweep"
mkdir -p "$MAIN_OUT" "$SCORE_OUT" "$OUT_ROOT/tables" "$OUT_ROOT/figures" "$OUT_ROOT/stats"

# 1) Main full-OOD eval (uncertainty) on existing checkpoints
for method in ce mcdropout edl fisher; do
  for seed in 1234 2345 3456; do
    ckpt="$CKPT_ROOT/${method}_seed${seed}/best_val_acc.pt"
    [[ -f "$ckpt" ]] || { echo "[WARN] missing $ckpt"; continue; }
    if [[ "$method" == "mcdropout" ]]; then mc_passes=10; else mc_passes=1; fi
    out_csv="$MAIN_OUT/${method}_seed${seed}.csv"
    $PY run_cifar_eval.py \
      --ckpt "$ckpt" \
      --method "$method" \
      --id-dataset cifar10 \
      --ood-datasets "$OODS" \
      --data-root ./data \
      --batch-size 256 \
      --num-workers 2 \
      --seed "$seed" \
      --score-type uncertainty \
      --calibration none \
      --mc-dropout-passes "$mc_passes" \
      --skip-missing-ood \
      --out-csv "$out_csv" \
      --wandb --wandb-project info-edl --wandb-name "${method}_seed${seed}_unc_fullood_eval"
  done
done

$PY - <<'PY'
import csv
from pathlib import Path
root = Path('results/benchmark_journal_gpu_e100_full_evalonly/cifar')
out = root / 'summary_all.csv'
rows=[]; header=None
for p in sorted(root.glob('*_seed*.csv')):
    with p.open('r', encoding='utf-8') as f:
        r=csv.DictReader(f)
        if header is None: header=r.fieldnames
        rows.extend(list(r))
if header and rows:
    with out.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    print('merged ->', out)
else:
    print('no rows for summary_all.csv')
PY

# 2) CE/MCDropout score-axis expansion on existing checkpoints
for method in ce mcdropout; do
  for seed in 1234 2345 3456; do
    ckpt="$CKPT_ROOT/${method}_seed${seed}/best_val_acc.pt"
    [[ -f "$ckpt" ]] || continue
    if [[ "$method" == "mcdropout" ]]; then mc_passes=10; else mc_passes=1; fi
    for score in msp energy; do
      for calib in none temperature; do
        out_csv="$SCORE_OUT/${method}_seed${seed}_${score}_${calib}.csv"
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
          --wandb --wandb-project info-edl --wandb-name "${method}_seed${seed}_${score}_${calib}_eval"
      done
    done
  done
done

$PY - <<'PY'
import csv
from pathlib import Path
root = Path('results/benchmark_journal_gpu_e100_full_evalonly/cifar_score_sweep')
out = root / 'summary_scores.csv'
rows=[]; header=None
for p in sorted(root.glob('*.csv')):
    if p.name == out.name: continue
    with p.open('r', encoding='utf-8') as f:
        r=csv.DictReader(f)
        if header is None: header=r.fieldnames
        rows.extend(list(r))
if header and rows:
    with out.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    print('merged ->', out)
else:
    print('no score rows')
PY

# 3) reports
$PY scripts/make_tables.py --inputs "$MAIN_OUT/summary_all.csv" --out-csv "$OUT_ROOT/tables/main_table.csv" --out-md "$OUT_ROOT/tables/main_table.md"
$PY scripts/plot_main_figures.py --eval-csv "$MAIN_OUT/summary_all.csv" --out-dir "$OUT_ROOT/figures"
$PY scripts/stat_tests.py --input-csv "$MAIN_OUT/summary_all.csv" --out-md "$OUT_ROOT/stats/stat_tests.md"

echo "[DONE] eval-only extended pipeline finished"
