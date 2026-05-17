#!/bin/bash
# Idea 2B 7-cell × 5-seed runner.
# Splits work: GPU 2 = non-GAT (5 cells × 5 seeds = 25 runs, 4-way parallel).
#              GPU 0 = GAT (2 cells × 5 seeds = 10 runs, 4-way parallel via cached base).
# Total expected wall: ~10-15 min (vs YelpChi-BWGNN already complete with 5 seeds).
#
# Per-cell run launches up to 4 seeds in parallel; remaining seed runs sequential
# after wait. User policy: max 4 simultaneous processes per GPU.

set -uo pipefail
mkdir -p logs/idea2b_8cell

run_cell () {
  local ds=$1 base=$2 gpu=$3
  local cfg=configs/phase2_reasoner/ablation/idea2b_learned_extractor_${ds}_${base}.yaml
  local seeds=(42 123 456 789 2026)

  # First 4 seeds in parallel
  for i in 0 1 2 3; do
    local s=${seeds[$i]}
    local logf=logs/idea2b_8cell/${ds}_${base}_seed${s}.log
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_phase2_reasoner.py \
      --config $cfg --seed $s --device cuda:0 \
      --run_name idea2b_learned_extractor \
      --base_ckpt_path artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${s}/base.pt \
      > $logf 2>&1 &
  done
  wait

  # 5th seed sequential
  local s=${seeds[4]}
  local logf=logs/idea2b_8cell/${ds}_${base}_seed${s}.log
  CUDA_VISIBLE_DEVICES=$gpu python scripts/train_phase2_reasoner.py \
    --config $cfg --seed $s --device cuda:0 \
    --run_name idea2b_learned_extractor \
    --base_ckpt_path artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${s}/base.pt \
    > $logf 2>&1
  echo "[CELL_DONE] ${ds}-${base} (GPU $gpu)"
}

# GPU 2: non-GAT cells (5 cells, sequential by cell, 4-parallel within cell)
(
  for cell in yelpchi:sage yelpchi:gcn amazon:bwgnn amazon:sage amazon:gcn; do
    ds="${cell%%:*}"; base="${cell##*:}"
    echo "[CELL_START] ${ds}-${base} on GPU 2"
    run_cell $ds $base 2
  done
  echo "[GPU2_DONE]"
) &

# GPU 0: GAT cells (2 cells, small per-job memory thanks to pre-cache)
(
  for cell in yelpchi:gat amazon:gat; do
    ds="${cell%%:*}"; base="${cell##*:}"
    echo "[CELL_START] ${ds}-${base} on GPU 0"
    run_cell $ds $base 0
  done
  echo "[GPU0_DONE]"
) &

wait
echo "[ALL_8_CELLS_DONE]"
