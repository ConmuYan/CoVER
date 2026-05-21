#!/usr/bin/env bash
# interp-A 4-condition incremental loss-component ablation
# YelpChi-{bwgnn|sage|gcn|gat} × 5 seeds × 4 conditions = 20 runs / base
#
# Usage:
#   bash scripts/run_interp_a_ablation.sh                  # bwgnn sequential
#   bash scripts/run_interp_a_ablation.sh PARALLEL         # bwgnn 4-GPU parallel
#   BASE=sage bash scripts/run_interp_a_ablation.sh PARALLEL  # other base
#   BASE=gcn  bash scripts/run_interp_a_ablation.sh PARALLEL
#
# Tests the genuinely locked 3-component loss form for C3:
#   L = L_bce^(K) + α_f · L_KL^(K) + λ_cbr · L_CBR^(K)
#
# All 4 conditions use `--mode det_mask_cbr` (top-K H(p_S) mask constant)
# + `--ablate_reliability --ablate_adaptive_bce` (drop Z1-falsified weighting);
# rKL condition adds `--ablate_mixed_kl` (η ≡ 0 → pure reverse-KL).
# Multi-head heads (alpha_r, alpha_g) permanently 0 (Z1: 0/8 sig vs det_mask).

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 123 456 789 2026)
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
K="${K:-2048}"
BASE="${BASE:-bwgnn}"
DATASET="${DATASET:-yelpchi}"

CONFIG="configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"
if [[ "$BASE" != "bwgnn" ]]; then
  # Naming: idea1_${dataset}_${base}_canonical_clsonly.yaml
  for candidate in \
    "configs/phase2_reasoner/ablation/idea1_${DATASET}_${BASE}_canonical_clsonly.yaml" \
    "configs/${DATASET}_${BASE}.yaml" \
    "configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"; do
    if [[ -f "$candidate" ]]; then CONFIG="$candidate"; break; fi
  done
fi
echo "[config] $CONFIG"

run_one() {
  local seed="$1" runname="$2" alpha_f="$3" cbr_lambda="$4" extra_flags="$5" device="$6"
  local teacher_ckpt="artifacts/checkpoints/${DATASET}/${BASE}/idea2b_learned_extractor/seed_${seed}/reasoner.pt"
  local teacher_ext="artifacts/checkpoints/${DATASET}/${BASE}/idea2b_learned_extractor/seed_${seed}/evidence_extractor.pt"
  local base_ckpt="artifacts/checkpoints/${DATASET}/${BASE}/fixed_v1_100ep/seed_${seed}/base.pt"
  local out="artifacts/results/${DATASET}/${BASE}/${runname}/seed_${seed}/stage3_metrics.json"

  if [[ -f "$out" ]]; then
    echo "[skip-exists] ${DATASET}/${BASE} seed=${seed} run=${runname}"
    return 0
  fi

  echo "[run device=${device}] ${DATASET}/${BASE} seed=${seed} run=${runname} α_f=${alpha_f} λ_cbr=${cbr_lambda}"
  PYTHONPATH=. python scripts/train_cbr_flash.py \
    --config "$CONFIG" \
    --seed "$seed" \
    --device "$device" \
    --mode det_mask_cbr \
    --teacher_ckpt "$teacher_ckpt" \
    --teacher_extractor_ckpt "$teacher_ext" \
    --base_ckpt_path "$base_ckpt" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --K "$K" \
    --alpha_f "$alpha_f" \
    --alpha_r 0.0 \
    --alpha_g 0.0 \
    --cbr_lambda "$cbr_lambda" \
    --cbr_weight_form exp \
    --ablate_reliability \
    --ablate_adaptive_bce \
    $extra_flags \
    --run_name "$runname" > "/tmp/interp_a_${DATASET}_${BASE}_${runname}_seed${seed}.log" 2>&1
}

run_condition_all_seeds() {
  local cond_name="$1" alpha_f="$2" cbr_lambda="$3" extra_flags="$4" device="$5"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" "$cond_name" "$alpha_f" "$cbr_lambda" "$extra_flags" "$device"
  done
  echo "[done device=${device}] ${DATASET}/${BASE} condition=${cond_name}: all 5 seeds complete"
}

MODE="${1:-SEQ}"

if [[ "$MODE" == "PARALLEL" ]]; then
  echo "=== PARALLEL mode: ${DATASET}/${BASE} 4 conditions on 4 GPUs (cuda:0..3) ==="
  run_condition_all_seeds "interp_a_bce"         0.0 0.0  ""                  "cuda:0" &
  PID0=$!
  run_condition_all_seeds "interp_a_bce_kl"      1.0 0.0  "--ablate_mixed_kl" "cuda:1" &
  PID1=$!
  run_condition_all_seeds "interp_a_bce_cbr"     0.0 1.0  ""                  "cuda:2" &
  PID2=$!
  run_condition_all_seeds "interp_a_bce_kl_cbr"  1.0 1.0  "--ablate_mixed_kl" "cuda:3" &
  PID3=$!
  wait $PID0 $PID1 $PID2 $PID3
  echo "[all done] ${DATASET}/${BASE} 4 conditions × 5 seeds = 20 runs complete (4-way parallel)"
else
  echo "=== SEQUENTIAL mode: ${DATASET}/${BASE} 4 conditions × 5 seeds on cuda:0 ==="
  DEVICE="${DEVICE:-cuda:0}"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" "interp_a_bce"         0.0 0.0  ""                  "$DEVICE"
    run_one "$seed" "interp_a_bce_kl"      1.0 0.0  "--ablate_mixed_kl" "$DEVICE"
    run_one "$seed" "interp_a_bce_cbr"     0.0 1.0  ""                  "$DEVICE"
    run_one "$seed" "interp_a_bce_kl_cbr"  1.0 1.0  "--ablate_mixed_kl" "$DEVICE"
  done
fi

echo ""
echo "Aggregate:"
echo "  PYTHONPATH=. python scripts/aggregate_interp_a.py"
