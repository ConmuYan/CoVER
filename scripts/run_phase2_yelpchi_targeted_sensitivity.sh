#!/usr/bin/env bash
# Targeted YelpChi Phase2 sensitivity runs.
#
# This intentionally varies only the theory-relevant knobs:
#   lambda_trust, lambda_sparse, lambda_align, alpha_max.
#
# Usage:
#   bash scripts/run_phase2_yelpchi_targeted_sensitivity.sh GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_phase2_yelpchi_targeted_sensitivity.sh 0 42 123 456 789 2026
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="$1"; shift
SEEDS=("$@")
PY="${PYTHON:-/data1/mq/conda_envs/gread-core/bin/python}"
CFG="configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E0_relgate.yaml"
JCFG="configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E2_judge_residual.yaml"

cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers
LOG="artifacts/sweeps/_drivers/phase2_yelpchi_targeted_gpu${GPU}.log"

run_one() {
  local seed="$1"; shift
  echo "[yelpchi-targeted] BEGIN seed=${seed} $* @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_phase2_reasoner.py \
    --seed "$seed" --device cuda:0 "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[yelpchi-targeted] END seed=${seed} rc=${rc} $* @ $(date -Is)" >> "$LOG"
  return "$rc"
}

overall_rc=0
echo "[yelpchi-targeted] start $(date -Is) gpu=${GPU} seeds=${SEEDS[*]}" >> "$LOG"

# S1: trust sensitivity, relation-only.
for item in 0:0.0 1em3:1.0e-3 3em3:3.0e-3 1em2:1.0e-2 3em2:3.0e-2; do
  label="${item%%:*}"
  trust="${item#*:}"
  run_name="phase2_yelp_s1_ltrust_${label}"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" --config "$CFG" --run_name "$run_name" \
      --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
      --lambda_trust "$trust" --lambda_sparse 1.0e-3 || overall_rc=$?
  done
done

# S2: gate sparsity sensitivity, relation-only.
for item in 0:0.0 3em4:3.0e-4 1em3:1.0e-3 3em3:3.0e-3; do
  label="${item%%:*}"
  sparse="${item#*:}"
  run_name="phase2_yelp_s2_lsparse_${label}"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" --config "$CFG" --run_name "$run_name" \
      --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
      --lambda_trust 3.0e-3 --lambda_sparse "$sparse" || overall_rc=$?
  done
done

# S3a: judge alignment strength, no judge residual.
for item in 0:0.0 1em3:1.0e-3 3em3:3.0e-3 1em2:1.0e-2; do
  label="${item%%:*}"
  align="${item#*:}"
  run_name="phase2_yelp_s3_lalign_${label}"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" --config "$JCFG" --run_name "$run_name" \
      --use_judge 1 --alpha_max 0.0 --lambda_align "$align" \
      --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3 || overall_rc=$?
  done
done

# S3b: conservative judge residual cap.
for item in 0:0.0 01:0.1 03:0.3; do
  label="${item%%:*}"
  alpha="${item#*:}"
  run_name="phase2_yelp_s3_alpha_${label}"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed" --config "$JCFG" --run_name "$run_name" \
      --use_judge 1 --alpha_max "$alpha" --lambda_align 1.0e-3 \
      --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3 || overall_rc=$?
  done
done

echo "[yelpchi-targeted] done $(date -Is) rc=${overall_rc}" >> "$LOG"
exit "$overall_rc"
