#!/usr/bin/env bash
# Lock-aware YelpChi Phase2 worker. Multiple instances may run concurrently.
#
# Usage:
#   bash scripts/run_phase2_yelpchi_queue_worker.sh GPU WORKER_NAME
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="$1"
WORKER="$2"
PY="${PYTHON:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export COVER_NUM_THREADS="${COVER_NUM_THREADS:-1}"
export COVER_INTEROP_THREADS="${COVER_INTEROP_THREADS:-1}"

CFG_E0="configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E0_relgate.yaml"
CFG_E2="configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E2_judge_residual.yaml"
RESULT_ROOT="artifacts/results/yelpchi/bwgnn"
LOCK_ROOT="artifacts/sweeps/locks/yelpchi_phase2_targeted"

cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers "$LOCK_ROOT"
LOG="artifacts/sweeps/_drivers/phase2_yelpchi_queue_${WORKER}_gpu${GPU}.log"
echo "[queue-worker] start $(date -Is) worker=${WORKER} gpu=${GPU}" >> "$LOG"

run_task() {
  local run_name="$1"; shift
  local seed="$1"; shift
  local metrics="${RESULT_ROOT}/${run_name}/seed_${seed}/stage3_metrics.json"
  local lock_dir="${LOCK_ROOT}/${run_name}__seed_${seed}.lock"

  if [ -s "$metrics" ]; then
    echo "[queue-worker] SKIP done run=${run_name} seed=${seed}" >> "$LOG"
    return 0
  fi
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[queue-worker] SKIP locked run=${run_name} seed=${seed}" >> "$LOG"
    return 0
  fi

  echo "[queue-worker] BEGIN run=${run_name} seed=${seed} @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_phase2_reasoner.py \
    --seed "$seed" --device cuda:0 --run_name "$run_name" \
    --eval_interval "$EVAL_INTERVAL" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[queue-worker] END run=${run_name} seed=${seed} rc=${rc} @ $(date -Is)" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "[queue-worker] FAILED run=${run_name} seed=${seed}; lock retained at ${lock_dir}" >> "$LOG"
    return "$rc"
  fi
  rmdir "$lock_dir" 2>/dev/null || true
  return 0
}

for seed in "${SEEDS[@]}"; do
  run_task phase2_yelp_s1_ltrust_0 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 0.0 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s1_ltrust_1em3 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 1.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s1_ltrust_3em3 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s1_ltrust_1em2 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 1.0e-2 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s1_ltrust_3em2 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-2 --lambda_sparse 1.0e-3

  run_task phase2_yelp_s2_lsparse_0 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 0.0
  run_task phase2_yelp_s2_lsparse_3em4 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 3.0e-4
  run_task phase2_yelp_s2_lsparse_1em3 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s2_lsparse_3em3 "$seed" --config "$CFG_E0" --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 3.0e-3

  run_task phase2_yelp_s3_lalign_0 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.0 --lambda_align 0.0 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s3_lalign_1em3 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.0 --lambda_align 1.0e-3 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s3_lalign_3em3 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.0 --lambda_align 3.0e-3 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s3_lalign_1em2 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.0 --lambda_align 1.0e-2 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3

  run_task phase2_yelp_s3_alpha_0 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.0 --lambda_align 1.0e-3 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s3_alpha_01 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.1 --lambda_align 1.0e-3 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
  run_task phase2_yelp_s3_alpha_03 "$seed" --config "$CFG_E2" --use_judge 1 --alpha_max 0.3 --lambda_align 1.0e-3 --lambda_trust 3.0e-3 --lambda_sparse 1.0e-3
done

echo "[queue-worker] done $(date -Is) worker=${WORKER} gpu=${GPU}" >> "$LOG"
