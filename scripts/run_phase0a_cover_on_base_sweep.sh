#!/bin/bash
# Phase 0a expansion: CoVER on D0-base strengths × 5 seeds
# 15 runs total = 5 seeds {42, 123, 456, 789, 2026} × 3 base strengths {200, 400, 1000} ep
# Champion config: --alpha_bias_init 0 --alpha_max 0.3 --lambda_align 3e-2 --single-stage
# GPU policy:
#   - cuda:1 (24GB free, idle): 8 runs serial
#   - cuda:2 (24GB free, idle): 7 runs serial
# Expected wall: ~4 min (8 × 30s on slower GPU)

set -e

CONFIG="artifacts/logs/yelpchi/bwgnn/phase2_yelpchi_bwgnn_revised_r3_full_alpha01_lint1em2/seed_42/repro_config.yaml"
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/phase0a_cover_on_base_$$
mkdir -p "$LOGDIR"

# Build the 15-job list ordered by (base_ep, seed) interleaved for load balancing.
JOBS=(
  "200 42  cuda:1"
  "200 123 cuda:2"
  "200 456 cuda:1"
  "200 789 cuda:2"
  "200 2026 cuda:1"
  "400 42  cuda:2"
  "400 123 cuda:1"
  "400 456 cuda:2"
  "400 789 cuda:1"
  "400 2026 cuda:2"
  "1000 42 cuda:1"
  "1000 123 cuda:2"
  "1000 456 cuda:1"
  "1000 789 cuda:2"
  "1000 2026 cuda:1"
)
# Allocation: cuda:1 = 8 runs, cuda:2 = 7 runs

# Group jobs by device, run each device's jobs serially in its own bg process
run_device_jobs() {
  local DEV="$1"; shift
  local jobs=("$@")
  for job in "${jobs[@]}"; do
    read EP SEED <<< "$job"
    local RUN_NAME="phase0a_cover_on_base${EP}ep"
    local CKPT="artifacts/checkpoints/yelpchi/bwgnn/d0_base_${EP}ep/seed_${SEED}/base.pt"
    local TAG="${RUN_NAME}_seed${SEED}_${DEV//:/}"
    local LOG="${LOGDIR}/${TAG}.log"
    echo "[launch] ${TAG}"
    JOB_START=$SECONDS
    CUDA_VISIBLE_DEVICES="${DEV#cuda:}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/train_phase2_reasoner.py \
      --config "$CONFIG" \
      --run_name "$RUN_NAME" \
      --seed "$SEED" \
      --device cuda:0 \
      --base_ckpt_path "$CKPT" \
      --alpha_bias_init 0.0 --alpha_max 0.3 --lambda_align 3e-2 \
      --single-stage \
      2>&1 | sed "s/^/[${TAG}] /" >"$LOG" || true
    if [ ! -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_${SEED}/phase2_diagnostics.json" ]; then
      echo "[FAIL]  ${TAG} — no diagnostics — see ${LOG}"
    else
      echo "[done]  ${TAG} ($((SECONDS-JOB_START))s)"
    fi
  done
}

# Filter jobs per device
cuda1_jobs=()
cuda2_jobs=()
for j in "${JOBS[@]}"; do
  read EP SEED DEV <<< "$j"
  if [ "$DEV" = "cuda:1" ]; then
    cuda1_jobs+=("$EP $SEED")
  else
    cuda2_jobs+=("$EP $SEED")
  fi
done

START_ALL=$SECONDS
echo "============================================================"
echo "[Phase 0a expansion] launching cuda:1 = ${#cuda1_jobs[@]} runs, cuda:2 = ${#cuda2_jobs[@]} runs"
echo "============================================================"

run_device_jobs "cuda:1" "${cuda1_jobs[@]}" &
PID1=$!
run_device_jobs "cuda:2" "${cuda2_jobs[@]}" &
PID2=$!
wait $PID1
wait $PID2

TOTAL=$((SECONDS - START_ALL))
echo
echo "============================================================"
echo "ALL ${#JOBS[@]} RUNS DONE in ${TOTAL}s"
echo "Per-job logs: ${LOGDIR}"
echo "============================================================"
