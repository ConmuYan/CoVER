#!/bin/bash
# Phase 2 CoVER verification on fixed Phase 1 base
# Tests champion config on top of {fixed_v1_100ep, fixed_v1_400ep} × {42, 123, 456, 789, 2026}
# 10 runs total, split across cuda:1 + cuda:2 (5 each, serial within)

set -e

CONFIG="artifacts/logs/yelpchi/bwgnn/phase2_yelpchi_bwgnn_revised_r3_full_alpha01_lint1em2/seed_42/repro_config.yaml"
# Wait — that config doesn't exist anymore (we deleted phase2_yelpchi_bwgnn_revised artifacts).
# Fall back to the canonical full_cover config.
if [ ! -f "$CONFIG" ]; then
  CONFIG="configs/phase2_reasoner/phase2_yelpchi_bwgnn_full_cover.yaml"
fi
echo "[config] $CONFIG"

PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/phase2_verif_$$
mkdir -p "$LOGDIR"

run_one() {
  local BASE_RUN="$1"
  local SEED="$2"
  local GPU="$3"
  local CKPT="artifacts/checkpoints/yelpchi/bwgnn/${BASE_RUN}/seed_${SEED}/base.pt"
  local RUN_NAME="phase2_on_${BASE_RUN}"
  local TAG="${RUN_NAME}_seed${SEED}_cuda${GPU}"
  local LOG="${LOGDIR}/${TAG}.log"
  echo "[launch] ${TAG}"
  JOB_START=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" \
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
  if [ -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_${SEED}/phase2_diagnostics.json" ]; then
    echo "[done]  ${TAG} ($((SECONDS-JOB_START))s)"
  else
    echo "[FAIL]  ${TAG} ($((SECONDS-JOB_START))s) — see ${LOG}"
  fi
}

run_device() {
  local GPU="$1"; shift
  for pair in "$@"; do
    read BASE SEED <<< "$pair"
    run_one "$BASE" "$SEED" "$GPU"
  done
}

# Distribute 10 runs across 2 GPUs (5 each, balanced by base strength)
CUDA1=(
  "fixed_v1_100ep 42"
  "fixed_v1_100ep 456"
  "fixed_v1_100ep 2026"
  "fixed_v1_400ep 123"
  "fixed_v1_400ep 789"
)
CUDA2=(
  "fixed_v1_100ep 123"
  "fixed_v1_100ep 789"
  "fixed_v1_400ep 42"
  "fixed_v1_400ep 456"
  "fixed_v1_400ep 2026"
)

START_ALL=$SECONDS
echo "============================================================"
echo "[Phase 2 verif] cuda:1=${#CUDA1[@]} runs, cuda:2=${#CUDA2[@]} runs"
echo "============================================================"

run_device "1" "${CUDA1[@]}" &
P1=$!
run_device "2" "${CUDA2[@]}" &
P2=$!
wait $P1
wait $P2

TOTAL=$((SECONDS - START_ALL))
echo
echo "============================================================"
echo "ALL 10 RUNS DONE in ${TOTAL}s"
echo "Logs: ${LOGDIR}"
echo "============================================================"
