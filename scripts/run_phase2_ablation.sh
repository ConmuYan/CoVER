#!/bin/bash
# Phase 2 ablation on fixed Phase 1 100ep BWGNN base (new champion).
#
# Two factor sweeps, anchored on the same fixed_v1_100ep base × 5 seed:
#   (1) Loss ablation (8 cells): full architecture, vary {L_int, L_sparse, L_align}
#   (2) Architecture ablation (2 new cells; A0=base reused, A3=L7=champion):
#       A1 relation-only: use_judge=0, alpha_max=0, lambda_align=0 (judge dependency)
#       A2 judge-only:    delta_rel_max=0
#
# 50 runs total (10 cells × 5 seeds), split 25/25 across cuda:1 + cuda:2.
# Each ~90s → ~38 min wall.

set -e

CONFIG="configs/phase2_reasoner/phase2_yelpchi_bwgnn_full_cover.yaml"
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/phase2_ablation_$$
mkdir -p "$LOGDIR"
BASE_RUN="fixed_v1_100ep"

# Champion HP (applied to every cell; matches phase2_verification overrides):
#   alpha_bias_init=0.0  (escape dead zone)
#   alpha_max=0.3        (overrides yaml 0.10)
#   lambda_align=3e-2    (overrides yaml 1e-2)

# Cell schema: cell_id  lam_int  lam_sparse  lam_align  use_judge  alpha_max  delta_rel_max
CELLS=(
  "L0  0     0     0      1  0.3  2.0"
  "L1  3e-3  0     0      1  0.3  2.0"
  "L2  0     1e-3  0      1  0.3  2.0"
  "L3  0     0     3e-2   1  0.3  2.0"
  "L4  3e-3  1e-3  0      1  0.3  2.0"
  "L5  3e-3  0     3e-2   1  0.3  2.0"
  "L6  0     1e-3  3e-2   1  0.3  2.0"
  "L7  3e-3  1e-3  3e-2   1  0.3  2.0"
  "A1  3e-3  1e-3  0      0  0.0  2.0"
  "A2  3e-3  1e-3  3e-2   1  0.3  0.0"
)

SEEDS=(42 123 456 789 2026)

run_one() {
  local CELL="$1" LINT="$2" LSPARSE="$3" LALIGN="$4"
  local UJ="$5" AMAX="$6" DRMAX="$7" SEED="$8" GPU="$9"
  local CKPT="artifacts/checkpoints/yelpchi/bwgnn/${BASE_RUN}/seed_${SEED}/base.pt"
  local RUN_NAME="ablation_${CELL}"
  local TAG="${RUN_NAME}_seed${SEED}_cuda${GPU}"
  local LOG="${LOGDIR}/${TAG}.log"
  echo "[launch] ${TAG}  (λint=${LINT} λsp=${LSPARSE} λal=${LALIGN} uj=${UJ} αmax=${AMAX} Δrmax=${DRMAX})"
  JOB_START=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY scripts/train_phase2_reasoner.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --device cuda:0 \
    --base_ckpt_path "$CKPT" \
    --alpha_bias_init 0.0 \
    --alpha_max "$AMAX" \
    --lambda_int "$LINT" \
    --lambda_sparse "$LSPARSE" \
    --lambda_align "$LALIGN" \
    --use_judge "$UJ" \
    --delta_rel_max "$DRMAX" \
    --single-stage \
    2>&1 | sed "s/^/[${TAG}] /" >"$LOG" || true
  if [ -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_${SEED}/phase2_diagnostics.json" ]; then
    echo "[done]  ${TAG} ($((SECONDS-JOB_START))s)"
  else
    echo "[FAIL]  ${TAG} ($((SECONDS-JOB_START))s) — see ${LOG}"
  fi
}

# Distribute 50 jobs alternately across 2 GPUs (25 serial each)
CUDA1_JOBS=()
CUDA2_JOBS=()
i=0
for cell_line in "${CELLS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if (( i % 2 == 0 )); then
      CUDA1_JOBS+=("${cell_line} ${seed}")
    else
      CUDA2_JOBS+=("${cell_line} ${seed}")
    fi
    i=$((i + 1))
  done
done

run_device() {
  local GPU="$1"; shift
  for job in "$@"; do
    read CELL LINT LSPARSE LALIGN UJ AMAX DRMAX SEED <<< "$job"
    run_one "$CELL" "$LINT" "$LSPARSE" "$LALIGN" "$UJ" "$AMAX" "$DRMAX" "$SEED" "$GPU"
  done
}

START_ALL=$SECONDS
TOTAL=${#CUDA1_JOBS[@]}+${#CUDA2_JOBS[@]}
echo "============================================================"
echo "[Phase 2 Ablation] cuda:1=${#CUDA1_JOBS[@]} runs, cuda:2=${#CUDA2_JOBS[@]} runs"
echo "[Base] ${BASE_RUN}  [Cells] ${#CELLS[@]}  [Seeds] ${#SEEDS[@]}"
echo "============================================================"

run_device "1" "${CUDA1_JOBS[@]}" &
P1=$!
run_device "2" "${CUDA2_JOBS[@]}" &
P2=$!
wait $P1
wait $P2

ELAPSED=$((SECONDS - START_ALL))
echo
echo "============================================================"
echo "ALL ${i} RUNS DONE in ${ELAPSED}s"
echo "Logs: ${LOGDIR}"
echo "============================================================"
