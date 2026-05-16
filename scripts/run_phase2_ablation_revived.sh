#!/bin/bash
# Phase 2 ablation v2 — using REVIVED judge packets (2000 nodes, base-uncertain).
# Compare 4 critical cells × 5 seeds × cover_rel_judge_revived:
#   L7_revived  : full champion with revived judge (NEW)
#   A1_revived  : relation-only baseline (judge off) — same as before, sanity check
#   A2_revived  : judge-only (Δ_rel=0) — should now show meaningful AUPRC (≠ base)
#   L7_legacy   : champion with old 120-node judge (for paper "before/after" delta)
#
# 4 cells × 5 seeds = 20 runs. Splits 10/10 across cuda:1 + cuda:2.

set -e
CONFIG="configs/phase2_reasoner/phase2_yelpchi_bwgnn_full_cover.yaml"
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/phase2_revived_$$
mkdir -p "$LOGDIR"
BASE_RUN="fixed_v1_100ep"

# Cell schema:  cell_id  lam_int  lam_sparse  lam_align  use_judge  alpha_max  delta_rel_max  judge_run_name
CELLS=(
  "L7_revived   3e-3  1e-3  3e-2   1  0.3  2.0   cover_rel_judge_revived"
  "A1_revived   3e-3  1e-3  0      0  0.0  2.0   cover_rel_judge_revived"
  "A2_revived   3e-3  1e-3  3e-2   1  0.3  0.0   cover_rel_judge_revived"
  "L7_legacy    3e-3  1e-3  3e-2   1  0.3  2.0   cover_rel_judge"
)

SEEDS=(42 123 456 789 2026)

run_one() {
  local CELL="$1" LINT="$2" LSPARSE="$3" LALIGN="$4"
  local UJ="$5" AMAX="$6" DRMAX="$7" JRN="$8" SEED="$9" GPU="${10}"
  local CKPT="artifacts/checkpoints/yelpchi/bwgnn/${BASE_RUN}/seed_${SEED}/base.pt"
  local RUN_NAME="ablation_${CELL}"
  local TAG="${RUN_NAME}_seed${SEED}_cuda${GPU}"
  local LOG="${LOGDIR}/${TAG}.log"
  echo "[launch] ${TAG}  (judge_run=${JRN})"
  JOB_START=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY scripts/train_phase2_reasoner.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --device cuda:0 \
    --base_ckpt_path "$CKPT" \
    --judge_run_name "$JRN" \
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

# Distribute 20 jobs alternately across 2 GPUs (10 serial each)
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
    read CELL LINT LSPARSE LALIGN UJ AMAX DRMAX JRN SEED <<< "$job"
    run_one "$CELL" "$LINT" "$LSPARSE" "$LALIGN" "$UJ" "$AMAX" "$DRMAX" "$JRN" "$SEED" "$GPU"
  done
}

START=$SECONDS
echo "============================================================"
echo "[Phase2 Revived Ablation] cuda:1=${#CUDA1_JOBS[@]} runs, cuda:2=${#CUDA2_JOBS[@]} runs"
echo "[Base] ${BASE_RUN}  [Cells] ${#CELLS[@]}  [Seeds] ${#SEEDS[@]}"
echo "============================================================"

run_device "1" "${CUDA1_JOBS[@]}" &
P1=$!
run_device "2" "${CUDA2_JOBS[@]}" &
P2=$!
wait $P1
wait $P2

ELAPSED=$((SECONDS - START))
echo
echo "============================================================"
echo "ALL ${i} RUNS DONE in ${ELAPSED}s"
echo "Logs: ${LOGDIR}"
echo "============================================================"
