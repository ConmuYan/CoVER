#!/bin/bash
# Fixed BWGNN Phase 1 sweep — pos_weight + threshold search applied
# Goals:
#   1. Reproduce original BWGNN paper HOMO YelpChi numbers (AUC=84.03, MaF1=71.00 at 100ep)
#   2. Re-establish base.pt at each saturation level {100, 200, 400, 1000}ep × 5 seed
#   3. Provide clean foundation for Phase 2 CoVER ablations
# GPU policy: split across cuda:1 + cuda:2 (both 24GB free, idle). Serial within each.
# Job distribution: round-robin to balance load (1000ep is bottleneck).

set -e

CONFIG="configs/yelpchi_bwgnn.yaml"
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/fixed_stage1_$$
mkdir -p "$LOGDIR"

# 20 jobs: ep × seed. Distribute alternately to balance long+short on each GPU.
# Time per run: 100ep~48s, 200ep~90s, 400ep~173s, 1000ep~423s.
# Total work ≈ 3670s → each GPU ≈ 30 min.
CUDA1_JOBS=(
  "1000 42"
  "1000 456"
  "1000 2026"
  "400 123"
  "400 789"
  "200 42"
  "200 456"
  "200 2026"
  "100 123"
  "100 789"
)
CUDA2_JOBS=(
  "1000 123"
  "1000 789"
  "400 42"
  "400 456"
  "400 2026"
  "200 123"
  "200 789"
  "100 42"
  "100 456"
  "100 2026"
)

run_device_jobs() {
  local GPU="$1"; shift
  local jobs=("$@")
  for job in "${jobs[@]}"; do
    read EP SEED <<< "$job"
    local RUN_NAME="fixed_base_${EP}ep"
    local TAG="${RUN_NAME}_seed${SEED}_cuda${GPU}"
    local LOG="${LOGDIR}/${TAG}.log"
    echo "[launch] ${TAG}"
    JOB_START=$SECONDS
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/train_stage1.py \
      --config "$CONFIG" \
      --run_name "$RUN_NAME" \
      --seed "$SEED" \
      --epochs "$EP" \
      --patience 9999 \
      --select_metric macro_f1 \
      --stratified \
      2>&1 | sed "s/^/[${TAG}] /" >"$LOG" || true
    if [ ! -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_${SEED}/stage1.json" ]; then
      echo "[FAIL]  ${TAG} ($((SECONDS-JOB_START))s) — see ${LOG}"
    else
      echo "[done]  ${TAG} ($((SECONDS-JOB_START))s)"
    fi
  done
}

START_ALL=$SECONDS
echo "============================================================"
echo "[Fixed-Phase1] cuda:1=${#CUDA1_JOBS[@]} runs, cuda:2=${#CUDA2_JOBS[@]} runs (parallel)"
echo "============================================================"

run_device_jobs "1" "${CUDA1_JOBS[@]}" &
P1=$!
run_device_jobs "2" "${CUDA2_JOBS[@]}" &
P2=$!
wait $P1
wait $P2

TOTAL=$((SECONDS - START_ALL))
echo
echo "============================================================"
echo "ALL 20 RUNS DONE in ${TOTAL}s"
echo "Logs: ${LOGDIR}"
echo "============================================================"
