#!/bin/bash
# D0: Compute-matched base saturation sweep
# Train BWGNN on YelpChi at 100/200/400/1000 epochs × 5 seeds (= 20 runs)
# Selection: val_auprc (apples-to-apples with Phase 2 / CoVER)
# Early stopping: DISABLED (--patience = 9999)
# Stratified split: MATCHES ORIGINAL PROTOCOL used to train base.pt
# GPU policy: cuda:1 only (24GB free), concurrency=2 to avoid OOM
#   - BWGNN forward on YelpChi peaks at ~8GB activation memory
#   - cuda:2/3 had 9-10GB free each but failed with 2 concurrent jobs
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation

set -e

CONFIG="configs/yelpchi_bwgnn.yaml"
GPU=1
MAX_CONCURRENT=1
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/d0_sweep_$$
mkdir -p "$LOGDIR"

# Track failures so silent OOMs surface in the summary.
FAILED=()

# Build job list: (epochs, seed)
JOBS=()
for EPOCHS in 100 200 400 1000; do
  for SEED in 42 123 456 789 2026; do
    JOBS+=("${EPOCHS} ${SEED}")
  done
done

echo "=========================================="
echo "[D0] launching ${#JOBS[@]} jobs on cuda:${GPU}, max ${MAX_CONCURRENT} concurrent"
echo "=========================================="
START_ALL=$SECONDS

for JOB in "${JOBS[@]}"; do
  read EPOCHS SEED <<< "$JOB"
  RUN_NAME="d0_base_${EPOCHS}ep"
  TAG="${RUN_NAME}_seed${SEED}"
  LOG="${LOGDIR}/${TAG}.log"

  while (( $(jobs -rp | wc -l) >= MAX_CONCURRENT )); do
    sleep 0.5
  done

  echo "[launch] ${TAG}"
  (
    set -e
    JOB_START=$SECONDS
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/train_stage1.py \
      --config "$CONFIG" \
      --run_name "$RUN_NAME" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --patience 9999 \
      --select_metric auprc \
      --stratified \
      2>&1 | sed "s/^/[${TAG}] /" >"$LOG"
    JOB_END=$((SECONDS - JOB_START))
    # Detect silent failures: stage1.json must exist
    if [ ! -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_${SEED}/stage1.json" ]; then
      echo "[FAIL]   ${TAG}  (${JOB_END}s) — no stage1.json — see ${LOG}"
      exit 2
    fi
    echo "[done]   ${TAG}  (${JOB_END}s)"
  ) &
done

wait
TOTAL=$((SECONDS - START_ALL))
echo
echo "=========================================="
echo "ALL ${#JOBS[@]} RUNS DONE in ${TOTAL}s"
echo "Per-job logs: ${LOGDIR}"
echo "=========================================="
