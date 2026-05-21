#!/bin/bash
# Run Idea 2C distillation adapter on 2 cells × 5 seeds sequentially.
# Max 2 concurrent jobs per cell (base outputs are large).
# Usage: bash scripts/run_distill_2cell_5seed_seq.sh [cuda:N]
set -e
PY="${PY:-/data1/mq/conda_envs/gread-core/bin/python}"
DEVICE="${1:-cuda:2}"
SEEDS=(42 123 456 789 2026)

declare -A CELL_CONFIG CELL_TEACHER
CELL_CONFIG[bwgnn]="configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"
CELL_CONFIG[gat]="configs/phase2_reasoner/ablation/idea1_yelpchi_gat_canonical_clsonly.yaml"
CELL_TEACHER[bwgnn]="artifacts/checkpoints/yelpchi/bwgnn/idea1_canonical_clsonly"
CELL_TEACHER[gat]="artifacts/checkpoints/yelpchi/gat/idea1_yelpchi_gat_canonical_clsonly"

CELLS=(bwgnn gat)

echo "[distill-2x5-seq] cells=${CELLS[*]} device=${DEVICE} START"
START=$SECONDS
PIDS=()

for CELL in "${CELLS[@]}"; do
  CFG="${CELL_CONFIG[$CELL]}"
  TEACHER_DIR="${CELL_TEACHER[$CELL]}"
  for SEED in "${SEEDS[@]}"; do
    TEACHER_CKPT="${TEACHER_DIR}/seed_${SEED}/reasoner.pt"
    BASE_CKPT="artifacts/checkpoints/yelpchi/${CELL}/fixed_v1_100ep/seed_${SEED}/base.pt"
    LOG="logs/_distill_${CELL}_${SEED}.log"

    # Check if already done
    RESULT="artifacts/results/yelpchi/${CELL}/idea2c_distill_adapter/seed_${SEED}/stage3_metrics.json"
    if [ -f "$RESULT" ]; then
      echo "  SKIP ${CELL} seed_${SEED} (already done)"
      continue
    fi

    echo "  [${CELL} seed_${SEED}] START"
    CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY scripts/train_distill_adapter.py \
        --config "$CFG" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --run_name idea2c_distill_adapter \
        --teacher_ckpt "$TEACHER_CKPT" \
        --base_ckpt_path "$BASE_CKPT" \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  [${CELL} seed_${SEED}] PID=$PID"

    # Wait for every 2 jobs to finish before launching next batch
    if [ ${#PIDS[@]} -ge 2 ]; then
      echo "  Waiting for batch of 2..."
      for P in "${PIDS[@]}"; do wait "$P" || true; done
      PIDS=()
    fi
  done
done

# Wait for remaining
if [ ${#PIDS[@]} -gt 0 ]; then
  echo "  Waiting for final batch..."
  for P in "${PIDS[@]}"; do wait "$P" || true; done
fi

DUR=$((SECONDS - START))
echo "[distill-2x5-seq] DONE in ${DUR}s"

# Report
echo ""
echo "=== Results ==="
for CELL in "${CELLS[@]}"; do
  echo "--- ${CELL} ---"
  for SEED in "${SEEDS[@]}"; do
    R="artifacts/results/yelpchi/${CELL}/idea2c_distill_adapter/seed_${SEED}/stage3_metrics.json"
    if [ -f "$R" ]; then
      AUPRC=$($PY -c "import json; d=json.load(open('$R')); print(f'{d[\"auprc\"]:.4f}')")
      echo "  seed_${SEED}: AUPRC=${AUPRC}"
    else
      echo "  seed_${SEED}: MISSING"
    fi
  done
done
