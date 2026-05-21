#!/bin/bash
# Run 5 seeds of one Idea-1 ablation config in PARALLEL on one GPU.
#
# Each reasoner job uses ~750 MiB / 24 GB; 5 in parallel ≈ 4 GB.
#
# Usage:
#   bash scripts/run_5seeds_parallel.sh <config_basename> <device>
#
# Example:
#   bash scripts/run_5seeds_parallel.sh idea1_yelpchi_gcn_canonical_clsonly cuda:0

set -e
PY="${PY:-/data1/mq/conda_envs/gread-core/bin/python}"
CONFIG_DIR="configs/phase2_reasoner/ablation"
SEEDS=(42 123 456 789 2026)

CFG_NAME="${1:?config basename required}"
DEVICE="${2:-cuda:0}"
CFG="${CONFIG_DIR}/${CFG_NAME}.yaml"

if [ ! -f "$CFG" ]; then echo "ERROR: $CFG not found" >&2; exit 1; fi

DS=$(grep -E "^  name:" "$CFG" | head -n1 | awk '{print $2}')
MODEL=$(grep -A20 "^model:" "$CFG" | grep "  name:" | head -n1 | awk '{print $2}')

echo "[parallel-5] config=${CFG_NAME} ds=${DS} model=${MODEL} device=${DEVICE} START"
START=$SECONDS

for SEED in "${SEEDS[@]}"; do
  CKPT="artifacts/checkpoints/${DS}/${MODEL}/fixed_v1_100ep/seed_${SEED}/base.pt"
  if [ ! -f "$CKPT" ]; then echo "ERROR: ckpt missing: $CKPT" >&2; exit 1; fi
  (
    $PY scripts/train_raer_teacher.py \
      --config "$CFG" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --run_name "$CFG_NAME" \
      --base_ckpt_path "$CKPT" \
      --single-stage \
      > "logs/_seed_${CFG_NAME}_${SEED}.log" 2>&1
  ) &
done
wait
DUR=$((SECONDS - START))
echo "[parallel-5] ${CFG_NAME} DONE in ${DUR}s (5 seeds parallel)"
