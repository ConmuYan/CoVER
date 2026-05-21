#!/bin/bash
# Generic Idea-1 cross-cell ablation runner — 5 seeds × 1 config × 1 cell.
#
# Looks up the base checkpoint path from the dataset/model_name embedded in
# the config (assumes BWGNN/SAGE/GCN/GAT × YelpChi/Amazon with the
# `fixed_v1_100ep` checkpoint family).
#
# Usage:
#   bash scripts/run_idea1_crosscell_5seed.sh <config_basename> <device>
#
# Example:
#   bash scripts/run_idea1_crosscell_5seed.sh idea1_yelpchi_sage_ablate_gate_uniform cuda:0
#   bash scripts/run_idea1_crosscell_5seed.sh idea1_amazon_bwgnn_ablate_evidence_no_proto cuda:2
#
# Parallel launch (2 GPUs):
#   bash scripts/run_idea1_crosscell_5seed.sh idea1_yelpchi_sage_canonical_clsonly cuda:0 &
#   bash scripts/run_idea1_crosscell_5seed.sh idea1_amazon_bwgnn_canonical_clsonly cuda:2 &
#   wait

set -e

PY="${PY:-/data1/mq/conda_envs/gread-core/bin/python}"
CONFIG_DIR="configs/phase2_reasoner/ablation"
SEEDS=(42 123 456 789 2026)

CFG_NAME="${1:?config basename required (without .yaml suffix)}"
DEVICE="${2:-cuda:0}"
CFG="${CONFIG_DIR}/${CFG_NAME}.yaml"

if [ ! -f "$CFG" ]; then
  echo "ERROR: config not found: $CFG" >&2; exit 1
fi

# Parse dataset + model from config (rough but sufficient for fixed-v1 layout)
DS=$(grep -E "^  name:" "$CFG" | head -n1 | awk '{print $2}')
MODEL=$(grep -A20 "^model:" "$CFG" | grep "  name:" | head -n1 | awk '{print $2}')
if [ -z "$DS" ] || [ -z "$MODEL" ]; then
  echo "ERROR: could not parse dataset/model from $CFG" >&2; exit 1
fi
echo "[crosscell] config=${CFG_NAME} dataset=${DS} base_model=${MODEL} device=${DEVICE}"

START_ALL=$SECONDS
for SEED in "${SEEDS[@]}"; do
  CKPT="artifacts/checkpoints/${DS}/${MODEL}/fixed_v1_100ep/seed_${SEED}/base.pt"
  if [ ! -f "$CKPT" ]; then
    echo "ERROR: base ckpt missing: $CKPT" >&2; exit 1
  fi
  echo
  echo "[${CFG_NAME}] seed=${SEED}"
  echo "------------------------------------------"
  START=$SECONDS
  $PY scripts/train_raer_teacher.py \
    --config "$CFG" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --run_name "$CFG_NAME" \
    --base_ckpt_path "$CKPT" \
    --single-stage \
    2>&1 | tail -15
  DUR=$((SECONDS - START))
  echo "[${CFG_NAME}] seed=${SEED} done in ${DUR}s"
done

TOTAL=$((SECONDS - START_ALL))
echo
echo "[${CFG_NAME}] ALL 5 SEEDS DONE in ${TOTAL}s on ${DEVICE}"
