#!/bin/bash
# 5-seed champion run for Full CoVER (YelpChi/BWGNN base)
# Recipe: --alpha_bias_init 0 --alpha_max 0.3 --lambda_align 3e-2 --single-stage
# Source config: artifacts/logs/yelpchi/bwgnn/phase2_yelpchi_bwgnn_revised_r3_full_alpha01_lint1em2/seed_42/repro_config.yaml

set -e

CONFIG="artifacts/logs/yelpchi/bwgnn/phase2_yelpchi_bwgnn_revised_r3_full_alpha01_lint1em2/seed_42/repro_config.yaml"
RUN_NAME="phase2_yelpchi_bwgnn_champion_v1"
DEVICE="cuda:1"
PY="/data1/mq/conda_envs/gread-core/bin/python"

START_ALL=$SECONDS
for SEED in 42 123 456 789 2026; do
  echo "=========================================="
  echo "[champion-v1] seed=${SEED}  device=${DEVICE}"
  echo "=========================================="
  START=$SECONDS
  $PY scripts/train_phase2_reasoner.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --run_name "$RUN_NAME" \
    --alpha_bias_init 0.0 \
    --alpha_max 0.3 \
    --lambda_align 3e-2 \
    --single-stage \
  2>&1 | tail -25
  DUR=$((SECONDS - START))
  echo "[champion-v1] seed=${SEED} done in ${DUR}s"
done
TOTAL=$((SECONDS - START_ALL))
echo
echo "=========================================="
echo "ALL 5 SEEDS DONE in ${TOTAL}s"
echo "=========================================="
