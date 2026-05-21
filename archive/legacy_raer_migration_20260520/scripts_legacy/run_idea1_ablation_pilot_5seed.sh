#!/bin/bash
# Idea-1 ablation pilot — YelpChi-BWGNN, 5 seeds × 3 configs (cls-only canonical).
#
# Configs (in configs/phase2_reasoner/ablation/):
#   idea1_canonical_clsonly.yaml         (★ baseline reference for paired-t)
#   idea1_ablate_gate_uniform.yaml       (gate switch:    softmax → uniform 1/R)
#   idea1_ablate_evidence_no_proto.yaml  (evidence drop:  group C prototype dims)
#
# Usage:
#   bash scripts/run_idea1_ablation_pilot_5seed.sh                                # sequential on cuda:0
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_canonical_clsonly cuda:0     # one config only
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_ablate_gate_uniform cuda:1
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_ablate_evidence_no_proto cuda:2
#
# Parallel launch (3 GPUs, wall ≈ 5 min on warm cache):
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_canonical_clsonly cuda:0 &
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_ablate_gate_uniform cuda:1 &
#   bash scripts/run_idea1_ablation_pilot_5seed.sh idea1_ablate_evidence_no_proto cuda:2 &
#   wait
#
# Seeds: 42, 123, 456, 789, 2026 (BWGNN protocol, df=4).

set -e

PY="${PY:-/data1/mq/conda_envs/gread-core/bin/python}"
CONFIG_DIR="configs/phase2_reasoner/ablation"

if [ -n "$1" ]; then
  CONFIGS=("$1")
else
  CONFIGS=(idea1_canonical_clsonly idea1_ablate_gate_uniform idea1_ablate_evidence_no_proto)
fi
DEVICE="${2:-cuda:0}"
SEEDS=(42 123 456 789 2026)

START_ALL=$SECONDS
for CFG_NAME in "${CONFIGS[@]}"; do
  CFG="$CONFIG_DIR/${CFG_NAME}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "ERROR: config not found: $CFG" >&2
    exit 1
  fi
  echo
  echo "##########################################"
  echo "# CONFIG: $CFG_NAME"
  echo "# DEVICE: $DEVICE"
  echo "##########################################"
  for SEED in "${SEEDS[@]}"; do
    echo
    echo "[${CFG_NAME}] seed=${SEED} device=${DEVICE}"
    echo "------------------------------------------"
    START=$SECONDS
    $PY scripts/train_raer_teacher.py \
      --config "$CFG" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --run_name "${CFG_NAME}" \
      --base_ckpt_path "artifacts/checkpoints/yelpchi/bwgnn/fixed_v1_100ep/seed_${SEED}/base.pt" \
      --single-stage \
      2>&1 | tail -20
    DUR=$((SECONDS - START))
    echo "[${CFG_NAME}] seed=${SEED} done in ${DUR}s"
  done
done

TOTAL=$((SECONDS - START_ALL))
echo
echo "##########################################"
echo "# ALL CONFIGS × 5 SEEDS DONE in ${TOTAL}s"
echo "##########################################"
echo
echo "Test metrics land under: artifacts/logs/yelpchi/bwgnn/<run_name>/seed_<s>/phase2_diagnostics.json"
echo "(field: test_metrics.{auprc,roc_auc,macro_f1,g_means})"
echo
echo "Paired-t analysis: load all three runs' phase2_diagnostics.json across 5 seeds,"
echo "use scipy.stats.ttest_rel(ablation, baseline) on per-seed AUPRC, df=4."
