#!/usr/bin/env bash
# Build relation features + train Stage3 anchor_gate (CoVER-REL-Gate) for SAGE base.
# Usage:
#   bash scripts/run_stage3_sage_gate.sh DATASET GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_stage3_sage_gate.sh yelpchi 2 42 123 456 789 2026
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; shift
GPU="$1"; shift

CFG="configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_${DATASET}_sage_nollm.yaml"
if [ ! -f "$ROOT/$CFG" ]; then
  echo "Config not found: $ROOT/$CFG"
  exit 3
fi

mkdir -p "$ROOT/artifacts/sweeps/_drivers"
LOG="$ROOT/artifacts/sweeps/_drivers/sage_stage3_gate_${DATASET}_gpu${GPU}.log"
cd "$ROOT"
echo "[sage_stage3_gate] start $(date -Is) ds=$DATASET gpu=$GPU seeds=$*" >> "$LOG"
overall_rc=0
for SEED in "$@"; do
  echo "[sage_stage3_gate] BEGIN ds=$DATASET seed=$SEED @ $(date -Is)" >> "$LOG"
  # Step 1: build relation features
  CUDA_VISIBLE_DEVICES=$GPU python scripts/build_relation_features.py \
    --config "$CFG" --seed "$SEED" --relation_set all >> "$LOG" 2>&1
  rc1=$?
  echo "[sage_stage3_gate] relation_features rc=$rc1 @ $(date -Is)" >> "$LOG"
  if [ $rc1 -ne 0 ]; then overall_rc=$rc1; continue; fi
  # Step 2: train stage3 anchor_gate
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_stage3.py \
    --config "$CFG" --seed "$SEED" --stratified \
    --use_relation_features \
    --run_name cover_rel_anchor_gate_nollm >> "$LOG" 2>&1
  rc2=$?
  echo "[sage_stage3_gate] train_stage3 rc=$rc2 @ $(date -Is)" >> "$LOG"
  if [ $rc2 -ne 0 ]; then overall_rc=$rc2; fi
done
echo "[sage_stage3_gate] all done $(date -Is) overall_rc=$overall_rc" >> "$LOG"
exit $overall_rc
