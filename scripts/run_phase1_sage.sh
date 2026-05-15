#!/usr/bin/env bash
# Train Phase1 SAGE base model for given dataset+gpu+seeds.
# Usage:
#   bash scripts/run_phase1_sage.sh DATASET GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_phase1_sage.sh yelpchi 2 42 123 456 789 2026
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; shift
GPU="$1"; shift

CFG="configs/${DATASET}_sage.yaml"
if [ ! -f "$ROOT/$CFG" ]; then
  echo "Config not found: $ROOT/$CFG"
  exit 3
fi

mkdir -p "$ROOT/artifacts/sweeps/_drivers"
LOG="$ROOT/artifacts/sweeps/_drivers/sage_phase1_${DATASET}_gpu${GPU}.log"
cd "$ROOT"
echo "[sage_phase1] start $(date -Is) ds=$DATASET gpu=$GPU seeds=$*" >> "$LOG"
overall_rc=0
for SEED in "$@"; do
  echo "[sage_phase1] BEGIN ds=$DATASET seed=$SEED @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_stage1.py \
    --config "$CFG" --seed "$SEED" --stratified --deterministic \
    --run_name base >> "$LOG" 2>&1
  rc=$?
  echo "[sage_phase1] END ds=$DATASET seed=$SEED rc=$rc @ $(date -Is)" >> "$LOG"
  if [ $rc -ne 0 ]; then overall_rc=$rc; fi
done
echo "[sage_phase1] all done $(date -Is) overall_rc=$overall_rc" >> "$LOG"
exit $overall_rc
