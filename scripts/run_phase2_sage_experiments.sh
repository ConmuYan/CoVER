#!/usr/bin/env bash
# Train Phase2 unified reasoner for SAGE base model.
# Usage:
#   bash scripts/run_phase2_sage_experiments.sh DATASET EXP GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_phase2_sage_experiments.sh yelpchi E0 2 42 123 456 789 2026
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; shift
EXP="$1"; shift
GPU="$1"; shift

CFG="configs/cover-rel-gj/phase2_ablations/phase2_${DATASET}_sage_${EXP}.yaml"
if [ ! -f "$ROOT/$CFG" ]; then
  echo "Config not found: $ROOT/$CFG"
  exit 3
fi

mkdir -p "$ROOT/artifacts/sweeps/_drivers"
LOG="$ROOT/artifacts/sweeps/_drivers/sage_phase2_${DATASET}_${EXP}_gpu${GPU}.log"
cd "$ROOT"
echo "[sage_phase2] start $(date -Is) ds=$DATASET exp=$EXP gpu=$GPU seeds=$*" >> "$LOG"
overall_rc=0
for SEED in "$@"; do
  echo "[sage_phase2] BEGIN ds=$DATASET exp=$EXP seed=$SEED @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_phase2_reasoner.py \
    --config "$CFG" --seed "$SEED" --device cuda:0 >> "$LOG" 2>&1
  rc=$?
  echo "[sage_phase2] END ds=$DATASET exp=$EXP seed=$SEED rc=$rc @ $(date -Is)" >> "$LOG"
  if [ $rc -ne 0 ]; then overall_rc=$rc; fi
done
echo "[sage_phase2] all done $(date -Is) overall_rc=$overall_rc" >> "$LOG"
exit $overall_rc
