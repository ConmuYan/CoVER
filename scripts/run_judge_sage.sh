#!/usr/bin/env bash
# Build judge packets + run Qwen LLM judge for SAGE base.
# Usage:
#   bash scripts/run_judge_sage.sh DATASET GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_judge_sage.sh yelpchi 2 42 123 456 789 2026
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
LOG="$ROOT/artifacts/sweeps/_drivers/sage_judge_${DATASET}_gpu${GPU}.log"
cd "$ROOT"
echo "[sage_judge] start $(date -Is) ds=$DATASET gpu=$GPU seeds=$*" >> "$LOG"
overall_rc=0
for SEED in "$@"; do
  echo "[sage_judge] BEGIN ds=$DATASET seed=$SEED @ $(date -Is)" >> "$LOG"
  # Step 1: build judge packets
  CUDA_VISIBLE_DEVICES=$GPU python scripts/build_judge_packets.py \
    --config "$CFG" \
    --gate_run_name cover_rel_anchor_gate_nollm \
    --output_run_name cover_rel_judge \
    --seed "$SEED" \
    --num_nodes 120 \
    --device cuda:0 >> "$LOG" 2>&1
  rc1=$?
  echo "[sage_judge] build_packets rc=$rc1 @ $(date -Is)" >> "$LOG"
  if [ $rc1 -ne 0 ]; then overall_rc=$rc1; continue; fi
  # Step 2: generate LLM judge outputs
  PACKETS_DIR="artifacts/judge_packets/$DATASET/sage/cover_rel_judge/seed_$SEED"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/generate_llm_judge.py \
    --packets_path "$PACKETS_DIR/judge_packets.jsonl" \
    --output_dir "$PACKETS_DIR" \
    --seed "$SEED" \
    --device cuda:0 >> "$LOG" 2>&1
  rc2=$?
  echo "[sage_judge] generate_judge rc=$rc2 @ $(date -Is)" >> "$LOG"
  if [ $rc2 -ne 0 ]; then overall_rc=$rc2; fi
done
echo "[sage_judge] all done $(date -Is) overall_rc=$overall_rc" >> "$LOG"
exit $overall_rc
