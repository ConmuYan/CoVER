#!/usr/bin/env bash
# Train stage3 CoVER-REL Judge (with real LLM features) for given dataset+gpu+seeds.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; shift
GPU="$1"; shift
mkdir -p "$ROOT/artifacts/sweeps/_drivers"
LOG="$ROOT/artifacts/sweeps/_drivers/judge_${DATASET}_gpu${GPU}.log"
cd "$ROOT"
if [ "$DATASET" = "yelpchi" ]; then
  CFG="configs/stage3_cover_rel_judge_yelpchi.yaml"
elif [ "$DATASET" = "amazon" ]; then
  CFG="configs/stage3_cover_rel_judge_amazon.yaml"
fi
echo "[judge] start $(date -Is) dataset=$DATASET gpu=$GPU seeds=$*" >> "$LOG"
for SEED in "$@"; do
  JF="artifacts/judge_packets/$DATASET/bwgnn/cover_rel_judge/seed_$SEED/judge_features.pt"
  JM="artifacts/judge_packets/$DATASET/bwgnn/cover_rel_judge/seed_$SEED/judge_packet_meta.json"
  echo "[judge] BEGIN seed=$SEED @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_stage3.py \
    --config "$CFG" --seed "$SEED" --stratified \
    --run_name rule --stage2_run_name rule \
    --stage3_run_name cover_rel_judge_strength_gate \
    --use_relation_features \
    --use_llm_judge \
    --judge_features_path "$JF" \
    --judge_feature_meta_path "$JM" \
    --device cuda:0 >> "$LOG" 2>&1
  rc=$?
  echo "[judge] END seed=$SEED rc=$rc @ $(date -Is)" >> "$LOG"
done
echo "[judge] all done $(date -Is)" >> "$LOG"
