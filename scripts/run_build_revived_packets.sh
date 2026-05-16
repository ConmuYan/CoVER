#!/bin/bash
# Build 2000 judge packets per seed using base-uncertainty selection.
# Anchored on fixed_v1_100ep BWGNN base. Skips stage2/stage3 deps (uniform gate).
# Output: artifacts/judge_packets/yelpchi/bwgnn/cover_rel_judge_revived/seed_{seed}/

set -e
PY="/data1/mq/conda_envs/gread-core/bin/python"
CONFIG="configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_gate_nollm.yaml"
NUM_NODES=2000
OUTPUT_RUN="cover_rel_judge_revived"

SEEDS=(42 123 456 789 2026)
LOGDIR=/tmp/build_packets_$$
mkdir -p "$LOGDIR"

START=$SECONDS
echo "[build_packets] num_nodes=${NUM_NODES} selection=base_uncertain base=fixed_v1_100ep"

for SEED in "${SEEDS[@]}"; do
  TAG="seed${SEED}"
  LOG="${LOGDIR}/${TAG}.log"
  OUTDIR="artifacts/judge_packets/yelpchi/bwgnn/${OUTPUT_RUN}/seed_${SEED}"
  mkdir -p "$OUTDIR"
  JOB_START=$SECONDS
  echo "[launch] ${TAG}"
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY scripts/build_judge_packets.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --base_run_name fixed_v1_100ep \
    --num_nodes ${NUM_NODES} \
    --selection_mode base_uncertain \
    --skip_stage_deps \
    --output_run_name "$OUTPUT_RUN" \
    --device cuda:0 \
    --output_dir "$OUTDIR" \
    2>&1 | sed "s/^/[${TAG}] /" >"$LOG"
  N=$(wc -l <"$OUTDIR/judge_packets.jsonl")
  echo "[done]  ${TAG} ($((SECONDS-JOB_START))s, ${N} packets)"
done

ELAPSED=$((SECONDS - START))
echo
echo "===== build_packets DONE in ${ELAPSED}s ====="
echo "Logs: ${LOGDIR}"
