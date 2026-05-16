#!/bin/bash
# Run LLM judge with vLLM backend, 5 seeds × 2000 packets each.
# Distribute across cuda:1 (3 seeds) + cuda:2 (2 seeds) for parallel throughput.
#
# Estimated wall: ~5-10 min total (vLLM batched continuous batching).

set -e
PY="/data1/mq/conda_envs/gread-core/bin/python"
MODEL_PATH="/data1/mq/models/Qwen3-4B-Instruct-2507"
OUTPUT_RUN="cover_rel_judge_revived"

LOGDIR=/tmp/llm_judge_revived_$$
mkdir -p "$LOGDIR"

CUDA1_SEEDS=(42 456 2026)
CUDA2_SEEDS=(123 789)

run_seed() {
  local SEED="$1"
  local GPU="$2"
  local PKT_DIR="artifacts/judge_packets/yelpchi/bwgnn/${OUTPUT_RUN}/seed_${SEED}"
  local PKT="${PKT_DIR}/judge_packets.jsonl"
  local TAG="seed${SEED}_cuda${GPU}"
  local LOG="${LOGDIR}/${TAG}.log"
  if [ ! -f "$PKT" ]; then
    echo "[skip]   ${TAG} — no packets at ${PKT}"
    return
  fi
  echo "[launch] ${TAG}  ($(wc -l <"$PKT") packets)"
  JOB_START=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PATH="/data1/mq/conda_envs/gread-core/bin:$PATH" \
  VLLM_USE_FLASHINFER=0 \
  VLLM_ATTENTION_BACKEND=XFORMERS \
  $PY scripts/generate_llm_judge.py \
    --packets_path "$PKT" \
    --output_dir "$PKT_DIR" \
    --model_path "$MODEL_PATH" \
    --backend vllm \
    --device cuda:0 \
    --dtype float16 \
    --gpu_memory_utilization 0.80 \
    --max_new_tokens 160 \
    --seed "$SEED" \
    2>&1 | sed "s/^/[${TAG}] /" >"$LOG"
  N=$(wc -l <"${PKT_DIR}/accepted_judge.jsonl" 2>/dev/null || echo 0)
  REJ=$(wc -l <"${PKT_DIR}/rejected_judge.jsonl" 2>/dev/null || echo 0)
  echo "[done]   ${TAG} ($((SECONDS-JOB_START))s) accepted=${N} rejected=${REJ}"
}

run_device() {
  local GPU="$1"; shift
  for SEED in "$@"; do
    run_seed "$SEED" "$GPU"
  done
}

START=$SECONDS
echo "============================================================"
echo "[LLM judge vLLM] cuda:1=${#CUDA1_SEEDS[@]} seeds, cuda:2=${#CUDA2_SEEDS[@]} seeds"
echo "============================================================"

run_device "1" "${CUDA1_SEEDS[@]}" &
P1=$!
run_device "2" "${CUDA2_SEEDS[@]}" &
P2=$!
wait $P1
wait $P2

ELAPSED=$((SECONDS - START))
echo
echo "===== LLM judge DONE in ${ELAPSED}s ====="
echo "Logs: ${LOGDIR}"
