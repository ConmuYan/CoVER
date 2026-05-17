#!/usr/bin/env bash
# Block 1 sanity pilot -- Phase 3 LEQA pilot wrapper.
#
# Per STAGE_2_IMPLEMENTATION_CHECKLIST.md section 5.
# chmod 644 intentionally -- user must chmod +x explicitly.
#
# Usage:
#   chmod +x scripts/run_phase3_block1_sanity.sh
#   bash scripts/run_phase3_block1_sanity.sh leqa 42 yelpchi
#
# Date: 2026-05-17

set -euo pipefail

PROPOSAL=${1:-leqa}
SEED=${2:-42}
DATASET=${3:-yelpchi}
BUDGET_GPU_H=${4:-1.0}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/data1/mq/conda_envs/gread-core/bin/python"

# Ensure conda env binaries (ninja, etc.) are on PATH so vLLM's flashinfer
# JIT compile can find them. Disable flashinfer sampler as belt-and-braces.
export PATH="/data1/mq/conda_envs/gread-core/bin:${PATH}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================"
echo "  Block 1 sanity pilot"
echo "  Proposal: $PROPOSAL"
echo "  Seed:     $SEED"
echo "  Dataset:  $DATASET"
echo "  Budget:   ${BUDGET_GPU_H} GPU-h"
echo "================================================================"

# ---- Step 1: Generate synthetic training data (< 5 min) ----
echo ""
echo "[Step 1/4] Generating synthetic training data..."
$PYTHON scripts/generate_synth_token_quality.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --n 2000 \
    --out "data/synth_token_quality_${DATASET}_seed${SEED}.jsonl"

echo "[Step 1/4] Done."

# ---- Step 2: LoRA train 1-epoch quick mode (~ 50 min on 1x RTX 3090) ----
echo ""
echo "[Step 2/4] Training LoRA (1-epoch quick mode)..."
$PYTHON scripts/train_lora_leqa.py \
    --base_model /data1/mq/models/Qwen3-4B-Instruct-2507 \
    --train_data "data/synth_token_quality_${DATASET}_seed${SEED}.jsonl" \
    --output_dir "artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/adapter" \
    --epochs 1 \
    --rank 16 \
    --alpha 16 \
    --lr 2e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_len 1024 \
    --bf16 \
    --seed "$SEED"

echo "[Step 2/4] Done."

# ---- Step 3: Inference on 100-packet sample (~ 5 min via vLLM) ----
echo ""
echo "[Step 3/4] Running LoRA inference on 100-packet sample..."

# Auto-detect packets path
PACKETS_PATH="artifacts/judge_packets/${DATASET}/bwgnn/cover_rel_judge_revived/seed_${SEED}/judge_packets.jsonl"
if [ ! -f "$PACKETS_PATH" ]; then
    PACKETS_PATH="artifacts/err_cache/${DATASET}/bwgnn/seed_${SEED}/teacher_payloads.jsonl"
fi

$PYTHON scripts/cache_lora_leqa_outputs.py \
    --adapter "artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/adapter" \
    --packets "$PACKETS_PATH" \
    --sample 100 \
    --out "artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/sanity_inference.parquet" \
    --seed "$SEED"

echo "[Step 3/4] Done."

# ---- Step 4: Pilot pass-criteria check (< 1 min) ----
echo ""
echo "[Step 4/4] Checking pilot pass criteria..."

mkdir -p "artifacts/diagnostics"

$PYTHON scripts/check_phase3_pilot.py \
    --inference "artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/sanity_inference.parquet" \
    --min_std 0.1 \
    --min_spearman 0.2 \
    --min_nonnone_pct 0.3 \
    --out "artifacts/diagnostics/phase3B_pilot_${DATASET}_seed${SEED}.json"

PILOT_PASS=$(jq -r '.pass' "artifacts/diagnostics/phase3B_pilot_${DATASET}_seed${SEED}.json")

echo ""
echo "================================================================"
if [ "$PILOT_PASS" = "true" ]; then
    echo "[OK] Block 1 sanity pilot PASSED for seed $SEED on $DATASET."
    echo "     Proceeding to Block 2 is now safe."
    echo "================================================================"
    exit 0
else
    echo "[FAIL] Block 1 sanity pilot FAILED."
    echo "       Auto-fallback per GATE_1_DECISION.md -> switch to Proposal A."
    echo ""
    jq '.' "artifacts/diagnostics/phase3B_pilot_${DATASET}_seed${SEED}.json"
    echo "================================================================"
    exit 1
fi
