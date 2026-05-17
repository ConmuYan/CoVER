#!/usr/bin/env bash
# Block 2: Phase 3 LEQA 5-seed x 7-cell sweep — YelpChi + BWGNN
#
# 35 runs total: 5 seeds x 7 cells (E0, E1, E2, E3, E4, E4prime, E5).
# Spread across GPU 0, 2, 3 (GPU 1 is busy).
# Resume-friendly: skips runs whose final_metrics.json already exists.
#
# chmod 644 intentionally -- user must chmod +x explicitly.
#
# Pre-flight checks:
#   - 5 LoRA inference caches
#   - 5 base checkpoints (fixed_v1_100ep)
#   - 7 phase3 config YAMLs
#
# Usage:
#   chmod +x scripts/run_phase3_block2_yelpchi_bwgnn_5seed_parallel.sh
#   bash scripts/run_phase3_block2_yelpchi_bwgnn_5seed_parallel.sh
#
# Date: 2026-05-17

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/data1/mq/conda_envs/gread-core/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="/tmp/phase3_block2_yelpchi_bwgnn_${TIMESTAMP}.log"

# Ensure conda env binaries (ninja, etc.) are on PATH and disable
# flashinfer JIT sampler (avoids ninja FileNotFoundError surfaced during
# pilot v3 vLLM init). PYTORCH_CUDA_ALLOC_CONF reduces fragmentation on
# 24GB cards during multi-cell parallel runs.
export PATH="/data1/mq/conda_envs/gread-core/bin:${PATH}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Configuration ----
SEEDS=(42 123 456 789 2026)
CELLS=(E0 E1 E2 E3 E4 E4prime E5)
CELL_CONFIGS=(E0.yaml E1.yaml E2.yaml E3.yaml E4.yaml E4prime.yaml E5.yaml)
AVAIL_GPUS=(0 2 3)
DATASET="yelpchi"
MODEL="bwgnn"

# Cells that need LoRA cache (lambda_audit > 0 and not random)
CELLS_NEED_CACHE=(E1 E3 E4 E5)
# Cell that uses random signal (E4prime)
CELL_RANDOM="E4prime"

echo "================================================================" | tee "$LOGFILE"
echo "  Phase 3 Block 2: 5-seed x 7-cell sweep" | tee -a "$LOGFILE"
echo "  Dataset:  $DATASET" | tee -a "$LOGFILE"
echo "  Model:    $MODEL" | tee -a "$LOGFILE"
echo "  Seeds:    ${SEEDS[*]}" | tee -a "$LOGFILE"
echo "  Cells:    ${CELLS[*]}" | tee -a "$LOGFILE"
echo "  GPUs:     ${AVAIL_GPUS[*]}" | tee -a "$LOGFILE"
echo "  Log:      $LOGFILE" | tee -a "$LOGFILE"
echo "  Started:  $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

# ---- Pre-flight checks ----
echo "" | tee -a "$LOGFILE"
echo "[Pre-flight] Checking required artifacts..." | tee -a "$LOGFILE"
PREFLIGHT_FAIL=0

# Check LoRA inference caches
for SEED in "${SEEDS[@]}"; do
    CACHE="artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/inference.parquet"
    if [ ! -f "$CACHE" ]; then
        echo "  MISSING: $CACHE" | tee -a "$LOGFILE"
        PREFLIGHT_FAIL=1
    fi
done

# Check base checkpoints
for SEED in "${SEEDS[@]}"; do
    CKPT="artifacts/checkpoints/${DATASET}/${MODEL}/fixed_v1_100ep/seed_${SEED}/base.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  MISSING: $CKPT" | tee -a "$LOGFILE"
        PREFLIGHT_FAIL=1
    fi
done

# Check config YAMLs
for CFG in "${CELL_CONFIGS[@]}"; do
    CFGPATH="configs/phase3/${DATASET}_${MODEL}/leqa/${CFG}"
    if [ ! -f "$CFGPATH" ]; then
        echo "  MISSING: $CFGPATH" | tee -a "$LOGFILE"
        PREFLIGHT_FAIL=1
    fi
done

if [ "$PREFLIGHT_FAIL" -eq 1 ]; then
    echo "" | tee -a "$LOGFILE"
    echo "ERROR: Pre-flight checks FAILED. Fix missing artifacts above." | tee -a "$LOGFILE"
    exit 1
fi
echo "[Pre-flight] All checks passed." | tee -a "$LOGFILE"

# ---- Helper: check if cell needs LoRA cache ----
needs_cache() {
    local cell="$1"
    for c in "${CELLS_NEED_CACHE[@]}"; do
        if [ "$c" = "$cell" ]; then
            return 0
        fi
    done
    return 1
}

# ---- Launch runs ----
echo "" | tee -a "$LOGFILE"
echo "[Launch] Starting 35 runs..." | tee -a "$LOGFILE"

TOTAL=0
SKIPPED=0
LAUNCHED=0
PIDS=()
PID_LABELS=()
START_EPOCH_TIME=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    for CELL_IDX in "${!CELLS[@]}"; do
        CELL="${CELLS[$CELL_IDX]}"
        CONFIG="configs/phase3/${DATASET}_${MODEL}/leqa/${CELL_CONFIGS[$CELL_IDX]}"

        # Read run_name from config to construct expected results path
        RUN_NAME="phase3_${DATASET}_${MODEL}_leqa_${CELL}"
        RESULTS_DIR="artifacts/results/${DATASET}/${MODEL}/${RUN_NAME}/seed_${SEED}"
        METRICS_FILE="${RESULTS_DIR}/final_metrics.json"

        TOTAL=$((TOTAL + 1))

        # Resume check
        if [ -f "$METRICS_FILE" ]; then
            echo "  SKIP: ${CELL}/seed_${SEED} (final_metrics.json exists)" | tee -a "$LOGFILE"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # GPU assignment: round-robin cell_idx % 3
        GPU_IDX=$((CELL_IDX % 3))
        GPU="${AVAIL_GPUS[$GPU_IDX]}"

        # Build command
        BASE_CKPT="artifacts/checkpoints/${DATASET}/${MODEL}/fixed_v1_100ep/seed_${SEED}/base.pt"
        CMD=(
            "$PYTHON" scripts/train_phase3_reasoner.py
            --config "$CONFIG"
            --seed "$SEED"
            --device "cuda:${GPU}"
            --base_ckpt_path "$BASE_CKPT"
            --run_name "$RUN_NAME"
        )

        # Add LoRA cache path for cells that need it
        if needs_cache "$CELL"; then
            LORA_CACHE="artifacts/cache/lora_leqa/${DATASET}/seed_${SEED}/inference.parquet"
            CMD+=(--lora_leqa_cache_path "$LORA_CACHE")
        fi

        # E4prime: random signal
        if [ "$CELL" = "$CELL_RANDOM" ]; then
            CMD+=(--use_random_lora_signal)
        fi

        RUN_LOG="/tmp/phase3_block2_${CELL}_seed${SEED}_${TIMESTAMP}.log"
        echo "  RUN:  ${CELL}/seed_${SEED} on GPU ${GPU} -> ${RUN_LOG}" | tee -a "$LOGFILE"

        # Launch in background
        "${CMD[@]}" > "$RUN_LOG" 2>&1 &
        PID=$!
        PIDS+=("$PID")
        PID_LABELS+=("${CELL}/seed_${SEED}")
        LAUNCHED=$((LAUNCHED + 1))

        # Throttle: if we have 3 concurrent jobs (one per GPU), wait for any to finish
        if [ "${#PIDS[@]}" -ge 3 ]; then
            # Wait for any one process to finish
            wait -n "${PIDS[@]}" 2>/dev/null || true
            # Clean up finished PIDs
            NEW_PIDS=()
            NEW_LABELS=()
            for i in "${!PIDS[@]}"; do
                if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    NEW_PIDS+=("${PIDS[$i]}")
                    NEW_LABELS+=("${PID_LABELS[$i]}")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            PID_LABELS=("${NEW_LABELS[@]}")
        fi
    done
done

# Wait for remaining background jobs
echo "" | tee -a "$LOGFILE"
echo "[Wait] Waiting for ${#PIDS[@]} remaining jobs..." | tee -a "$LOGFILE"
for PID in "${PIDS[@]}"; do
    wait "$PID" 2>/dev/null || true
done

END_EPOCH_TIME=$(date +%s)
WALL_SECONDS=$((END_EPOCH_TIME - START_EPOCH_TIME))
WALL_HOURS=$(echo "scale=2; $WALL_SECONDS / 3600" | bc 2>/dev/null || echo "N/A")

# ---- Summary ----
echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "  SUMMARY" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

PASS_COUNT=0
FAIL_COUNT=0
MISSING_RUNS=""

for SEED in "${SEEDS[@]}"; do
    for CELL_IDX in "${!CELLS[@]}"; do
        CELL="${CELLS[$CELL_IDX]}"
        RUN_NAME="phase3_${DATASET}_${MODEL}_leqa_${CELL}"
        METRICS_FILE="artifacts/results/${DATASET}/${MODEL}/${RUN_NAME}/seed_${SEED}/final_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            MISSING_RUNS="${MISSING_RUNS}  - ${CELL}/seed_${SEED}\n"
        fi
    done
done

echo "  Total runs:     $TOTAL" | tee -a "$LOGFILE"
echo "  Passed:         $PASS_COUNT" | tee -a "$LOGFILE"
echo "  Failed/missing: $FAIL_COUNT" | tee -a "$LOGFILE"
echo "  Skipped (pre):  $SKIPPED" | tee -a "$LOGFILE"
echo "  Launched:       $LAUNCHED" | tee -a "$LOGFILE"
echo "  Wall time:      ${WALL_SECONDS}s (~${WALL_HOURS} h)" | tee -a "$LOGFILE"
echo "  GPU-hours est:  ~$(echo "scale=1; $WALL_SECONDS * 3 / 3600" | bc 2>/dev/null || echo "N/A") (3 GPUs)" | tee -a "$LOGFILE"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "" | tee -a "$LOGFILE"
    echo "  Missing runs:" | tee -a "$LOGFILE"
    echo -e "$MISSING_RUNS" | tee -a "$LOGFILE"
fi

echo "" | tee -a "$LOGFILE"
echo "  Log: $LOGFILE" | tee -a "$LOGFILE"
echo "  Finished: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
