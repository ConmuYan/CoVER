#!/usr/bin/env bash
# Launch Idea 3 AL experiments: YelpChi-BWGNN × 5 seeds × 4 AFs × 5 budgets
# Usage: bash scripts/run_idea3_al_yelpchi_bwgnn.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
SEEDS=(42 123 456 789 2026)
AFS=(random uncertainty mitigate_af rel_af)
BUDGETS=(1 5 10 20 40)
MAX_PARALLEL=4  # 4 jobs per GPU

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2

LOGDIR=artifacts/logs/idea3_al_yelpchi_bwgnn
mkdir -p "$LOGDIR"

echo "[Launcher] GPU=$GPU  Seeds=${SEEDS[*]}  AFs=${AFS[*]}  Budgets=${BUDGETS[*]}"
echo "[Launcher] Max parallel=$MAX_PARALLEL  Total runs=$(( ${#SEEDS[@]} * ${#AFS[@]} * ${#BUDGETS[@]} ))"

running=0
pids=()

wait_for_slot() {
    # Wait for any one background job to finish
    while [ "$running" -ge "$MAX_PARALLEL" ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                unset 'pids[i]'
                running=$((running - 1))
            fi
        done
        if [ "$running" -ge "$MAX_PARALLEL" ]; then
            sleep 2
        fi
    done
}

total=0
for seed in "${SEEDS[@]}"; do
    for af in "${AFS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            wait_for_slot

            logfile="$LOGDIR/yelpchi_bwgnn_${af}_seed${seed}_budget${budget}.log"
            echo "[Launch] seed=$seed af=$af budget=$budget% → $logfile"

            python scripts/al_loop.py \
                --dataset yelpchi \
                --base_model bwgnn \
                --seed "$seed" \
                --budget_pct "$budget" \
                --af "$af" \
                --device "cuda:$GPU" \
                --al_epochs 30 \
                > "$logfile" 2>&1 &

            pids+=($!)
            running=$((running + 1))
            total=$((total + 1))
        done
    done
done

echo "[Launcher] All $total jobs launched. Waiting for completion..."
wait
echo "[Launcher] All jobs done!"
