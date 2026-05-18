#!/usr/bin/env bash
# Launch full Idea 3 AL experiments: YelpChi × {BWGNN, GAT} × 5 seeds × 4 AFs × 5 budgets
# 4-way parallel per GPU × 4 GPUs = 16 concurrent jobs
# Skips jobs that already have learning_curve.json
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 123 456 789 2026)
AFS=(random uncertainty mitigate_af rel_af)
BUDGETS=(1 5 10 20 40)
GPUS=(0 1 2 3)
MAX_PER_GPU=4

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

BASEMODELS=()
if [[ "${1:-all}" == "all" || "${1:-all}" == "bwgnn" ]]; then
    BASEMODELS+=(bwgnn)
fi
if [[ "${1:-all}" == "all" || "${1:-all}" == "gat" ]]; then
    BASEMODELS+=(gat)
fi

LOGDIR=artifacts/logs/idea3_al_full
mkdir -p "$LOGDIR"

RESULTS_BASE="artifacts/results/al/yelpchi"

echo "============================================================"
echo "[Launcher] Idea 3 Full Experiment"
echo "  Models:   ${BASEMODELS[*]}"
echo "  Seeds:    ${SEEDS[*]}"
echo "  AFs:      ${AFS[*]}"
echo "  Budgets:  ${BUDGETS[*]}"
echo "  GPUs:     ${GPUS[*]}  (${MAX_PER_GPU} per GPU)"
echo "============================================================"

# Build job list, skipping existing results
jobs=()
skipped=0
for bm in "${BASEMODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for af in "${AFS[@]}"; do
            for budget in "${BUDGETS[@]}"; do
                result_file="$RESULTS_BASE/$bm/$af/seed_$seed/budget_$budget/learning_curve.json"
                if [[ -f "$result_file" ]]; then
                    skipped=$((skipped + 1))
                    continue
                fi
                jobs+=("$bm $seed $af $budget")
            done
        done
    done
done

total=${#jobs[@]}
echo "[Launcher] Total new jobs: $total  (skipped $skipped existing)"
echo ""

if [[ $total -eq 0 ]]; then
    echo "[Launcher] Nothing to run. All results exist."
    exit 0
fi

# Track PIDs per GPU: gpu_pids[gpu_idx]="pid1 pid2 pid3"
declare -A gpu_pids
for g_idx in "${!GPUS[@]}"; do
    gpu_pids[$g_idx]=""
done

wait_for_slot() {
    while true; do
        for g_idx in "${!GPUS[@]}"; do
            local running=""
            for pid in ${gpu_pids[$g_idx]}; do
                if kill -0 "$pid" 2>/dev/null; then
                    running="$running $pid"
                fi
            done
            gpu_pids[$g_idx]="$running"
            local count
            count=$(echo "$running" | wc -w)
            if [[ $count -lt $MAX_PER_GPU ]]; then
                echo "$g_idx"
                return 0
            fi
        done
        sleep 3
    done
}

launched=0
for job in "${jobs[@]}"; do
    read -r bm seed af budget <<< "$job"

    free_gpu=$(wait_for_slot)
    gpu_id="${GPUS[$free_gpu]}"

    logfile="$LOGDIR/yelpchi_${bm}_${af}_seed${seed}_budget${budget}.log"

    python scripts/al_loop.py \
        --dataset yelpchi \
        --base_model "$bm" \
        --seed "$seed" \
        --budget_pct "$budget" \
        --af "$af" \
        --device "cuda:$gpu_id" \
        --al_epochs 30 \
        > "$logfile" 2>&1 &

    pid=$!
    gpu_pids[$free_gpu]="${gpu_pids[$free_gpu]} $pid"
    launched=$((launched + 1))
    echo "[Launch $launched/$total] $bm seed=$seed af=$af budget=${budget}% → GPU $gpu_id (pid=$pid)"
done

echo ""
echo "[Launcher] All $launched jobs launched. Waiting for completion..."
wait
echo "[Launcher] All jobs done!"

# Count results
n_results=$(find "$RESULTS_BASE" -name "learning_curve.json" | wc -l)
echo "[Launcher] Total result files: $n_results"
