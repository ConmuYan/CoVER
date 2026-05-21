#!/usr/bin/env bash
# E8: Hyperparameter sweep — YelpChi care_712 split
# Sweeps cbr_lambda × K for the CBR-Flash student stage.
# Reuses E1 care_712 base + teacher checkpoints.
#
# Grid:
#   cbr_lambda: [0.0, 0.1, 0.5, 1.0]
#   K (budget batch size): [256, 512, 1024, 2048]
#   = 16 configs × 2 models × 5 seeds = 160 runs
#
# Usage:
#   bash scripts/experiments/E8_run_hyperparam.sh        # default GPUs
#   bash scripts/experiments/E8_run_hyperparam.sh 2 3    # specific GPUs
set -uo pipefail

GPUS=("${@:-2 3}")
NUM_GPUS=${#GPUS[@]}
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="yelpchi"
MODELS=(sage gcn)
CKPT_ROOT="artifacts/checkpoints"
CONFIG="configs/raer_fd/experiments/E1_benchmark/care_712/student"
LOG="artifacts/logs/E8_hyperparam.log"

CBR_LAMBDAS=(0.0 0.1 0.5 1.0)
K_VALUES=(256 512 1024 2048)

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

gpu_worker() {
    set +e
    local gpu=$1
    shift
    for task in "$@"; do
        IFS='|' read -r model cbr_lambda K seed <<< "${task}"

        base_ckpt="${CKPT_ROOT}/${DATASET}/${model}/base/seed_${seed}/base.pt"
        teacher_ckpt="${CKPT_ROOT}/${DATASET}/${model}/raer_lree/seed_${seed}/raer_teacher.pt"
        if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
            log "[skip] missing deps: ${model} s=${seed}"
            continue
        fi

        run_tag="cbr_l${cbr_lambda}_K${K}"
        ckpt="${CKPT_ROOT}/${DATASET}/${model}/cbr_flash_${run_tag}/seed_${seed}/cbr_flash_student.pt"
        if [[ -f "${ckpt}" ]]; then
            log "[skip] ${model}/${run_tag} s=${seed}"
            continue
        fi

        cfg="${CONFIG}/${DATASET}_${model}.yaml"
        args=(
            scripts/train_cbr_flash.py
            --config "${cfg}" --seed "${seed}"
            --device cuda:0 --run_name "cbr_flash_${run_tag}"
            --teacher_ckpt "${teacher_ckpt}"
            --base_ckpt_path "${base_ckpt}"
            --cbr_lambda "${cbr_lambda}"
            --K "${K}"
        )
        lree="${CKPT_ROOT}/${DATASET}/${model}/raer_lree/seed_${seed}/lree.pt"
        [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")

        log "[E8] GPU=${gpu} ${model} λ=${cbr_lambda} K=${K} s=${seed}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
            >> "${LOG}" 2>&1
        [[ $? -ne 0 ]] && log "[ERROR] ${model} λ=${cbr_lambda} K=${K} s=${seed}"
    done
}

log "=== E8 Hyperparameter Sweep | gpus=${GPUS[*]} | $(date) ==="

# Build task list
tasks=()
for model in "${MODELS[@]}"; do
    for cbr_lambda in "${CBR_LAMBDAS[@]}"; do
        for K in "${K_VALUES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                tasks+=("${model}|${cbr_lambda}|${K}|${seed}")
            done
        done
    done
done

log "Total tasks: ${#tasks[@]}"

# Distribute tasks across GPUs
pids=()
gi=0
for (( i=0; i<NUM_GPUS; i++ )); do
    eval "gpu_tasks_${i}=()"
done
for task in "${tasks[@]}"; do
    eval "gpu_tasks_${gi}+=(\"${task}\")"
    gi=$(( (gi + 1) % NUM_GPUS ))
done

pids=()
for (( i=0; i<NUM_GPUS; i++ )); do
    eval "tasks_i=(\"\${gpu_tasks_${i}[@]}\")"
    if [[ ${#tasks_i[@]} -gt 0 ]]; then
        gpu_worker "${GPUS[$i]}" "${tasks_i[@]}" &
        pids+=($!)
    fi
done

for pid in "${pids[@]}"; do wait "${pid}" || true; done

log ""
log "=== E8 Hyperparameter Sweep complete | $(date) ==="
