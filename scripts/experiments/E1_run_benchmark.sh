#!/usr/bin/env bash
# E1: Benchmark — YelpChi & Amazon with CARE-GNN (7:1:2) and BWGNN semi-supervised (1:33:66) splits.
# Usage:
#   bash scripts/experiments/E1_run_benchmark.sh                  # run both splits
#   bash scripts/experiments/E1_run_benchmark.sh care_712         # run only CARE-GNN split
#   bash scripts/experiments/E1_run_benchmark.sh bwgnn_semi       # run only BWGNN semi split
#   bash scripts/experiments/E1_run_benchmark.sh care_712 0       # use single GPU
set -euo pipefail

SPLIT="${1:-all}"
NUM_GPUS="${2:-4}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASETS=(yelpchi amazon)
MODELS=(bwgnn sage gcn gat)
CONFIG_ROOT="configs/raer_fd/experiments/E1_benchmark"

echo "=== E1 Benchmark | split=${SPLIT} | gpus=${NUM_GPUS} ==="

# --- Helpers ---------------------------------------------------------------

run_base_on_gpu() {
    local gpu=$1; shift
    for config in "$@"; do
        for seed in "${SEEDS[@]}"; do
            echo "[E1-base] GPU=${gpu} config=${config} seed=${seed}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_base_detector.py \
                --config "${config}" \
                --seed "${seed}" \
                --run_name base \
                --stratified \
                || echo "[E1-base ERROR] failed: ${config} seed=${seed}"
        done
    done
}

run_teacher_on_gpu() {
    local gpu=$1; shift
    for config in "$@"; do
        for seed in "${SEEDS[@]}"; do
            dataset=$(grep "name:" "${config}" | head -1 | awk '{print $2}')
            model=$(grep "name:" "${config}" | sed -n '2p' | awk '{print $2}')
            base_ckpt="artifacts/checkpoints/${dataset}/${model}/base/seed_${seed}/base.pt"
            if [[ ! -f "${base_ckpt}" ]]; then
                echo "[E1-teacher skip] missing base ckpt ${base_ckpt}"
                continue
            fi
            echo "[E1-teacher] GPU=${gpu} config=${config} seed=${seed}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
                --config "${config}" \
                --seed "${seed}" \
                --device cuda:0 \
                --run_name raer_lree \
                --base_ckpt_path "${base_ckpt}" \
                || echo "[E1-teacher ERROR] failed: ${config} seed=${seed}"
        done
    done
}

run_student_on_gpu() {
    local gpu=$1; shift
    for config in "$@"; do
        for seed in "${SEEDS[@]}"; do
            dataset=$(grep "name:" "${config}" | head -1 | awk '{print $2}')
            model=$(grep "name:" "${config}" | sed -n '2p' | awk '{print $2}')
            base_ckpt="artifacts/checkpoints/${dataset}/${model}/base/seed_${seed}/base.pt"
            teacher_ckpt="artifacts/checkpoints/${dataset}/${model}/raer_lree/seed_${seed}/raer_teacher.pt"
            lree_ckpt="artifacts/checkpoints/${dataset}/${model}/raer_lree/seed_${seed}/lree.pt"

            if [[ ! -f "${base_ckpt}" ]]; then
                echo "[E1-student skip] missing base ckpt ${base_ckpt}"
                continue
            fi
            if [[ ! -f "${teacher_ckpt}" ]]; then
                echo "[E1-student skip] missing teacher ckpt ${teacher_ckpt}"
                continue
            fi

            args=(
                scripts/train_cbr_flash.py
                --config "${config}"
                --seed "${seed}"
                --device cuda:0
                --run_name cbr_flash
                --teacher_ckpt "${teacher_ckpt}"
                --base_ckpt_path "${base_ckpt}"
            )
            if [[ -f "${lree_ckpt}" ]]; then
                args+=(--teacher_extractor_ckpt "${lree_ckpt}")
            fi

            echo "[E1-student] GPU=${gpu} config=${config} seed=${seed}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
                || echo "[E1-student ERROR] failed: ${config} seed=${seed}"
        done
    done
}

distribute_configs() {
    # Read all matching config files, distribute across GPUs
    local stage_dir=$1
    local configs=()
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            cfg="${stage_dir}/${ds}_${model}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done

    if [[ ${#configs[@]} -eq 0 ]]; then
        echo "[warn] no configs found in ${stage_dir}"
        return
    fi

    local n=${#configs[@]}
    local gpu=0
    local -a gpu_cfgs=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        eval "gpu_cfgs_${i}=()"
    done

    for cfg in "${configs[@]}"; do
        eval "gpu_cfgs_${gpu}+=(\"${cfg}\")"
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done

    echo "  Distributing ${n} configs across ${NUM_GPUS} GPUs"
}

run_stage_parallel() {
    local stage=$1
    local split_dir=$2
    local run_fn=$3

    if [[ "${stage}" == "base" ]]; then
        local stage_dir="${split_dir}/base_detectors"
    elif [[ "${stage}" == "teacher" ]]; then
        local stage_dir="${split_dir}/teacher/raer_lree"
    elif [[ "${stage}" == "student" ]]; then
        local stage_dir="${split_dir}/student"
    fi

    local pids=()
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        eval "local cfgs=(\"\${gpu_cfgs_${g}[@]}\")"
        if [[ ${#cfgs[@]} -gt 0 ]]; then
            ${run_fn} "${g}" "${cfgs[@]}" &
            pids+=($!)
        fi
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || echo "[warn] PID ${pid} exited with error"
    done
}

# --- Main loop -------------------------------------------------------------

splits_to_run=()
if [[ "${SPLIT}" == "all" ]]; then
    splits_to_run=(care_712 bwgnn_semi)
else
    splits_to_run=("${SPLIT}")
fi

for split_name in "${splits_to_run[@]}"; do
    split_dir="${CONFIG_ROOT}/${split_name}"
    if [[ ! -d "${split_dir}" ]]; then
        echo "[skip] missing split dir ${split_dir}"
        continue
    fi

    echo ""
    echo ">>> Split: ${split_name}"

    # Stage 1: Base detectors
    echo "--- Stage 1: Base Detectors ---"
    distribute_configs "${split_dir}/base_detectors"
    # Re-distribute for this stage
    configs=()
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            cfg="${split_dir}/base_detectors/${ds}_${model}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done
    pids=()
    gpu=0
    for cfg in "${configs[@]}"; do
        eval "gpu_cfgs_${gpu}+=(\"${cfg}\")"
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        eval "local_cfgs=(\"\${gpu_cfgs_${g}[@]}\")"
        [[ ${#local_cfgs[@]} -gt 0 ]] && run_base_on_gpu "${g}" "${local_cfgs[@]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    # Reset gpu arrays
    for g in $(seq 0 $((NUM_GPUS - 1))); do eval "gpu_cfgs_${g}=()"; done

    # Stage 2: RAER-LREE teachers
    echo "--- Stage 2: RAER-LREE Teachers ---"
    configs=()
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            cfg="${split_dir}/teacher/raer_lree/${ds}_${model}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done
    pids=()
    gpu=0
    for cfg in "${configs[@]}"; do
        eval "gpu_cfgs_${gpu}+=(\"${cfg}\")"
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        eval "local_cfgs=(\"\${gpu_cfgs_${g}[@]}\")"
        [[ ${#local_cfgs[@]} -gt 0 ]] && run_teacher_on_gpu "${g}" "${local_cfgs[@]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    for g in $(seq 0 $((NUM_GPUS - 1))); do eval "gpu_cfgs_${g}=()"; done

    # Stage 3: CBR-Flash students
    echo "--- Stage 3: CBR-Flash Students ---"
    configs=()
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            cfg="${split_dir}/student/${ds}_${model}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done
    pids=()
    gpu=0
    for cfg in "${configs[@]}"; do
        eval "gpu_cfgs_${gpu}+=(\"${cfg}\")"
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        eval "local_cfgs=(\"\${gpu_cfgs_${g}[@]}\")"
        [[ ${#local_cfgs[@]} -gt 0 ]] && run_student_on_gpu "${g}" "${local_cfgs[@]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done

    echo ">>> Split ${split_name} complete"
done

echo ""
echo "=== E1 Benchmark complete ==="
