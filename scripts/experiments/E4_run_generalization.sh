#!/usr/bin/env bash
# E4: Generalization — TFinance & TSocial with bwgnn_424 split.
# Usage:
#   bash scripts/experiments/E4_run_generalization.sh           # all
#   bash scripts/experiments/E4_run_generalization.sh tfinance   # only TFinance
set -euo pipefail

DS_FILTER="${1:-all}"
NUM_GPUS="${2:-4}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASETS=(tfinance tsocial)
MODELS=(gcn gat sage)
CONFIG_ROOT="configs/raer_fd/experiments/E4_generalization/bwgnn_424"

echo "=== E4 Generalization | dataset=${DS_FILTER} | gpus=${NUM_GPUS} ==="

run_base() {
    local gpu=$1; local config=$2;
    local ds=$(grep "  name:" "${config}" | head -1 | awk '{print $2}')
    for seed in "${SEEDS[@]}"; do
        echo "[E4-base] GPU=${gpu} ${ds} seed=${seed}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_base_detector.py \
            --config "${config}" --seed "${seed}" --run_name base --stratified
    done
}

run_teacher() {
    local gpu=$1; local config=$2;
    local ds=$(grep "  name:" "${config}" | head -1 | awk '{print $2}')
    local model=$(grep "  name:" "${config}" | sed -n '2p' | awk '{print $2}')
    for seed in "${SEEDS[@]}"; do
        local base_ckpt="artifacts/checkpoints/${ds}/${model}/base/seed_${seed}/base.pt"
        [[ ! -f "${base_ckpt}" ]] && { echo "[skip] ${base_ckpt}"; continue; }
        echo "[E4-teacher] GPU=${gpu} ${ds}/${model} seed=${seed}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
            --config "${config}" --seed "${seed}" --device cuda:0 \
            --run_name raer_lree --base_ckpt_path "${base_ckpt}"
    done
}

run_student() {
    local gpu=$1; local config=$2;
    local ds=$(grep "  name:" "${config}" | head -1 | awk '{print $2}')
    local model=$(grep "  name:" "${config}" | sed -n '2p' | awk '{print $2}')
    for seed in "${SEEDS[@]}"; do
        local base_ckpt="artifacts/checkpoints/${ds}/${model}/base/seed_${seed}/base.pt"
        local teacher_ckpt="artifacts/checkpoints/${ds}/${model}/raer_lree/seed_${seed}/raer_teacher.pt"
        local lree_ckpt="artifacts/checkpoints/${ds}/${model}/raer_lree/seed_${seed}/lree.pt"
        [[ ! -f "${base_ckpt}" || ! -f "${teacher_ckpt}" ]] && continue
        local args=(scripts/train_cbr_flash.py --config "${config}" --seed "${seed}"
            --device cuda:0 --run_name cbr_flash --teacher_ckpt "${teacher_ckpt}"
            --base_ckpt_path "${base_ckpt}")
        [[ -f "${lree_ckpt}" ]] && args+=(--teacher_extractor_ckpt "${lree_ckpt}")
        echo "[E4-student] GPU=${gpu} ${ds}/${model} seed=${seed}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}"
    done
}

datasets=()
if [[ "${DS_FILTER}" == "all" ]]; then
    datasets=("${DATASETS[@]}")
else
    datasets=("${DS_FILTER}")
fi

for stage in base teacher student; do
    echo "--- Stage: ${stage} ---"
    if [[ "${stage}" == "base" ]]; then
        stage_dir="${CONFIG_ROOT}/base_detectors"
        run_fn="run_base"
    elif [[ "${stage}" == "teacher" ]]; then
        stage_dir="${CONFIG_ROOT}/teacher/raer_lree"
        run_fn="run_teacher"
    else
        stage_dir="${CONFIG_ROOT}/student"
        run_fn="run_student"
    fi

    configs=()
    for ds in "${datasets[@]}"; do
        for model in "${MODELS[@]}"; do
            cfg="${stage_dir}/${ds}_${model}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done
    [[ ${#configs[@]} -eq 0 ]] && { echo "[skip] no configs for stage ${stage}"; continue; }

    pids=()
    for i in "${!configs[@]}"; do
        gpu=$(( i % NUM_GPUS ))
        ${run_fn} "${gpu}" "${configs[$i]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
done

echo ""
echo "=== E4 Generalization complete ==="
