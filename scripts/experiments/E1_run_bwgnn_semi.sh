#!/usr/bin/env bash
# E1 BWGNN Semi-supervised split (1:33:66) — YelpChi & Amazon
# Run names use "_semi" suffix to avoid overwriting care_712 checkpoints.
# Usage:
#   bash scripts/experiments/E1_run_bwgnn_semi.sh        # 3 GPUs (1,2,3)
#   bash scripts/experiments/E1_run_bwgnn_semi.sh 0 1 2 3 # 4 GPUs
set -uo pipefail

GPUS=("${@:-1 2 3}")
NUM_GPUS=${#GPUS[@]}
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASETS=(yelpchi amazon)
MODELS=(bwgnn sage gcn gat)
CONFIG_ROOT="configs/raer_fd/experiments/E1_benchmark/bwgnn_semi"
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E1_bwgnn_semi.log"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

# Per-GPU worker: processes assigned configs sequentially for a given stage.
# Args: gpu stage config1 [config2 ...]
gpu_worker() {
    set +e
    local gpu=$1
    local stage=$2
    shift 2
    for config in "$@"; do
        base=$(basename "${config}" .yaml)
        dataset="${base%%_*}"
        model="${base#*_}"

        for seed in "${SEEDS[@]}"; do
            case "${stage}" in
                base)
                    ckpt="${CKPT_ROOT}/${dataset}/${model}/base_semi/seed_${seed}/base.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${dataset}/${model}/base_semi s=${seed}"
                        continue
                    fi
                    log "[E1-semi-base] GPU=${gpu} ${dataset}/${model} s=${seed}"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_base_detector.py \
                        --config "${config}" --seed "${seed}" \
                        --run_name base_semi --stratified \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] base ${dataset}/${model} s=${seed}"
                    ;;
                teacher)
                    base_ckpt="${CKPT_ROOT}/${dataset}/${model}/base_semi/seed_${seed}/base.pt"
                    if [[ ! -f "${base_ckpt}" ]]; then
                        log "[skip-teacher] no base: ${base_ckpt}"
                        continue
                    fi
                    ckpt="${CKPT_ROOT}/${dataset}/${model}/raer_lree_semi/seed_${seed}/raer_teacher.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${dataset}/${model}/raer_lree_semi s=${seed}"
                        continue
                    fi
                    log "[E1-semi-teacher] GPU=${gpu} ${dataset}/${model} s=${seed}"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
                        --config "${config}" --seed "${seed}" \
                        --device cuda:0 --run_name raer_lree_semi \
                        --base_ckpt_path "${base_ckpt}" \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] teacher ${dataset}/${model} s=${seed}"
                    ;;
                student)
                    base_ckpt="${CKPT_ROOT}/${dataset}/${model}/base_semi/seed_${seed}/base.pt"
                    teacher_ckpt="${CKPT_ROOT}/${dataset}/${model}/raer_lree_semi/seed_${seed}/raer_teacher.pt"
                    if [[ ! -f "${base_ckpt}" ]]; then
                        log "[skip-student] no base"
                        continue
                    fi
                    if [[ ! -f "${teacher_ckpt}" ]]; then
                        log "[skip-student] no teacher: ${teacher_ckpt}"
                        continue
                    fi
                    ckpt="${CKPT_ROOT}/${dataset}/${model}/cbr_flash_semi/seed_${seed}/cbr_flash_student.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${dataset}/${model}/cbr_flash_semi s=${seed}"
                        continue
                    fi
                    args=(
                        scripts/train_cbr_flash.py
                        --config "${config}" --seed "${seed}"
                        --device cuda:0 --run_name cbr_flash_semi
                        --teacher_ckpt "${teacher_ckpt}"
                        --base_ckpt_path "${base_ckpt}"
                    )
                    lree_ckpt="${CKPT_ROOT}/${dataset}/${model}/raer_lree_semi/seed_${seed}/lree.pt"
                    if [[ -f "${lree_ckpt}" ]]; then
                        args+=(--teacher_extractor_ckpt "${lree_ckpt}")
                    fi
                    log "[E1-semi-student] GPU=${gpu} ${dataset}/${model} s=${seed}"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] student ${dataset}/${model} s=${seed}"
                    ;;
            esac
        done
    done
}

# Distribute configs across GPUs and launch one worker per GPU.
# Args: stage config_dir_suffix
run_stage() {
    local stage=$1
    local subdir=$2
    shift 2

    # Collect configs
    local configs=()
    for ds in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            cfg="${CONFIG_ROOT}/${subdir}/${ds}_${m}.yaml"
            [[ -f "${cfg}" ]] && configs+=("${cfg}")
        done
    done
    if [[ ${#configs[@]} -eq 0 ]]; then
        log "[warn] no configs for ${subdir}"
        return
    fi

    # Assign configs to GPUs (round-robin)
    local -a gpu_configs=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        eval "gpu_configs_${i}=()"
    done
    local gi=0
    for cfg in "${configs[@]}"; do
        eval "gpu_configs_${gi}+=(\"${cfg}\")"
        gi=$(( (gi + 1) % NUM_GPUS ))
    done

    # Launch one worker per GPU
    local pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        eval "local cfgs=(\"\${gpu_configs_${i}[@]}\")"
        if [[ ${#cfgs[@]} -gt 0 ]]; then
            gpu_worker "${GPUS[$i]}" "${stage}" "${cfgs[@]}" &
            pids+=($!)
        fi
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

log "=== E1 BWGNN Semi (1:33:66) | gpus=${GPUS[*]} | $(date) ==="

# --- Stage 1: Base Detectors ---
log "--- Stage 1: Base Detectors ---"
run_stage base "base_detectors"

# --- Stage 2: RAER-LREE Teachers ---
log "--- Stage 2: RAER-LREE Teachers ---"
# Pre-cache GAT base outputs to avoid teacher OOM on dense graphs
log "[info] Pre-caching GAT base outputs..."
for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        base_ckpt="${CKPT_ROOT}/${ds}/gat/base_semi/seed_${seed}/base.pt"
        [[ ! -f "${base_ckpt}" ]] && continue
        cache_dir="artifacts/base_outputs/${ds}/gat/seed_${seed}"
        # Check if already cached (override path matching base_semi)
        cached=$(ls "${cache_dir}"/_override_base_semi_seed_${seed}_base.pt 2>/dev/null)
        if [[ -n "${cached}" ]]; then
            log "[skip-cache] ${ds}/gat s=${seed}"
            continue
        fi
        g=${GPUS[0]}
        log "  [cache] ${ds}/gat s=${seed} GPU=${g}"
        CUDA_VISIBLE_DEVICES="${g}" "${PYTHON_BIN}" scripts/experiments/cache_gat_base.py \
            --config "${CONFIG_ROOT}/base_detectors/${ds}_gat.yaml" \
            --seed "${seed}" --base_ckpt_path "${base_ckpt}" \
            >> "${LOG}" 2>&1 || log "[warn] GAT cache failed ${ds} s=${seed}"
    done
done
run_stage teacher "teacher/raer_lree"

# --- Stage 3: CBR-Flash Students ---
log "--- Stage 3: CBR-Flash Students ---"
run_stage student "student"

log "=== E1 BWGNN Semi complete | $(date) ==="
