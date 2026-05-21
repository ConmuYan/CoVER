#!/usr/bin/env bash
# E2: Scarcity — YelpChi care_712 with scarcity_ratio in [1%, 5%, 10%, 20%, 50%]
# Models: bwgnn, sage, gcn, gat × 3 stages × 5 seeds = 300 runs
# Usage:
#   bash scripts/experiments/E2_run_scarcity.sh        # default GPUs
#   bash scripts/experiments/E2_run_scarcity.sh 2 3    # specific GPUs
set -uo pipefail

GPUS=("${@:-2 3}")
NUM_GPUS=${#GPUS[@]}
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="yelpchi"
MODELS=(bwgnn sage gcn gat)
PCTS=(1 5 10 20 50)
CONFIG_ROOT="configs/raer_fd/experiments/E2_scarcity"
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E2_scarcity.log"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

# Per-GPU worker: processes assigned configs sequentially
gpu_worker() {
    set +e
    local gpu=$1 stage=$2 pct=$3
    shift 3
    for config in "$@"; do
        model=$(basename "${config}" .yaml | sed "s/${DATASET}_//")
        for seed in "${SEEDS[@]}"; do
            case "${stage}" in
                base)
                    ckpt="${CKPT_ROOT}/${DATASET}/${model}/base_scarcity_${pct}pct/seed_${seed}/base.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${model}/base_s${pct}pct s=${seed}"
                        continue
                    fi
                    log "[E2-base] GPU=${gpu} ${model} s=${seed} scarcity=${pct}%"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_base_detector.py \
                        --config "${config}" --seed "${seed}" \
                        --run_name "base_scarcity_${pct}pct" --stratified \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] base ${model} s=${seed} p=${pct}"
                    ;;
                teacher)
                    base_ckpt="${CKPT_ROOT}/${DATASET}/${model}/base_scarcity_${pct}pct/seed_${seed}/base.pt"
                    if [[ ! -f "${base_ckpt}" ]]; then
                        log "[skip-t] ${model} no base s=${seed} p=${pct}"
                        continue
                    fi
                    ckpt="${CKPT_ROOT}/${DATASET}/${model}/raer_lree_scarcity_${pct}pct/seed_${seed}/raer_teacher.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${model}/teacher_s${pct}pct s=${seed}"
                        continue
                    fi
                    log "[E2-teacher] GPU=${gpu} ${model} s=${seed} scarcity=${pct}%"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
                        --config "${config}" --seed "${seed}" \
                        --device cuda:0 --run_name "raer_lree_scarcity_${pct}pct" \
                        --base_ckpt_path "${base_ckpt}" \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] teacher ${model} s=${seed} p=${pct}"
                    ;;
                student)
                    base_ckpt="${CKPT_ROOT}/${DATASET}/${model}/base_scarcity_${pct}pct/seed_${seed}/base.pt"
                    teacher_ckpt="${CKPT_ROOT}/${DATASET}/${model}/raer_lree_scarcity_${pct}pct/seed_${seed}/raer_teacher.pt"
                    if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
                        log "[skip-s] ${model} missing deps s=${seed} p=${pct}"
                        continue
                    fi
                    ckpt="${CKPT_ROOT}/${DATASET}/${model}/cbr_flash_scarcity_${pct}pct/seed_${seed}/cbr_flash_student.pt"
                    if [[ -f "${ckpt}" ]]; then
                        log "[skip] ${model}/student_s${pct}pct s=${seed}"
                        continue
                    fi
                    args=(
                        scripts/train_cbr_flash.py
                        --config "${config}" --seed "${seed}"
                        --device cuda:0 --run_name "cbr_flash_scarcity_${pct}pct"
                        --teacher_ckpt "${teacher_ckpt}"
                        --base_ckpt_path "${base_ckpt}"
                    )
                    lree="${CKPT_ROOT}/${DATASET}/${model}/raer_lree_scarcity_${pct}pct/seed_${seed}/lree.pt"
                    [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")
                    log "[E2-student] GPU=${gpu} ${model} s=${seed} scarcity=${pct}%"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
                        >> "${LOG}" 2>&1
                    [[ $? -ne 0 ]] && log "[ERROR] student ${model} s=${seed} p=${pct}"
                    ;;
            esac
        done
    done
}

# Run one (stage, pct): distribute configs across GPUs, one worker per GPU
run_stage_pct() {
    local stage=$1 pct=$2 subdir=$3
    local configs=()
    for m in "${MODELS[@]}"; do
        cfg="${CONFIG_ROOT}/care_712_scarcity_${pct}pct/${subdir}/${DATASET}_${m}.yaml"
        [[ -f "${cfg}" ]] && configs+=("${cfg}")
    done
    [[ ${#configs[@]} -eq 0 ]] && return

    local pids=()
    local gi=0
    for cfg in "${configs[@]}"; do
        g=${GPUS[$((gi % NUM_GPUS))]}
        gpu_worker "${g}" "${stage}" "${pct}" "${cfg}" &
        pids+=($!)
        gi=$((gi + 1))
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

log "=== E2 Scarcity | gpus=${GPUS[*]} | $(date) ==="

for pct in "${PCTS[@]}"; do
    log ""
    log ">>> Scarcity = ${pct}%"

    log "--- Stage 1: Base (p=${pct}%) ---"
    run_stage_pct base "${pct}" "base_detectors"

    log "--- Stage 2: Teacher (p=${pct}%) ---"
    # Pre-cache GAT base outputs for this pct level
    for seed in "${SEEDS[@]}"; do
        base_ckpt="${CKPT_ROOT}/${DATASET}/gat/base_scarcity_${pct}pct/seed_${seed}/base.pt"
        [[ ! -f "${base_ckpt}" ]] && continue
        cache_path="artifacts/base_outputs/${DATASET}/gat/seed_${seed}/_override_base_scarcity_${pct}pct_seed_${seed}_base.pt"
        [[ -f "${cache_path}" ]] && continue
        g=${GPUS[0]}
        log "  [cache-gat] s=${seed} GPU=${g} pct=${pct}"
        CUDA_VISIBLE_DEVICES="${g}" "${PYTHON_BIN}" scripts/experiments/cache_gat_base.py \
            --config "${CONFIG_ROOT}/care_712_scarcity_${pct}pct/base_detectors/${DATASET}_gat.yaml" \
            --seed "${seed}" --base_ckpt_path "${base_ckpt}" \
            >> "${LOG}" 2>&1 || log "[warn] GAT cache failed s=${seed} p=${pct}"
    done
    run_stage_pct teacher "${pct}" "teacher/raer_lree"

    log "--- Stage 3: Student (p=${pct}%) ---"
    run_stage_pct student "${pct}" "student"
done

log ""
log "=== E2 Sccarcity complete | $(date) ==="
