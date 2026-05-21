#!/usr/bin/env bash
# E4: TFinance SAGE (full-graph, bwgnn_424 split)
# GCN/GAT OOM on TFinance (42M edges) — only SAGE works.
# Usage:
#   bash scripts/experiments/E4_run_tfinance.sh        # default GPU 2
#   bash scripts/experiments/E4_run_tfinance.sh 2      # specific GPU
set -uo pipefail

GPU="${1:-2}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="tfinance"
MODEL="sage"
CONFIG_ROOT="configs/raer_fd/experiments/E4_generalization/bwgnn_424"
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E4_tfinance.log"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

log "=== E4 TFinance SAGE | gpu=${GPU} | $(date) ==="

# Stage 1: Base
BASE_CFG="${CONFIG_ROOT}/base_detectors/${DATASET}_${MODEL}.yaml"
if [[ ! -f "${BASE_CFG}" ]]; then
    log "[ERROR] missing base config: ${BASE_CFG}"
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_bwgnn_424/seed_${seed}/base.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/base_bwgnn_424 s=${seed}"
        continue
    fi
    log "[E4-base] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/train_base_detector.py \
        --config "${BASE_CFG}" --seed "${seed}" \
        --run_name base_bwgnn_424 --stratified \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] base ${MODEL} s=${seed}"
done

# Stage 2: Teacher
TEACHER_CFG="${CONFIG_ROOT}/teacher/raer_lree/${DATASET}_${MODEL}.yaml"
for seed in "${SEEDS[@]}"; do
    base_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_bwgnn_424/seed_${seed}/base.pt"
    if [[ ! -f "${base_ckpt}" ]]; then
        log "[skip-t] no base: ${MODEL} s=${seed}"
        continue
    fi
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_bwgnn_424/seed_${seed}/raer_teacher.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/raer_lree_bwgnn_424 s=${seed}"
        continue
    fi
    log "[E4-teacher] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
        --config "${TEACHER_CFG}" --seed "${seed}" \
        --device cuda:0 --run_name raer_lree_bwgnn_424 \
        --base_ckpt_path "${base_ckpt}" \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] teacher ${MODEL} s=${seed}"
done

# Stage 3: Student
STUDENT_CFG="${CONFIG_ROOT}/student/${DATASET}_${MODEL}.yaml"
for seed in "${SEEDS[@]}"; do
    base_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_bwgnn_424/seed_${seed}/base.pt"
    teacher_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_bwgnn_424/seed_${seed}/raer_teacher.pt"
    if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
        log "[skip-s] missing deps: ${MODEL} s=${seed}"
        continue
    fi
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/cbr_flash_bwgnn_424/seed_${seed}/cbr_flash_student.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/cbr_flash_bwgnn_424 s=${seed}"
        continue
    fi
    args=(
        scripts/train_cbr_flash.py
        --config "${STUDENT_CFG}" --seed "${seed}"
        --device cuda:0 --run_name cbr_flash_bwgnn_424
        --teacher_ckpt "${teacher_ckpt}"
        --base_ckpt_path "${base_ckpt}"
    )
    lree="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_bwgnn_424/seed_${seed}/lree.pt"
    [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")
    log "[E4-student] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${args[@]}" \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] student ${MODEL} s=${seed}"
done

log "=== E4 TFinance SAGE complete | $(date) ==="
