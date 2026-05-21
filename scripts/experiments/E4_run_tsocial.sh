#!/usr/bin/env bash
# E4: TSocial SAGE (mini-batch pipeline, bwgnn_424 split)
# 5.7M nodes / 146M edges — must use mini-batch base + scalable LREE teacher.
# Usage:
#   bash scripts/experiments/E4_run_tsocial.sh        # default GPU 0
#   bash scripts/experiments/E4_run_tsocial.sh 0      # specific GPU
set -uo pipefail

GPU="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="tsocial"
MODEL="sage"
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E4_tsocial.log"

# Configs — reuses existing large_graph pipeline
BASE_CFG="configs/raer_fd/large_graph/base_detectors/tsocial_sage_neighbor_mb.yaml"
TEACHER_CFG="configs/raer_fd/large_graph/teacher/raer_lree_scalable/tsocial_sage.yaml"
STUDENT_CFG="configs/raer_fd/large_graph/student/cbr_flash_tsocial_sage_lree_scalable.yaml"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

log "=== E4 TSocial SAGE (mini-batch) | gpu=${GPU} | $(date) ==="

# Stage 1: Mini-batch Base
log "--- Stage 1: Mini-batch Base ---"
for seed in "${SEEDS[@]}"; do
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_neighbor_mb/seed_${seed}/base.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/base_neighbor_mb s=${seed}"
        continue
    fi
    log "[E4-base-mb] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/train_base_detector_minibatch.py \
        --config "${BASE_CFG}" --seed "${seed}" \
        --run_name base_neighbor_mb --device cuda:0 \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] base-mb ${MODEL} s=${seed}"
done

# Stage 2: Scalable LREE Teacher
log "--- Stage 2: Scalable LREE Teacher ---"
for seed in "${SEEDS[@]}"; do
    base_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_neighbor_mb/seed_${seed}/base.pt"
    if [[ ! -f "${base_ckpt}" ]]; then
        log "[skip-t] no base: ${MODEL} s=${seed}"
        continue
    fi
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_scalable/seed_${seed}/raer_teacher.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/raer_lree_scalable s=${seed}"
        continue
    fi
    log "[E4-teacher] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
        --config "${TEACHER_CFG}" --seed "${seed}" \
        --device cuda:0 --run_name raer_lree_scalable \
        --base_ckpt_path "${base_ckpt}" \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] teacher ${MODEL} s=${seed}"
done

# Stage 3: CBR-Flash Student
log "--- Stage 3: CBR-Flash Student ---"
for seed in "${SEEDS[@]}"; do
    base_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base_neighbor_mb/seed_${seed}/base.pt"
    teacher_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_scalable/seed_${seed}/raer_teacher.pt"
    if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
        log "[skip-s] missing deps: ${MODEL} s=${seed}"
        continue
    fi
    ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/cbr_flash_lree_scalable/seed_${seed}/cbr_flash_student.pt"
    if [[ -f "${ckpt}" ]]; then
        log "[skip] ${MODEL}/cbr_flash_lree_scalable s=${seed}"
        continue
    fi
    args=(
        scripts/train_cbr_flash.py
        --config "${STUDENT_CFG}" --seed "${seed}"
        --device cuda:0 --run_name cbr_flash_lree_scalable
        --teacher_ckpt "${teacher_ckpt}"
        --base_ckpt_path "${base_ckpt}"
    )
    lree="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree_scalable/seed_${seed}/lree.pt"
    [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")
    log "[E4-student] GPU=${GPU} ${MODEL} s=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${args[@]}" \
        >> "${LOG}" 2>&1
    [[ $? -ne 0 ]] && log "[ERROR] student ${MODEL} s=${seed}"
done

log "=== E4 TSocial SAGE complete | $(date) ==="
