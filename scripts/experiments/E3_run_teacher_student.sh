#!/usr/bin/env bash
# E3: Teacher + Student for YelpNYC/YelpZip SAGE remaining seeds
# Assumes base_neighbor_mb checkpoints already exist.
# Uses existing large_graph pipeline configs.
#
# Usage:
#   bash scripts/experiments/E3_run_teacher_student.sh        # default GPU
#   bash scripts/experiments/E3_run_teacher_student.sh 3      # specific GPU
set -uo pipefail

GPU="${1:-3}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASETS=(yelpnyc yelpzip)
MODEL="sage"
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E3_teacher_student.log"

TEACHER_CFG="configs/raer_fd/large_graph/teacher/raer_lree_scalable"
STUDENT_CFG="configs/raer_fd/large_graph/student"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

log "=== E3 Teacher+Student | gpu=${GPU} | $(date) ==="

for ds in "${DATASETS[@]}"; do
    log ""
    log "--- ${ds} ---"

    teacher_cfg="${TEACHER_CFG}/${ds}_sage.yaml"
    student_cfg="${STUDENT_CFG}/cbr_flash_${ds}_sage_lree_scalable.yaml"

    if [[ ! -f "${teacher_cfg}" ]] || [[ ! -f "${student_cfg}" ]]; then
        log "[skip] missing config for ${ds}"
        continue
    fi

    # Stage 2: Teacher
    for seed in "${SEEDS[@]}"; do
        base_ckpt="${CKPT_ROOT}/${ds}/${MODEL}/base_neighbor_mb/seed_${seed}/base.pt"
        if [[ ! -f "${base_ckpt}" ]]; then
            log "[skip-t] no base: ${ds} s=${seed}"
            continue
        fi
        ckpt="${CKPT_ROOT}/${ds}/${MODEL}/raer_lree_scalable/seed_${seed}/raer_teacher.pt"
        if [[ -f "${ckpt}" ]]; then
            log "[skip] ${ds}/teacher s=${seed}"
            continue
        fi
        log "[E3-teacher] GPU=${GPU} ${ds} s=${seed}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
            --config "${teacher_cfg}" --seed "${seed}" \
            --device cuda:0 --run_name raer_lree_scalable \
            --base_ckpt_path "${base_ckpt}" \
            >> "${LOG}" 2>&1
        [[ $? -ne 0 ]] && log "[ERROR] teacher ${ds} s=${seed}"
    done

    # Stage 3: Student
    for seed in "${SEEDS[@]}"; do
        base_ckpt="${CKPT_ROOT}/${ds}/${MODEL}/base_neighbor_mb/seed_${seed}/base.pt"
        teacher_ckpt="${CKPT_ROOT}/${ds}/${MODEL}/raer_lree_scalable/seed_${seed}/raer_teacher.pt"
        if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
            log "[skip-s] missing deps: ${ds} s=${seed}"
            continue
        fi
        ckpt="${CKPT_ROOT}/${ds}/${MODEL}/cbr_flash_lree_scalable/seed_${seed}/cbr_flash_student.pt"
        if [[ -f "${ckpt}" ]]; then
            log "[skip] ${ds}/student s=${seed}"
            continue
        fi
        args=(
            scripts/train_cbr_flash.py
            --config "${student_cfg}" --seed "${seed}"
            --device cuda:0 --run_name cbr_flash_lree_scalable
            --teacher_ckpt "${teacher_ckpt}"
            --base_ckpt_path "${base_ckpt}"
        )
        lree="${CKPT_ROOT}/${ds}/${MODEL}/raer_lree_scalable/seed_${seed}/lree.pt"
        [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")
        log "[E3-student] GPU=${GPU} ${ds} s=${seed}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${args[@]}" \
            >> "${LOG}" 2>&1
        [[ $? -ne 0 ]] && log "[ERROR] student ${ds} s=${seed}"
    done
done

log ""
log "=== E3 Teacher+Student complete | $(date) ==="
