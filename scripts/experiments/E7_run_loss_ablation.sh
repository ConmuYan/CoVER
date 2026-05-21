#!/usr/bin/env bash
# E7: Full 3-loss ablation — 2^3-1 = 7 variants
#
# L_CBR = L_final + bce_lambda * L_sup + cbr_lambda * L_budget
#
# Variants:
#   v1_full     L_final + L_sup + L_budget   (Full CBR-Flash)
#   v2_d_s      L_final + L_sup              (standard KD)
#   v3_d_b      L_final + L_budget           (distill + budget)
#   v4_s_b      L_sup + L_budget             (supervise + teacher intervention guide)
#   v5_d_only   L_final                      (pure distillation)
#   v6_s_only   L_sup                        (pure supervision)
#   v7_b_only   L_budget                     (budget only → near Freeze-Base)
#
# Dataset: YelpChi care_712, SAGE
# Seeds: 42 123 456 789 2026
# Total: 7 variants × 5 seeds = 35 runs
#
# Usage:
#   bash scripts/experiments/E7_run_loss_ablation.sh        # default GPU 2
#   bash scripts/experiments/E7_run_loss_ablation.sh 0      # specific GPU
set -uo pipefail

GPU="${1:-2}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="yelpchi"
MODEL="sage"
CKPT_ROOT="artifacts/checkpoints"
CONFIG_DIR="configs/raer_fd/experiments/E7_loss"
LOG="artifacts/logs/E7_loss_ablation.log"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

VARIANTS=(v1_full v2_d_s v3_d_b v4_s_b v5_d_only v6_s_only v7_b_only)

log "=== E7 Full Loss Ablation | gpu=${GPU} | $(date) ==="
log "Variants: ${VARIANTS[*]}"
log "Seeds: ${SEEDS[*]}"

for variant in "${VARIANTS[@]}"; do
    cfg="${CONFIG_DIR}/${variant}.yaml"
    if [[ ! -f "${cfg}" ]]; then
        log "[skip] no config: ${cfg}"
        continue
    fi

    run_name=$(grep -A50 "^cbr_flash:" "${cfg}" | grep "run_name:" | head -1 | awk '{print $2}')
    log ""
    log "--- ${variant} (${run_name}) ---"

    for seed in "${SEEDS[@]}"; do
        base_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/base/seed_${seed}/base.pt"
        teacher_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree/seed_${seed}/raer_teacher.pt"
        if [[ ! -f "${base_ckpt}" ]] || [[ ! -f "${teacher_ckpt}" ]]; then
            log "[skip] missing deps: s=${seed}"
            continue
        fi

        student_ckpt="${CKPT_ROOT}/${DATASET}/${MODEL}/${run_name}/seed_${seed}/cbr_flash_student.pt"
        if [[ -f "${student_ckpt}" ]]; then
            log "[skip] ${variant} s=${seed} (exists)"
            continue
        fi

        args=(
            scripts/train_cbr_flash.py
            --config "${cfg}" --seed "${seed}"
            --device cuda:0 --run_name "${run_name}"
            --teacher_ckpt "${teacher_ckpt}"
            --base_ckpt_path "${base_ckpt}"
        )
        lree="${CKPT_ROOT}/${DATASET}/${MODEL}/raer_lree/seed_${seed}/lree.pt"
        [[ -f "${lree}" ]] && args+=(--teacher_extractor_ckpt "${lree}")

        log "[E7] ${variant} GPU=${GPU} s=${seed}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${args[@]}" \
            >> "${LOG}" 2>&1
        rc=$?
        if [[ ${rc} -ne 0 ]]; then
            log "[ERROR] ${variant} s=${seed} rc=${rc}"
        fi
    done
done

log ""
log "=== E7 Full Loss Ablation complete | $(date) ==="
