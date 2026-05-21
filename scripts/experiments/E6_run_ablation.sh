#!/usr/bin/env bash
# E6: Ablation — YelpChi care_712 split.
# Variant 1: Freeze-Base (reuse E1 care_712 base results)
# Variant 2: RAER-HC (hand-crafted evidence)
# Variant 3: RAER-LREE (reuse E1 care_712 LREE results)
# Only Variant 2 needs new runs. Models: gcn, gat, sage.
# Usage:
#   bash scripts/experiments/E6_run_ablation.sh        # default GPUs
#   bash scripts/experiments/E6_run_ablation.sh 0 1    # specific GPUs
set -uo pipefail

GPUS=("${@:-0 1}")
NUM_GPUS=${#GPUS[@]}
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
SEEDS=(42 123 456 789 2026)
DATASET="yelpchi"
MODELS=(gcn gat sage)
CKPT_ROOT="artifacts/checkpoints"
LOG="artifacts/logs/E6_ablation.log"

mkdir -p artifacts/logs

log() { echo "$@" | tee -a "${LOG}"; }

log "=== E6 Ablation | gpus=${GPUS[*]} | $(date) ==="

# --- Variant 1: Freeze-Base (reuse E1 care_712) ---
log "--- Variant 1: Freeze-Base (reusing E1 care_712 base results) ---"
for m in "${MODELS[@]}"; do
    count=0
    for s in "${SEEDS[@]}"; do
        [[ -f "${CKPT_ROOT}/${DATASET}/${m}/base/seed_${s}/base.pt" ]] && count=$((count+1))
    done
    log "  ${m}: ${count}/5 base checkpoints exist"
done

# --- Variant 2: RAER-HC ---
log "--- Variant 2: RAER-HC (hand-crafted evidence) ---"
gpu_worker_hc() {
    set +e
    local gpu=$1; shift
    for config in "$@"; do
        model=$(basename "${config}" .yaml | sed 's/yelpchi_//')
        for seed in "${SEEDS[@]}"; do
            base_ckpt="${CKPT_ROOT}/${DATASET}/${model}/base/seed_${seed}/base.pt"
            if [[ ! -f "${base_ckpt}" ]]; then
                log "[skip-hc] no base: ${model} s=${seed}"
                continue
            fi
            ckpt="${CKPT_ROOT}/${DATASET}/${model}/raer_hc/seed_${seed}/raer_teacher.pt"
            if [[ -f "${ckpt}" ]]; then
                log "[skip] ${model}/raer_hc s=${seed}"
                continue
            fi
            log "[E6-hc] GPU=${gpu} ${model} s=${seed}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
                --config "${config}" --seed "${seed}" \
                --device cuda:0 --run_name raer_hc \
                --base_ckpt_path "${base_ckpt}" \
                >> "${LOG}" 2>&1
            [[ $? -ne 0 ]] && log "[ERROR] hc ${model} s=${seed}"
        done
    done
}

HC_DIR="configs/raer_fd/experiments/E6_ablation/hc_teacher"
configs=()
for m in "${MODELS[@]}"; do
    cfg="${HC_DIR}/${DATASET}_${m}.yaml"
    [[ -f "${cfg}" ]] && configs+=("${cfg}")
done

if [[ ${#configs[@]} -gt 0 ]]; then
    # Distribute across GPUs
    local pids=()
    gi=0
    for cfg in "${configs[@]}"; do
        g=${GPUS[$((gi % NUM_GPUS))]}
        gpu_worker_hc "${g}" "${cfg}" &
        pids+=($!)
        gi=$((gi + 1))
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
fi

# --- Variant 3: RAER-LREE (reuse E1 care_712) ---
log "--- Variant 3: RAER-LREE (reusing E1 care_712 LREE results) ---"
for m in "${MODELS[@]}"; do
    count=0
    for s in "${SEEDS[@]}"; do
        [[ -f "${CKPT_ROOT}/${DATASET}/${m}/raer_lree/seed_${s}/raer_teacher.pt" ]] && count=$((count+1))
    done
    log "  ${m}: ${count}/5 LREE checkpoints exist"
done

log "=== E6 Ablation complete | $(date) ==="
