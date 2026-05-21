#!/usr/bin/env bash
# T7 - CBR-Flash deployment-shift evaluation
#
# Tests OPD's exposure-bias claim: a student trained against base_v1
# should degrade *less* under CBR-Flash than under vanilla off-policy KD
# when its base is upgraded to base_v2 at deployment time.
#
# Setup:
#   - base_v1 = seed=42 base ckpt (training base)
#   - base_v2 = seed=2026 base ckpt (deployment base — different init)
#   - Student trained with base_v1 cache; evaluated on base_v2 outputs.
#
# Three modes compared (per design v3 §6 T7):
#   off_policy   — v1 vanilla KL distill baseline
#   det_mask     — v1 deterministic entropy mask
#   g_opd_flash  - historical mode name for full CBR-Flash
#
# Runs across 8 cells, 5 seeds for training base_v1 = (42, 123, 456, 789, 2026)
# each paired with one fixed alternate deployment base (rotated by +1 within
# the seed list, with wrap-around).
#
# Usage:
#   bash scripts/run_cbr_flash_deployshift.sh [TEACHER_RUN_NAME]

set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER_RUN_NAME="${1:-idea1_canonical_clsonly_smoke}"
DATASETS=(yelpchi amazon)
BASES=(bwgnn sage gcn gat)
SEEDS=(42 123 456 789 2026)
DEPLOY_SEEDS=(123 456 789 2026 42)  # rotated by +1 from SEEDS (wrap-around)
MODES=(off_policy det_mask g_opd_flash)
GPUS=(0 1 2 3)
N_GPUS=${#GPUS[@]}

config_for() {
    local ds="$1" base="$2"
    if [[ "$ds" == "yelpchi" && "$base" == "bwgnn" ]]; then
        echo "configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"
    else
        echo "configs/phase2_reasoner/ablation/idea1_${ds}_${base}_canonical_clsonly.yaml"
    fi
}

LOG_ROOT="artifacts/logs/g_opd_flash_deployshift/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
echo "[T7] Logs → $LOG_ROOT"
echo "[T7] Teacher = $TEACHER_RUN_NAME"

idx=0
for ds in "${DATASETS[@]}"; do
    for base in "${BASES[@]}"; do
        cfg=$(config_for "$ds" "$base")
        if [[ ! -f "$cfg" ]]; then
            echo "[T7] SKIP cell ${ds}/${base} — config missing"
            continue
        fi
        for i in "${!SEEDS[@]}"; do
            seed="${SEEDS[$i]}"
            deploy_seed="${DEPLOY_SEEDS[$i]}"
            teacher="artifacts/checkpoints/${ds}/${base}/${TEACHER_RUN_NAME}/seed_${seed}/reasoner.pt"
            base_v1="artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt"
            base_v2="artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${deploy_seed}/base.pt"
            if [[ ! -f "$teacher" || ! -f "$base_v1" || ! -f "$base_v2" ]]; then
                echo "[T7] SKIP ${ds}/${base}/seed_${seed}→${deploy_seed} — missing ckpt"
                continue
            fi
            for mode in "${MODES[@]}"; do
                gpu_id="${GPUS[$((idx % N_GPUS))]}"
                idx=$((idx + 1))
                runlog="$LOG_ROOT/${ds}_${base}_v1seed${seed}_v2seed${deploy_seed}_${mode}.log"
                (
                    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$gpu_id" \
                        python scripts/train_cbr_flash.py \
                        --config "$cfg" --seed "$seed" --device cuda:0 \
                        --mode "$mode" \
                        --teacher_ckpt "$teacher" \
                        --base_ckpt_path "$base_v1" \
                        --base_v2_ckpt_path "$base_v2" \
                        --epochs 80 --patience 10 --K 2048 \
                        --run_name "g_opd_flash_deployshift_${mode}_v1${seed}_v2${deploy_seed}" \
                        > "$runlog" 2>&1
                    echo "[T7] DONE gpu${gpu_id} ${ds}/${base} v1=${seed} v2=${deploy_seed} ${mode}"
                ) &
                if (( idx % 8 == 0 )); then
                    wait
                fi
            done
        done
    done
done

wait
echo "[T7] All deployment-shift runs dispatched. Logs: $LOG_ROOT"
echo "[T7] Aggregate via: python scripts/aggregate_cbr_flash_deployshift.py"
