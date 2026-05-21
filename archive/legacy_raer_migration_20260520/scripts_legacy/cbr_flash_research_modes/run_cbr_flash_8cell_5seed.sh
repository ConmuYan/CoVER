#!/usr/bin/env bash
# T5 - CBR-Flash 8-cell x 5-seed benchmark
#
# Layout per design v3.2 §6 + §8 ablation matrix:
#
#   cells (8) = {yelpchi, amazon} × {bwgnn, sage, gcn, gat}
#   seeds (5) = 42, 123, 456, 789, 2026
#   modes (6) = off_policy / all_node_mh / det_mask / g_opd_flash /
#               opd_action_strict / opd_action_strict_mh  (historical names)
#
# Total runs = 8 × 5 × 6 = 240.  Sharded 4-way across all RTX 3090s.
#
# Prereqs:
#   - Teacher checkpoints at  artifacts/checkpoints/{ds}/{base}/idea1_canonical_clsonly_smoke/seed_{s}/reasoner.pt  (or override)
#   - Base checkpoints at     artifacts/checkpoints/{ds}/{base}/fixed_v1_100ep/seed_{s}/base.pt
#   - Optional T4 priors:     artifacts/teacher_curriculum_prior.json / artifacts/teacher_calibration_bins.json
#
# Usage:
#   bash scripts/run_cbr_flash_8cell_5seed.sh [TEACHER_RUN_NAME]
#
# TEACHER_RUN_NAME (optional, default: idea1_canonical_clsonly_smoke) selects
# the teacher dir under artifacts/checkpoints/{ds}/{base}/{TEACHER_RUN_NAME}/.
# Use idea2b_learned_extractor* for LREE-teacher (T5 arm 8 of design v3 §8).

set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER_RUN_NAME="${1:-idea1_canonical_clsonly_smoke}"
DATASETS=(yelpchi amazon)
BASES=(bwgnn sage gcn gat)
SEEDS=(42 123 456 789 2026)
MODES=(det_mask_cbr)
GPUS=(0 1 2 3)
N_GPUS=${#GPUS[@]}

# Map (dataset, base) → config path.
config_for() {
    local ds="$1" base="$2"
    if [[ "$ds" == "yelpchi" && "$base" == "bwgnn" ]]; then
        echo "configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"
    else
        echo "configs/phase2_reasoner/ablation/idea1_${ds}_${base}_canonical_clsonly.yaml"
    fi
}

LOG_ROOT="artifacts/logs/g_opd_flash_5seed_run/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
echo "[T5] Logs -> $LOG_ROOT"
echo "[T5] Teacher run name = $TEACHER_RUN_NAME"

idx=0
for ds in "${DATASETS[@]}"; do
    for base in "${BASES[@]}"; do
        cfg=$(config_for "$ds" "$base")
        if [[ ! -f "$cfg" ]]; then
            echo "[T5] SKIP cell ${ds}/${base} — config missing: $cfg"
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            teacher="artifacts/checkpoints/${ds}/${base}/${TEACHER_RUN_NAME}/seed_${seed}/reasoner.pt"
            base_ckpt="artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt"
            if [[ ! -f "$teacher" ]]; then
                echo "[T5] SKIP ${ds}/${base}/seed_${seed} — teacher missing: $teacher"
                continue
            fi
            if [[ ! -f "$base_ckpt" ]]; then
                echo "[T5] SKIP ${ds}/${base}/seed_${seed} — base missing: $base_ckpt"
                continue
            fi
            for mode in "${MODES[@]}"; do
                gpu_id="${GPUS[$((idx % N_GPUS))]}"
                idx=$((idx + 1))
                runlog="$LOG_ROOT/${ds}_${base}_seed${seed}_${mode}.log"
                # LREE teacher detection: if an evidence_extractor.pt sits next
                # to the reasoner ckpt, pass it through so the teacher uses
                # LREE-learned evidence path.  Otherwise the teacher
                # uses hand-crafted 9-dim relation features.
                extractor_path="$(dirname "$teacher")/evidence_extractor.pt"
                extractor_arg=""
                if [[ -f "$extractor_path" ]]; then
                    extractor_arg="--teacher_extractor_ckpt $extractor_path"
                fi
                (
                    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$gpu_id" \
                        python scripts/train_cbr_flash.py \
                        --config "$cfg" --seed "$seed" --device cuda:0 \
                        --mode "$mode" \
                        --teacher_ckpt "$teacher" \
                        $extractor_arg \
                        --base_ckpt_path "$base_ckpt" \
                        --epochs 80 --patience 10 --K 2048 \
                        --run_name "g_opd_flash_${mode}" \
                        > "$runlog" 2>&1
                    echo "[T5] DONE gpu${gpu_id} ${ds}/${base}/seed_${seed}/${mode}"
                ) &
                # Throttle: keep at most 4*2 = 8 concurrent runs (avoid OOM if cells are big).
                if (( idx % 8 == 0 )); then
                    wait
                fi
            done
        done
    done
done

wait
echo "[T5] All 240 runs dispatched. Logs: $LOG_ROOT"
echo "[T5] Aggregate via: python scripts/aggregate_cbr_flash.py"
