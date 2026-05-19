#!/usr/bin/env bash
# Properly-engineered launcher for v3 P0+P1 400-run ablation
#
# Per-GPU concurrency control:
#   - 3 GPUs (1, 2, 3) each run their own parallel pool
#   - Max 5 concurrent runs per GPU = 15 total concurrent
# Round-robin job assignment to balance GPU load.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_ROOT="artifacts/logs/v3_p0p1_ablation/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

GPUS=(1 2 3)
PER_GPU_CONCURRENCY=5

DATASETS=(yelpchi amazon)
BASES=(bwgnn sage gcn gat)
SEEDS=(42 123 456 789 2026)

# (extra_args, run_name, teacher_kind): 10 batches × 40 = 400 jobs
BATCHES=(
    "--cbr_lambda 1.0|v3_cbr_l10|lree"
    "--cbr_lambda 1.0 --cbr_symmetric_beta 0.5|v3_cbr_sym|lree"
    "--cbr_lambda 1.0 --cbr_weight_form sq|v3_cbr_sq|lree"
    "--cbr_lambda 1.0 --cbr_weight_form exp|v3_cbr_exp|lree"
    "--cbr_lambda 1.0 --cbr_weight_form bin|v3_cbr_bin|lree"
    "--cbr_lambda 1.0 --mask_criterion topk_H_T|v3_cbr_mask_HT|lree"
    "--cbr_lambda 1.0 --mask_criterion topk_disagree|v3_cbr_mask_disagree|lree"
    "--cbr_lambda 1.0 --mask_criterion random_k|v3_cbr_mask_rand|lree"
    "--cbr_lambda 1.0 --alpha_r_for_cbr 0.3 --alpha_g_for_cbr 0.2|v3_cbr_mh|lree"
    "--cbr_lambda 1.0|v3_cbr_hc_teacher|hc"
)

get_cfg() {
    local ds="$1" base="$2"
    if [[ "$ds" == "yelpchi" && "$base" == "bwgnn" ]]; then
        echo "configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml"
    else
        echo "configs/phase2_reasoner/ablation/idea1_${ds}_${base}_canonical_clsonly.yaml"
    fi
}

get_hc_teacher() {
    local ds="$1" base="$2" seed="$3"
    for run in "idea1_${ds}_${base}_canonical_clsonly" idea1_canonical_clsonly idea1_canonical_clsonly_smoke; do
        local p="artifacts/checkpoints/${ds}/${base}/${run}/seed_${seed}/reasoner.pt"
        if [[ -f "$p" ]]; then echo "$p"; return; fi
    done
    echo ""
}

# Build the full job list, one shell command per line, partitioned by GPU
mkdir -p "$LOG_ROOT/cmd"
for g in "${GPUS[@]}"; do
    : > "$LOG_ROOT/cmd/gpu${g}.sh"
done

idx=0
total=0
for batch in "${BATCHES[@]}"; do
    IFS='|' read -r extra run_name teacher_kind <<< "$batch"
    for ds in "${DATASETS[@]}"; do
        for base in "${BASES[@]}"; do
            cfg=$(get_cfg "$ds" "$base")
            if [[ ! -f "$cfg" ]]; then
                echo "[v3] WARN: missing config for ${ds}/${base} ($cfg) — skipping" >&2
                continue
            fi
            for seed in "${SEEDS[@]}"; do
                base_ckpt="artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt"
                if [[ "$teacher_kind" == "hc" ]]; then
                    teacher=$(get_hc_teacher "$ds" "$base" "$seed")
                    if [[ -z "$teacher" ]]; then continue; fi
                    extractor_arg=""
                else
                    teacher="artifacts/checkpoints/${ds}/${base}/idea2b_learned_extractor/seed_${seed}/reasoner.pt"
                    extractor_arg="--teacher_extractor_ckpt artifacts/checkpoints/${ds}/${base}/idea2b_learned_extractor/seed_${seed}/evidence_extractor.pt"
                fi
                if [[ ! -f "$teacher" || ! -f "$base_ckpt" ]]; then continue; fi

                gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
                idx=$((idx + 1))
                total=$((total + 1))
                runlog="$LOG_ROOT/${ds}_${base}_seed${seed}_${run_name}.log"

                # Emit a single self-contained shell command (no shell metacharacters in args)
                cat >> "$LOG_ROOT/cmd/gpu${gpu}.sh" <<EOF
PYTHONPATH=. CUDA_VISIBLE_DEVICES=${gpu} python scripts/train_g_opd_flash.py --config "${cfg}" --seed ${seed} --device cuda:0 --mode det_mask_cbr --teacher_ckpt "${teacher}" ${extractor_arg} --base_ckpt_path "${base_ckpt}" --epochs 80 --patience 10 --K 2048 ${extra} --run_name "${run_name}" > "${runlog}" 2>&1 && echo "[v3] DONE gpu${gpu} ${ds}/${base}/seed_${seed}/${run_name}" || echo "[v3] FAIL gpu${gpu} ${ds}/${base}/seed_${seed}/${run_name}"
EOF
            done
        done
    done
done

echo "[v3] Generated ${total} jobs, partitioned across ${#GPUS[@]} GPUs"
for g in "${GPUS[@]}"; do
    n=$(wc -l < "$LOG_ROOT/cmd/gpu${g}.sh")
    echo "[v3]   gpu${g}: ${n} jobs"
done

# Run each GPU's pool with -P 5 concurrency, all 3 pools in parallel
echo "[v3] Launching ${#GPUS[@]} pools (each max ${PER_GPU_CONCURRENCY} concurrent)"
for g in "${GPUS[@]}"; do
    (
        # Each line of gpuN.sh is one full command; pipe to xargs which runs N at a time
        # Use bash -c so the pipelines (>, &&, ||) are honored per-line
        awk 'NF' "$LOG_ROOT/cmd/gpu${g}.sh" | xargs -d '\n' -P ${PER_GPU_CONCURRENCY} -I {} bash -c '{}' > "$LOG_ROOT/pool_gpu${g}.log" 2>&1
        echo "[v3] POOL gpu${g} drained"
    ) &
done

wait
echo "[v3] All ${total} runs done. Logs: $LOG_ROOT"
