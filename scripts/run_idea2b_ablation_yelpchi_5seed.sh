#!/bin/bash
# Idea-2B module ablation: 3 switches × 4 bases × 5 seeds = 60 runs
# Usage: bash scripts/run_idea2b_ablation_yelpchi_5seed.sh [GPU_ID]
set -euo pipefail

GPU=${1:-2}
SEEDS="42 123 456 789 2026"
SWITCHES="drop_gcn drop_proto encoder_shared"
BASES="bwgnn sage gcn gat"
LOGDIR="logs/idea2b_ablation"
mkdir -p "$LOGDIR"

echo "[ORCH] GPU=$GPU  seeds=$SEEDS"
echo "[ORCH] switches=$SWITCHES"
echo "[ORCH] bases=$BASES"
echo "[ORCH] total=$((3 * 4 * 5)) runs"

run_cell() {
  local switch=$1 base=$2 seed=$3
  local config="configs/phase2_reasoner/ablation/idea2b_ablate_${switch}_yelpchi_${base}.yaml"
  local run_name="idea2b_ablate_${switch}"
  local logf="${LOGDIR}/${run_name}_${base}_seed${seed}.log"
  local result="artifacts/results/yelpchi/${base}/${run_name}/seed_${seed}/stage3_metrics.json"

  if [ -f "$result" ]; then
    echo "[SKIP] ${switch}/${base}/seed_${seed} (already done)"
    return 0
  fi

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python scripts/train_phase2_reasoner.py \
    --config "$config" \
    --seed "$seed" \
    --device cuda:0 \
    --run_name "$run_name" \
    --base_ckpt_path "artifacts/checkpoints/yelpchi/${base}/fixed_v1_100ep/seed_${seed}/base.pt" \
    > "$logf" 2>&1

  if [ -f "$result" ]; then
    local auprc=$(python -c "import json; print(json.load(open('$result'))['auprc']:.4f)")
    echo "[DONE] ${switch}/${base}/seed_${seed}  AUPRC=$auprc"
  else
    echo "[FAIL] ${switch}/${base}/seed_${seed}  (no result)"
  fi
}

# Run: for each base, 4 seeds in parallel (cap at GPU mem), sequential 5th
for base in $BASES; do
  echo ""
  echo "[CELL_START] yelpchi-${base} on GPU $GPU"
  for switch in $SWITCHES; do
    pids=()
    for seed in $SEEDS; do
      run_cell "$switch" "$base" "$seed" &
      pids+=($!)
      # Cap at 4 parallel
      if [ ${#pids[@]} -ge 4 ]; then
        wait "${pids[0]}" 2>/dev/null || true
        pids=("${pids[@]:1}")
      fi
    done
    # Wait remaining
    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  done
  echo "[CELL_DONE] yelpchi-${base}"
done

echo ""
echo "[ALL_60_DONE]"
