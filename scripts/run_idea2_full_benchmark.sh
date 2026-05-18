#!/bin/bash
# Idea-2 full benchmark: distillation (6 remaining cells) + inference bench
# Usage: bash scripts/run_idea2_full_benchmark.sh [GPU_ID]
set -euo pipefail
GPU=${1:-1}
SEEDS="42 123 456 789 2026"
LOGDIR="logs/idea2_full_benchmark"
mkdir -p "$LOGDIR"

declare -A CELL_CFG CELL_TEACHER
CELL_CFG[yelpchi-bwgnn]=configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml
CELL_CFG[yelpchi-sage]=configs/phase2_reasoner/ablation/idea1_yelpchi_sage_canonical_clsonly.yaml
CELL_CFG[yelpchi-gcn]=configs/phase2_reasoner/ablation/idea1_yelpchi_gcn_canonical_clsonly.yaml
CELL_CFG[yelpchi-gat]=configs/phase2_reasoner/ablation/idea1_yelpchi_gat_canonical_clsonly.yaml
CELL_CFG[amazon-bwgnn]=configs/phase2_reasoner/ablation/idea1_amazon_bwgnn_canonical_clsonly.yaml
CELL_CFG[amazon-sage]=configs/phase2_reasoner/ablation/idea1_amazon_sage_canonical_clsonly.yaml
CELL_CFG[amazon-gcn]=configs/phase2_reasoner/ablation/idea1_amazon_gcn_canonical_clsonly.yaml
CELL_CFG[amazon-gat]=configs/phase2_reasoner/ablation/idea1_amazon_gat_canonical_clsonly.yaml

CELL_TEACHER[yelpchi-bwgnn]=artifacts/checkpoints/yelpchi/bwgnn/idea1_canonical_clsonly
CELL_TEACHER[yelpchi-sage]=artifacts/checkpoints/yelpchi/sage/idea1_yelpchi_sage_canonical_clsonly
CELL_TEACHER[yelpchi-gcn]=artifacts/checkpoints/yelpchi/gcn/idea1_yelpchi_gcn_canonical_clsonly
CELL_TEACHER[yelpchi-gat]=artifacts/checkpoints/yelpchi/gat/idea1_yelpchi_gat_canonical_clsonly
CELL_TEACHER[amazon-bwgnn]=artifacts/checkpoints/amazon/bwgnn/idea1_amazon_bwgnn_canonical_clsonly
CELL_TEACHER[amazon-sage]=artifacts/checkpoints/amazon/sage/idea1_amazon_sage_canonical_clsonly
CELL_TEACHER[amazon-gcn]=artifacts/checkpoints/amazon/gcn/idea1_amazon_gcn_canonical_clsonly
CELL_TEACHER[amazon-gat]=artifacts/checkpoints/amazon/gat/idea1_amazon_gat_canonical_clsonly

launch_distill() {
  local ds=$1 base=$2 seed=$3
  local cell="${ds}-${base}"
  local result="artifacts/results/${ds}/${base}/idea2c_distill_adapter/seed_${seed}/stage3_metrics.json"
  if [ -f "$result" ]; then echo "[SKIP] ${cell}/seed_${seed}"; return; fi
  local logf="${LOGDIR}/distill_${cell}_seed${seed}.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python scripts/train_distill_adapter.py \
    --config "${CELL_CFG[$cell]}" --seed "$seed" --device cuda:0 \
    --run_name idea2c_distill_adapter \
    --teacher_ckpt "${CELL_TEACHER[$cell]}/seed_${seed}/reasoner.pt" \
    --base_ckpt_path "artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt" \
    > "$logf" 2>&1 &
  echo "[LAUNCH] distill ${cell}/seed_${seed} (pid $!)"
}

# Distillation: skip already-done BWGNN+SAGE, run remaining 6 cells
pids=()
for ds_base in "yelpchi-gcn" "yelpchi-gat" "amazon-bwgnn" "amazon-sage" "amazon-gcn" "amazon-gat"; do
  ds=${ds_base%%-*}; base=${ds_base##*-}
  for seed in $SEEDS; do
    launch_distill "$ds" "$base" "$seed"
    pids+=($!)
    if [ ${#pids[@]} -ge 4 ]; then
      wait "${pids[0]}" 2>/dev/null || true
      pids=("${pids[@]:1}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
echo "[DISTILL_ALL_DONE]"

# Inference benchmark: 3 cells × 5 seeds
echo ""
echo "[BENCH_START]"
for cell in "yelpchi-bwgnn" "yelpchi-sage" "yelpchi-gat"; do
  ds=${cell%%-*}; base=${cell##*-}
  for seed in $SEEDS; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python scripts/bench_inference_speed.py \
      --config "${CELL_CFG[$cell]}" --seed "$seed" --device cuda:0 \
      --teacher_ckpt "${CELL_TEACHER[$cell]}/seed_${seed}/reasoner.pt" \
      --base_ckpt_path "artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt" \
      --adapter_ckpt "artifacts/checkpoints/${ds}/${base}/idea2c_distill_adapter/seed_${seed}/adapter.pt" \
      > "${LOGDIR}/bench_${cell}_seed${seed}.log" 2>&1 &
    echo "[BENCH] ${cell}/seed_${seed} (pid $!)"
    wait $! 2>/dev/null || true
  done
done
echo "[BENCH_ALL_DONE]"
echo "[ALL_DONE]"
