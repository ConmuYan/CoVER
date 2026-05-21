#!/bin/bash
# Robust distill launcher — no set -e, handles NaN gracefully
GPU=${1:-1}
LOGDIR="/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/logs/idea2_full_benchmark"
mkdir -p "$LOGDIR"

declare -A CFG TEACHER
CFG[yelpchi-bwgnn]=configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml
CFG[yelpchi-sage]=configs/phase2_reasoner/ablation/idea1_yelpchi_sage_canonical_clsonly.yaml
CFG[yelpchi-gcn]=configs/phase2_reasoner/ablation/idea1_yelpchi_gcn_canonical_clsonly.yaml
CFG[yelpchi-gat]=configs/phase2_reasoner/ablation/idea1_yelpchi_gat_canonical_clsonly.yaml
CFG[amazon-bwgnn]=configs/phase2_reasoner/ablation/idea1_amazon_bwgnn_canonical_clsonly.yaml
CFG[amazon-sage]=configs/phase2_reasoner/ablation/idea1_amazon_sage_canonical_clsonly.yaml
CFG[amazon-gcn]=configs/phase2_reasoner/ablation/idea1_amazon_gcn_canonical_clsonly.yaml
CFG[amazon-gat]=configs/phase2_reasoner/ablation/idea1_amazon_gat_canonical_clsonly.yaml
TEACHER[yelpchi-bwgnn]=artifacts/checkpoints/yelpchi/bwgnn/idea1_canonical_clsonly
TEACHER[yelpchi-sage]=artifacts/checkpoints/yelpchi/sage/idea1_yelpchi_sage_canonical_clsonly
TEACHER[yelpchi-gcn]=artifacts/checkpoints/yelpchi/gcn/idea1_yelpchi_gcn_canonical_clsonly
TEACHER[yelpchi-gat]=artifacts/checkpoints/yelpchi/gat/idea1_yelpchi_gat_canonical_clsonly
TEACHER[amazon-bwgnn]=artifacts/checkpoints/amazon/bwgnn/idea1_amazon_bwgnn_canonical_clsonly
TEACHER[amazon-sage]=artifacts/checkpoints/amazon/sage/idea1_amazon_sage_canonical_clsonly
TEACHER[amazon-gcn]=artifacts/checkpoints/amazon/gcn/idea1_amazon_gcn_canonical_clsonly
TEACHER[amazon-gat]=artifacts/checkpoints/amazon/gat/idea1_amazon_gat_canonical_clsonly

run_one() {
  local ds=$1 base=$2 seed=$3
  local cell="${ds}-${base}"
  local result="artifacts/results/${ds}/${base}/idea2c_distill_adapter/seed_${seed}/stage3_metrics.json"
  [ -f "$result" ] && return 0
  local logf="${LOGDIR}/distill_${cell}_seed${seed}.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python scripts/train_distill_adapter.py \
    --config "${CFG[$cell]}" --seed "$seed" --device cuda:0 \
    --run_name idea2c_distill_adapter \
    --teacher_ckpt "${TEACHER[$cell]}/seed_${seed}/reasoner.pt" \
    --base_ckpt_path "artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt" \
    > "$logf" 2>&1
  if [ -f "$result" ]; then
    echo "[OK] ${cell}/seed_${seed}"
  else
    echo "[FAIL] ${cell}/seed_${seed}"
  fi
}

CELLS="yelpchi-bwgnn yelpchi-sage yelpchi-gcn yelpchi-gat amazon-bwgnn amazon-sage amazon-gcn amazon-gat"
SEEDS="42 123 456 789 2026"

for cell in $CELLS; do
  ds=${cell%%-*}; base=${cell##*-}
  for seed in $SEEDS; do
    run_one "$ds" "$base" "$seed" &
    # 4-way parallel cap
    while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 2; done
  done
done
wait
echo "[ALL_DONE]"
