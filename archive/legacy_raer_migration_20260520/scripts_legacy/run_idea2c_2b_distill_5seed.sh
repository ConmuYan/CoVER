#!/bin/bash
# Distill Idea-2B learned-extractor teachers into lightweight adapters.
# Usage: bash scripts/run_idea2c_2b_distill_5seed.sh

set -euo pipefail

PY=/data1/mq/conda_envs/gread-core/bin/python
RUN_NAME=idea2c_distill_adapter_2b
SEEDS="42 123 456 789 2026"
LOGDIR=logs/idea2c_2b_distill
mkdir -p "$LOGDIR"

declare -A CFG
CFG[yelpchi-bwgnn]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_bwgnn.yaml
CFG[yelpchi-sage]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_sage.yaml
CFG[yelpchi-gcn]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_gcn.yaml
CFG[yelpchi-gat]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_gat.yaml
CFG[amazon-bwgnn]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_bwgnn.yaml
CFG[amazon-sage]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_sage.yaml
CFG[amazon-gcn]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_gcn.yaml
CFG[amazon-gat]=configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_gat.yaml

launch_one() {
  local cell=$1
  local seed=$2
  local device=$3
  local ds=${cell%%-*}
  local base=${cell##*-}
  local result="artifacts/results/${ds}/${base}/${RUN_NAME}/seed_${seed}/stage3_metrics.json"
  if [ -f "$result" ]; then
    echo "[SKIP] ${cell}/seed_${seed}"
    return 0
  fi
  local teacher_dir="artifacts/checkpoints/${ds}/${base}/idea2b_learned_extractor/seed_${seed}"
  "$PY" scripts/train_distill_adapter.py \
    --config "${CFG[$cell]}" \
    --seed "$seed" \
    --device "$device" \
    --run_name "$RUN_NAME" \
    --teacher_ckpt "${teacher_dir}/reasoner.pt" \
    --teacher_extractor_ckpt "${teacher_dir}/evidence_extractor.pt" \
    --base_ckpt_path "artifacts/checkpoints/${ds}/${base}/fixed_v1_100ep/seed_${seed}/base.pt"
  echo "[DONE] ${cell}/seed_${seed} on ${device}"
}

cells=(
  yelpchi-bwgnn yelpchi-sage yelpchi-gcn yelpchi-gat
  amazon-bwgnn amazon-sage amazon-gcn amazon-gat
)

i=0
for cell in "${cells[@]}"; do
  for seed in $SEEDS; do
    if [ $((i % 2)) -eq 0 ]; then device=cuda:1; else device=cuda:3; fi
    launch_one "$cell" "$seed" "$device"
    i=$((i + 1))
  done
done
echo "[DONE] ${RUN_NAME}"
