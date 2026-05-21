#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"

# Full-graph base training is intended for the compact canonical set.
# YelpNYC/YelpZip/TSocial/TFinance need the sampling guidance in
# docs/plans/RAER_FD_DATASET_SCALING_PLAN.md before large reruns.
datasets=(yelpchi amazon)
bases=(bwgnn sage gcn gat)
seeds=(42 123 456 789 2026)

for dataset in "${datasets[@]}"; do
  for base in "${bases[@]}"; do
    config="configs/raer_fd/base_detectors/${dataset}_${base}.yaml"
    if [[ ! -f "${config}" ]]; then
      echo "[skip] missing config ${config}"
      continue
    fi
    for seed in "${seeds[@]}"; do
      echo "[run] base ${dataset}/${base} seed=${seed}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/train_base_detector.py \
        --config "${config}" \
        --seed "${seed}" \
        --run_name base \
        --stratified
    done
  done
done
