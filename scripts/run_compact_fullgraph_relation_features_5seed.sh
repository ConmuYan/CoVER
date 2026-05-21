#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"

# Hand-crafted relation-feature extraction can be CPU/RAM heavy on YelpNYC,
# YelpZip, TFinance, and TSocial. Use explicit single-run commands for those
# after checking docs/plans/RAER_FD_DATASET_SCALING_PLAN.md.
datasets=(yelpchi amazon)
bases=(bwgnn sage gcn gat)
seeds=(42 123 456 789 2026)

for dataset in "${datasets[@]}"; do
  for base in "${bases[@]}"; do
    config="configs/raer_fd/teacher/raer_hc/${dataset}_${base}.yaml"
    if [[ ! -f "${config}" ]]; then
      echo "[skip] missing config ${config}"
      continue
    fi
    for seed in "${seeds[@]}"; do
      echo "[run] relation features ${dataset}/${base} seed=${seed}"
      "${PYTHON_BIN}" scripts/build_relation_features.py \
        --config "${config}" \
        --seed "${seed}" \
        --relation_set all
    done
  done
done
