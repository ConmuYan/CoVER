#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:-raer_lree}"
GPU_ID="${2:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"

if [[ "${FAMILY}" != "raer_hc" && "${FAMILY}" != "raer_lree" ]]; then
  echo "FAMILY must be raer_hc or raer_lree" >&2
  exit 2
fi

# Full-graph RAER-LREE is intended for the compact canonical set. For
# YelpNYC/YelpZip/TFinance/TSocial, use the scaling plan before reruns.
datasets=(yelpchi amazon)
bases=(bwgnn sage gcn gat)
seeds=(42 123 456 789 2026)

for dataset in "${datasets[@]}"; do
  for base in "${bases[@]}"; do
    config="configs/raer_fd/teacher/${FAMILY}/${dataset}_${base}.yaml"
    base_ckpt="artifacts/checkpoints/${dataset}/${base}/base/seed_%SEED%/base.pt"
    if [[ ! -f "${config}" ]]; then
      echo "[skip] missing config ${config}"
      continue
    fi
    for seed in "${seeds[@]}"; do
      ckpt="${base_ckpt//%SEED%/${seed}}"
      if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] missing base checkpoint ${ckpt}"
        continue
      fi
      echo "[run] ${FAMILY} ${dataset}/${base} seed=${seed}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/train_raer_teacher.py \
        --config "${config}" \
        --seed "${seed}" \
        --device cuda:0 \
        --run_name "${FAMILY}" \
        --base_ckpt_path "${ckpt}"
    done
  done
done
