#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
TEACHER_FAMILY="${TEACHER_FAMILY:-raer_lree}"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"

# Full-graph CBR-Flash is safe after cached teacher evidence, but this batch
# script keeps the compact canonical set aligned with base/teacher reruns.
datasets=(yelpchi amazon)
bases=(bwgnn sage gcn gat)
seeds=(42 123 456 789 2026)

for dataset in "${datasets[@]}"; do
  for base in "${bases[@]}"; do
    config="configs/raer_fd/student/cbr_flash_${dataset}_${base}.yaml"
    if [[ ! -f "${config}" ]]; then
      echo "[skip] missing config ${config}"
      continue
    fi
    for seed in "${seeds[@]}"; do
      base_ckpt="artifacts/checkpoints/${dataset}/${base}/base/seed_${seed}/base.pt"
      teacher_dir="artifacts/checkpoints/${dataset}/${base}/${TEACHER_FAMILY}/seed_${seed}"
      teacher_ckpt="${teacher_dir}/raer_teacher.pt"
      lree_ckpt="${teacher_dir}/lree.pt"

      if [[ ! -f "${base_ckpt}" ]]; then
        echo "[skip] missing base checkpoint ${base_ckpt}"
        continue
      fi
      if [[ ! -f "${teacher_ckpt}" ]]; then
        echo "[skip] missing RAER teacher ${teacher_ckpt}"
        continue
      fi

      args=(
        scripts/train_cbr_flash.py
        --config "${config}"
        --seed "${seed}"
        --device cuda:0
        --run_name cbr_flash
        --teacher_ckpt "${teacher_ckpt}"
        --base_ckpt_path "${base_ckpt}"
      )
      if [[ -f "${lree_ckpt}" ]]; then
        args+=(--teacher_extractor_ckpt "${lree_ckpt}")
      fi

      echo "[run] CBR-Flash ${dataset}/${base} seed=${seed}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${args[@]}"
    done
  done
done
