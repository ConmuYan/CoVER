#!/usr/bin/env bash
# Run one Phase2 sweep suite, aggregate metrics, and render figures.
#
# Usage:
#   bash scripts/run_phase2_sweep_suite.sh yelpchi GPU SEED1 [SEED2 ...]
#   bash scripts/run_phase2_sweep_suite.sh amazon  GPU SEED1 [SEED2 ...]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; shift
GPU="$1"; shift
SEEDS=("$@")
PY="${PYTHON:-/data1/mq/conda_envs/gread-core/bin/python}"

cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers
LOG="artifacts/sweeps/_drivers/phase2_${DATASET}_suite_gpu${GPU}.log"

echo "[phase2-suite] start $(date -Is) dataset=${DATASET} gpu=${GPU} seeds=${SEEDS[*]}" >> "$LOG"

if [ "$DATASET" = "yelpchi" ]; then
  bash scripts/run_phase2_yelpchi_targeted_sensitivity.sh "$GPU" "${SEEDS[@]}" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[phase2-suite] yelpchi training failed rc=${rc}" >> "$LOG"
    exit "$rc"
  fi
  RUNS=(
    phase2_yelp_s1_ltrust_0
    phase2_yelp_s1_ltrust_1em3
    phase2_yelp_s1_ltrust_3em3
    phase2_yelp_s1_ltrust_1em2
    phase2_yelp_s1_ltrust_3em2
    phase2_yelp_s2_lsparse_0
    phase2_yelp_s2_lsparse_3em4
    phase2_yelp_s2_lsparse_1em3
    phase2_yelp_s2_lsparse_3em3
    phase2_yelp_s3_lalign_0
    phase2_yelp_s3_lalign_1em3
    phase2_yelp_s3_lalign_3em3
    phase2_yelp_s3_lalign_1em2
    phase2_yelp_s3_alpha_0
    phase2_yelp_s3_alpha_01
    phase2_yelp_s3_alpha_03
  )
  PREFIX="artifacts/tables/yelpchi_phase2_targeted_sensitivity"
  FIGDIR="artifacts/figures/phase2_sweeps/yelpchi_targeted_sensitivity"
  "$PY" scripts/aggregate_phase2_custom_runs.py \
    --dataset yelpchi \
    --runs "${RUNS[@]}" \
    --seeds "${SEEDS[@]}" \
    --out-prefix "$PREFIX" >> "$LOG" 2>&1
  "$PY" scripts/plot_phase2_sweep_metrics.py \
    --summary-csv "${PREFIX}_summary.csv" \
    --suite yelpchi_targeted \
    --out-dir "$FIGDIR" >> "$LOG" 2>&1
elif [ "$DATASET" = "amazon" ]; then
  bash scripts/run_phase2_amazon_default_candidates.sh "$GPU" "${SEEDS[@]}" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[phase2-suite] amazon training failed rc=${rc}" >> "$LOG"
    exit "$rc"
  fi
  RUNS=(
    phase2_amz_c1_drel05_ltrust1em2_tau18_lsp0
    phase2_amz_c2_drel075_ltrust1em2_tau18_lsp0
    phase2_amz_c3_drel05_ltrust3em3_tau18_lsp0
    phase2_amz_c4_drel05_ltrust1em2_tau13_lsp3em4
    phase2_amz_c5_drel05_align_only
    phase2_amz_c6_drel05_judge_residual
  )
  PREFIX="artifacts/tables/amazon_phase2_default_candidates"
  FIGDIR="artifacts/figures/phase2_sweeps/amazon_default_candidates"
  "$PY" scripts/aggregate_phase2_custom_runs.py \
    --dataset amazon \
    --runs "${RUNS[@]}" \
    --seeds "${SEEDS[@]}" \
    --out-prefix "$PREFIX" >> "$LOG" 2>&1
  "$PY" scripts/plot_phase2_sweep_metrics.py \
    --summary-csv "${PREFIX}_summary.csv" \
    --suite amazon_candidates \
    --out-dir "$FIGDIR" >> "$LOG" 2>&1
else
  echo "Unknown dataset: $DATASET" >&2
  exit 2
fi

echo "[phase2-suite] done $(date -Is) dataset=${DATASET}" >> "$LOG"
