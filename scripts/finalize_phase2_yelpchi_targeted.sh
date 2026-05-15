#!/usr/bin/env bash
# Wait for queue workers, aggregate YelpChi targeted sweep, and render figures.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/data1/mq/conda_envs/gread-core/bin/python}"
cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers
LOG="artifacts/sweeps/_drivers/phase2_yelpchi_queue_finalize.log"

echo "[queue-finalize] wait start $(date -Is)" >> "$LOG"
while tmux ls 2>/dev/null | grep -q '^cover_yelp_q'; do
  done_count=$(find artifacts/results/yelpchi/bwgnn -maxdepth 3 -path '*phase2_yelp_*' -name stage3_metrics.json | wc -l)
  echo "[queue-finalize] waiting done_count=${done_count}/80 @ $(date -Is)" >> "$LOG"
  sleep 30
done

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
  --seeds 42 123 456 789 2026 \
  --out-prefix "$PREFIX" >> "$LOG" 2>&1
"$PY" scripts/plot_phase2_sweep_metrics.py \
  --summary-csv "${PREFIX}_summary.csv" \
  --suite yelpchi_targeted \
  --out-dir "$FIGDIR" >> "$LOG" 2>&1

echo "[queue-finalize] done $(date -Is)" >> "$LOG"
