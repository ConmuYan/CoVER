#!/usr/bin/env bash
# Launch the Phase2 YelpChi and Amazon sweeps in tmux.
#
# Usage:
#   bash scripts/launch_phase2_sweeps_tmux.sh [YELP_GPU] [AMAZON_GPU]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
YELP_GPU="${1:-0}"
AMAZON_GPU="${2:-2}"
SEEDS=(42 123 456 789 2026)

cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers

tmux new-session -d -s cover_yelp_phase2_sens \
  "cd '$ROOT' && bash scripts/run_phase2_sweep_suite.sh yelpchi '$YELP_GPU' ${SEEDS[*]}"

tmux new-session -d -s cover_amz_phase2_candidates \
  "cd '$ROOT' && bash scripts/run_phase2_sweep_suite.sh amazon '$AMAZON_GPU' ${SEEDS[*]}"

tmux ls
