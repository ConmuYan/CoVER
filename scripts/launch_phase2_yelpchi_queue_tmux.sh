#!/usr/bin/env bash
# Launch lock-aware YelpChi queue workers across GPUs 0/2/3.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

rm -rf artifacts/sweeps/locks/yelpchi_phase2_targeted

for item in 0:a 2:a 3:a; do
  gpu="${item%%:*}"
  tag="${item#*:}"
  session="cover_yelp_q_gpu${gpu}_${tag}"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && bash scripts/run_phase2_yelpchi_queue_worker.sh '$gpu' '${session}'"
done

tmux new-session -d -s cover_yelp_q_finalize \
  "cd '$ROOT' && bash scripts/finalize_phase2_yelpchi_targeted.sh"

tmux ls
