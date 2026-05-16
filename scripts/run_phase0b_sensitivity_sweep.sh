#!/bin/bash
# Phase 0b: 4-knob single-seed sensitivity on NEW champion v2 (base 1000ep)
# Vary one knob at a time, holding others at champion values:
#   λ_align ∈ {1e-2, 1e-1, 3e-1}        (3e-2 is champion, already have)
#   λ_int   ∈ {3e-3, 3e-2, 1e-1}        (1e-2 is champion, already have)
#   λ_sparse ∈ {3e-4, 3e-3, 1e-2}        (1e-3 is champion, already have)
#   α_max   ∈ {0.1, 0.2, 0.5}            (0.3 is champion, already have)
# Total: 4 knobs × 3 new points = 12 single-seed (seed=42) runs
# Base = D0 1000ep, all CoVER-on-base1000ep_seed42 (existing) is the champion anchor

set -e

CONFIG="artifacts/logs/yelpchi/bwgnn/phase2_yelpchi_bwgnn_revised_r3_full_alpha01_lint1em2/seed_42/repro_config.yaml"
BASE_CKPT="artifacts/checkpoints/yelpchi/bwgnn/d0_base_1000ep/seed_42/base.pt"
PY="/data1/mq/conda_envs/gread-core/bin/python"
LOGDIR=/tmp/phase0b_sens_$$
mkdir -p "$LOGDIR"

# (knob_name, knob_value, device)
JOBS=(
  "lambda_align 1e-2 cuda:1"
  "lambda_align 1e-1 cuda:1"
  "lambda_align 3e-1 cuda:2"
  "lambda_int   3e-3 cuda:2"
  "lambda_int   3e-2 cuda:1"
  "lambda_int   1e-1 cuda:1"
  "lambda_sparse 3e-4 cuda:2"
  "lambda_sparse 3e-3 cuda:2"
  "lambda_sparse 1e-2 cuda:1"
  "alpha_max    0.1  cuda:1"
  "alpha_max    0.2  cuda:2"
  "alpha_max    0.5  cuda:2"
)

run_jobs_on_device() {
  local DEV="$1"; shift
  local jobs=("$@")
  for job in "${jobs[@]}"; do
    read KNOB VAL <<< "$job"
    # Construct CLI overrides: pass the swept knob, others stay at champion via config defaults
    local KNOB_CLI=""
    case $KNOB in
      lambda_align) KNOB_CLI="--lambda_align $VAL" ;;
      lambda_int)   KNOB_CLI="--lambda_int $VAL"   ;;
      lambda_sparse) KNOB_CLI="--lambda_sparse $VAL" ;;
      alpha_max)    KNOB_CLI="--alpha_max $VAL"    ;;
    esac
    local VAL_TAG=$(echo "$VAL" | sed 's/\./p/g; s/-/m/g')
    local RUN_NAME="phase0b_sens_${KNOB}_${VAL_TAG}"
    local TAG="${RUN_NAME}_${DEV//:/}"
    local LOG="${LOGDIR}/${TAG}.log"
    echo "[launch] ${TAG}  (${KNOB}=${VAL})"
    JOB_START=$SECONDS
    CUDA_VISIBLE_DEVICES="${DEV#cuda:}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/train_phase2_reasoner.py \
      --config "$CONFIG" \
      --run_name "$RUN_NAME" \
      --seed 42 \
      --device cuda:0 \
      --base_ckpt_path "$BASE_CKPT" \
      --alpha_bias_init 0.0 --alpha_max 0.3 --lambda_align 3e-2 \
      --single-stage \
      $KNOB_CLI \
      2>&1 | sed "s/^/[${TAG}] /" >"$LOG" || true
    if [ -f "artifacts/logs/yelpchi/bwgnn/${RUN_NAME}/seed_42/phase2_diagnostics.json" ]; then
      echo "[done]  ${TAG} ($((SECONDS-JOB_START))s)"
    else
      echo "[FAIL]  ${TAG} — see ${LOG}"
    fi
  done
}

cuda1=(); cuda2=()
for j in "${JOBS[@]}"; do
  IFS=' ' read -ra parts <<< "$j"
  KNOB="${parts[0]}"; VAL="${parts[1]}"; DEV="${parts[2]}"
  pair="$KNOB $VAL"
  [ "$DEV" = "cuda:1" ] && cuda1+=("$pair") || cuda2+=("$pair")
done

echo "============================================================"
echo "[Phase 0b] launching ${#cuda1[@]} on cuda:1, ${#cuda2[@]} on cuda:2 (sensitivity)"
echo "============================================================"
START_ALL=$SECONDS
run_jobs_on_device "cuda:1" "${cuda1[@]}" &
P1=$!
run_jobs_on_device "cuda:2" "${cuda2[@]}" &
P2=$!
wait $P1
wait $P2
TOTAL=$((SECONDS - START_ALL))
echo
echo "============================================================"
echo "ALL ${#JOBS[@]} SENSITIVITY RUNS DONE in ${TOTAL}s"
echo "Logs: ${LOGDIR}"
echo "============================================================"
