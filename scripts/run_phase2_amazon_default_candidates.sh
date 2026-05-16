#!/usr/bin/env bash
# Amazon Phase2 default-candidate search.
#
# Theory behind these candidates:
#   Amazon BWGNN is already near-saturated. The next default should avoid
#   large relation residuals and keep relation routing distributed. Therefore
#   this search varies delta_rel_max, lambda_trust, tau_gate, and sparsity,
#   while keeping the judge residual mostly off until the relation-only
#   candidate is chosen.
#
# Usage:
#   bash scripts/run_phase2_amazon_default_candidates.sh GPU SEED1 [SEED2 ...]
# Example:
#   bash scripts/run_phase2_amazon_default_candidates.sh 0 42 123 456 789 2026
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="$1"; shift
SEEDS=("$@")
PY="${PYTHON:-/data1/mq/conda_envs/gread-core/bin/python}"
CFG="configs/cover-rel-gj/phase2_ablations/phase2_amazon_E0_relgate.yaml"
JCFG="configs/cover-rel-gj/phase2_ablations/phase2_amazon_E2_judge_residual.yaml"

cd "$ROOT" || exit 1
mkdir -p artifacts/sweeps/_drivers
LOG="artifacts/sweeps/_drivers/phase2_amazon_default_candidates_gpu${GPU}.log"

run_one() {
  local seed="$1"; shift
  echo "[amazon-candidates] BEGIN seed=${seed} $* @ $(date -Is)" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_phase2_reasoner.py \
    --seed "$seed" --device cuda:0 "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[amazon-candidates] END seed=${seed} rc=${rc} $* @ $(date -Is)" >> "$LOG"
  return "$rc"
}

overall_rc=0
echo "[amazon-candidates] start $(date -Is) gpu=${GPU} seeds=${SEEDS[*]}" >> "$LOG"

# Relation-only defaults. Keep alpha/judge off while selecting the residual regime.
for seed in "${SEEDS[@]}"; do
  run_one "$seed" --config "$CFG" --run_name phase2_amz_c1_drel05_ltrust1em2_tau18_lsp0 \
    --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
    --delta_rel_max 0.5 --tau_gate 1.8 --lambda_trust 1.0e-2 \
    --lambda_sparse 0.0 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?

  run_one "$seed" --config "$CFG" --run_name phase2_amz_c2_drel075_ltrust1em2_tau18_lsp0 \
    --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
    --delta_rel_max 0.75 --tau_gate 1.8 --lambda_trust 1.0e-2 \
    --lambda_sparse 0.0 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?

  run_one "$seed" --config "$CFG" --run_name phase2_amz_c3_drel05_ltrust3em3_tau18_lsp0 \
    --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
    --delta_rel_max 0.5 --tau_gate 1.8 --lambda_trust 3.0e-3 \
    --lambda_sparse 0.0 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?

  run_one "$seed" --config "$CFG" --run_name phase2_amz_c4_drel05_ltrust1em2_tau13_lsp3em4 \
    --use_judge 0 --alpha_max 0.0 --lambda_align 0.0 \
    --delta_rel_max 0.5 --tau_gate 1.3 --lambda_trust 1.0e-2 \
    --lambda_sparse 3.0e-4 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?
done

# Judge-light candidates after relation-only regimes. These should be compared
# against c1/c2 rather than treated as primary models.
for seed in "${SEEDS[@]}"; do
  run_one "$seed" --config "$JCFG" --run_name phase2_amz_c5_drel05_align_only \
    --use_judge 1 --alpha_max 0.0 --lambda_align 1.0e-3 \
    --delta_rel_max 0.5 --tau_gate 1.8 --lambda_trust 1.0e-2 \
    --lambda_sparse 0.0 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?

  run_one "$seed" --config "$JCFG" --run_name phase2_amz_c6_drel05_judge_residual \
    --use_judge 1 --alpha_max 0.05 --lambda_align 1.0e-3 \
    --delta_rel_max 0.5 --tau_gate 1.8 --lambda_trust 1.0e-2 \
    --lambda_sparse 0.0 --lr 1.0e-3 --patience 50 --early_stop_metric val_auprc || overall_rc=$?
done

echo "[amazon-candidates] done $(date -Is) rc=${overall_rc}" >> "$LOG"
exit "$overall_rc"
