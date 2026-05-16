## Handoff: team-plan → team-exec (SAGE Base Adaptation)

### Mission
Add SAGE (GraphSAGE) as a new base model to CoVER-FD, mirroring the BWGNN pipeline end-to-end:
Phase1 deterministic base → Relation features → Stage3 anchor_gate (CoVER-REL-Gate) → Judge packets + Qwen LLM judge → Phase2 unified reasoner training (E0/E1/E2) → Aggregate + safety audit.

### Decided
- **No new model code**: existing `models/gnn.py:SAGEDetector` already mirrors GraphSAGE paper (mean-aggregator, 2 SAGEConv, hidden=64, dropout=0.5, Linear head). Reuse as-is.
- **External clone is reference-only**: `external/williamleif_GraphSAGE/` for hyper-param audit trail. Do NOT import from it.
- **SAGE hyperparameters** (BWGNN-paper aligned for fair comparison):
  hidden=64, num_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, optimizer=adam, epochs=100,
  train_ratio=0.4, val_test_ratio=[1,2], stratified=true, select_metric=macro_f1, patience=100.
- **Seeds**: `[42, 123, 456, 789, 2026]` (matches BWGNN baseline for fair same-seed Δ).
- **Determinism**: `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + cuDNN deterministic.
- **GPU allocation** (数据集并行): YelpChi → GPU 2; Amazon → GPU 3 (matches BWGNN convention; both have ≥16GB free at planning time).
- **Strict gate at Phase2 E0**: 1-seed E0 smoke first; only proceed to 5-seed × E0/E1/E2 if mean ΔAUPRC > 0. Never burn Qwen GPU time on a regression.
- **Phase2 reasoner config** (relation gate):
  - YelpChi `[RUR,RSR,RTR]/anchor=RUR/tau=0.7/delta_rel_max=2.0` (mirror existing BWGNN E0)
  - Amazon `[UPU,USU,UVU]/anchor=UVU/tau=1.3/delta_rel_max=1.5` (mirror existing BWGNN E0)
- **Judge packet construction**: must use SAGE's own anchor_gate run, not BWGNN's, for per-base coherence.
- **Test scope**: existing Phase2 tests are model-agnostic. SAGE only needs smoke pass via `scripts/run_model_smoke_tests.py --debug` and `pytest -q` green.

### Rejected
- **Rewriting SAGEDetector**: in-repo impl matches paper; rewrite risks regression and wastes budget.
- **Reusing BWGNN judge_packets for SAGE**: judge packets contain anchor_gate scores; mixing bases violates per-base coherence.
- **Running all-4-experiments full sweep without 1-seed gate**: Qwen judge gen is the heaviest GPU cost; gating saves 60–80% on a negative result.
- **Dropout=0.3** (BWGNN value) for SAGE: keep paper-default 0.5.
- **New `retrain_baseline.py`**: extend existing `train_stage1.py` with `--deterministic` flag instead.

### Risks
- **R1 GPU contention**: GPU 2 had 7.6GB used at plan time; SAGE uses <4GB so OK, but watch nvidia-smi. Fallback: GPU 0 has more free.
- **R2 Negative Phase2 result**: SAGE base is weaker than BWGNN (no spectral filter); relation evidence may give larger Δ (more headroom) OR smaller Δ (lacks BWGNN's high-freq complement). Either is publishable.
- **R3 Stale `artifacts/checkpoints/yelpchi/sage/`**: may exist from old smoke tests. Audit and back-up before overwrite.
- **R4 Qwen latency**: ~5–10 min/seed × 120 nodes × Qwen3-4B. 5 seeds × 2 datasets ≤ 100 min total.
- **R5 Score-blind / forbidden-field safety**: every new artifact must pass `judge_forbidden_field_audit.json` and have `score_blind=True`.

### Files (planned new/modified)
- `external/williamleif_GraphSAGE/` (cloned, reference-only)
- `docs/sage_paper_alignment.md` (hyper-param source-of-truth)
- `configs/yelpchi_sage.yaml` (rewrite to BWGNN-format with `gpu`, `run`, `stage2`, `evidence`, `llm`, `reasoner`, `eval` blocks; `model.name=sage`)
- `configs/amazon_sage.yaml` (new)
- `configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_yelpchi_sage_nollm.yaml`, `configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_amazon_sage_nollm.yaml`
- `configs/phase2_yelpchi_sage_E{0,1,2}.yaml`, `configs/phase2_amazon_sage_E{0,1,2}.yaml`
- `scripts/train_stage1.py` (add `--deterministic` flag; preserve existing behaviour)
- `scripts/run_phase1_sage.sh`, `scripts/run_stage3_sage_gate.sh`, `scripts/run_judge_sage.sh`, `scripts/run_phase2_sage_experiments.sh`
- `scripts/aggregate_phase2_sage.py`
- `artifacts/checkpoints/{yelpchi,amazon}/sage/base/seed_X/base.pt`
- `artifacts/relation_features/{yelpchi,amazon}/sage/seed_X/all/...`
- `artifacts/checkpoints/{yelpchi,amazon}/sage/cover_rel_anchor_gate_nollm/seed_X/reasoner.pt`
- `artifacts/judge_packets/{yelpchi,amazon}/sage/cover_rel_judge/seed_X/...`
- `artifacts/checkpoints/{yelpchi,amazon}/sage/phase2_E{0,1,2}/seed_X/reasoner.pt`
- `artifacts/tables/phase2_sage_5seed_summary.{csv,md}` + `artifacts/reports/phase2_sage_first_round_conclusion.md`
- `PROGRESS.md` append "Task 10: SAGE Base Adaptation" section

### Strict-gate decision tree (for Phase2 workers)
After T8 (1-seed E0 smoke for {dataset}):
- if `mean_AUPRC_E0 - mean_AUPRC_base > 0`: proceed to T9 (5-seed × E0/E1/E2 for that dataset)
- if `≤ 0`: stop at E0; mark dataset Phase2 as "negative-result"; verifier writes negative-result memo; do NOT spend Qwen on E1/E2 for that dataset.

### Communication contract
- Workers report each completed stage via SendMessage to `team-lead` with seed counts, mean ROC-AUC/AUPRC, file paths.
- Workers MUST NOT mark a task `completed` if any seed's `score_blind=False` or `max_abs_alpha_llm_rejected ≥ 1e-6`.
- Workers MUST run `pytest -q tests/test_phase2_*.py tests/test_score_blind.py` before claiming any reasoner task complete.

### Remaining for team-exec
1. **bootstrap**: clone external + write configs + smoke + driver scripts.
2. **yelpchi-sage worker**: drive YelpChi pipeline (Phase1 → relation_features → Stage3 anchor_gate → judge packets → Qwen judge → Phase2 E0 1-seed gate → 5-seed × E0/E1/E2 if pass).
3. **amazon-sage worker**: drive Amazon pipeline on GPU 3.
4. **verifier**: safety audit + 5-seed aggregation + report.
5. **writer**: PROGRESS.md update.
