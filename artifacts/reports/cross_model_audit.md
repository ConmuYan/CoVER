# Cross-Model Audit Report

## Audit Cycle 0 — Baseline (2026-05-15 22:40)

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### BWGNN Immutability
- `git status artifacts/checkpoints/yelpchi/bwgnn/ artifacts/checkpoints/amazon/bwgnn/ scripts/run_paper_result_consolidation.py`
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

### Current Executor State
| Task | Status | Stage A Checkpoints |
|------|--------|-------------------|
| #1 exec-yc-gcn | in_progress | seed_42 present |
| #3 exec-yc-gat | in_progress | seed_42 present |
| #5 exec-am-gcn | in_progress | no gcn/gat checkpoints yet |
| #6 exec-am-gat | pending (blocked by #3) | not started |

### Notes
- err_cache: only `yelpchi/bwgnn/` exists. No gcn/gat err_cache yet.
- Amazon: only bwgnn and sage checkpoints exist. No gcn/gat dirs yet.
- Waiting for executor Stage A completion notifications to run determinism + score-blind audits.

---

## Audit Cycle 1 — YelpChi GCN Stage A Complete (2026-05-15 22:50)

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

### YelpChi GCN Stage A — 5 Seeds Confirmed
All 5 seed checkpoints present: `seed_42`, `seed_123`, `seed_456`, `seed_789`, `seed_2026`

### Bit-Exact Reproducibility (yelpchi/gcn/seed_42)
- Command: `python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --seed 42 --stratified --run_name base_rerun`
- Comparison: `base/seed_42/base.pt` vs `base_rerun/seed_42/base.pt`
- Per-parameter diffs:
  - convs.0.bias: 0.000000e+00
  - convs.0.lin.weight: 0.000000e+00
  - convs.1.bias: 0.000000e+00
  - convs.1.lin.weight: 0.000000e+00
  - head.weight: 0.000000e+00
  - head.bias: 0.000000e+00
- **max_abs_diff: 0.000000e+00**
- **Status**: ✅ PASS (bit-exact, perfectly deterministic)

---

## Audit Cycle 2 — YelpChi GCN Stage C Score-Blind (2026-05-15 22:55)

### Score-Blind Audit: teacher_payloads.jsonl
- Command: `grep -lE 'base_score|base_prob|base_logit|"confidence"|"label"|"split"|"is_FN"|"is_FP"|"ground_truth"|"final_pred"' artifacts/err_cache/yelpchi/gcn/rule/seed_*/teacher_payloads.jsonl`
- **Result**: Empty output (0 matches)
- **Status**: ✅ PASS — teacher payloads are fully score-blind

### Score-Blind Audit: evidence_cards.jsonl
- Command: `grep -lE '...' artifacts/err_cache/yelpchi/gcn/rule/seed_*/evidence_cards.jsonl`
- **Result**: All 5 seeds match — field `calibration.base_score` present (1000 occurrences per file)
- **Analysis**: `base_score` exists ONLY in `.calibration.base_score` inside evidence_cards.jsonl. This is an intermediate calibration artifact. It does NOT flow into teacher_payloads.jsonl or LLM prompts. Teacher payloads contain only graph-structural/behavioral tokens (degree_level, neighbor_consistency, detector_signal, etc.)
- **Status**: ⚠️ INFO — intermediate artifact contains calibration metadata by design; no leak to LLM pathway
- **Source-code double-check** (confirmed by team-lead as by-design):
  - `grep -n "calibration\b" scripts/train_phase2_reasoner.py scripts/train_stage3.py` → **0 matches** ✅
  - `grep -n "base_score" scripts/train_phase2_reasoner.py scripts/train_stage3.py` → **0 matches** ✅
  - `base_logits` in training scripts comes from `output.logits.detach()` (frozen detector forward pass), not from evidence_cards
  - **Conclusion**: calibration.base_score in evidence_cards is internal metadata for polarity computation, never traverses teacher/LLM/reasoner training boundary. Verified by source-code grep.

### Score-Blind Audit: prompts.jsonl
- prompts.jsonl not present in Stage C err_cache (generated in Stage E)
- **Status**: N/A

### Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 1.000 |
| 123  | 1.000 |
| 456  | 1.000 |
| 789  | 1.000 |
| 2026 | 1.000 |
- **Status**: ✅ PASS — all seeds 100% acceptance (threshold: ≥ 0.80)

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

---

## Audit Cycle 3 — Amazon GCN Stage C Score-Blind (2026-05-15 23:05)

### Score-Blind Audit: teacher_payloads.jsonl
- Command: `grep -lE 'base_score|base_prob|base_logit|"confidence"|"label"|"split"|"is_FN"|"is_FP"|"ground_truth"|"final_pred"' artifacts/err_cache/amazon/gcn/rule/seed_*/teacher_payloads.jsonl`
- **Result**: Empty output (0 matches, grep exit=1)
- **Status**: ✅ PASS — teacher payloads are fully score-blind

### Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 1.000 |
| 123  | 1.000 |
| 456  | 1.000 |
| 789  | 1.000 |
| 2026 | 1.000 |
- **Status**: ✅ PASS — all seeds 100% acceptance (threshold: ≥ 0.80)

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

---

## Audit Cycle 4 — attention_heads Passthrough Bug Fix Verification (2026-05-15 23:20)

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### Git Diff Analysis (3 files modified, 1 claimed but unchanged)
**scripts/train_stage1.py** — LARGER THAN EXPECTED changes:
  - ✅ `attention_heads` passthrough via `extra_kwargs` pattern (safe: only adds if present in config)
  - ⚠️ Added `CUBLAS_WORKSPACE_CONFIG` env var before torch import
  - ⚠️ Added `--deterministic` CLI flag with `retraining_metrics.json` output
  - ⚠️ Added `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`
  - ⚠️ Added `torch.cuda.manual_seed_all(seed)`
  - Analysis: Deterministic training changes are ALWAYS-ON (not gated by `--deterministic` flag). This means ALL future train_stage1 runs use deterministic CUDA ops. The `--deterministic` flag only controls retraining_metrics.json output.

**scripts/generate_stage2_err.py** — MINIMAL, as expected:
  - ✅ `attention_heads` passthrough via `extra_kwargs` pattern (safe)

**scripts/build_judge_packets.py** — SLIGHTLY BROADER refactor:
  - ✅ `attention_heads` passthrough (safe)
  - ✅ `num_bands` made conditional (safe: only adds if present in config, matching attention_heads pattern)
  - Refactored from positional to dict-based `detector_kwargs` construction

**scripts/train_phase2_reasoner.py** — NO CHANGES (0 diff). Lead mentioned 4 scripts, only 3 modified.

### BWGNN Smoke Tests
- `train_stage1.py --debug --seed 42`: ✅ PASS (completed, checkpoint saved)
- `generate_stage2_err.py --debug --seed 42`: ✅ PASS (42 cards, 42 ERR, 100% acceptance, Score-blind: PASSED)

### ⚠️ BWGNN Checkpoint Corruption Incident
- **Cause**: `--debug` smoke test saved to production path `artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt`
- **Detection**: File size mismatch — 73,571 bytes (debug/synthetic) vs 77,667 bytes (production, other seeds)
- **Note**: BWGNN checkpoints are NOT git-tracked (gitignored). `git status` immutability check only covers tracked files.
- **Restoration**: Retrained on GPU 3 with full dataset: `CUDA_VISIBLE_DEVICES=3 python scripts/train_stage1.py --config configs/yelpchi_bwgnn.yaml --seed 42 --stratified --run_name base`
- **Post-restore size**: 77,667 bytes ✅ (matches all other seeds)
- **Post-restore metrics**: roc_auc=0.7958, macro_f1=0.6305 (reasonable for BWGNN/YelpChi)
- **Caveat**: Restored checkpoint was trained with NEW script (deterministic flags added). Original was trained with old script. Weights may differ from original, though file size matches. Downstream BWGNN phase2 results were computed with the original checkpoint.
- **Recommendation**: For future smoke tests, use a separate `--run_name smoke_test` to avoid overwriting production checkpoints.

### BWGNN Immutability (post-restore)
- All 5 seeds: 77,667 bytes ✅
- git status: `nothing to commit, working tree clean`

### Audit Cycle 4 Follow-up: train_stage1.py Debug Guard Added
- **Fix**: Added guard at line 88-90: `if args.debug and args.run_name == "base": args.run_name = "debug"` with print warning
- **Verification**: `python scripts/train_stage1.py --config configs/yelpchi_bwgnn.yaml --debug --seed 42`
  - Output: `[DEBUG] Forcing run_name=debug to avoid overwriting production checkpoints`
  - Saved to: `artifacts/checkpoints/yelpchi/bwgnn/debug/seed_42/base.pt` ✅ (NOT base/)
  - Production `base/seed_42/base.pt` untouched (77,667 bytes) ✅
- **pytest post-fix**: 323 passed, 2 skipped ✅
- **Cleanup**: `rm -rf artifacts/checkpoints/yelpchi/bwgnn/debug/` ✅
- **Status**: ✅ Future-proofed

---

## Audit Cycle 5 — YelpChi GCN Stage E Judge Safety (2026-05-15 23:30)

**First full pipeline completion audit (Task #1 exec-yc-gcn COMPLETED)**

### Forbidden Field Audit: judge_packets.jsonl
- Command: `grep -lE '...' artifacts/judge_packets/yelpchi/gcn/cover_rel_judge/seed_*/judge_packets.jsonl`
- **Result**: 0 matches ✅
- Note: `prompts.jsonl` does not exist in this pipeline variant (uses `judge_packet_texts.jsonl` instead)

### Forbidden Field Audit: judge_packet_texts.jsonl (actual LLM input)
- Command: `grep -lE '...' artifacts/judge_packets/yelpchi/gcn/cover_rel_judge/seed_*/judge_packet_texts.jsonl`
- **Result**: 0 matches ✅

### Built-in Pipeline Audit (judge_forbidden_field_audit.json, seed_42)
```json
{
  "passed": true,
  "accepted_outputs_passed": true,
  "prompt_packet_passed": true,
  "num_accepted_violations": 0,
  "num_prompt_packet_violations": 0,
  "num_generated_violations": 1,
  "num_rejected_forbidden_violations": 1,
  "num_audited": 120
}
```
- 1 violation was detected in raw LLM output and correctly REJECTED (never accepted) ✅
- All 120 audited packets clean ✅

### Judge Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 0.900 |
| 123  | 0.917 |
| 456  | 0.925 |
| 789  | 0.950 |
| 2026 | 0.925 |
| **Mean** | **0.923** |
- **Status**: ✅ PASS — all seeds ≥ 0.80 threshold

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

### Pipeline Results (reported by exec-yc-gcn)
- Base AUPRC: 0.1756
- Gate AUPRC: 0.4657±0.0141 (ΔAUPRC vs Base: +0.2901)
- Judge AUPRC: 0.4654±0.0131 (ΔAUPRC vs Gate: -0.0002)

---

## Audit Cycle 6 — YelpChi GAT Stage C Score-Blind + GAT Reproducibility (2026-05-15 23:40)

### Score-Blind Audit: teacher_payloads.jsonl
- Command: `grep -lE '...' artifacts/err_cache/yelpchi/gat/rule/seed_*/teacher_payloads.jsonl`
- **Result**: 0 matches (grep exit=1)
- **Status**: ✅ PASS — teacher payloads are fully score-blind

### Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 1.000 |
| 123  | 1.000 |
| 456  | 1.000 |
| 789  | 1.000 |
| 2026 | 1.000 |
- **Status**: ✅ PASS — all seeds 100% acceptance

### GAT Reproducibility (non-deterministic mode)
- Command: `python scripts/train_stage1.py --config configs/yelpchi_gat.yaml --seed 42 --stratified --run_name base_rerun`
- GAT skips `torch.use_deterministic_algorithms` (scatter_add_ OOM on dense graphs)
- Per-parameter diffs:
  - convs.0.att_src: 2.980232e-08
  - convs.0.att_dst: 2.980232e-08
  - convs.0.bias: 1.862645e-09
  - convs.0.lin.weight: 2.980232e-08
  - convs.1.att_src: 7.450581e-09
  - convs.1.att_dst: 2.980232e-08
  - convs.1.bias: 9.313226e-10
  - convs.1.lin.weight: 5.289912e-07
  - head.weight: 7.450581e-09
  - head.bias: 0.000000e+00
- **max_abs_diff: 5.289912e-07** (< 1e-3 threshold)
- **Status**: ✅ PASS
- **Comparison**: GCN was bit-exact (0e+00). GAT shows ~5e-07 drift from non-deterministic cuDNN attention reductions — expected and acceptable.
- Cleanup: `rm -rf artifacts/checkpoints/yelpchi/gat/base_rerun/` ✅

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

---

## Audit Cycle 7 — Amazon GCN Stage E Judge Safety (2026-05-15 23:50)

**Second full pipeline completion (Task #5 exec-am-gcn COMPLETED)**

### Forbidden Field Audit: judge_packets.jsonl + judge_packet_texts.jsonl
- Command: `grep -lE '...' artifacts/judge_packets/amazon/gcn/cover_rel_judge/seed_*/{judge_packets.jsonl,judge_packet_texts.jsonl}`
- **Result**: 0 matches (grep exit=1)
- **Status**: ✅ PASS — all judge artifacts are fully score-blind

### Judge Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 0.858 |
| 123  | 0.942 |
| 456  | 0.900 |
| 789  | 0.908 |
| 2026 | 0.917 |
| **Mean** | **0.905** |
- **Status**: ✅ PASS — all seeds ≥ 0.80 threshold
- **Note**: seed 42 lowest at 0.858, still comfortably above 0.80

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

### Pipeline Results (reported by exec-am-gcn)
- Base AUPRC: 0.2514
- Gate AUPRC: 0.2515 (ΔAUPRC vs Base: +0.0001) — neutral
- Judge AUPRC: 0.2513 (ΔAUPRC vs Gate: -0.0002) — neutral

---

## Audit Cycle 8 — train_stage3.py Bug Fix Verification (2026-05-16 00:00)

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### Git Diff Analysis: train_stage3.py
- `git diff scripts/train_stage3.py` → NO unstaged changes (fix already committed in 720dad7)
- attention_heads passthrough at lines 1012-1021: same `extra_kwargs` dict pattern as other scripts
- Code review: ONLY `attention_heads` kwargs passthrough added, no logic changes ✅

### BWGNN Smoke Test: train_stage3.py --debug
- **Result**: FAILED — `RuntimeError: size mismatch for input_proj.0.weight: [64,32] vs [64,16]`
- **Root cause**: Pre-existing debug mode limitation, NOT caused by attention_heads fix
  - tiny synthetic graph: 16 features
  - real YelpChi data: 32 features
  - train_stage3 debug mode creates tiny graph but loads real checkpoint → dimension mismatch
  - BWGNN config has NO `attention_heads` field → `extra_kwargs` is empty → fix is a no-op for BWGNN
- **Verdict**: ⚠️ Pre-existing issue, NOT a regression. The attention_heads fix does not affect BWGNN behavior.

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

---

## Audit Cycle 9 — YelpChi GAT Stage E Judge Safety (2026-05-16 00:10)

**Third full pipeline completion (Task #3 exec-yc-gat COMPLETED)**

### Forbidden Field Audit: judge_packets.jsonl + judge_packet_texts.jsonl
- Command: `grep -lE '...' artifacts/judge_packets/yelpchi/gat/cover_rel_judge/seed_*/{judge_packets.jsonl,judge_packet_texts.jsonl}`
- **Result**: 0 matches (grep exit=1)
- **Status**: ✅ PASS — all judge artifacts are fully score-blind

### Judge Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 0.917 |
| 123  | 0.942 |
| 456  | 0.933 |
| 789  | 0.917 |
| 2026 | 0.950 |
| **Mean** | **0.932** |
- **Status**: ✅ PASS — all seeds ≥ 0.80 threshold

### pytest
- **Result**: 323 passed, 2 skipped, 0 failed
- **Status**: ✅ PASS

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

### Pipeline Results (reported by exec-yc-gat) — NOVEL FINDING
- Base AUPRC: 0.1453±0.0061
- Gate AUPRC: 0.1598±0.0134 (ΔAUPRC vs Base: +0.0145)
- **Judge AUPRC: 0.3630±0.0546 (ΔAUPRC vs Gate: +0.2032, vs Base: +0.2176)**
- GAT shows massive judge benefit (+0.2032 over Gate), unlike GCN/BWGNN where judge was neutral
- This suggests the LLM judge provides strongest value when the base detector is weakest

---

## Audit Cycle 10 — Amazon GAT Stage C Score-Blind + Reproducibility (2026-05-16 00:20)

### Score-Blind Audit: teacher_payloads.jsonl
- Command: `grep -lE '...' artifacts/err_cache/amazon/gat/rule/seed_*/teacher_payloads.jsonl`
- **Result**: 0 matches (grep exit=1)
- **Status**: ✅ PASS — teacher payloads are fully score-blind

### Acceptance Rate
| Seed | acceptance_rate |
|------|----------------|
| 42   | 1.000 |
| 123  | 1.000 |
| 456  | 1.000 |
| 789  | 1.000 |
| 2026 | 1.000 |
- **Status**: ✅ PASS — all seeds 100% acceptance

### Amazon GAT Reproducibility (non-deterministic mode)
- Command: `python scripts/train_stage1.py --config configs/amazon_gat.yaml --seed 42 --stratified --run_name base_rerun`
- Per-parameter diffs:
  - convs.0.att_src: 7.863234e-02
  - convs.0.att_dst: 2.604837e-01
  - convs.0.bias: 2.012927e-01
  - convs.0.lin.weight: 2.233224e-01
  - convs.1.att_src: 8.784072e-02
  - convs.1.att_dst: 2.547853e-01
  - convs.1.bias: 8.976457e-01
  - convs.1.lin.weight: 3.053082e-01
  - head.weight: 3.612487e-01
  - head.bias: 6.930102e-01
- **max_abs_diff: 8.976457e-01** (EXCEEDS 1e-3 threshold)
- **Status**: ⚠️ WARN (warn_only=True per audit matrix)
- **Analysis**: Amazon GAT shows massive non-determinism vs YelpChi GAT (5.29e-07). ALL parameters diverge (0.07–0.90). Likely causes:
  - GAT skips `use_deterministic_algorithms` (scatter_add_ OOM risk)
  - Amazon graph structure amplifies non-deterministic cuDNN attention reductions more than YelpChi
  - Non-deterministic ops accumulate differently over 100 epochs on this graph topology
- **Comparison across models**:
  - GCN (deterministic): 0.000000e+00 (bit-exact)
  - YelpChi GAT (non-det): 5.289912e-07 (near-exact)
  - Amazon GAT (non-det): 8.976457e-01 (large drift)
- Cleanup: `rm -rf artifacts/checkpoints/amazon/gat/base_rerun/` ✅

### BWGNN Immutability
- **Result**: `nothing to commit, working tree clean`
- **Status**: ✅ PASS

---

---

## Final Audit (Stage F completion, 2026-05-16)

### pytest

`python -m pytest -q --tb=no`: 4 FAILED, but all are **CUDA OOM**, not code regressions:

| Test | Failure Mode |
|------|-------------|
| `test_train_stage1_deterministic.py::test_deterministic_flag_creates_retraining_metrics` | `torch.AcceleratorError: CUDA out of memory` |
| `test_train_stage1_deterministic.py::test_no_deterministic_flag_skips_retraining_metrics` | `torch.AcceleratorError: CUDA out of memory` |
| `test_stage3_debug.py::test_stage3_debug_creates_artifacts` | `torch.AcceleratorError: CUDA out of memory` |
| `test_real_sanity_scripts.py::test_generate_stage2_trace_size_arg` | `torch.AcceleratorError: CUDA out of memory` |

**Root cause**: GPU 0 is fully occupied (24,061 MiB / 24,576 MiB) by an external process (PID 1353208, not owned by this project). Tests default to `cuda` and cannot allocate. These same tests passed in audit cycles 0–11 (323 passed, 2 skipped consistently). No code regression in our Stage F doc-only changes (`PROGRESS.md`, `AGENTS.md`, `artifacts/paper/method_section_draft.md`, `artifacts/tables/*`).

### Final Stage F Artifacts

| Artifact | Lines | Status |
|----------|------:|--------|
| `artifacts/tables/am_gat_phase2_judge_5seed.md` | 35 | NEW |
| `artifacts/tables/paper_cross_model_results.csv` | 19 | NEW |
| `artifacts/tables/paper_cross_model_results.md` | 71 | NEW |
| `PROGRESS.md` (Task 11) | +354 lines | UPDATED |
| `AGENTS.md` (cross-model table) | +12 lines | UPDATED |
| `artifacts/paper/method_section_draft.md` | +6 lines (cross-model section) | UPDATED |

### Cleanup

- `artifacts/checkpoints/{yelpchi,amazon}/*/base_rerun/` — removed (verifier audit scratch).

### Verdict

Task 11 GCN/GAT Cross-Model Validation pipeline complete. All 20 base.pt + 20 phase2_E0 + 20 phase2_E2 artifacts present. All 7 result tables generated. Cross-model paper artifacts written. pytest CUDA-OOM failures are environment-only (external GPU contention) and not caused by Stage F changes.
