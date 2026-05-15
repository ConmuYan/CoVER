# SAGE Pipeline Safety Audit

**Auditor**: verifier agent
**Date**: 2026-05-15
**Scope**: SAGE base adaptation — Phase1 through Phase2 E0/E1/E2 (YelpChi full, Amazon E0-only gate-fail)

---

## YelpChi (5 seeds: 42, 123, 456, 789, 2026)

### 1a. Judge Forbidden Field Audit

| Seed | passed | accepted_outputs_passed | Result | Artifact Path |
|------|--------|------------------------|--------|---------------|
| 42 | true | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_42/judge_forbidden_field_audit.json` |
| 123 | true | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_123/judge_forbidden_field_audit.json` |
| 456 | true | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_456/judge_forbidden_field_audit.json` |
| 789 | true | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_789/judge_forbidden_field_audit.json` |
| 2026 | true | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_2026/judge_forbidden_field_audit.json` |

### 1b. Relation Feature Meta (score_blind)

| Seed | score_blind | target_label_used | val_label_used | test_label_used | Result | Artifact Path |
|------|-------------|-------------------|----------------|-----------------|--------|---------------|
| 42 | true | false | false | false | PASS | `artifacts/relation_features/yelpchi/sage/seed_42/all/rel_feature_meta.json` |
| 123 | true | false | false | false | PASS | `artifacts/relation_features/yelpchi/sage/seed_123/all/rel_feature_meta.json` |
| 456 | true | false | false | false | PASS | `artifacts/relation_features/yelpchi/sage/seed_456/all/rel_feature_meta.json` |
| 789 | true | false | false | false | PASS | `artifacts/relation_features/yelpchi/sage/seed_789/all/rel_feature_meta.json` |
| 2026 | true | false | false | false | PASS | `artifacts/relation_features/yelpchi/sage/seed_2026/all/rel_feature_meta.json` |

### 1c. Judge Packet Meta (score_blind)

| Seed | score_blind | Result | Artifact Path |
|------|-------------|--------|---------------|
| 42 | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_42/judge_packet_meta.json` |
| 123 | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_123/judge_packet_meta.json` |
| 456 | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_456/judge_packet_meta.json` |
| 789 | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_789/judge_packet_meta.json` |
| 2026 | true | PASS | `artifacts/judge_packets/yelpchi/sage/cover_rel_judge/seed_2026/judge_packet_meta.json` |

### 2. LLM Gate Leak Check (Phase2 E1 + E2)

| Experiment | Seed | max_abs_alpha_llm_rejected | Result | Artifact Path |
|------------|------|---------------------------|--------|---------------|
| E1 | 42 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E1_judge_align/seed_42/phase2_diagnostics.json` |
| E1 | 123 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E1_judge_align/seed_123/phase2_diagnostics.json` |
| E1 | 456 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E1_judge_align/seed_456/phase2_diagnostics.json` |
| E1 | 789 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E1_judge_align/seed_789/phase2_diagnostics.json` |
| E1 | 2026 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E1_judge_align/seed_2026/phase2_diagnostics.json` |
| E2 | 42 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E2_judge_residual/seed_42/phase2_diagnostics.json` |
| E2 | 123 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E2_judge_residual/seed_123/phase2_diagnostics.json` |
| E2 | 456 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E2_judge_residual/seed_456/phase2_diagnostics.json` |
| E2 | 789 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E2_judge_residual/seed_789/phase2_diagnostics.json` |
| E2 | 2026 | 0.0 | PASS | `artifacts/logs/yelpchi/sage/phase2_E2_judge_residual/seed_2026/phase2_diagnostics.json` |

### YelpChi Score-Blind Summary: 15/15 checks — all PASS

---

## Amazon (5 seeds: 42, 123, 456, 789, 2026)

### 1a. Judge Forbidden Field Audit

| Seed | passed | accepted_outputs_passed | Result | Artifact Path |
|------|--------|------------------------|--------|---------------|
| 42 | true | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_42/judge_forbidden_field_audit.json` |
| 123 | true | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_123/judge_forbidden_field_audit.json` |
| 456 | true | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_456/judge_forbidden_field_audit.json` |
| 789 | true | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_789/judge_forbidden_field_audit.json` |
| 2026 | true | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_2026/judge_forbidden_field_audit.json` |

### 1b. Relation Feature Meta (score_blind)

| Seed | score_blind | target_label_used | val_label_used | test_label_used | Result | Artifact Path |
|------|-------------|-------------------|----------------|-----------------|--------|---------------|
| 42 | true | false | false | false | PASS | `artifacts/relation_features/amazon/sage/seed_42/all/rel_feature_meta.json` |
| 123 | true | false | false | false | PASS | `artifacts/relation_features/amazon/sage/seed_123/all/rel_feature_meta.json` |
| 456 | true | false | false | false | PASS | `artifacts/relation_features/amazon/sage/seed_456/all/rel_feature_meta.json` |
| 789 | true | false | false | false | PASS | `artifacts/relation_features/amazon/sage/seed_789/all/rel_feature_meta.json` |
| 2026 | true | false | false | false | PASS | `artifacts/relation_features/amazon/sage/seed_2026/all/rel_feature_meta.json` |

### 1c. Judge Packet Meta (score_blind)

| Seed | score_blind | Result | Artifact Path |
|------|-------------|--------|---------------|
| 42 | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_42/judge_packet_meta.json` |
| 123 | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_123/judge_packet_meta.json` |
| 456 | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_456/judge_packet_meta.json` |
| 789 | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_789/judge_packet_meta.json` |
| 2026 | true | PASS | `artifacts/judge_packets/amazon/sage/cover_rel_judge/seed_2026/judge_packet_meta.json` |

### 2. LLM Gate Leak Check

Amazon Phase2 E1/E2 were NOT executed (strict gate failed at E0). No diagnostics to check.

Amazon E0 seed 42 diagnostics: `max_abs_alpha_llm_rejected=0.0` — PASS.
Source: `artifacts/logs/amazon/sage/phase2_E0_relgate/seed_42/phase2_diagnostics.json`

### Amazon Score-Blind Summary: 15/15 checks (judge/rel_feature/packet) — all PASS; E0 diagnostics — PASS

---

## 3. Determinism Reproducibility Check

| Check | Result | Details |
|-------|--------|---------|
| Re-run | YelpChi SAGE seed 42, `--deterministic` flag | `CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 python scripts/train_stage1.py --config configs/yelpchi_sage.yaml --seed 42 --stratified --deterministic --run_name verify_determinism` |
| Original checkpoint | `artifacts/checkpoints/yelpchi/sage/base/seed_42/base.pt` | 8 tensor keys |
| Verify checkpoint | `artifacts/checkpoints/yelpchi/sage/verify_determinism/seed_42/base.pt` | 8 tensor keys |
| Max absolute difference | **0.00e+00** | Bit-exact match |
| Keys match | true | All 8 keys present in both |
| Result | **PASS** | Deterministic training is bit-exact reproducible |
| Cleanup | verify_determinism artifacts deleted after successful comparison | |

---

## 4. Pytest Sanity

| Check | Result | Command |
|-------|--------|---------|
| Full suite | **323 passed, 2 skipped, 0 failed** | `pytest --tb=no -q` |
| Phase2 tests | Included in above (all pass) | `tests/test_phase2_*.py` |
| Score-blind tests | Included in above (all pass) | `tests/test_score_blind*.py` |

---

## 5. Phase2 Gate Decisions

| Dataset | Phase2 E0 ΔAUPRC vs SAGE base | Gate Decision | E1/E2 Executed |
|---------|-------------------------------|---------------|----------------|
| YelpChi | +0.2258 (5/5 seeds positive) | **PASS** | Yes (5 seeds each) |
| Amazon | +0.0000 (seed 42 only, ΔAUPRC ≈ 0) | **FAIL** (strict gate) | No — saved ~80 min Qwen |

---

## Overall Safety Audit Summary

```
+--------------------------------------------------+
|                                                  |
|   SAGE Safety Audit: ALL CHECKS PASS             |
|                                                  |
|   Score-blind checks:       30/30 PASS           |
|   LLM gate leak checks:    10/10 PASS (E1+E2)   |
|   Amazon E0 diagnostics:    1/1  PASS            |
|   Determinism:              PASS (bit-exact)     |
|   pytest:                   323 passed, 0 failed |
|   Gate decisions:           Correctly applied     |
|                                                  |
|   No safety violations detected.                 |
|                                                  |
+--------------------------------------------------+
```
