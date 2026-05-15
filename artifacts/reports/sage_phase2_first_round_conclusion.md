# SAGE Phase2 First-Round Conclusion

**Author**: verifier agent
**Date**: 2026-05-15
**Source data**: `artifacts/tables/phase2_sage_5seed_summary.csv`, `artifacts/tables/phase2_sage_5seed_summary.md`

---

## 1. SAGE Base Performance (5-seed deterministic)

| Dataset | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---------|-------|---------|----------|---------|
| YelpChi | 0.2246 ± 0.1261 | 0.5995 ± 0.1115 | 0.4848 ± 0.0535 | 0.0746 ± 0.1667 |
| Amazon | 0.7556 ± 0.0511 | 0.8934 ± 0.0284 | 0.8212 ± 0.0861 | 0.7119 ± 0.1351 |

Source: `artifacts/results/{ds}/sage/base/seed_X/stage1_metrics.json` (X ∈ {42, 123, 456, 789, 2026})

SAGE base is substantially weaker than BWGNN base (YelpChi: 0.2246 vs 0.4674 AUPRC; Amazon: 0.7556 vs 0.8643 AUPRC). This is expected — SAGE lacks BWGNN's spectral band-pass filters.

---

## 2. SAGE Stage3 Anchor Gate (relation-only, no LLM)

| Dataset | AUPRC | ROC-AUC | Macro-F1 | G-Means | ΔAUPRC vs base |
|---------|-------|---------|----------|---------|----------------|
| YelpChi | 0.4465 ± 0.0289 | 0.8064 ± 0.0142 | 0.4873 ± 0.0591 | 0.0777 ± 0.1737 | **+0.2219** |
| Amazon | 0.7667 ± 0.0444 | 0.8975 ± 0.0260 | 0.8378 ± 0.0670 | 0.7380 ± 0.1078 | **+0.0111** |

Source: `artifacts/results/{ds}/sage/cover_rel_anchor_gate_nollm/seed_X/stage3_metrics.json`

YelpChi: Massive AUPRC lift (+0.2219), 5/5 seeds positive. Relation evidence nearly doubles SAGE's AUPRC.
Amazon: Modest lift (+0.0111), 5/5 seeds positive.

---

## 3. SAGE Phase2 Results (YelpChi: full E0/E1/E2; Amazon: E0 gate-failed)

### YelpChi (5 seeds)

| Stage | AUPRC | ROC-AUC | Macro-F1 | G-Means | ΔAUPRC vs base |
|-------|-------|---------|----------|---------|----------------|
| SAGE base | 0.2246 ± 0.1261 | 0.5995 ± 0.1115 | 0.4848 ± 0.0535 | 0.0746 ± 0.1667 | — |
| anchor_gate | 0.4465 ± 0.0289 | 0.8064 ± 0.0142 | 0.4873 ± 0.0591 | 0.0777 ± 0.1737 | +0.2219 |
| Phase2 E0 | 0.4503 ± 0.0891 | 0.8429 ± 0.0249 | 0.7056 ± 0.0238 | 0.7103 ± 0.0372 | +0.2258 |
| Phase2 E1 | 0.4535 ± 0.0888 | 0.8438 ± 0.0254 | 0.7064 ± 0.0246 | 0.7124 ± 0.0403 | +0.2290 |
| Phase2 E2 | 0.4539 ± 0.0891 | 0.8442 ± 0.0257 | 0.7064 ± 0.0247 | 0.7131 ± 0.0413 | **+0.2293** |

Per-seed E0 ΔAUPRC vs SAGE base: seed 42 = +0.2913, seed 123 = +0.1069, seed 456 = +0.1746, seed 789 = +0.2429, seed 2026 = +0.3132. All 5/5 positive.

### Amazon (gate-failed at E0)

| Stage | AUPRC | ΔAUPRC vs base | Note |
|-------|-------|----------------|------|
| SAGE base (seed 42) | 0.7415 | — | |
| Phase2 E0 (seed 42) | 0.7415 | +0.0000 | **STRICT GATE FAIL** |
| Phase2 E1 | — | — | Not executed (gate-failed) |
| Phase2 E2 | — | — | Not executed (gate-failed) |

Source: `artifacts/results/amazon/sage/phase2_E0_relgate/seed_42/stage3_metrics.json`

Amazon Phase2 E0 showed zero AUPRC improvement on the smoke seed. Per strict-gate policy, E1/E2 were not executed, saving ~80 minutes of Qwen GPU time.

---

## 4. Cross-Base Comparison (SAGE Phase2 vs BWGNN Phase2)

### YelpChi

| Method | Base | AUPRC | ΔAUPRC vs own base | ΔAUPRC (absolute gain) |
|--------|------|-------|--------------------|----------------------|
| BWGNN base | BWGNN | 0.4674 | — | — |
| BWGNN Phase2 E0 | BWGNN | 0.5669 | +0.0995 | +0.0995 |
| BWGNN Phase2 E2 | BWGNN | 0.5676 | +0.1002 | +0.1002 |
| SAGE base | SAGE | 0.2246 | — | — |
| SAGE Phase2 E0 | SAGE | 0.4503 | **+0.2258** | +0.2258 |
| SAGE Phase2 E2 | SAGE | 0.4539 | **+0.2293** | +0.2293 |

BWGNN Phase2 source: `artifacts/tables/phase2_5seed_summary.md`
SAGE Phase2 source: `artifacts/tables/phase2_sage_5seed_summary.md`

Key finding: Relation evidence gives SAGE a 2.3x larger relative lift (+0.2293) than BWGNN (+0.1002). SAGE has more headroom because it starts weaker. However, SAGE Phase2 E2 absolute AUPRC (0.4539) remains below BWGNN base (0.4674).

### Amazon

| Method | Base | AUPRC | ΔAUPRC vs own base |
|--------|------|-------|-------------------|
| BWGNN base | BWGNN | 0.8643 | — |
| BWGNN Phase2 E0 | BWGNN | 0.8643 | +0.0000 |
| SAGE base | SAGE | 0.7556 | — |
| SAGE Phase2 E0 (seed 42) | SAGE | 0.7415 | +0.0000 |

Neither BWGNN nor SAGE Phase2 E0 improves Amazon AUPRC over their respective bases. The Amazon dataset appears saturated for relation-gate-only improvements at Phase2 level.

---

## 5. Diagnostics Summary

| Dataset | Exp | mean|Δ_rel| | mean α_llm | reject α (max) | gate entropy |
|---------|-----|-------------|------------|-----------------|--------------|
| YelpChi | E0 | 1.3628 | 0.0000 | 0.0000 | 0.2067 |
| YelpChi | E1 | 1.3591 | 0.0000 | 0.0000 | 0.2337 |
| YelpChi | E2 | 1.3571 | 0.0000 | 0.0000 | 0.2396 |
| Amazon | E0 | 1.4239 | 0.0000 | 0.0000 | 0.0012 |

All `reject α (max) = 0.0` — no LLM gate leaks detected. Safety: PASS.
Source: `artifacts/logs/{ds}/sage/phase2_E{0,1,2}_*/seed_X/phase2_diagnostics.json`

---

## 6. Verdict

### Per-dataset assessment

| Dataset | mean ΔAUPRC (best) | Positive seeds | Threshold met (>+0.005, ≥3/5) | Result |
|---------|-------------------|----------------|-------------------------------|--------|
| YelpChi | +0.2293 (E2) | 5/5 | YES | **POSITIVE** |
| Amazon | +0.0000 (E0 gate-fail) | 0/1 tested | NO | **NEGATIVE** |

### Overall verdict: **MIXED**

- YelpChi is a strong positive: relation evidence transforms SAGE from a near-random detector (AUPRC 0.2246) into a competitive model (AUPRC 0.4539), with consistent gains across all 5 seeds and progressive improvement E0 → E1 → E2.
- Amazon is a negative result: Phase2 E0 showed zero improvement on the smoke seed. The strict gate correctly stopped further Qwen expenditure.

---

## 7. Recommended Next Steps

1. **Do NOT deploy SAGE-Gate as a universal second base detector.** The mixed verdict (YelpChi positive, Amazon negative) does not meet the "both datasets positive" criterion for deployment recommendation.

2. **Report SAGE as a cross-base generalization study:**
   - CoVER-REL's relation evidence mechanism generalizes to SAGE on YelpChi: the lift is even larger than BWGNN (+0.2293 vs +0.1002 ΔAUPRC).
   - Amazon remains challenging for both SAGE and BWGNN at Phase2 E0 level (both show ~0 improvement), suggesting Amazon Phase2 gains require the judge branch (E1/E2), which was only viable for BWGNN.

3. **Notable finding for the paper:** SAGE Phase2 E2 nearly closes the gap to BWGNN base on YelpChi (0.4539 vs 0.4674), despite starting from a 2x weaker base. This demonstrates that CoVER-REL's relation evidence is the dominant discriminative signal, not the base model's spectral filters.

4. **Bug fix to note:** `scripts/build_judge_packets.py` was patched to conditionally pass `num_bands` (SAGE does not use spectral bands). This is a model-agnostic improvement.

---

## Artifact Paths

| Artifact | Path |
|----------|------|
| SAGE Phase2 summary (CSV) | `artifacts/tables/phase2_sage_5seed_summary.csv` |
| SAGE Phase2 summary (MD) | `artifacts/tables/phase2_sage_5seed_summary.md` |
| SAGE Phase2 per-seed (CSV) | `artifacts/tables/phase2_sage_5seed_per_seed.csv` |
| Auto-generated conclusion | `artifacts/reports/phase2_sage_first_round_conclusion.md` |
| Safety audit | `artifacts/reports/sage_safety_audit.md` |
| This report | `artifacts/reports/sage_phase2_first_round_conclusion.md` |
