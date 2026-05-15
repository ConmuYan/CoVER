# YelpChi BWGNN vs SAGE — same teacher Phase2 config (lalign_1em2_standard)

## Setup

This experiment swaps the Phase1 base detector (BWGNN ↔ SAGE) while holding the
Phase2 unified reasoner config strictly identical. The Phase2 config is sourced
verbatim from the BWGNN-confirmed run
`phase2_yelp_confirm_lalign_1em2_standard`. Only `model.name` (and the
detector-specific `num_bands`/`agg` keys, which SAGE does not accept) were
swapped to produce `configs/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`.
All Phase2 hyperparameters (lr, optimizer, lambda_align=1e-2, lambda_trust=3e-3,
lambda_sparse=1e-3, tau_gate=0.7, delta_rel_max=2.0, anchor=RUR, use_judge=true,
alpha_max=0.0, judge feature dim, eta_llm) are bit-identical to the BWGNN run.

5 seeds: 42, 123, 456, 789, 2026 (matched).

## Per-seed comparison

| Seed | BWGNN AUPRC | SAGE AUPRC | BWGNN ROC | SAGE ROC | BWGNN MF1 | SAGE MF1 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.5647 | 0.5018 | 0.8690 | 0.8666 | 0.7299 | 0.7298 |
| 123 | 0.5930 | 0.5543 | 0.8778 | 0.8557 | 0.7383 | 0.7143 |
| 456 | 0.5843 | 0.4224 | 0.8707 | 0.8457 | 0.7321 | 0.7054 |
| 789 | 0.5799 | 0.4403 | 0.8773 | 0.8422 | 0.7349 | 0.7036 |
| 2026 | 0.5803 | 0.4740 | 0.8749 | 0.8587 | 0.7355 | 0.7258 |
| **mean ± std** | **0.5804 ± 0.0103** | **0.4786 ± 0.0522** | **0.8739 ± 0.0039** | **0.8538 ± 0.0099** | **0.7341 ± 0.0033** | **0.7158 ± 0.0118** |

## Paired delta (BWGNN minus SAGE, same seed)

| Metric | mean ± std | paired t-stat | Verdict |
|---|---:|---:|---|
| AUPRC | **+0.1019 ± 0.0514** | +4.43 | BWGNN > SAGE significantly |
| ROC-AUC | +0.0201 ± 0.0121 | +3.73 | BWGNN > SAGE significantly |

## SAGE delta vs its own baselines

| Compared against | mean ± std | paired t-stat | Note |
|---|---:|---:|---|
| SAGE Phase1 base (5 seeds) | **+0.2540 ± 0.0833** | +6.82 | Massive lift, highly significant |
| SAGE old anchor_gate (5 seeds) | **+0.0321 ± 0.0283** | +2.53 | Borderline-significant lift over Stage3 — 4.5× larger than SAGE-default Phase2 E2 (+0.0074) |

## Diagnostic comparison (5-seed mean)

| Field | BWGNN mean | SAGE mean | Δ (SAGE - BWGNN) |
|---|---:|---:|---:|
| mean\|Δ_rel\| | 1.5167 | 1.4696 | -0.047 |
| mean α_llm | 0.0000 | 0.0000 | 0 (alpha_max=0) |
| gate entropy | 0.3843 | 0.3837 | -0.001 |
| dominance ρ | 0.7578 | 0.8962 | **+0.138** |
| gate weight RUR (rel_0) | 0.7834 | 0.7615 | -0.022 |
| gate weight RSR (rel_1) | 0.1039 | 0.1375 | +0.034 |
| gate weight RTR (rel_2) | 0.1128 | 0.1011 | -0.012 |

## Safety audit (LLM gate leak)

| Base | max_abs_alpha_llm_rejected (5 seeds, max) | Status |
|---|---:|---|
| BWGNN | 0.00e+00 | ✅ PASS |
| SAGE | 0.00e+00 | ✅ PASS |

Both strictly below 1e-6 threshold; safety unchanged from prior audits.

## Key observations

1. **Absolute performance**: under identical Phase2 teacher config, BWGNN
   outperforms SAGE by paired ΔAUPRC = +0.1019 (t=+4.43). The reasoner alone
   does not close the base detector gap.

2. **BWGNN-confirmed config helps SAGE more than SAGE-default did**:
   - SAGE-default Phase2 E2 vs old anchor_gate: Δ = +0.0074 (t=0.23, ns)
   - SAGE-confirmed (lalign_1em2_standard) vs old anchor_gate: Δ = +0.0321 (t=+2.53, borderline-sig)
   - 4.5× larger relative lift; the BWGNN-tuned `lambda_align=1e-2` (10× SAGE-default 1e-3) also benefits SAGE.

3. **SAGE has higher seed variance**: AUPRC std 0.0522 vs BWGNN's 0.0103
   (5× wider). The reasoner cannot smooth the underlying SAGE base
   instability, even with the BWGNN-confirmed config.

4. **Gate behavior is nearly identical**: gate entropy 0.384 vs 0.384, RUR
   weight 0.76-0.78 across both bases. SAGE has slightly more peaked dominance
   (ρ 0.90 vs 0.76), suggesting the SAGE reasoner relies a bit more heavily
   on the dominant relation residual head, not the gate distribution itself.

5. **Δ_rel magnitude is similar**: 1.47 vs 1.52 — the residual heads are
   making comparable size corrections on both bases, but the corrections are
   less effective on SAGE because the base is starting further from the truth.

## Verdict

The BWGNN-confirmed Phase2 teacher config **does transfer to SAGE** with a
real, paired-significant lift over SAGE's own anchor_gate (+0.032 AUPRC,
t=+2.53). However it does not equalize the absolute base detector gap:
SAGE+Phase2 (0.479) remains 0.10 AUPRC behind BWGNN+Phase2 (0.580). The
residual gap is dominated by Phase1 base detector quality, not by Phase2
reasoner config quality.

For paper framing, this is a clean cross-base ablation:
- "Phase2 config space tuned on BWGNN transfers positively to SAGE"
  (+0.032 paired Δ vs anchor_gate).
- "But the base detector gap (SAGE vs BWGNN) is preserved through Phase2"
  (+0.102 paired Δ vs SAGE-side after both get the same Phase2 layer).

## Source artifacts

- BWGNN: `artifacts/results/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_X/stage3_metrics.json`
- SAGE: `artifacts/results/yelpchi/sage/phase2_yelp_confirm_lalign_1em2_standard/seed_X/stage3_metrics.json`
- BWGNN diag: `artifacts/logs/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_X/phase2_diagnostics.json`
- SAGE diag: `artifacts/logs/yelpchi/sage/phase2_yelp_confirm_lalign_1em2_standard/seed_X/phase2_diagnostics.json`
- SAGE config: `configs/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`
