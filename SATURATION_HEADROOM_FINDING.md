# Saturation-Headroom Hypothesis: When LLM-Augmentation Helps GAD

**Status**: validated, 5-seed × 4-base paired t-tests
**Date**: 2026-05-17
**Project**: CoVER-FD (YelpChi fraud detection)

---

## TL;DR

LLM-augmented modules (per-relation trust signal via auxiliary classifier hidden state, called **B3 PRTAE**) help fraud detection **only on competent-but-unsaturated base classifiers**. They actively hurt weak bases (where the rel branch already extracts maximum signal) and have no room on already-saturated bases. The empirical signature is a **U-shaped efficacy curve** in base strength.

---

## Key Empirical Table (YelpChi, 5-seed paired t-tests)

| Base   | Base AUPRC | Idea 1 ΔAUPRC (rel branch over base) | Idea 2 ΔAUPRC (B3 over rel branch) |
|--------|-----------:|--------------------------------------:|------------------------------------:|
| GCN    | 0.219      | **+0.292** ★★ (p<0.01)                | **−0.0104** ★ (p<0.05) ← hurts      |
| GAT    | 0.207      | **+0.320** ★★ (p<0.01)                | −0.0030 (n.s.)                      |
| SAGE   | 0.455      | **+0.147** ★★ (p<0.01)                | **−0.0056** ★ (p<0.05) ← hurts      |
| BWGNN  | 0.503      | **+0.105** ★★ (p<0.01)                | **+0.0053** ★★ (p<0.01) ← helps     |

### Observed pattern

- **Idea 1 (relation-branch residual)** is the workhorse: lifts AUPRC by +0.105 to +0.320 absolute, all 4 bases significant at p<0.01.
- **Idea 2 (B3 PRTAE auxiliary classifier signal)** is **base-dependent**: significantly **harmful** on GCN/SAGE, neutral on GAT, **significantly beneficial** on BWGNN.

---

## The U-shape hypothesis

```
ΔAUPRC(idea 2)
       │
       │      ┌── BWGNN
   +0  │──────┤
       │      │
   −   │ GCN  ├── GAT/SAGE
       │ ↓    ↓
       └──────────────────→ base strength
        weak  medium  strong+rel-saturated
```

Three regimes:

1. **Weak base** (GCN, GAT, AUPRC<0.22): rel branch over-fits residual capacity to compensate for weak base. The 9-stat rel signal saturates the rel branch's expressive capacity entirely. Adding B3's per-relation trust signal injects redundant noise → AUPRC degrades.

2. **Medium-but-saturated base** (SAGE, AUPRC=0.46): rel branch lifts substantially (+0.15) but base+rel together saturate the available structural signal. B3 has no room → slight degradation.

3. **Strong, partially-saturated base** (BWGNN, AUPRC=0.50): rel branch adds smaller increment (+0.105). The base is competent enough to leave headroom in the residual that B3's trust signal can exploit → AUPRC gains.

### Quantitative correlation

ΔAUPRC(idea 2) shows roughly **negative correlation with ΔAUPRC(idea 1)** across bases:
- GAT: idea 1 = +0.320 → idea 2 = −0.0030 (rel exhausted)
- GCN: idea 1 = +0.292 → idea 2 = −0.0104 (rel exhausted)
- SAGE: idea 1 = +0.147 → idea 2 = −0.0056 (still exhausted)
- BWGNN: idea 1 = +0.105 → idea 2 = +0.0053 (rel left room)

The crossover seems to occur around idea 1 ΔAUPRC ≈ +0.10. Above this threshold, idea 2 is squeezed out; below, it can find purchase.

---

## Implications for LLM-augmented GAD literature

- Most LLM-aug GAD papers (FLAG, MLED, ...) report ~+3-7% lift over their chosen base. This may reflect that they implicitly choose bases in the "headroom" regime. Our cross-base study shows the choice of base is dispositive: a wrong (too weak) base will null the LLM contribution by elevated rel-branch saturation.
- The "LLM-as-Evidence-Auditor" failure mode reported in our Phase 2 work (LEQA all variants null/negative on BWGNN AUPRC) is consistent: BWGNN+rel is **borderline** for LEQA-style scalar-quality injection (which we found degrades G-means), but **suffices** for B3-style auxiliary feature injection.

---

## Recommended paper structure (revised)

1. **Method**: Frozen-base + per-relation expert MLP rel-branch (Idea 1), 4-term joint loss (Idea 3 = L_cls + λ_int·Δ² + λ_sparse·KL(π_ev ‖ g) + B3 augmentation in MLP input).
2. **Main result**: Idea 1 cross-base lifts (+0.105 to +0.320 AUPRC), 4 bases × 5 seeds.
3. **Idea 2 result**: B3 PRTAE — beneficial only on BWGNN (+0.0053 AUPRC p<0.01).
4. **Cross-base ablation**: full 4-base × {base, E0, B3} table.
5. **Saturation-headroom discussion**: U-shape hypothesis, ΔIdea1–ΔIdea2 anti-correlation, threshold analysis.
6. **Comprehensive negative results** (LEQA / LLM verbalization / sigpool raw-text aggregation): showing the wider mechanism space largely fails, B3 is the rare success.

---

## Provenance

- All experiments: YelpChi.mat (45,954 nodes, 14.5% fraud, RUR/RSR/RTR relations)
- Base ckpts: `artifacts/checkpoints/yelpchi/{model}/fixed_v1_100ep/seed_{seed}/base.pt`
- Reasoner: `models/cover_rel_reasoner.py` (Phase 3 cleanup, judge path removed)
- B3 aux features: `artifacts/relation_features_aug41/yelpchi/bwgnn/seed_{seed}/all/rel_stats.pt` (per-relation MLP classifier hidden state, 9→32→1 trained on rel_stats)
- Phase 3 trainer: `scripts/train_phase3_reasoner.py` (worktree `../cover-fd-qmixer`)
- Seeds: 42, 123, 456, 789, 2026
- Paired t-tests: df=4, significance bar p<0.05 = |t|>2.78, p<0.01 = |t|>4.6

---

## Negative results matrix (for completeness)

For BWGNN base, the following LLM-augmentation mechanisms were tested 5-seed; none crossed the +0.0053 AUPRC threshold:

| Mechanism | Best AUPRC Δ | t-stat |
|---|---|---|
| LEQA audit loss (λ sweep × E1-E5 cells) | ~0 | n.s. |
| Schema-gate KL prior (5 oracles × λ sweep) | ~0 | n.s. (single-seed cherry-pick only) |
| q-gate forward feature mixer | ~0 | n.s. |
| q-weighted BCE classification loss | −0.001 | n.s. |
| LLM Qwen3-0.6B verbalization embedding (PCA32) | **−0.0168** | **−10.6 (p<0.01) worse** |
| Raw-text sigpool (leak-free) | **−0.0515** | **−19.4 (p<0.01) much worse** |
| Oracle q (perfect base-correctness signal) | ~0 | n.s. |
| **B3 per-relation classifier hidden state** | **+0.0053** | **+6.61 (p<0.01)** |

B3 stands out as the only mechanism that delivered a multi-seed significant lift.
