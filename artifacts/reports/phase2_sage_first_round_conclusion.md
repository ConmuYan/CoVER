# Phase2 SAGE First-Round Conclusion

## 1. E0 relation-gate improvement over SAGE base

| Dataset | E0 AUPRC (SAGE + relation gate) | SAGE base | Δ |
|---|---|---|---|
| yelpchi | 0.4503 | 0.2246 | 0.2258 |
| amazon | 0.7415 | 0.7556 | -0.0141 |

Pass criterion for proceeding to E1/E2: Δ > 0.

## 2. Main result table (5-seed mean ± std)

### yelpchi

| Experiment | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| E0 | 0.4503 ± 0.0891 | 0.8429 ± 0.0249 | 0.7056 ± 0.0238 | 0.7103 ± 0.0372 |
| E1 | 0.4535 ± 0.0888 | 0.8438 ± 0.0254 | 0.7064 ± 0.0246 | 0.7124 ± 0.0403 |
| E2 | 0.4539 ± 0.0891 | 0.8442 ± 0.0257 | 0.7064 ± 0.0247 | 0.7131 ± 0.0413 |

### amazon

| Experiment | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| E0 | 0.7415 ± 0.0000 | 0.8660 ± 0.0000 | 0.8766 ± 0.0000 | 0.8189 ± 0.0000 |
| E1 | — ± — | — ± — | — ± — | — ± — |
| E2 | — ± — | — ± — | — ± — | — ± — |

## 3. Comparison with BWGNN baselines

### yelpchi

| Method | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| SAGE base | 0.2246 | 0.5995 | 0.4848 | 0.0746 |
| BWGNN base (BWGNN) | 0.4674 | 0.8076 | 0.6483 | 0.5105 |
| anchor_gate (BWGNN) | 0.4998 | 0.8177 | 0.6598 | 0.5253 |
| judge_strength_gate (BWGNN) | 0.5006 | 0.8180 | 0.6650 | 0.5359 |
| SAGE Phase2 E0 | 0.4503 | 0.8429 | 0.7056 | 0.7103 |
| SAGE Phase2 E1 | 0.4535 | 0.8438 | 0.7064 | 0.7124 |
| SAGE Phase2 E2 | 0.4539 | 0.8442 | 0.7064 | 0.7131 |

### amazon

| Method | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| SAGE base | 0.7556 | 0.8934 | 0.8212 | 0.7119 |
| BWGNN base (BWGNN) | 0.8643 | 0.9747 | 0.9168 | 0.8815 |
| anchor_gate (BWGNN) | 0.8661 | 0.9752 | 0.9174 | 0.8832 |
| judge_strength_gate (BWGNN) | 0.8663 | 0.9751 | 0.9168 | 0.8831 |
| SAGE Phase2 E0 | 0.7415 | 0.8660 | 0.8766 | 0.8189 |
| SAGE Phase2 E1 | — | — | — | — |
| SAGE Phase2 E2 | — | — | — | — |

## 4. Diagnostics & safety checks

| Dataset | Exp | mean|Δ_rel| | mean α | accepted α | rejected/missing α (max) | gate entropy | dominance ρ |
|---|---|---|---|---|---|---|---|
| yelpchi | E0 | 1.3628 | 0.0000 | — | 0.0000 | 0.2067 | 0.7358 |
| yelpchi | E1 | 1.3591 | 0.0000 | 0.0000 | 0.0000 | 0.2337 | 0.7635 |
| yelpchi | E2 | 1.3571 | 0.0000 | 0.0351 | 0.0000 | 0.2396 | 0.7535 |
| amazon | E0 | 1.4239 | 0.0000 | — | 0.0000 | 0.0012 | 0.6003 |
| amazon | E1 | — | — | — | — | — | — |
| amazon | E2 | — | — | — | — | — | — |

Safety assertion: rejected/missing α (max) must be 0 (≤ 1e-6). Any non-zero value indicates a leak in the LLM gate.

## 5. Conclusion (auto-template — fill in once numbers land)

- **E0 (relation gate only)**: see Section 1 — does relation evidence improve SAGE base?
- **E1 (judge alignment)**: compare to E0 — does L_align improve gate alignment?
- **E2 (judge residual)**: compare to E1 — does small α residual buy further AUPRC?
- **Cross-model comparison**: compare SAGE E0/E1/E2 with BWGNN E0/E1/E2 — does the method generalize?

