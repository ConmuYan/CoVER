# Phase2 First-Round Conclusion

## 1. E0 reproduction check vs existing anchor_gate

| Dataset | E0 AUPRC (new reasoner) | anchor_gate baseline | Δ |
|---|---|---|---|
| yelpchi | 0.5669 | 0.4998 | 0.0671 |
| amazon | 0.8643 | 0.8661 | -0.0018 |

Pass criterion: |Δ| ≤ 0.005. If Δ < -0.005 the new reasoner regresses; investigate softmax-gate / loss differences.

## 2. Main result table (5-seed mean ± std)

### yelpchi

| Experiment | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| E0 | 0.5669 ± 0.0151 | 0.8665 ± 0.0049 | 0.7279 ± 0.0055 | 0.7110 ± 0.0118 |
| E1 | 0.5675 ± 0.0145 | 0.8672 ± 0.0046 | 0.7278 ± 0.0046 | 0.7030 ± 0.0160 |
| E2 | 0.5676 ± 0.0144 | 0.8673 ± 0.0045 | 0.7279 ± 0.0045 | 0.7095 ± 0.0174 |
| E3 | 0.5672 ± 0.0146 | 0.8668 ± 0.0045 | 0.7285 ± 0.0046 | 0.7070 ± 0.0122 |

### amazon

| Experiment | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| E0 | 0.8643 ± 0.0185 | 0.9748 ± 0.0065 | 0.9113 ± 0.0033 | 0.8885 ± 0.0175 |
| E1 | 0.8644 ± 0.0186 | 0.9748 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 |
| E2 | 0.8644 ± 0.0186 | 0.9748 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 |
| E3 | 0.8640 ± 0.0188 | 0.9746 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 |

## 3. Comparison with prior baselines

### yelpchi

| Method | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| BWGNN base | 0.4674 | 0.8076 | 0.6483 | 0.5105 |
| anchor_gate | 0.4998 | 0.8177 | 0.6598 | 0.5253 |
| judge_strength_gate | 0.5006 | 0.8180 | 0.6650 | 0.5359 |
| Phase2 E0 | 0.5669 | 0.8665 | 0.7279 | 0.7110 |
| Phase2 E1 | 0.5675 | 0.8672 | 0.7278 | 0.7030 |
| Phase2 E2 | 0.5676 | 0.8673 | 0.7279 | 0.7095 |
| Phase2 E3 | 0.5672 | 0.8668 | 0.7285 | 0.7070 |

### amazon

| Method | AUPRC | ROC-AUC | Macro-F1 | G-Means |
|---|---|---|---|---|
| BWGNN base | 0.8643 | 0.9747 | 0.9168 | 0.8815 |
| anchor_gate | 0.8661 | 0.9752 | 0.9174 | 0.8832 |
| judge_strength_gate | 0.8663 | 0.9751 | 0.9168 | 0.8831 |
| Phase2 E0 | 0.8643 | 0.9748 | 0.9113 | 0.8885 |
| Phase2 E1 | 0.8644 | 0.9748 | 0.9133 | 0.8888 |
| Phase2 E2 | 0.8644 | 0.9748 | 0.9133 | 0.8888 |
| Phase2 E3 | 0.8640 | 0.9746 | 0.9133 | 0.8888 |

## 4. Diagnostics & safety checks

| Dataset | Exp | mean|Δ_rel| | mean α | accepted α | rejected/missing α (max) | gate entropy | dominance ρ |
|---|---|---|---|---|---|---|---|
| yelpchi | E0 | 1.4913 | 0.0000 | — | 0.0000 | 0.0046 | 0.7210 |
| yelpchi | E1 | 1.4724 | 0.0000 | 0.0000 | 0.0000 | 0.2745 | 0.7049 |
| yelpchi | E2 | 1.4610 | 0.0001 | 0.0912 | 0.0000 | 0.2591 | 0.6969 |
| yelpchi | E3 | 1.5086 | 0.0001 | 0.0915 | 0.0000 | 0.2059 | 0.7700 |
| amazon | E0 | 1.1391 | 0.0000 | — | 0.0000 | 0.3708 | 0.5095 |
| amazon | E1 | 1.1481 | 0.0000 | 0.0000 | 0.0000 | 0.5766 | 0.4077 |
| amazon | E2 | 1.1481 | 0.0000 | 0.0056 | 0.0000 | 0.5766 | 0.4077 |
| amazon | E3 | 1.1530 | 0.0001 | 0.0120 | 0.0000 | 0.5945 | 0.4740 |

Safety assertion: rejected/missing α (max) must be 0 (≤ 1e-6). Any non-zero value indicates a leak in the LLM gate.

## 5. Conclusion (auto-template — fill in once numbers land)

- **E0 reproduction**: see Section 1.
- **E1 (judge alignment only)**: compare to E0 — does L_align improve gate alignment without changing the logit?
- **E2 (judge residual)**: compare to E1 — does small α residual buy further AUPRC?
- **E3 (no trust)**: compare to E2 — does removing L_trust cause regression / instability (look at mean|Δ_rel| inflation)?
- **YelpChi vs Amazon**: confirm RUR-dominant vs UVU-distributed gate behavior.

