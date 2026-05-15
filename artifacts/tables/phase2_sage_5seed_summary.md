# Phase2 SAGE 5-seed Summary

## yelpchi
| Experiment | run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean|Δrel| | mean α | gate H | reject α (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | `phase2_E0_relgate` | 5 | 0.4503 ± 0.0891 | 0.8429 ± 0.0249 | 0.7056 ± 0.0238 | 0.7103 ± 0.0372 | 1.3628 | 0.0000 | 0.2067 | 0.00 |
| E1 | `phase2_E1_judge_align` | 5 | 0.4535 ± 0.0888 | 0.8438 ± 0.0254 | 0.7064 ± 0.0246 | 0.7124 ± 0.0403 | 1.3591 | 0.0000 | 0.2337 | 0.00 |
| E2 | `phase2_E2_judge_residual` | 5 | 0.4539 ± 0.0891 | 0.8442 ± 0.0257 | 0.7064 ± 0.0247 | 0.7131 ± 0.0413 | 1.3571 | 0.0000 | 0.2396 | 0.00 |

### Baselines
- **SAGE base**: AUPRC 0.2246 | ROC-AUC 0.5995 | Macro-F1 0.4848 | G-Means 0.0746
- **BWGNN base (BWGNN)**: AUPRC 0.4674 | ROC-AUC 0.8076 | Macro-F1 0.6483 | G-Means 0.5105
- **anchor_gate (BWGNN)**: AUPRC 0.4998 | ROC-AUC 0.8177 | Macro-F1 0.6598 | G-Means 0.5253
- **judge_strength_gate (BWGNN)**: AUPRC 0.5006 | ROC-AUC 0.8180 | Macro-F1 0.6650 | G-Means 0.5359

## amazon
| Experiment | run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean|Δrel| | mean α | gate H | reject α (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | `phase2_E0_relgate` | 1 | 0.7415 ± 0.0000 | 0.8660 ± 0.0000 | 0.8766 ± 0.0000 | 0.8189 ± 0.0000 | 1.4239 | 0.0000 | 0.0012 | 0.00 |
| E1 | `phase2_E1_judge_align` | 0 | — ± — | — ± — | — ± — | — ± — | — | — | — | — |
| E2 | `phase2_E2_judge_residual` | 0 | — ± — | — ± — | — ± — | — ± — | — | — | — | — |

### Baselines
- **SAGE base**: AUPRC 0.7556 | ROC-AUC 0.8934 | Macro-F1 0.8212 | G-Means 0.7119
- **BWGNN base (BWGNN)**: AUPRC 0.8643 | ROC-AUC 0.9747 | Macro-F1 0.9168 | G-Means 0.8815
- **anchor_gate (BWGNN)**: AUPRC 0.8661 | ROC-AUC 0.9752 | Macro-F1 0.9174 | G-Means 0.8832
- **judge_strength_gate (BWGNN)**: AUPRC 0.8663 | ROC-AUC 0.9751 | Macro-F1 0.9168 | G-Means 0.8831

