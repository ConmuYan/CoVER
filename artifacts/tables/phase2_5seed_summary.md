# Phase2 5-seed Summary

## yelpchi
| Experiment | run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean|Δrel| | mean α | gate H | reject α (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | `phase2_E0_relgate` | 5 | 0.5669 ± 0.0151 | 0.8665 ± 0.0049 | 0.7279 ± 0.0055 | 0.7110 ± 0.0118 | 1.4913 | 0.0000 | 0.0046 | 0.00 |
| E1 | `phase2_E1_judge_align` | 5 | 0.5675 ± 0.0145 | 0.8672 ± 0.0046 | 0.7278 ± 0.0046 | 0.7030 ± 0.0160 | 1.4724 | 0.0000 | 0.2745 | 0.00 |
| E2 | `phase2_E2_judge_residual` | 5 | 0.5676 ± 0.0144 | 0.8673 ± 0.0045 | 0.7279 ± 0.0045 | 0.7095 ± 0.0174 | 1.4610 | 0.0001 | 0.2591 | 0.00 |
| E3 | `phase2_E3_no_trust` | 5 | 0.5672 ± 0.0146 | 0.8668 ± 0.0045 | 0.7285 ± 0.0046 | 0.7070 ± 0.0122 | 1.5086 | 0.0001 | 0.2059 | 0.00 |

### Baselines
- **BWGNN base**: AUPRC 0.4674 | ROC-AUC 0.8076 | Macro-F1 0.6483 | G-Means 0.5105
- **anchor_gate**: AUPRC 0.4998 | ROC-AUC 0.8177 | Macro-F1 0.6598 | G-Means 0.5253
- **judge_strength_gate**: AUPRC 0.5006 | ROC-AUC 0.8180 | Macro-F1 0.6650 | G-Means 0.5359

## amazon
| Experiment | run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean|Δrel| | mean α | gate H | reject α (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | `phase2_E0_relgate` | 5 | 0.8643 ± 0.0185 | 0.9748 ± 0.0065 | 0.9113 ± 0.0033 | 0.8885 ± 0.0175 | 1.1391 | 0.0000 | 0.3708 | 0.00 |
| E1 | `phase2_E1_judge_align` | 5 | 0.8644 ± 0.0186 | 0.9748 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 | 1.1481 | 0.0000 | 0.5766 | 0.00 |
| E2 | `phase2_E2_judge_residual` | 5 | 0.8644 ± 0.0186 | 0.9748 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 | 1.1481 | 0.0000 | 0.5766 | 0.00 |
| E3 | `phase2_E3_no_trust` | 5 | 0.8640 ± 0.0188 | 0.9746 ± 0.0065 | 0.9133 ± 0.0033 | 0.8888 ± 0.0156 | 1.1530 | 0.0001 | 0.5945 | 0.00 |

### Baselines
- **BWGNN base**: AUPRC 0.8643 | ROC-AUC 0.9747 | Macro-F1 0.9168 | G-Means 0.8815
- **anchor_gate**: AUPRC 0.8661 | ROC-AUC 0.9752 | Macro-F1 0.9174 | G-Means 0.8832
- **judge_strength_gate**: AUPRC 0.8663 | ROC-AUC 0.9751 | Macro-F1 0.9168 | G-Means 0.8831

