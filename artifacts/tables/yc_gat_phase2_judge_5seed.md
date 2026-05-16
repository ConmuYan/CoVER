# YelpChi GAT (heads=1) — 5-seed Phase2 Results

## Summary Table (mean +/- std over seeds 42, 123, 456, 789, 2026)

| Stage | ROC-AUC | AUPRC | Macro-F1 |
|-------|---------|-------|----------|
| Base (Stage A) | 0.4993+/-0.0128 | 0.1453+/-0.0061 | 0.4608+/-0.0000 |
| Gate (E0) | 0.5449+/-0.0428 | 0.1598+/-0.0134 | 0.4608+/-0.0000 |
| Judge (E2) | 0.8134+/-0.0274 | 0.3630+/-0.0546 | 0.6767+/-0.0187 |

## Deltas

| Comparison | AUPRC | ROC-AUC | Macro-F1 |
|------------|-------|---------|----------|
| Judge vs Base | +0.2176 | +0.3141 | +0.2158 |
| Judge vs Gate | +0.2032 | +0.2685 | +0.2158 |
| Gate vs Base  | +0.0145 | +0.0456 | +0.0000 |

## Per-seed Breakdown

### Base (Stage A)
| Seed | ROC-AUC | AUPRC | Macro-F1 |
|------|---------|-------|----------|
| 42   | 0.5152  | 0.1551 | 0.4608  |
| 123  | 0.5045  | 0.1438 | 0.4608  |
| 456  | 0.4769  | 0.1370 | 0.4608  |
| 789  | 0.4955  | 0.1421 | 0.4608  |
| 2026 | 0.5043  | 0.1487 | 0.4608  |

### Gate (E0 relgate)
| Seed | ROC-AUC | AUPRC | Macro-F1 |
|------|---------|-------|----------|
| 42   | 0.5290  | 0.1597 | 0.4608  |
| 123  | 0.6266  | 0.1843 | 0.4608  |
| 456  | 0.5007  | 0.1445 | 0.4608  |
| 789  | 0.5307  | 0.1518 | 0.4608  |
| 2026 | 0.5376  | 0.1588 | 0.4608  |

### Judge (E2 judge_residual)
| Seed | ROC-AUC | AUPRC | Macro-F1 |
|------|---------|-------|----------|
| 42   | 0.8094  | 0.3318 | 0.6750  |
| 123  | 0.7987  | 0.3102 | 0.6640  |
| 456  | 0.7741  | 0.3303 | 0.6555  |
| 789  | 0.8300  | 0.3807 | 0.6785  |
| 2026 | 0.8546  | 0.4619 | 0.7103  |

## LLM Judge Acceptance Rates
| Seed | Accepted | Total | Rate |
|------|----------|-------|------|
| 42   | 110      | 120   | 91.7% |
| 123  | 113      | 120   | 94.2% |
| 456  | 112      | 120   | 93.3% |
| 789  | 110      | 120   | 91.7% |
| 2026 | 114      | 120   | 95.0% |

## Notes
- GAT uses heads=1 due to attention saturation on YelpChi's 7.7M-edge graph
- Base model is near-random (ROC ~0.50, all-negative predictions)
- Phase2 Judge rescues from near-random to ROC=0.81, validating model-agnostic claim
- torch.use_deterministic_algorithms disabled for GAT (scatter_add_ 3x memory overhead)
