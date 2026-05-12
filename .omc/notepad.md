# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->
Task 8.1 Complete: Qwen teacher adds no value over Rule teacher (Δ ROC-AUC: +0.0000, Δ F1: +0.0001). Both reasoners hurt F1 (Rule: -0.1483, Qwen: -0.1387). Issue is in reasoner architecture/training, not teacher quality. Next: investigate F1 regression root cause.

## Working Memory
<!-- Session notes. Auto-pruned after 7 days. -->
### 2026-05-12 18:16
## Task 8.1 Results Summary

### What Was Done
1. Fixed pytest.ini to exclude external/ and artifacts/
2. Ran YelpChi Qwen controlled experiment (3 seeds: 123, 456, 789, trace_size=32)
3. Aggregated results with fair same-seed comparisons
4. Generated evidence quality reports

### Key Findings
- Qwen vs Rule: Δ ROC-AUC: +0.0000, Δ F1: +0.0001 (nearly identical)
- Qwen vs BWGNN: Δ ROC-AUC: +0.0000, Δ F1: -0.1387 (F1 regression)
- Rule vs BWGNN: Δ ROC-AUC: +0.0000, Δ F1: -0.1388 (F1 regression)
- Both reasoners hurt F1 performance
- Low ERR diversity: 93/96 Qwen ERRs have structural_discrepancy
- F1=0 anomaly: Seed 123 in both reasoners

### Output Files
- artifacts/tables/controlled_experiments_metrics.md
- artifacts/tables/evidence_quality_summary.md
- artifacts/reports/yelpchi/bwgnn/method_comparison.md

### Next Steps
1. Investigate F1 regression root cause (lambda_evi, threshold, ERR diversity)
2. Run Amazon Qwen experiment
3. Increase trace_size for more diverse evidence
4. Ablation studies


## MANUAL
<!-- User content. Never auto-pruned. -->

