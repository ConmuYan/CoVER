# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->
Idea 3 三 track 全部完成（已 commit 726214f 之前 + 后续新产出未 commit）:
Track 1 CAAFE: Ours vs CAAFE +0.019 ★★, vs PromptFE +0.023 ★ (40 pairs df=39)
Track 2 Multi-LLM: 关键发现——只有 instruct 模型可设计公式，scaling law 不成立
Track 3 OpenFE+SR: Ours vs OpenFE +0.066 ★ ✨, vs GP +0.031 ★★, vs Random +0.027 ★, vs Systematic +0.041 ★★
仍需: (1) 更新 res.md 加 OpenFE 数据 (2) 跑 Codex 终审 (3) commit + push (4) 决定是否需要 PLM 修复
Idea 1+2 ✅ committed phase3/commit1-v2; Idea 3 on idea3/llm-case-retrieval

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

