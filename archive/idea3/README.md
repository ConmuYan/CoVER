# Archive: Idea 3 (LLM-driven Feature Design)

**Status**: ARCHIVED on 2026-05-19.
**Reason**: Decision to focus the TKDE 2026 submission on Idea 1 (Relation-Aware Evidence Reasoning) + Idea 2 (Learnable Relational Evidence Extractor + OPD-Flash Distill). Idea 3 created internal narrative tension (Base+LLM 0.666 ≥ Base+CoVER-REL 0.609 on single seed) that conflicts with the RAER framing.

**Do not import / extend / cite anything in this directory for the current paper.** It exists as historical record and may be revisited in a separate follow-up submission (KDD 2026 LLM-for-graphs workshop or similar).

## Contents

```
archive/idea3/
├── scripts/        — 9 Python / shell scripts (idea3_*, run_idea3_*, aggregate_idea3, idea3b_llm_rule_pilot)
├── tables/         — 6 markdown / json paired-t tables (CAAFE, OpenFE, AutoFE, multi-LLM, AL learning curves, full benchmark)
├── results/        — 2 result directories: idea3_scaling (multi-LLM raw outputs) and al (active-learning learning curves)
├── logs/           — 6 run logs (Qwen3 reruns, OpenFE full, AutoFE baselines, PLM rerun)
└── docs/           — empty placeholder for future Idea 3 design notes if revisited
```

## Notable bugs caught (preserved as audit evidence)

Idea 3 surfaced TWO independently fatal silent-failure bugs that have been fixed in the live codebase:

1. **`phase4_plm_adapter`** had 4 cascading bugs (hard-coded `selected_indices`, divergent loss, detached numpy weights, LR re-fit on fixed columns) making BERT and RoBERTa converge to identical test AUPRC at 17-decimal precision. Fixed in commit `13fb854` with REINFORCE-based candidate selection.
2. **`parse_llm_formulas`** silently mis-extracted prompt-echoed feature descriptions as formulas; `build_composite_features` silently substituted zero columns. Result: 4 base Qwen3 models colliding on AUPRC = 0.5002955616217414 = base-only. Fixed in commit `cc642c7` with sympy validation, strict-mode parser, and `fail_mode='raise'` default.

Both fixes are retained in the live codebase (the parser and `build_composite_features` were modified in-place; the post-mortem note in `res.md` §8.7 documents the history). The archived `idea3_multi_llm_scaling.py` here is the *final fixed version* (post-cc642c7) for reproducibility, NOT the buggy original.

## res.md is intentionally NOT modified

`res.md` retains all Idea 3 sections (§7 AL, §8 LLM feature design, §8.5.1 split-by-dataset, §8.7 multi-LLM scaling). These are kept as the honest experimental record and historical evidence. The paper draft for TKDE will explicitly exclude these sections — but the archive of `res.md` itself stays whole.

## How to reference (if needed)

If a reviewer asks about LLM-feature-design baselines, point to the archived data:

- CAAFE / PromptFE comparison: `archive/idea3/tables/idea3_caafe_comparison.{md,json}`
- OpenFE / gplearn-SR shootout: `archive/idea3/tables/idea3_openfe_comparison.{md,json}`
- Multi-LLM scaling (post-fix): `archive/idea3/results/idea3_scaling/quick_eval_yelpchi_bwgnn_seed42.json`
- REL-curriculum active learning (negative result): `archive/idea3/results/al/yelpchi/`

Cite as: *"In separate experiments not included in this paper, we explored LLM-driven composite feature design… see supplementary archive."*

## Restore protocol

To resurrect Idea 3 for a future submission:

```bash
git mv archive/idea3/scripts/idea3_*.py scripts/
git mv archive/idea3/scripts/run_idea3_*.sh scripts/
git mv archive/idea3/scripts/aggregate_idea3.py scripts/
mv archive/idea3/results/idea3_scaling artifacts/results/
mv archive/idea3/results/al artifacts/results/
mv archive/idea3/tables/* artifacts/tables/
mv archive/idea3/logs/* logs/
```

Re-add Idea 3 sections to AGENTS.md and notepad if needed.

---

*This README is the single source of truth for what is and isn't part of the current TKDE 2026 submission. New sessions should treat the archive as read-only context, not actionable code.*
