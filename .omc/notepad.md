# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->
TKDE 2026 投稿主线 = Idea 1 (CoVER-REL → RAER 范式) + Idea 2 (LREE 最优 + OPD-Flash Distill)。Idea 3 已 ARCHIVE 到 archive/idea3/ — 新 session 不要 touch idea3 任何 script/artifact。当前主要任务: 实施 OPD-Flash (docs/OPD_FLASH_DESIGN.md v1) — T1-T5 subtasks. 已修复 5 个 silent failure bugs (parser/builder/phase/max_new_tokens/PLM) 保留在 live codebase. 投稿 plan: docs/TKDE_2026_SUBMISSION_PLAN.md. 当前 branch: idea3/llm-case-retrieval (历史名, 不再代表当前 focus). 新 session 应当: 读 AGENTS.md §14 OPD-Flash + §15 Archive Note 先.

## Working Memory
<!-- Session notes. Auto-pruned after 7 days. -->

### 2026-05-19 17:00 — Idea 3 archive + OPD-Flash pivot
Decision: archive Idea 3 (LLM-driven feature design); refocus TKDE 2026 paper on
Idea 1 (RAER) + Idea 2 (LREE + OPD-Flash). All idea3 scripts/tables/results/logs
moved to archive/idea3/. AGENTS.md gains §14 (OPD-Flash design) + §15 (Archive
note). Live codebase keeps all 5 silent-failure fixes from the Idea 3 audit pass.
Next: implement OPD-Flash T1-T5 per docs/OPD_FLASH_DESIGN.md.


## MANUAL
<!-- User content. Never auto-pruned. -->

