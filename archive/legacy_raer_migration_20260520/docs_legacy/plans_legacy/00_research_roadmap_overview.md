# RAER-FD Research Roadmap Overview

## Purpose

This directory turns the current research narrative into executable planning documents for:

1. Cleaning the project into a defensible evidence chain.
2. Freezing Work 2's research content, goals, contributions, and experiment claims.
3. Writing a logically coherent graduation thesis from introduction to conclusion.

No code or artifact should be deleted only because it is messy. Cleanup must preserve every result needed to defend the final claims.

## Unified Research Story

The current project should be framed as a two-work research progression:

> Graph fraud detection is not solved by blindly stacking stronger GNNs. The key signal is relation-conditioned anomaly evidence. PriorF-GNN internalizes this signal inside message passing; RAER-FD externalizes it as a bounded, interpretable, and distillable residual correction interface.

### Work 1: PriorF-GNN

PriorF-GNN shows that relation-aware structural discrepancy can build a strong detector.

- HSD makes local structural discrepancy explicit.
- ASDA uses discrepancy to route low-frequency context and high-frequency residual information.
- SDCL focuses supervision on high-discrepancy regions.
- It provides the strongest existing base and proves the value of relation-aware discrepancy.

### Work 2: RAER-FD

RAER-FD lifts the same principle from detector construction to detector correction.

- It freezes a base detector.
- It extracts score-blind relation evidence.
- It learns a bounded residual correction.
- It distills the teacher into a lightweight adapter with CBR.
- It behaves conservatively on strong relation-aware bases such as PriorF-GNN.

## Canonical Naming

| Historical Name | Paper/Thesis Name | Meaning |
|---|---|---|
| Idea-1 canonical | RAER-HC | Hand-crafted relation evidence teacher |
| Idea-2B | RAER-LREE | Learnable relation evidence teacher |
| Idea-2C / distill | Flash-RAER | Lightweight student distilled from teacher |
| det_mask_cbr / cbr_best | CBR-Flash | Contract-Budgeted Residual student objective |
| PriorF-GNN 404020 adaptation | Strong-base saturation check | Conservative residual sanity check |

## Core Claim Stack

| Claim | Evidence Source | Status |
|---|---|---|
| Relation evidence residual improves weaker or incomplete base detectors. | `artifacts/tables/idea1_ablation_FINAL_7cell.md` | Supported |
| Learned relation evidence improves over hand-crafted evidence. | `artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md` | Supported |
| CBR improves residual distillation stability and student performance. | `artifacts/tables/c3_4metric_8cell_5seed.md`, `artifacts/tables/g_opd_flash_8cell_5seed.md` | Supported |
| On strong relation-aware PriorF-GNN, RAER-FD stays near identity instead of inventing artificial gains. | `artifacts/logs/yelpchi/priorfgnn/priorfgnn_idea2b_404020_gpu/seed_42/phase2_summary.json` | Supported as sanity check |

## Plan Documents

- `01_project_cleanup_plan.md`: what to keep, archive, rename, or delete.
- `02_work2_research_plan.md`: full Work 2 research formulation and contribution plan.
- `03_graduation_thesis_plan.md`: chapter-by-chapter writing plan.

## Execution Order

1. Review and approve the cleanup categories.
2. Freeze canonical names and contribution wording.
3. Move or archive negative/debris files only after approval.
4. Generate thesis outline and figures from the cleaned evidence chain.
5. Draft chapters in order: Introduction, Related Work, Method, Experiments, Conclusion.

## Current Important Caveat

The PriorF-GNN adaptation summary currently records the actual 40/20/40 run but its `data_protocol` text still says `70/10/20`. The loaded path and log confirm 40/20/40:

- `datasets/unified_404020/yelpchi/seed_42/data.pt`
- train/val/test = `18381 / 9191 / 18382`

This should be fixed in documentation or code comments before final thesis compilation, but no cleanup should rely on the stale text.
