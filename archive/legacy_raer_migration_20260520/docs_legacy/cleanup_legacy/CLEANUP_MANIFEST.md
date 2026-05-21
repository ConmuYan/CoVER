# Cleanup Manifest

Date: 2026-05-20

## Final Narrative Names

| Final name | Role | Canonical code |
|---|---|---|
| RAER teacher | Two-stage relation-aware evidence reasoning teacher | `models/raer_teacher.py`, `scripts/train_raer_teacher.py` |
| LREE | Learnable relation evidence extractor | `evidence/lree.py` |
| CBR-Flash | Contract-budgeted residual student/distillation stage | `models/cbr_flash_adapter.py`, `scripts/train_cbr_flash.py` |
| RAER on PriorF-GNN | Strong-base saturation check | `scripts/train_raer_priorfgnn.py` |

## Archived Negative Or Debris Routes

These files were moved, not deleted.

| Old path | New path | Reason |
|---|---|---|
| `scripts/al_loop.py` | `archive/negative_routes/active_learning/al_loop.py` | Active-learning route is not part of the final Work 2 contribution. |
| `artifacts/logs/idea3_al_yelpchi_bwgnn` | `archive/negative_routes/active_learning/idea3_al_yelpchi_bwgnn` | Negative/side route evidence. |
| `artifacts/logs/idea3_al_full` | `archive/negative_routes/active_learning/idea3_al_full` | Negative/side route evidence. |
| `artifacts/logs/yelpchi/bwgnn/idea3b_llm_aug` | `archive/negative_routes/llm_aug/idea3b_llm_aug` | LLM-augmentation route is not in the final method. |
| `artifacts/relation_features/yelpchi/bwgnn_llm_aug` | `archive/negative_routes/llm_aug/yelpchi_bwgnn_llm_aug_features` | Feature cache tied to removed LLM augmentation. |
| `artifacts/relation_features/amazon/gat.backup_1779002705` | `archive/debris/relation_features/amazon_gat_backup_1779002705` | Duplicate backup cache. |
| `artifacts/relation_features/amazon/gcn.backup_1779002705` | `archive/debris/relation_features/amazon_gcn_backup_1779002705` | Duplicate backup cache. |
| `artifacts/relation_features/amazon/sage.backup_1779002705` | `archive/debris/relation_features/amazon_sage_backup_1779002705` | Duplicate backup cache. |

## Canonical Renames

| Old exploratory path | Canonical path | Compatibility |
|---|---|---|
| `models/cover_rel_reasoner.py` | `models/raer_teacher.py` | Old path now re-exports `CoVERRelReasoner` and `RAERTeacher`. |
| `evidence/learned_extractor.py` | `evidence/lree.py` | Old path now re-exports `LREE` and builder utilities. |
| `models/flash_adapter.py` | `models/cbr_flash_adapter.py` | Old path now re-exports `CBRFlashAdapter`, `FlashAdapter`, and `RelDistillAdapter`. |
| `scripts/train_phase2_reasoner.py` | `scripts/train_raer_teacher.py` | Old script remains as an executable Python wrapper. |
| `scripts/train_phase2_priorfgnn.py` | `scripts/train_raer_priorfgnn.py` | Old script remains as an executable Python wrapper. |
| `scripts/train_g_opd_flash.py` | `scripts/train_cbr_flash.py` | Old script remains as an executable Python wrapper. |
| `scripts/aggregate_g_opd_flash.py` | `scripts/aggregate_cbr_flash.py` | Old script remains as an executable Python wrapper. |
| `scripts/aggregate_g_opd_flash_4metric.py` | `scripts/aggregate_cbr_flash_4metric.py` | Old script remains as an executable Python wrapper. |
| `scripts/aggregate_g_opd_flash_deployshift.py` | `scripts/aggregate_cbr_flash_deployshift.py` | Old script remains as an executable Python wrapper. |
| `scripts/run_g_opd_flash_8cell_5seed.sh` | `scripts/run_cbr_flash_8cell_5seed.sh` | Old shell script remains as a wrapper. |
| `scripts/run_g_opd_flash_deployshift.sh` | `scripts/run_cbr_flash_deployshift.sh` | Old shell script remains as a wrapper. |
| `tests/test_opd_flash_contracts.py` | `tests/test_cbr_flash_contracts.py` | Test name now matches CBR-Flash. |

## Intentionally Preserved Historical Names

Some result and config names still contain `idea1`, `idea2b`, or `g_opd_flash`.
They are retained because they are already baked into finished metrics,
checkpoint paths, paper tables, and aggregation scripts.

| Historical name | Paper-facing label |
|---|---|
| `idea1_canonical_clsonly` | RAER-HC |
| `idea2b_learned_extractor` | RAER-LREE |
| `g_opd_flash_det_mask_cbr` | CBR-Flash |
| `priorfgnn_idea2b_404020_gpu` | RAER-LREE on PriorF-GNN strong base |

Future runs should prefer the canonical script/module names above, while
published result paths can keep historical names for provenance.
