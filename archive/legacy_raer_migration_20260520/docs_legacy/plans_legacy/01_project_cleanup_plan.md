# Project Cleanup Plan

## Goal

Transform the current exploratory project into a clean, defensible research repository for Work 2 without losing evidence needed for the thesis.

The cleanup target is not minimal file count. The target is a clear separation between:

- canonical method implementation,
- canonical evidence artifacts,
- negative routes,
- historical/debug debris.

## Non-Negotiable Rules

- Do not delete any result table before it is mapped to a thesis claim or explicitly marked irrelevant.
- Do not overwrite checkpoints or metrics from completed runs.
- Do not rename result directories until dependent scripts are checked.
- Archive negative evidence rather than deleting it if it may help defend design choices.
- Preserve seed, split, dataset, and base-model information for every retained artifact.

## Keep As Canonical

### Method Code

| Path | Reason |
|---|---|
| `models/raer_teacher.py` | RAER teacher implementation |
| `evidence/lree.py` | LREE implementation |
| `models/cbr_flash_adapter.py` | CBR-Flash student implementation |
| `scripts/train_raer_teacher.py` | Main RAER teacher trainer for standard bases |
| `scripts/train_cbr_flash.py` | Student distillation and CBR trainer |
| `training/phase2_losses.py` | Current cls-only teacher loss contract |
| `training/priorfgnn_losses.py` | PriorF-GNN integration support |
| `models/priorfgnn.py` | PriorF-GNN port into current project |
| `scripts/train_raer_priorfgnn.py` | PriorF-GNN strong-base saturation experiment |

### Canonical Configs

| Path | Reason |
|---|---|
| `configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml` | RAER-HC canonical baseline |
| `configs/phase2_reasoner/ablation/idea2b_learned_extractor_*.yaml` | RAER-LREE configs |
| `configs/phase2_reasoner/priorfgnn_idea2b_yelpchi_404020.yaml` | Strong-base saturation check |
| `configs/yelpchi_priorfgnn_404020.yaml` | PriorF-GNN 40/20/40 base config |
| `configs/yelpchi_*.yaml`, `configs/amazon_*.yaml` | Standard base detector configs |

### Canonical Result Tables

| Path | Claim |
|---|---|
| `artifacts/tables/idea1_ablation_FINAL_7cell.md` | RAER architecture and evidence/gate value |
| `artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md` | LREE vs hand-crafted evidence |
| `artifacts/tables/idea2b_ablation_4base_5seed.md` | LREE internal ablation caveats |
| `artifacts/tables/g_opd_flash_8cell_5seed.md` | Student captures teacher and deployment value |
| `artifacts/tables/c3_4metric_8cell_5seed.md` | CBR cross-metric evidence |
| `artifacts/tables/c3_interp_a_ablation_yelpchi_bwgnn_sage_gcn_5seed.md` | CBR component ablation |
| `artifacts/tables/idea2d_2b_speed_benchmark.md` | Inference speed evidence |
| `artifacts/tables/paper_negative_routes.md` | Negative-route defense |

### Canonical PriorF-GNN Adaptation Artifacts

| Path | Reason |
|---|---|
| `artifacts/checkpoints/yelpchi/priorfgnn/h64_404020/seed_42/base.pt` | Strong PriorF-GNN base checkpoint |
| `artifacts/results/yelpchi/priorfgnn/h64_404020/seed_42/stage1_metrics.json` | PriorF-GNN base test metrics |
| `artifacts/logs/yelpchi/priorfgnn/priorfgnn_idea2b_404020_gpu/seed_42/phase2_summary.json` | RAER-FD on strong base |
| `artifacts/results/yelpchi/priorfgnn/priorfgnn_idea2b_404020_gpu/seed_42/stage3_metrics.json` | Strong-base saturation metrics |
| `artifacts/launch_logs/priorfgnn_idea2b_404020_gpu_seed42.log` | Launch and epoch log |

## Archive As Negative Evidence

Move these into a future `archive/negative_routes/` area after approval.

| Category | Examples | Why Archive |
|---|---|---|
| LLM judge route | judge fusion, `alpha_llm`, old judge packet routes | Falsified; useful to explain why final method is rel-only |
| Auxiliary losses | `L_intervention`, `L_sparse`, `L_align` variants | Retired; supports cls-only design |
| Active learning route | REL-AF / Idea-3 outputs | Underperformed; appendix only |
| Old prompt/LLM feature composites | judge packets, LEQA, PRTAE traces | Not aligned with final contribution |
| Failed or partial distillation variants | off-policy variants that clearly underperform | May support CBR motivation if summarized |

## Candidates For Deletion Or Compression

Only delete after a second pass confirms they are not referenced.

| Category | Check Before Action |
|---|---|
| stale TensorBoard logs | verify no unique metrics are only in event files |
| duplicate backup relation feature dirs | compare `rel_feature_meta.json` and `rel_stats.pt` checksum |
| debug checkpoints | confirm corresponding final run exists |
| empty logs | confirm no process still writing |
| old `.omc`/agent session debris | preserve only if needed for provenance |

## Rename Mapping For Paper Tables

| Current Label | Final Label |
|---|---|
| `idea1_canonical_clsonly` | `RAER-HC` |
| `idea2b_learned_extractor` | `RAER-LREE` |
| `g_opd_flash` | `Flash-RAER` |
| `det_mask_cbr` | `CBR-Flash-K1` |
| `cbr_best` | `CBR-Flash` |
| `priorfgnn_idea2b_404020_gpu` | `PriorF-GNN + RAER-LREE` |

## Cleanup Phases

### Phase 1: Inventory

- List all checkpoints, result JSONs, tables, logs, and configs.
- Assign each item to: canonical, negative evidence, duplicate, debug, unknown.
- Produce a machine-readable manifest before moving files.

### Phase 2: Evidence Lock

- For each thesis claim, identify the exact source artifact.
- Copy or link the canonical artifacts into a stable `artifacts/final_evidence/` area.
- Record seed counts, datasets, base models, and split protocol.

### Phase 3: Archive

- Move negative and historical routes into `archive/negative_routes/`.
- Move debug-only runs into `archive/debug_runs/`.
- Keep a README in each archive folder explaining why the route is not canonical.

### Phase 4: Delete Or Compress

- Delete only empty logs, temporary smoke outputs, and confirmed duplicates.
- Compress large non-canonical event logs if disk pressure matters.
- Never delete raw datasets, canonical checkpoints, or final result tables.

### Phase 5: Documentation

- Update a top-level method map.
- Update the result-to-claim table.
- Record the LREE caching policy.
- Record the PriorF-GNN saturation interpretation.

## Validation Checklist

- All canonical scripts still import.
- All canonical result tables still exist.
- Every final claim maps to one artifact.
- No final thesis table depends on archived paths without a note.
- PriorF-GNN 40/20/40 saturation result is preserved.
- Negative-route archive explains why each route was retired.

## Expected Outcome

After cleanup, the project should communicate:

1. What the final method is.
2. Which experiments support each claim.
3. Which routes were tried and falsified.
4. Why Work 2 is a relation-evidence residual correction framework rather than a collection of exploratory variants.
