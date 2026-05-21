# RAER-FD Rerun Plan

This plan describes the clean rerun sequence after the RAER-FD naming reset.

## Goal

Produce a coherent Work 2 result set around three paper-facing claims:

1. RAER teacher learns bounded relation-evidence residual corrections over
   frozen graph fraud detectors.
2. LREE supplies score-blind learnable relation evidence and improves the
   teacher interface over fixed evidence where the base has remaining error.
3. CBR-Flash compresses the teacher correction behavior into a lightweight
   student while preserving the residual contract.

## Rerun Order

Active configs cover `yelpchi`, `amazon`, `yelpnyc`, `yelpzip`, `tfinance`,
and `tsocial`.

The default batch scripts run the compact full-graph set only:
`yelpchi` and `amazon`. Use explicit single-run commands, and first read
`docs/plans/RAER_FD_DATASET_SCALING_PLAN.md`, for `yelpnyc`, `yelpzip`,
`tfinance`, and `tsocial`.

### 1. Base Detectors

```bash
scripts/run_compact_fullgraph_base_detectors_5seed.sh 0
```

Expected checkpoint:

```text
artifacts/checkpoints/{dataset}/{base}/base/seed_{seed}/base.pt
```

Expected metric file:

```text
artifacts/results/{dataset}/{base}/base/seed_{seed}/base_metrics.json
```

### 2. Relation Features

Build hand-crafted relation features after base checkpoints exist:

```bash
scripts/run_compact_fullgraph_relation_features_5seed.sh
```

Expected feature cache:

```text
artifacts/relation_features/{dataset}/{base}/seed_{seed}/all/rel_stats.pt
```

### 3. RAER Teachers

RAER-HC:

```bash
scripts/run_compact_fullgraph_raer_teachers_5seed.sh raer_hc 0
```

RAER-LREE:

```bash
scripts/run_compact_fullgraph_raer_teachers_5seed.sh raer_lree 0
```

Expected checkpoints:

```text
artifacts/checkpoints/{dataset}/{base}/raer_hc/seed_{seed}/raer_teacher.pt
artifacts/checkpoints/{dataset}/{base}/raer_lree/seed_{seed}/raer_teacher.pt
artifacts/checkpoints/{dataset}/{base}/raer_lree/seed_{seed}/lree.pt
```

### 4. CBR-Flash

```bash
scripts/run_compact_fullgraph_cbr_flash_5seed.sh 0
```

Expected checkpoint:

```text
artifacts/checkpoints/{dataset}/{base}/cbr_flash/seed_{seed}/cbr_flash_student.pt
```

### 5. Strong-Base Saturation

Use `configs/raer_fd/strong_base/priorfgnn_raer_lree_yelpchi_404020.yaml` with
`scripts/train_raer_priorfgnn.py` after the PriorF-GNN base checkpoint exists.
This result supports the saturation sanity-check claim rather than an
improvement claim.

## Aggregation

```bash
/data1/mq/conda_envs/gread-core/bin/python scripts/aggregate_cbr_flash.py
```

Output:

```text
artifacts/tables/cbr_flash_compact_fullgraph_5seed.{csv,json,md}
```

## Naming Rules

- Use `base`, `raer_hc`, `raer_lree`, `cbr_flash`, and
  `strong_base_priorfgnn_raer_lree_404020` as active run names.
- Use `test_metrics.json` for the primary test result of every active run.
- Use `raer_summary.json` and `cbr_flash_summary.json` for method-specific
  summaries.
- Keep historical materials inside `archive/legacy_raer_migration_20260520/`.
