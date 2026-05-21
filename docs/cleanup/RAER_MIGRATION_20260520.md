# RAER-FD Migration Manifest

Date: 2026-05-20

This migration resets the active project surface around the final RAER-FD
narrative. Historical code, configs, logs, result artifacts, and planning notes
were archived rather than deleted.

## Archive Root

```text
archive/legacy_raer_migration_20260520/
```

## Active Naming

| Role | Active name |
|---|---|
| Project | RAER-FD |
| Hand-crafted evidence teacher | RAER-HC |
| Learnable evidence teacher | RAER-LREE |
| Learnable evidence extractor | LREE |
| Distilled student | CBR-Flash |
| Strong-base check | PriorF-GNN RAER-LREE saturation |

## Active Paths

| Component | Path |
|---|---|
| Base trainer | `scripts/train_base_detector.py` |
| RAER teacher trainer | `scripts/train_raer_teacher.py` |
| PriorF-GNN RAER trainer | `scripts/train_raer_priorfgnn.py` |
| CBR-Flash trainer | `scripts/train_cbr_flash.py` |
| RAER teacher | `models/raer_teacher.py` |
| LREE | `evidence/lree.py` |
| CBR-Flash student | `models/cbr_flash_adapter.py` |
| Configs | `configs/raer_fd/` |
| Result aggregation | `scripts/aggregate_cbr_flash.py` |

## Active Artifact Layout

```text
artifacts/checkpoints/{dataset}/{base}/base/seed_{seed}/base.pt
artifacts/checkpoints/{dataset}/{base}/raer_hc/seed_{seed}/raer_teacher.pt
artifacts/checkpoints/{dataset}/{base}/raer_lree/seed_{seed}/raer_teacher.pt
artifacts/checkpoints/{dataset}/{base}/raer_lree/seed_{seed}/lree.pt
artifacts/checkpoints/{dataset}/{base}/cbr_flash/seed_{seed}/cbr_flash_student.pt

artifacts/results/{dataset}/{base}/{run}/seed_{seed}/test_metrics.json
artifacts/results/{dataset}/{base}/raer_lree/seed_{seed}/raer_summary.json
artifacts/results/{dataset}/{base}/cbr_flash/seed_{seed}/cbr_flash_summary.json
```

## Archive Contents

| Archived item | Reason |
|---|---|
| `configs_legacy/` | Historical experiment configs retained for provenance. |
| `artifacts_legacy/` | Historical checkpoints, logs, tables, and results retained for provenance. |
| `logs_legacy/` | Historical launch logs retained for audit. |
| `scripts_legacy/` | Earlier exploratory trainers, aggregators, launchers, and wrappers. |
| `modules_legacy/` | Earlier module names replaced by canonical RAER-FD modules. |
| `docs_legacy/` | Earlier planning and cleanup notes with historical naming. |
| legacy test files | Tests tied to retired routes or retired public APIs. |

## Retained Rule

Archived material should not be imported by active code. Use it only to trace
old results back to paper claims or to inspect negative evidence.
