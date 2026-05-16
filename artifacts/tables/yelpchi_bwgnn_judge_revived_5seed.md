# YelpChi BWGNN — Judge Revival Ablation (5 seeds)

**Base**: `fixed_v1_100ep` (BWGNN, paper-faithful, frozen)  
**Judge revival**: 120 packets random → 2000 packets base-uncertain + vLLM (Qwen3-4B)  
**Seeds**: [42, 123, 456, 789, 2026]

## Cell legend

| Cell | Variant | Formula | Note |
|---|---|---|---|
| **A0_base** | base only | `Phase 1 fixed_v1_100ep` | — |
| **L7_legacy** | full (120 pkt) | `z = b + Δ_rel + α·Δ_llm` | old judge, ~50 accepted/seed |
| **L7_revived** | full (2000 pkt) | `z = b + Δ_rel + α·Δ_llm` | vLLM judge, ~800 accepted/seed |
| **A1_revived** | rel-only | `z = b + Δ_rel` | judge off |
| **A2_revived** | judge-only | `z = b + α·Δ_llm` | Δ_rel=0 |

## Headline metrics (mean ± std)

| Cell | AUROC | AUPRC | Macro-F1 | G-Means |
|---|---:|---:|---:|---:|
| A0_base | 0.8276 ± 0.0095 | 0.5034 ± 0.0155 | 0.6990 ± 0.0072 | 0.6753 ± 0.0122 |
| L7_legacy ★(old) | 0.8807 ± 0.0036 | 0.6076 ± 0.0092 | 0.7480 ± 0.0048 | 0.7239 ± 0.0191 |
| L7_revived 🆕 | 0.8808 ± 0.0036 | 0.6077 ± 0.0102 | 0.7480 ± 0.0068 | 0.7301 ± 0.0237 |
| A1_revived | 0.8815 ± 0.0034 | 0.6088 ± 0.0088 | 0.7488 ± 0.0050 | 0.7386 ± 0.0170 |
| A2_revived | 0.8280 ± 0.0095 | 0.5037 ± 0.0155 | 0.6983 ± 0.0080 | 0.6781 ± 0.0210 |

## Critical paired t-tests — AUPRC

| Comparison | Δ (paired t-test) | Interpretation |
|---|---|---|
| L7_revived vs L7_legacy | Δ=+0.0000 (t=+0.03, p=0.976) n.s. | Did 2000-pkt judge help? |
| L7_revived vs A1_revived | Δ=-0.0012 (t=-0.86, p=0.436) n.s. | **Is judge alive in full?** |
| A2_revived vs A0_base    | Δ=+0.0003 (t=+7.10, p=0.00208) ** | **Judge-only beat base?** |
| L7_revived vs A0_base    | Δ=+0.1043 (t=+30.90, p=6.53e-06) *** | Total CoVER lift |

## Judge activation diagnostics (L7_revived vs L7_legacy)

| Metric | L7_legacy | L7_revived | Δ |
|---|---:|---:|---:|
| |Δ_rel| (relation residual) | 1.09638 ± 0.05615 | 1.06278 ± 0.04533 | -0.03359 |
| |α·Δ_llm| full-sample | 0.00021 ± 0.00004 | 0.00255 ± 0.00008 | +0.00234 |
| |α·Δ_llm| on accepted | 0.20935 ± 0.01150 | 0.20883 ± 0.00578 | -0.00052 |
| α on accepted | 0.29672 ± 0.00283 | 0.29786 ± 0.00341 | +0.00114 |
| total intervention |z-b| | 1.09651 ± 0.05615 | 1.06449 ± 0.04531 | -0.03202 |

## Per-seed CSV: `artifacts/tables/yelpchi_bwgnn_judge_revived_per_seed.csv`

---
**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` p≥0.05