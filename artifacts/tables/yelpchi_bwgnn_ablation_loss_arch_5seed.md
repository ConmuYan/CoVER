# YelpChi BWGNN — Phase 2 Loss × Architecture Ablation (5 seeds)

**Base**: `fixed_v1_100ep` (BWGNN, paper-faithful, frozen)  
**Reference cell**: `L0` (cls-only, *retained* canonical)  
**Seeds**: [42, 123, 456, 789, 2026]  
**Statistical test**: paired t-test (per-seed Δ vs L0), 5 seeds → df=4  
**Falsification ledger**: every L_i / L7 / judge cell fails the 5-seed paired-t bar vs L0; see AGENTS.md §§7.1, 9.

## Cell legend (historical 4-term schema)

| Cell | Variant | Historical formula | HP override | Note |
|---|---|---|---|---|
| **A0** | base only | `z = b` | — | reuse fixed_v1_100ep (Phase 1) |
| **L0** | cls only | `z = b + Δ_rel + α·Δ_llm` | λ_int=0, λ_sp=0, λ_al=0 | ★ canonical reference (= retained loss) |
| **L1** | +int | `z = b + Δ_rel + α·Δ_llm` | λ_int=3e-3, λ_sp=0, λ_al=0 | falsified vs L0 (p=0.81) |
| **L2** | +sparse | `z = b + Δ_rel + α·Δ_llm` | λ_int=0, λ_sp=1e-3, λ_al=0 | falsified vs L0 (p=0.061) |
| **L3** | +align | `z = b + Δ_rel + α·Δ_llm` | λ_int=0, λ_sp=0, λ_al=3e-2 | falsified vs L0 (p=0.32) |
| **L4** | +int +sp | `z = b + Δ_rel + α·Δ_llm` | λ_int=3e-3, λ_sp=1e-3, λ_al=0 | falsified vs L0 (p=0.36) |
| **L5** | +int +al | `z = b + Δ_rel + α·Δ_llm` | λ_int=3e-3, λ_sp=0, λ_al=3e-2 | falsified vs L0 (p=0.51) |
| **L6** | +sp +al | `z = b + Δ_rel + α·Δ_llm` | λ_int=0, λ_sp=1e-3, λ_al=3e-2 | falsified vs L0 (p=0.54) |
| **L7** | full 4-term | `z = b + Δ_rel + α·Δ_llm` | λ_int=3e-3, λ_sp=1e-3, λ_al=3e-2 | historical champion (≡ L0 at p=0.991) |
| **A1** | rel-only | `z = b + Δ_rel` | use_judge=0, α_max=0, λ_al=0 | judge-off arch switch |
| **A2** | judge-only | `z = b + α·Δ_llm` | Δ_rel_max=0 | rel-off arch switch |

## Results (mean ± std, 5 seeds)

| Cell | AUROC | AUPRC | Macro-F1 | G-Means |
|---|---:|---:|---:|---:|
| A0 | 0.8276 ± 0.0095 | 0.5034 ± 0.0155 | 0.6990 ± 0.0072 | 0.6753 ± 0.0122 |
| L0 **★** | 0.8822 ± 0.0049 | 0.6076 ± 0.0089 | 0.7496 ± 0.0081 | 0.7355 ± 0.0260 |
| L1 | 0.8819 ± 0.0038 | 0.6082 ± 0.0076 | 0.7486 ± 0.0052 | 0.7298 ± 0.0155 |
| L2 | 0.8817 ± 0.0037 | 0.6100 ± 0.0095 | 0.7465 ± 0.0090 | 0.7240 ± 0.0308 |
| L3 | 0.8803 ± 0.0039 | 0.6072 ± 0.0097 | 0.7476 ± 0.0046 | 0.7296 ± 0.0081 |
| L4 | 0.8812 ± 0.0024 | 0.6091 ± 0.0096 | 0.7486 ± 0.0053 | 0.7339 ± 0.0155 |
| L5 | 0.8807 ± 0.0036 | 0.6079 ± 0.0099 | 0.7477 ± 0.0043 | 0.7321 ± 0.0211 |
| L6 | 0.8802 ± 0.0035 | 0.6079 ± 0.0099 | 0.7484 ± 0.0060 | 0.7354 ± 0.0192 |
| L7 | 0.8808 ± 0.0039 | 0.6076 ± 0.0094 | 0.7478 ± 0.0051 | 0.7245 ± 0.0154 |
| A1 | 0.8814 ± 0.0035 | 0.6090 ± 0.0087 | 0.7478 ± 0.0046 | 0.7243 ± 0.0220 |
| A2 | 0.8276 ± 0.0095 | 0.5034 ± 0.0155 | 0.6983 ± 0.0079 | 0.6782 ± 0.0210 |

## Paired t-test vs L0 (cls-only canonical) — AUPRC

| Cell | Variant | AUPRC paired Δ vs L0 |
|---|---|---|
| A0 | base only | Δ=-0.1042 (t=-31.10, p=6.37e-06) *** |
| L0 | cls only | reference |
| L1 | +int | Δ=+0.0006 (t=+0.39, p=0.718) n.s. |
| L2 | +sparse | Δ=+0.0025 (t=+1.02, p=0.364) n.s. |
| L3 | +align | Δ=-0.0004 (t=-0.19, p=0.859) n.s. |
| L4 | +int +sp | Δ=+0.0015 (t=+0.51, p=0.637) n.s. |
| L5 | +int +al | Δ=+0.0003 (t=+0.13, p=0.905) n.s. |
| L6 | +sp +al | Δ=+0.0003 (t=+0.12, p=0.909) n.s. |
| L7 | full 4-term | Δ=-0.0000 (t=-0.01, p=0.991) n.s. |
| A1 | rel-only | Δ=+0.0014 (t=+0.66, p=0.543) n.s. |
| A2 | judge-only | Δ=-0.1042 (t=-31.08, p=6.38e-06) *** |

## Per-seed CSV: `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_per_seed.csv`

---

**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` p≥0.05
