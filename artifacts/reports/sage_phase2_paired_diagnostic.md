# SAGE Phase2 Paired Diagnostic

Read-only diagnostic over saved per-seed JSON metrics + phase2_diagnostics.
No model re-inference. Source paths cited below per dataset/experiment.

## yelpchi

| Exp | n | Phase2 AUPRC | ΔAUPRC vs base (paired) | t-stat | ΔAUPRC vs anchor_gate (paired) | t-stat | mean\|Δ_rel\| | gate H | mean α | judge↔gate cos |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 5 | 0.4503 ± 0.0891 | 0.2258 ± 0.0851 | 5.93 | 0.0039 ± 0.0695 | 0.12 | 1.3628 ± 0.1617 | 0.2067 ± 0.1970 | 0.000000 ± 0.000000 | 0.9699 ± 0.0365 |
| E1 | 5 | 0.4535 ± 0.0888 | 0.2290 ± 0.0872 | 5.87 | 0.0071 ± 0.0702 | 0.22 | 1.3591 ± 0.1488 | 0.2337 ± 0.1598 | 0.000000 ± 0.000000 | 0.9787 ± 0.0239 |
| E2 | 5 | 0.4539 ± 0.0891 | 0.2293 ± 0.0875 | 5.86 | 0.0074 ± 0.0705 | 0.23 | 1.3571 ± 0.1467 | 0.2396 ± 0.1613 | 0.000031 ± 0.000016 | 0.9775 ± 0.0251 |

### Per-seed Δ AUPRC vs SAGE base
| Exp | seed 42 | seed 123 | seed 456 | seed 789 | seed 2026 |
|---|---|---|---|---|---|
| E0 | 0.2913 | 0.1069 | 0.1746 | 0.2429 | 0.3132 |
| E1 | 0.2941 | 0.1070 | 0.1741 | 0.2527 | 0.3168 |
| E2 | 0.2953 | 0.1071 | 0.1741 | 0.2526 | 0.3174 |

### Per-seed Δ AUPRC vs old anchor_gate (Stage3)
| Exp | seed 42 | seed 123 | seed 456 | seed 789 | seed 2026 |
|---|---|---|---|---|---|
| E0 | 0.0534 | 0.0600 | -0.1107 | -0.0086 | 0.0252 |
| E1 | 0.0562 | 0.0601 | -0.1112 | 0.0013 | 0.0288 |
| E2 | 0.0574 | 0.0602 | -0.1112 | 0.0012 | 0.0294 |

### Gate vs Judge relation distribution (RUR, RSR, RTR)
| Exp | gate distribution | judge distribution | cosine sim |
|---|---|---|---|
| E0 | RUR=0.837 ± 0.155, RSR=0.127 ± 0.132, RTR=0.036 ± 0.047 | RUR=1.000 ± 0.000, RSR=0.000 ± 0.000, RTR=0.000 ± 0.000 | 0.9699 ± 0.0365 |
| E1 | RUR=0.843 ± 0.124, RSR=0.109 ± 0.108, RTR=0.047 ± 0.041 | RUR=1.000 ± 0.000, RSR=0.000 ± 0.000, RTR=0.000 ± 0.000 | 0.9787 ± 0.0239 |
| E2 | RUR=0.840 ± 0.125, RSR=0.113 ± 0.111, RTR=0.048 ± 0.039 | RUR=1.000 ± 0.000, RSR=0.000 ± 0.000, RTR=0.000 ± 0.000 | 0.9775 ± 0.0251 |

### Safety check (LLM gate leak)
- **E0**: max_abs_alpha_llm_rejected (max across seeds) = 0.00000000 (must be < 1e-6 for safety pass)
- **E1**: max_abs_alpha_llm_rejected (max across seeds) = 0.00000000 (must be < 1e-6 for safety pass)
- **E2**: max_abs_alpha_llm_rejected (max across seeds) = 0.00000000 (must be < 1e-6 for safety pass)

## amazon

| Exp | n | Phase2 AUPRC | ΔAUPRC vs base (paired) | t-stat | ΔAUPRC vs anchor_gate (paired) | t-stat | mean\|Δ_rel\| | gate H | mean α | judge↔gate cos |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 1 | 0.7415 ± 0.0000 | 0.0000 ± 0.0000 | — | -0.0138 ± 0.0000 | — | 1.4239 ± 0.0000 | 0.0012 ± 0.0000 | 0.000000 ± 0.000000 | 0.0996 ± 0.0000 |
| E1 | 0 | — | — | — | — | — | — | — | — | — |
| E2 | 0 | — | — | — | — | — | — | — | — | — |

### Per-seed Δ AUPRC vs SAGE base
| Exp | seed 42 | seed 123 | seed 456 | seed 789 | seed 2026 |
|---|---|---|---|---|---|
| E0 | 0.0000 | — | — | — | — |
| E1 | — | — | — | — | — |
| E2 | — | — | — | — | — |

### Per-seed Δ AUPRC vs old anchor_gate (Stage3)
| Exp | seed 42 | seed 123 | seed 456 | seed 789 | seed 2026 |
|---|---|---|---|---|---|
| E0 | -0.0138 | — | — | — | — |
| E1 | — | — | — | — | — |
| E2 | — | — | — | — | — |

### Gate vs Judge relation distribution (UPU, USU, UVU)
| Exp | gate distribution | judge distribution | cosine sim |
|---|---|---|---|
| E0 | UPU=0.000 ± 0.000, USU=1.000 ± 0.000, UVU=0.000 ± 0.000 | UPU=0.000 ± 0.000, USU=0.091 ± 0.000, UVU=0.909 ± 0.000 | 0.0996 ± 0.0000 |
| E1 | UPU=—, USU=—, UVU=— | UPU=—, USU=—, UVU=— | — |
| E2 | UPU=—, USU=—, UVU=— | UPU=—, USU=—, UVU=— | — |

### Safety check (LLM gate leak)
- **E0**: max_abs_alpha_llm_rejected (max across seeds) = 0.00000000 (must be < 1e-6 for safety pass)
- **E1**: max_abs_alpha_llm_rejected (max across seeds) = — (must be < 1e-6 for safety pass)
- **E2**: max_abs_alpha_llm_rejected (max across seeds) = — (must be < 1e-6 for safety pass)

## Source paths (per dataset/experiment/seed)

- `artifacts/results/{ds}/sage/base/seed_X/stage1_metrics.json` — Phase1 base AUPRC
- `artifacts/results/{ds}/sage/cover_rel_anchor_gate_nollm/seed_X/stage3_metrics.json` — old anchor_gate AUPRC
- `artifacts/results/{ds}/sage/{run_name}/seed_X/stage3_metrics.json` — Phase2 reasoner AUPRC
- `artifacts/logs/{ds}/sage/{run_name}/seed_X/phase2_diagnostics.json` — diagnostics (gate, alpha, delta_rel)
- `artifacts/judge_packets/{ds}/sage/cover_rel_judge/seed_X/accepted_judge.jsonl` — judge key_relation distribution
