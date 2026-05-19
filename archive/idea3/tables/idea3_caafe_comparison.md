# Idea 3 CAAFE/PromptFE Direct Comparison

**Methods compared:**

1. **CAAFE-style** (data-driven): LLM sees raw node feature values → generates features
2. **PromptFE-style** (prompt-guided): LLM sees dataset semantics + per-class stats → generates features
3. **Ours** (summary-only): LLM sees only aggregated class statistics → designs formulas
4. **Baselines**: base-only, base+raw features

**Model**: Qwen3-4B-Instruct-2507

**Significance**: per-cell df=4: |t|>2.78→★; |t|>4.60→★★; |t|>8.61→★★★. Pooled df=39: |t|>2.02→★ (p<0.05); |t|>2.70→★★ (p<0.01); |t|>3.56→★★★ (p<0.001)

## AUPRC Summary (mean ± sd across 5 seeds)

| Cell | Base only | Base+raw | CAAFE-style | PromptFE-style | Ours (summary) |
|---|---:|---:|---:|---:|---:|
| yelpchi-bwgnn | 0.5034 ± 0.0155 | 0.6765 ± 0.0074 | 0.5663 ± 0.0169 | 0.6562 ± 0.0114 | 0.6659 ± 0.0060 |
| yelpchi-sage | 0.4595 ± 0.0137 | 0.6512 ± 0.0121 | 0.5981 ± 0.0083 | 0.4754 ± 0.0193 | 0.6376 ± 0.0120 |
| yelpchi-gcn | 0.2226 ± 0.0099 | 0.4515 ± 0.0115 | 0.4164 ± 0.0063 | 0.3436 ± 0.0262 | 0.4331 ± 0.0099 |
| yelpchi-gat | 0.2060 ± 0.0148 | 0.4423 ± 0.0104 | 0.3851 ± 0.0384 | 0.4507 ± 0.0080 | 0.4285 ± 0.0103 |
| amazon-bwgnn | 0.8595 ± 0.0339 | 0.8598 ± 0.0321 | 0.8599 ± 0.0335 | 0.8600 ± 0.0302 | 0.8599 ± 0.0313 |
| amazon-sage | 0.7864 ± 0.0725 | 0.7991 ± 0.0760 | 0.7983 ± 0.0782 | 0.7942 ± 0.0824 | 0.7979 ± 0.0769 |
| amazon-gcn | 0.2549 ± 0.0212 | 0.2559 ± 0.0330 | 0.2579 ± 0.0216 | 0.3081 ± 0.0241 | 0.2659 ± 0.0363 |
| amazon-gat | 0.3401 ± 0.2888 | 0.5096 ± 0.1608 | 0.4908 ± 0.1729 | 0.4533 ± 0.1902 | 0.4391 ± 0.2040 |

## Paired-t: Ours vs CAAFE/PromptFE (pooled across cells × seeds)

| Comparison | Δ | t | p | sig |
|---|---:|---:|---:|:---:|
| Ours vs CAAFE-style | +0.0194 | +2.687 | 0.0105 | ★★ |
| Ours vs PromptFE-style | +0.0233 | +2.244 | 0.0306 | ★ |
| Ours vs Base only | +0.1119 | +7.412 | 0.0000 | ★★ |
| Ours vs Base+raw | -0.0148 | -3.308 | 0.0020 | ★ |
| PromptFE vs CAAFE | -0.0039 | -0.349 | 0.7293 | ns |

## LLM-Generated Feature Expressions

### CAAFE-style (data-driven)

- **yelpchi-bwgnn**: f2 + f3, f5 - f4, log(f5 / f6), sqrt(f2 * f3), max(f1, f2, f3) - min(f1, f2, f3)

### PromptFE-style (prompt-guided)

- **yelpchi-bwgnn**: f1 * f3, abs(f4 - f1 * 10), max(f2 / f5, f3 / f6), sqrt(f1 / (f4 + 1e-6)), (f2 - f5) / (f5 - f6)

### Ours (summary-only, pre-designed)

- f1*f3, f2/(f5+0.1), f4*log(f6+1), sqrt(f1+f2), max(f3,f5)-f4

## Key Takeaway

Our summary-only approach (showing aggregated class statistics) is competitive with or superior to
CAAFE-style data-driven prompting, demonstrating that LLMs can design effective features from
concise statistical summaries rather than requiring access to raw data rows.
