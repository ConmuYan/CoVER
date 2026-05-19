# Idea 3 Multi-LLM Scaling Comparison

**Task**: Design 5 composite fraud features from graph statistics.
**Evaluation**: LR(base_logit + 5 features) on YelpChi-BWGNN seed_42.

## Quick Screen Results

| Model | Params | AUPRC | ROC-AUC | n_formulas |
|---|---:|---:|---:|---:|
| qwen3-0.6b | 600M | 0.5003 | — | 5 |
| qwen3-4b | 4.0B | 0.5003 | — | 5 |
| qwen3-4b-instruct | 4.0B | 0.6641 | — | 5 |
| qwen3-8b | 8.0B | 0.5003 | — | 5 |
| *base+only* | — | 0.5003 | — | — |
| *base+raw* | — | 0.6782 | — | — |
| *base+rel* | — | 0.6085 | — | — |

## Model-Designed Formulas

### qwen3-0.6b (0.6B)
  1. `- f1: fraction of RUR neighbors labeled fraud`
  2. `- f2: fraction of RSR neighbors labeled fraud`
  3. `- f3: fraction of RTR neighbors labeled fraud`
  4. `- f4: log(1 + degree in RUR)`
  5. `- f5: log(1 + degree in RSR)`

### qwen3-4b (4.0B)
  1. `- f1: fraud neighbor rate in RUR (same-user reviews)`
  2. `- f2: fraud neighbor rate in RSR (same-product same-star)`
  3. `- f3: fraud neighbor rate in RTR (same-product same-month)`
  4. `- f4: log(1 + degree in RUR)`
  5. `- f5: log(1 + degree in RSR)`

### qwen3-4b-instruct (4.0B)
  1. `f1 + f3 - f2`
  2. `sqrt(f4 * f5) / (f6 + 1)`
  3. `max(f1, f2, f3) - min(f1, f2, f3)`
  4. `log(1 + f1 * f5)`
  5. `f4 / f6 - f5 / f6`

### qwen3-8b (8.0B)
  1. `- f1: fraud neighbor rate (RUR) – fraction of RUR neighbors labeled fraud.`
  2. `- f2: same for RSR.`
  3. `- f3: same for RTR.`
  4. `- f4: log(1 + degree in RUR)`
  5. `- f5: log(1 + degree in RSR)`

## Scaling Analysis

- Pearson correlation (log10(params) vs AUPRC): r=0.180
- Smallest (qwen3-0.6b, 0.6B) AUPRC: 0.5003
- Largest (qwen3-8b, 8.0B) AUPRC: 0.5003
- Δ = +0.0000