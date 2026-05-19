# Idea 3 AutoFE Baselines — LLM vs XGBoost/LightGBM/Random/Systematic

**Comparison**: LLM-designed composites vs automated feature engineering baselines.

**All methods**: LR(base_logit + features) on test set.

## AUPRC Summary (mean ± sd across 5 seeds)

| Cell | Base only | Base+raw (LR) | XGBoost (raw) | LightGBM (raw) | Random formula (best of 100) | Systematic transforms (best) | Base+LLM composites |
|---|---:|---:|---:|---:|---:|---:|---:|
| yelpchi-bwgnn | 0.5034 ± 0.0155 | 0.6778 ± 0.0075 | 0.6639 ± 0.0111 | 0.6578 ± 0.0129 | 0.6204 ± 0.0078 | 0.5976 ± 0.0122 | 0.6660 ± 0.0059 |
| yelpchi-sage | 0.4595 ± 0.0137 | 0.6512 ± 0.0121 | 0.6330 ± 0.0112 | 0.6268 ± 0.0163 | 0.5938 ± 0.0151 | 0.5689 ± 0.0082 | 0.6376 ± 0.0120 |
| yelpchi-gcn | 0.2226 ± 0.0099 | 0.4515 ± 0.0115 | 0.4189 ± 0.0101 | 0.4134 ± 0.0165 | 0.3561 ± 0.0059 | 0.3349 ± 0.0113 | 0.4331 ± 0.0099 |
| yelpchi-gat | 0.2060 ± 0.0148 | 0.4450 ± 0.0111 | 0.4167 ± 0.0124 | 0.4085 ± 0.0093 | 0.3509 ± 0.0080 | 0.3379 ± 0.0049 | 0.4288 ± 0.0100 |
| amazon-bwgnn | 0.8595 ± 0.0339 | 0.8602 ± 0.0321 | 0.8731 ± 0.0245 | 0.8762 ± 0.0275 | 0.8611 ± 0.0326 | 0.8609 ± 0.0342 | 0.8600 ± 0.0313 |
| amazon-sage | 0.7864 ± 0.0725 | 0.7991 ± 0.0760 | 0.8214 ± 0.0671 | 0.8183 ± 0.0650 | 0.8037 ± 0.0780 | 0.7949 ± 0.0719 | 0.7979 ± 0.0769 |
| amazon-gcn | 0.2549 ± 0.0212 | 0.4088 ± 0.0143 | 0.4746 ± 0.0230 | 0.4740 ± 0.0276 | 0.3470 ± 0.0245 | 0.3442 ± 0.0240 | 0.3272 ± 0.0317 |
| amazon-gat | 0.3401 ± 0.2888 | 0.5096 ± 0.1608 | 0.5532 ± 0.1423 | 0.5464 ± 0.1549 | 0.4412 ± 0.2129 | 0.4266 ± 0.2193 | 0.4391 ± 0.2040 |

## Paired-t: Base+LLM vs baselines (8 cells × 5 seeds = 40 pairs)

| Comparison | Δ | t | p | sig |
|---|---:|---:|---:|:---:|
| Base+LLM vs Base only | +0.1196 | +8.543 | 0.0000 | ★★ |
| Base+LLM vs Base+raw (LR) | -0.0267 | -4.681 | 0.0000 | ★★ |
| Base+LLM vs XGBoost (raw) | -0.0331 | -3.243 | 0.0024 | ★ |
| Base+LLM vs LightGBM (raw) | -0.0290 | -2.776 | 0.0084 | ns |
| Base+LLM vs Random formula (best of 100) | +0.0269 | +4.306 | 0.0001 | ★ |
| Base+LLM vs Systematic transforms (best) | +0.0405 | +5.627 | 0.0000 | ★★ |