# Idea 3 Full Benchmark — LLM Feature Composites vs AutoFE Baselines

**Method**: Qwen3-4B-Instruct designs 5 composite features from graph statistics.

**Baselines**: raw features, all pairwise/log/sqrt transforms, random formulas.

**Significance (df=4)**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★

## AUPRC Summary (mean ± sd across 5 seeds)

| Cell | Base only | Base+REL | Base+raw | Base+pairwise | Base+random | Base+LLM | Base+REL+LLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| yelpchi-bwgnn | 0.5034 ± 0.0155 | 0.6079 ± 0.0079 | 0.6778 ± 0.0075 | 0.6859 ± 0.0095 | 0.5999 ± 0.0351 | 0.6660 ± 0.0059 | 0.6680 ± 0.0023 |
| yelpchi-sage | 0.4595 ± 0.0137 | 0.5967 ± 0.0110 | 0.6512 ± 0.0121 | 0.6581 ± 0.0104 | 0.5709 ± 0.0292 | 0.6376 ± 0.0120 | 0.6497 ± 0.0127 |
| yelpchi-gcn | 0.2226 ± 0.0099 | 0.4954 ± 0.0149 | 0.4515 ± 0.0115 | 0.4612 ± 0.0102 | 0.3569 ± 0.0428 | 0.4331 ± 0.0099 | 0.5868 ± 0.0133 |
| yelpchi-gat | 0.2060 ± 0.0148 | 0.5328 ± 0.0158 | 0.4450 ± 0.0111 | 0.4562 ± 0.0115 | 0.3450 ± 0.0545 | 0.4288 ± 0.0100 | 0.5972 ± 0.0145 |
| amazon-bwgnn | 0.8595 ± 0.0339 | 0.8652 ± 0.0258 | 0.8602 ± 0.0321 | 0.8510 ± 0.0264 | 0.8611 ± 0.0325 | 0.8600 ± 0.0313 | 0.8671 ± 0.0235 |
| amazon-sage | 0.7864 ± 0.0725 | 0.8289 ± 0.0506 | 0.7991 ± 0.0760 | 0.7961 ± 0.0755 | 0.7967 ± 0.0731 | 0.7979 ± 0.0769 | 0.8295 ± 0.0512 |
| amazon-gcn | 0.2549 ± 0.0212 | 0.4720 ± 0.1304 | 0.4088 ± 0.0143 | 0.4189 ± 0.0186 | 0.3671 ± 0.0332 | 0.3272 ± 0.0317 | 0.5013 ± 0.1039 |
| amazon-gat | 0.3401 ± 0.2888 | 0.4613 ± 0.3068 | 0.5096 ± 0.1608 | 0.5161 ± 0.1544 | 0.4889 ± 0.1791 | 0.4391 ± 0.2040 | 0.5285 ± 0.2248 |

## Paired-t: Base+LLM vs baselines (across all 8 cells × 5 seeds = 40 pairs)

| Comparison | Δ | t | p | sig |
|---|---:|---:|---:|:---:|
| Base+LLM vs Base only | +0.1196 | +8.543 | 0.0000 | ★★ |
| Base+LLM vs Base+REL | -0.0338 | -2.201 | 0.0337 | ns |
| Base+LLM vs Base+raw | -0.0267 | -4.681 | 0.0000 | ★★ |
| Base+LLM vs Base+pairwise | -0.0317 | -5.043 | 0.0000 | ★★ |
| Base+LLM vs Base+random | +0.0254 | +2.688 | 0.0105 | ns |
| Base+REL+LLM vs Base+REL | +0.0460 | +6.753 | 0.0000 | ★★ |