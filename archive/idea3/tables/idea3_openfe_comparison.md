# Idea 3 OpenFE + Symbolic Regression Comparison

**Goal**: Prove LLM-designed composites > SOTA AutoFE methods (OpenFE, Symbolic Regression).

**Methods compared**:

- **OpenFE** (NeurIPS 2023): Automatic feature engineering via mutual information + gradient boosting.

- **Symbolic Regression** (gplearn): Genetic programming to discover mathematical formulas.

- **LLM composites**: Qwen3-4B-Instruct designs 5 fraud-aware composite features.

- **Random formula**: Best of 100 random binary operations (val-selected).

- **Systematic**: Best of all pairwise products/ratios + log/sqrt transforms (val-selected).

- All evaluated: LR(base_logit + features) on test set.

- **Significance (df=4)**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★

## AUPRC Summary (mean ± sd across 5 seeds)

| Cell | Base only | Base+raw (LR) | Random formula (best of 100) | Systematic transforms (best) | Base+OpenFE | Base+GP (gplearn) | Base+LLM composites |
|---|---:|---:|---:|---:|---:|---:|---:|
| yelpchi-bwgnn | 0.5034 ± 0.0155 | 0.6778 ± 0.0075 | 0.6204 ± 0.0078 | 0.5976 ± 0.0122 | 0.6163 ± 0.0292 | 0.5976 ± 0.0119 | 0.6660 ± 0.0059 |
| yelpchi-sage | 0.4595 ± 0.0137 | 0.6512 ± 0.0121 | 0.5938 ± 0.0151 | 0.5689 ± 0.0082 | 0.4816 ± 0.0584 | 0.5700 ± 0.0088 | 0.6376 ± 0.0120 |
| yelpchi-gcn | 0.2226 ± 0.0099 | 0.4515 ± 0.0115 | 0.3561 ± 0.0059 | 0.3349 ± 0.0113 | 0.2299 ± 0.0070 | 0.3879 ± 0.0306 | 0.4331 ± 0.0099 |
| yelpchi-gat | 0.2060 ± 0.0148 | 0.4450 ± 0.0111 | 0.3509 ± 0.0080 | 0.3379 ± 0.0049 | 0.2124 ± 0.0156 | 0.3812 ± 0.0250 | 0.4288 ± 0.0100 |
| amazon-bwgnn | 0.8595 ± 0.0339 | 0.8602 ± 0.0321 | 0.8611 ± 0.0326 | 0.8609 ± 0.0342 | 0.8605 ± 0.0344 | 0.8597 ± 0.0339 | 0.8600 ± 0.0313 |
| amazon-sage | 0.7864 ± 0.0725 | 0.7991 ± 0.0760 | 0.8037 ± 0.0780 | 0.7949 ± 0.0719 | 0.7911 ± 0.0754 | 0.7871 ± 0.0727 | 0.7979 ± 0.0769 |
| amazon-gcn | 0.2549 ± 0.0212 | 0.4088 ± 0.0143 | 0.3470 ± 0.0245 | 0.3442 ± 0.0240 | 0.3884 ± 0.0134 | 0.3351 ± 0.0481 | 0.3272 ± 0.0317 |
| amazon-gat | 0.3401 ± 0.2888 | 0.5096 ± 0.1608 | 0.4412 ± 0.2129 | 0.4266 ± 0.2193 | 0.4847 ± 0.1778 | 0.4196 ± 0.2081 | 0.4391 ± 0.2040 |

## Overall Mean AUPRC (across all cells)

| Method | Mean ± SD | N |
|---|---:|---:|
| Base only | 0.4541 ± 0.2577 | 40 |
| Base+raw (LR) | 0.6004 ± 0.1730 | 40 |
| Random formula (best of 100) | 0.5468 ± 0.2091 | 40 |
| Systematic transforms (best) | 0.5332 ± 0.2119 | 40 |
| Base+OpenFE | 0.5081 ± 0.2351 | 40 |
| Base+GP (gplearn) | 0.5423 ± 0.2011 | 40 |
| Base+LLM composites | 0.5737 ± 0.1975 | 40 |

## Paired-t: Base+LLM vs baselines

| Comparison | Δ | t | p | sig |
|---|---:|---:|---:|:---:|
| Base+LLM vs Base only | +0.1196 | +8.543 | 0.0000 (40 pairs) | ★★ |
| Base+LLM vs Base+raw (LR) | -0.0267 | -4.681 | 0.0000 (40 pairs) | ★★ |
| Base+LLM vs Random formula (best of 100) | +0.0269 | +4.306 | 0.0001 (40 pairs) | ★ |
| Base+LLM vs Systematic transforms (best) | +0.0405 | +5.627 | 0.0000 (40 pairs) | ★★ |
| Base+LLM vs Base+OpenFE | +0.0656 | +3.829 | 0.0005 (40 pairs) | ★ |
| Base+LLM vs Base+GP (gplearn) | +0.0314 | +5.150 | 0.0000 (40 pairs) | ★★ |

## LLM Win Rate vs Baselines

| vs Method | Win | Tie | Loss | Win% |
|---|---:|---:|---:|---:|
| vs Base only | 37 | 0 | 3 | 92% |
| vs Base+raw (LR) | 4 | 0 | 36 | 10% |
| vs Random formula (best of 100) | 26 | 0 | 14 | 65% |
| vs Systematic transforms (best) | 30 | 0 | 10 | 75% |
| vs Base+OpenFE | 29 | 0 | 11 | 72% |
| vs Base+GP (gplearn) | 31 | 0 | 9 | 78% |

## Per-Cell Detailed Results

| Cell | Seed | Base | Raw | Random | Systematic | OpenFE | GP | LLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yelpchi-bwgnn | 42 | 0.5003 | 0.6788 | 0.6284 | 0.5965 | 0.6109 | 0.5965 | 0.6669 |
| yelpchi-bwgnn | 123 | 0.5232 | 0.6901 | 0.6256 | 0.6152 | 0.6270 | 0.6144 | 0.6749 |
| yelpchi-bwgnn | 456 | 0.4997 | 0.6719 | 0.6238 | 0.5958 | 0.6592 | 0.5952 | 0.6610 |
| yelpchi-bwgnn | 789 | 0.5121 | 0.6761 | 0.6115 | 0.5995 | 0.6037 | 0.6008 | 0.6671 |
| yelpchi-bwgnn | 2026 | 0.4816 | 0.6721 | 0.6127 | 0.5810 | 0.5808 | 0.5814 | 0.6602 |
| yelpchi-sage | 42 | 0.4473 | 0.6485 | 0.5981 | 0.5635 | 0.4491 | 0.5639 | 0.6334 |
| yelpchi-sage | 123 | 0.4573 | 0.6514 | 0.5824 | 0.5687 | 0.4609 | 0.5703 | 0.6369 |
| yelpchi-sage | 456 | 0.4504 | 0.6387 | 0.5950 | 0.5626 | 0.4506 | 0.5626 | 0.6261 |
| yelpchi-sage | 789 | 0.4601 | 0.6463 | 0.5776 | 0.5669 | 0.4617 | 0.5685 | 0.6337 |
| yelpchi-sage | 2026 | 0.4822 | 0.6711 | 0.6161 | 0.5828 | 0.5855 | 0.5847 | 0.6577 |
| yelpchi-gcn | 42 | 0.2181 | 0.4441 | 0.3542 | 0.3213 | 0.2272 | 0.3843 | 0.4277 |
| yelpchi-gcn | 123 | 0.2092 | 0.4531 | 0.3612 | 0.3446 | 0.2205 | 0.3737 | 0.4327 |
| yelpchi-gcn | 456 | 0.2349 | 0.4469 | 0.3530 | 0.3260 | 0.2396 | 0.3914 | 0.4289 |
| yelpchi-gcn | 789 | 0.2222 | 0.4425 | 0.3488 | 0.3352 | 0.2301 | 0.4364 | 0.4261 |
| yelpchi-gcn | 2026 | 0.2288 | 0.4709 | 0.3630 | 0.3471 | 0.2321 | 0.3537 | 0.4504 |
| yelpchi-gat | 42 | 0.2058 | 0.4405 | 0.3546 | 0.3355 | 0.2186 | 0.3724 | 0.4250 |
| yelpchi-gat | 123 | 0.2028 | 0.4505 | 0.3595 | 0.3429 | 0.2149 | 0.3650 | 0.4313 |
| yelpchi-gat | 456 | 0.2035 | 0.4409 | 0.3441 | 0.3403 | 0.2059 | 0.3916 | 0.4242 |
| yelpchi-gat | 789 | 0.2295 | 0.4322 | 0.3408 | 0.3306 | 0.2324 | 0.4197 | 0.4187 |
| yelpchi-gat | 2026 | 0.1886 | 0.4611 | 0.3557 | 0.3403 | 0.1904 | 0.3572 | 0.4447 |
| amazon-bwgnn | 42 | 0.8829 | 0.8836 | 0.8809 | 0.8844 | 0.8824 | 0.8829 | 0.8830 |
| amazon-bwgnn | 123 | 0.8396 | 0.8458 | 0.8469 | 0.8417 | 0.8414 | 0.8396 | 0.8473 |
| amazon-bwgnn | 456 | 0.8129 | 0.8152 | 0.8135 | 0.8129 | 0.8120 | 0.8129 | 0.8156 |
| amazon-bwgnn | 789 | 0.8647 | 0.8598 | 0.8661 | 0.8671 | 0.8676 | 0.8657 | 0.8586 |
| amazon-bwgnn | 2026 | 0.8975 | 0.8968 | 0.8981 | 0.8985 | 0.8990 | 0.8975 | 0.8956 |
| amazon-sage | 42 | 0.8030 | 0.8085 | 0.8149 | 0.8093 | 0.8043 | 0.8064 | 0.8056 |
| amazon-sage | 123 | 0.8030 | 0.8330 | 0.8396 | 0.8306 | 0.8113 | 0.8030 | 0.8334 |
| amazon-sage | 456 | 0.8158 | 0.8237 | 0.8251 | 0.8199 | 0.8234 | 0.8158 | 0.8245 |
| amazon-sage | 789 | 0.6612 | 0.6678 | 0.6692 | 0.6686 | 0.6608 | 0.6612 | 0.6649 |
| amazon-sage | 2026 | 0.8491 | 0.8625 | 0.8697 | 0.8460 | 0.8557 | 0.8491 | 0.8609 |
| amazon-gcn | 42 | 0.2286 | 0.4173 | 0.3672 | 0.3430 | 0.3838 | 0.2657 | 0.3223 |
| amazon-gcn | 123 | 0.2827 | 0.3902 | 0.3508 | 0.3579 | 0.3687 | 0.3416 | 0.3761 |
| amazon-gcn | 456 | 0.2394 | 0.3980 | 0.3085 | 0.3077 | 0.3880 | 0.3148 | 0.3021 |
| amazon-gcn | 789 | 0.2618 | 0.4135 | 0.3402 | 0.3402 | 0.4026 | 0.3610 | 0.3378 |
| amazon-gcn | 2026 | 0.2621 | 0.4250 | 0.3681 | 0.3720 | 0.3988 | 0.3926 | 0.2979 |
| amazon-gat | 42 | 0.5754 | 0.6092 | 0.5979 | 0.5841 | 0.6061 | 0.5517 | 0.5694 |
| amazon-gat | 123 | 0.0688 | 0.3649 | 0.2718 | 0.2622 | 0.3335 | 0.2280 | 0.2803 |
| amazon-gat | 456 | 0.1419 | 0.4027 | 0.3016 | 0.2840 | 0.3682 | 0.2950 | 0.3148 |
| amazon-gat | 789 | 0.7198 | 0.7431 | 0.7377 | 0.7349 | 0.7375 | 0.7198 | 0.7358 |
| amazon-gat | 2026 | 0.1946 | 0.4282 | 0.2971 | 0.2679 | 0.3780 | 0.3034 | 0.2951 |