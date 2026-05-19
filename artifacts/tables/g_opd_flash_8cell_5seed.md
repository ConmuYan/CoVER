# G-OPD-Flash — 8-cell × 5-seed benchmark

Baseline mode for paired-t: `off_policy`  (one-sided, H1: mode > baseline)

Capture % = mean student AUPRC / mean teacher AUPRC × 100

## Per-cell results

| Dataset | Base | Mode | n | Student AUPRC | Capture % | t vs baseline | p | sig | Teacher AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| yelpchi | bwgnn | `off_policy` | 5 | 0.6152±0.0069 | 93.8% |  |  |  | 0.6558 |
| yelpchi | bwgnn | `all_node_mh` | 5 | 0.6212±0.0114 | 94.7% | +2.70 | 0.0271 | ★ | 0.6558 |
| yelpchi | bwgnn | `det_mask` | 5 | 0.6195±0.0158 | 94.5% | +0.94 | 0.1994 |  | 0.6558 |
| yelpchi | bwgnn | `g_opd_flash` | 5 | 0.6197±0.0107 | 94.5% | +2.45 | 0.0351 | ★ | 0.6558 |
| yelpchi | bwgnn | `opd_action_strict` | 5 | 0.6224±0.0121 | 94.9% | +2.30 | 0.0417 | ★ | 0.6558 |
| yelpchi | bwgnn | `opd_action_strict_mh` | 5 | 0.6173±0.0097 | 94.1% | +1.36 | 0.1225 |  | 0.6558 |
| yelpchi | sage | `off_policy` | 5 | 0.5972±0.0117 | 91.9% |  |  |  | 0.6497 |
| yelpchi | sage | `all_node_mh` | 5 | 0.6153±0.0105 | 94.7% | +3.59 | 0.0114 | ★ | 0.6497 |
| yelpchi | sage | `det_mask` | 5 | 0.6258±0.0185 | 96.3% | +2.41 | 0.0369 | ★ | 0.6497 |
| yelpchi | sage | `g_opd_flash` | 5 | 0.6124±0.0106 | 94.3% | +3.36 | 0.0142 | ★ | 0.6497 |
| yelpchi | sage | `opd_action_strict` | 5 | 0.6150±0.0097 | 94.7% | +2.73 | 0.0263 | ★ | 0.6497 |
| yelpchi | sage | `opd_action_strict_mh` | 5 | 0.6054±0.0087 | 93.2% | +3.00 | 0.0199 | ★ | 0.6497 |
| yelpchi | gcn | `off_policy` | 5 | 0.5308±0.0109 | 91.6% |  |  |  | 0.5792 |
| yelpchi | gcn | `all_node_mh` | 5 | 0.5471±0.0209 | 94.5% | +1.95 | 0.0613 |  | 0.5792 |
| yelpchi | gcn | `det_mask` | 5 | 0.5777±0.0217 | 99.7% | +5.34 | 0.0030 | ★★ | 0.5792 |
| yelpchi | gcn | `g_opd_flash` | 5 | 0.5476±0.0200 | 94.5% | +2.18 | 0.0475 | ★ | 0.5792 |
| yelpchi | gcn | `opd_action_strict` | 5 | 0.5431±0.0158 | 93.8% | +1.74 | 0.0784 |  | 0.5792 |
| yelpchi | gcn | `opd_action_strict_mh` | 5 | 0.5375±0.0152 | 92.8% | +1.18 | 0.1523 |  | 0.5792 |
| yelpchi | gat | `off_policy` | 5 | 0.5579±0.0112 | 88.9% |  |  |  | 0.6278 |
| yelpchi | gat | `all_node_mh` | 5 | 0.5809±0.0137 | 92.5% | +3.06 | 0.0188 | ★ | 0.6278 |
| yelpchi | gat | `det_mask` | 5 | 0.6180±0.0126 | 98.4% | +10.32 | 0.0002 | ★★★ | 0.6278 |
| yelpchi | gat | `g_opd_flash` | 5 | 0.5814±0.0133 | 92.6% | +4.48 | 0.0055 | ★★ | 0.6278 |
| yelpchi | gat | `opd_action_strict` | 5 | 0.5860±0.0063 | 93.3% | +8.28 | 0.0006 | ★★★ | 0.6278 |
| yelpchi | gat | `opd_action_strict_mh` | 5 | 0.5724±0.0085 | 91.2% | +2.93 | 0.0214 | ★ | 0.6278 |
| amazon | bwgnn | `off_policy` | 5 | 0.8653±0.0282 | 97.1% |  |  |  | 0.8907 |
| amazon | bwgnn | `all_node_mh` | 5 | 0.8660±0.0283 | 97.2% | +0.98 | 0.1907 |  | 0.8907 |
| amazon | bwgnn | `det_mask` | 5 | 0.8657±0.0312 | 97.2% | +0.25 | 0.4074 |  | 0.8907 |
| amazon | bwgnn | `g_opd_flash` | 5 | 0.8658±0.0289 | 97.2% | +0.65 | 0.2761 |  | 0.8907 |
| amazon | bwgnn | `opd_action_strict` | 5 | 0.8656±0.0311 | 97.2% | +0.16 | 0.4399 |  | 0.8907 |
| amazon | bwgnn | `opd_action_strict_mh` | 5 | 0.8643±0.0314 | 97.0% | -0.49 | 0.6745 |  | 0.8907 |
| amazon | sage | `off_policy` | 5 | 0.8392±0.0185 | 99.1% |  |  |  | 0.8466 |
| amazon | sage | `all_node_mh` | 5 | 0.8438±0.0138 | 99.7% | +0.96 | 0.1960 |  | 0.8466 |
| amazon | sage | `det_mask` | 5 | 0.8489±0.0121 | 100.3% | +1.58 | 0.0946 |  | 0.8466 |
| amazon | sage | `g_opd_flash` | 5 | 0.8423±0.0148 | 99.5% | +0.61 | 0.2889 |  | 0.8466 |
| amazon | sage | `opd_action_strict` | 5 | 0.8396±0.0178 | 99.2% | +0.13 | 0.4520 |  | 0.8466 |
| amazon | sage | `opd_action_strict_mh` | 5 | 0.8377±0.0187 | 99.0% | -0.52 | 0.6861 |  | 0.8466 |
| amazon | gcn | `off_policy` | 5 | 0.6039±0.2033 | 263.5% |  |  |  | 0.2292 |
| amazon | gcn | `all_node_mh` | 5 | 0.6733±0.2507 | 293.8% | +1.79 | 0.0740 |  | 0.2292 |
| amazon | gcn | `det_mask` | 5 | 0.6856±0.2577 | 299.2% | +1.89 | 0.0658 |  | 0.2292 |
| amazon | gcn | `g_opd_flash` | 5 | 0.6726±0.2502 | 293.5% | +1.77 | 0.0753 |  | 0.2292 |
| amazon | gcn | `opd_action_strict` | 5 | 0.6673±0.2478 | 291.2% | +1.59 | 0.0932 |  | 0.2292 |
| amazon | gcn | `opd_action_strict_mh` | 5 | 0.6377±0.2502 | 278.3% | +0.57 | 0.3007 |  | 0.2292 |
| amazon | gat | `off_policy` | 5 | 0.5164±0.3806 | 62.9% |  |  |  | 0.8204 |
| amazon | gat | `all_node_mh` | 5 | 0.5299±0.3912 | 64.6% | +1.67 | 0.0853 |  | 0.8204 |
| amazon | gat | `det_mask` | 5 | 0.5298±0.3924 | 64.6% | +1.48 | 0.1062 |  | 0.8204 |
| amazon | gat | `g_opd_flash` | 5 | 0.5264±0.3899 | 64.2% | +1.56 | 0.0964 |  | 0.8204 |
| amazon | gat | `opd_action_strict` | 5 | 0.5277±0.3898 | 64.3% | +1.11 | 0.1652 |  | 0.8204 |
| amazon | gat | `opd_action_strict_mh` | 5 | 0.5176±0.3820 | 63.1% | +0.19 | 0.4285 |  | 0.8204 |

## Cross-cell summary (per mode)

**Statistical convention**: per-cell paired-t only (5-seed, one-sided H1: mode > baseline).  Cells are NOT pooled — pooling treats cells as IID and inflates apparent sample size (Critic round-4 fix).

| Mode | Cells | Mean AUPRC across cells | Directional + cells | Sig cells (p<0.05) | Sig cells (p<0.01) |
|---|---:|---:|---:|---:|---:|
| `off_policy` | 8 | 0.6407±0.1353 | 0/8 | 0/8 | 0/8 |
| `all_node_mh` | 8 | 0.6597±0.1286 | 8/8 | 3/8 | 0/8 |
| `det_mask` | 8 | 0.6714±0.1230 | 8/8 | 3/8 | 2/8 |
| `g_opd_flash` | 8 | 0.6585±0.1289 | 8/8 | 4/8 | 1/8 |
| `opd_action_strict` | 8 | 0.6583±0.1280 | 8/8 | 3/8 | 1/8 |
| `opd_action_strict_mh` | 8 | 0.6487±0.1312 | 6/8 | 2/8 | 0/8 |
