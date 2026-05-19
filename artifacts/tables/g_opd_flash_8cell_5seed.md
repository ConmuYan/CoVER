# G-OPD-Flash — 8-cell × 5-seed benchmark

Baseline mode for paired-t: `off_policy`  (one-sided, H1: mode > baseline)

Capture % = mean student AUPRC / mean teacher AUPRC × 100

## Per-cell results

| Dataset | Base | Mode | n | Student AUPRC | Capture % | t vs `off_policy` | p | sig | t vs `det_mask` | p | sig | Teacher AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| yelpchi | bwgnn | `off_policy` | 5 | 0.6152±0.0069 | 94.8% |  |  |  | -0.94 | 0.8006 |  | 0.6489 |
| yelpchi | bwgnn | `all_node_mh` | 5 | 0.6212±0.0114 | 95.7% | +2.70 | 0.0271 | ★ | +0.46 | 0.3334 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask` | 5 | 0.6195±0.0158 | 95.5% | +0.94 | 0.1994 |  |  |  |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_mh` | 5 | 0.6135±0.0145 | 94.5% | -0.49 | 0.6738 |  | -3.66 | 0.9892 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_no_rel` | 5 | 0.6195±0.0158 | 95.5% | +0.98 | 0.1904 |  | +0.02 | 0.4918 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_fixed_bce` | 5 | 0.6189±0.0149 | 95.4% | +0.98 | 0.1903 |  | -0.28 | 0.6024 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_rev_only` | 5 | 0.6195±0.0151 | 95.5% | +1.09 | 0.1692 |  | +0.05 | 0.4797 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_single_denom` | 5 | 0.6183±0.0160 | 95.3% | +0.73 | 0.2541 |  | -0.92 | 0.7944 |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.6489 |
| yelpchi | bwgnn | `det_mask_cbr` | 5 | 0.6216±0.0172 | 95.8% | +1.18 | 0.1520 |  | +0.88 | 0.2132 |  | 0.6489 |
| yelpchi | bwgnn | `g_opd_flash` | 5 | 0.6197±0.0107 | 95.5% | +2.45 | 0.0351 | ★ | +0.07 | 0.4751 |  | 0.6489 |
| yelpchi | bwgnn | `opd_action_strict` | 5 | 0.6224±0.0121 | 95.9% | +2.30 | 0.0417 | ★ | +0.61 | 0.2860 |  | 0.6489 |
| yelpchi | bwgnn | `opd_action_strict_mh` | 5 | 0.6173±0.0097 | 95.1% | +1.36 | 0.1225 |  | -0.56 | 0.6974 |  | 0.6489 |
| yelpchi | sage | `off_policy` | 5 | 0.5972±0.0117 | 91.6% |  |  |  | -2.41 | 0.9631 |  | 0.6520 |
| yelpchi | sage | `all_node_mh` | 5 | 0.6153±0.0105 | 94.4% | +3.59 | 0.0114 | ★ | -1.39 | 0.8822 |  | 0.6520 |
| yelpchi | sage | `det_mask` | 5 | 0.6258±0.0185 | 96.0% | +2.41 | 0.0369 | ★ |  |  |  | 0.6520 |
| yelpchi | sage | `det_mask_mh` | 5 | 0.6185±0.0221 | 94.9% | +1.68 | 0.0840 |  | -2.66 | 0.9717 |  | 0.6520 |
| yelpchi | sage | `det_mask_no_rel` | 5 | 0.6251±0.0204 | 95.9% | +2.24 | 0.0445 | ★ | -0.72 | 0.7440 |  | 0.6520 |
| yelpchi | sage | `det_mask_fixed_bce` | 5 | 0.6216±0.0181 | 95.3% | +2.28 | 0.0424 | ★ | -1.23 | 0.8566 |  | 0.6520 |
| yelpchi | sage | `det_mask_rev_only` | 5 | 0.6243±0.0206 | 95.8% | +2.07 | 0.0537 |  | -1.04 | 0.8210 |  | 0.6520 |
| yelpchi | sage | `det_mask_single_denom` | 5 | 0.6260±0.0215 | 96.0% | +2.26 | 0.0432 | ★ | +0.10 | 0.4640 |  | 0.6520 |
| yelpchi | sage | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.6520 |
| yelpchi | sage | `det_mask_cbr` | 5 | 0.6320±0.0153 | 96.9% | +3.54 | 0.0120 | ★ | +2.42 | 0.0364 | ★ | 0.6520 |
| yelpchi | sage | `g_opd_flash` | 5 | 0.6124±0.0106 | 93.9% | +3.36 | 0.0142 | ★ | -1.65 | 0.9128 |  | 0.6520 |
| yelpchi | sage | `opd_action_strict` | 5 | 0.6150±0.0097 | 94.3% | +2.73 | 0.0263 | ★ | -1.65 | 0.9130 |  | 0.6520 |
| yelpchi | sage | `opd_action_strict_mh` | 5 | 0.6054±0.0087 | 92.9% | +3.00 | 0.0199 | ★ | -2.20 | 0.9538 |  | 0.6520 |
| yelpchi | gcn | `off_policy` | 5 | 0.5308±0.0109 | 89.9% |  |  |  | -5.34 | 0.9970 |  | 0.5904 |
| yelpchi | gcn | `all_node_mh` | 5 | 0.5471±0.0209 | 92.7% | +1.95 | 0.0613 |  | -8.01 | 0.9993 |  | 0.5904 |
| yelpchi | gcn | `det_mask` | 5 | 0.5777±0.0217 | 97.8% | +5.34 | 0.0030 | ★★ |  |  |  | 0.5904 |
| yelpchi | gcn | `det_mask_mh` | 5 | 0.5737±0.0245 | 97.2% | +4.12 | 0.0073 | ★★ | -1.12 | 0.8376 |  | 0.5904 |
| yelpchi | gcn | `det_mask_no_rel` | 5 | 0.5723±0.0202 | 96.9% | +5.00 | 0.0037 | ★★ | -2.83 | 0.9762 |  | 0.5904 |
| yelpchi | gcn | `det_mask_fixed_bce` | 5 | 0.5649±0.0155 | 95.7% | +5.00 | 0.0038 | ★★ | -3.44 | 0.9869 |  | 0.5904 |
| yelpchi | gcn | `det_mask_rev_only` | 5 | 0.5765±0.0224 | 97.7% | +5.16 | 0.0034 | ★★ | -0.69 | 0.7369 |  | 0.5904 |
| yelpchi | gcn | `det_mask_single_denom` | 5 | 0.5694±0.0207 | 96.4% | +4.48 | 0.0055 | ★★ | -3.40 | 0.9863 |  | 0.5904 |
| yelpchi | gcn | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.5904 |
| yelpchi | gcn | `det_mask_cbr` | 5 | 0.5827±0.0204 | 98.7% | +6.28 | 0.0016 | ★★ | +3.76 | 0.0099 | ★★ | 0.5904 |
| yelpchi | gcn | `g_opd_flash` | 5 | 0.5476±0.0200 | 92.7% | +2.18 | 0.0475 | ★ | -8.18 | 0.9994 |  | 0.5904 |
| yelpchi | gcn | `opd_action_strict` | 5 | 0.5431±0.0158 | 92.0% | +1.74 | 0.0784 |  | -12.22 | 0.9999 |  | 0.5904 |
| yelpchi | gcn | `opd_action_strict_mh` | 5 | 0.5375±0.0152 | 91.0% | +1.18 | 0.1523 |  | -10.21 | 0.9997 |  | 0.5904 |
| yelpchi | gat | `off_policy` | 5 | 0.5579±0.0112 | 88.0% |  |  |  | -10.32 | 0.9998 |  | 0.6340 |
| yelpchi | gat | `all_node_mh` | 5 | 0.5809±0.0137 | 91.6% | +3.06 | 0.0188 | ★ | -4.07 | 0.9924 |  | 0.6340 |
| yelpchi | gat | `det_mask` | 5 | 0.6180±0.0126 | 97.5% | +10.32 | 0.0002 | ★★★ |  |  |  | 0.6340 |
| yelpchi | gat | `det_mask_mh` | 5 | 0.6062±0.0141 | 95.6% | +15.61 | 0.0000 | ★★★ | -1.72 | 0.9202 |  | 0.6340 |
| yelpchi | gat | `det_mask_no_rel` | 5 | 0.6159±0.0104 | 97.1% | +10.85 | 0.0002 | ★★★ | -1.46 | 0.8913 |  | 0.6340 |
| yelpchi | gat | `det_mask_fixed_bce` | 5 | 0.6148±0.0084 | 97.0% | +9.72 | 0.0003 | ★★★ | -1.29 | 0.8664 |  | 0.6340 |
| yelpchi | gat | `det_mask_rev_only` | 5 | 0.6157±0.0114 | 97.1% | +10.24 | 0.0003 | ★★★ | -1.30 | 0.8690 |  | 0.6340 |
| yelpchi | gat | `det_mask_single_denom` | 5 | 0.6139±0.0115 | 96.8% | +8.70 | 0.0005 | ★★★ | -2.36 | 0.9611 |  | 0.6340 |
| yelpchi | gat | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.6340 |
| yelpchi | gat | `det_mask_cbr` | 5 | 0.6260±0.0111 | 98.7% | +11.11 | 0.0002 | ★★★ | +4.48 | 0.0055 | ★★ | 0.6340 |
| yelpchi | gat | `g_opd_flash` | 5 | 0.5814±0.0133 | 91.7% | +4.48 | 0.0055 | ★★ | -4.72 | 0.9954 |  | 0.6340 |
| yelpchi | gat | `opd_action_strict` | 5 | 0.5860±0.0063 | 92.4% | +8.28 | 0.0006 | ★★★ | -5.15 | 0.9966 |  | 0.6340 |
| yelpchi | gat | `opd_action_strict_mh` | 5 | 0.5724±0.0085 | 90.3% | +2.93 | 0.0214 | ★ | -8.30 | 0.9994 |  | 0.6340 |
| amazon | bwgnn | `off_policy` | 5 | 0.8653±0.0282 | 99.8% |  |  |  | -0.25 | 0.5926 |  | 0.8671 |
| amazon | bwgnn | `all_node_mh` | 5 | 0.8660±0.0283 | 99.9% | +0.98 | 0.1907 |  | +0.16 | 0.4414 |  | 0.8671 |
| amazon | bwgnn | `det_mask` | 5 | 0.8657±0.0312 | 99.8% | +0.25 | 0.4074 |  |  |  |  | 0.8671 |
| amazon | bwgnn | `det_mask_mh` | 5 | 0.8655±0.0306 | 99.8% | +0.12 | 0.4562 |  | -0.59 | 0.7065 |  | 0.8671 |
| amazon | bwgnn | `det_mask_no_rel` | 5 | 0.8664±0.0317 | 99.9% | +0.58 | 0.2978 |  | +2.45 | 0.0352 | ★ | 0.8671 |
| amazon | bwgnn | `det_mask_fixed_bce` | 5 | 0.8661±0.0314 | 99.9% | +0.44 | 0.3415 |  | +2.08 | 0.0532 |  | 0.8671 |
| amazon | bwgnn | `det_mask_rev_only` | 5 | 0.8654±0.0309 | 99.8% | +0.08 | 0.4682 |  | -1.71 | 0.9192 |  | 0.8671 |
| amazon | bwgnn | `det_mask_single_denom` | 5 | 0.8657±0.0316 | 99.8% | +0.22 | 0.4200 |  | +0.02 | 0.4926 |  | 0.8671 |
| amazon | bwgnn | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.8671 |
| amazon | bwgnn | `det_mask_cbr` | 5 | 0.8670±0.0311 | 100.0% | +1.11 | 0.1653 |  | +2.12 | 0.0508 |  | 0.8671 |
| amazon | bwgnn | `g_opd_flash` | 5 | 0.8658±0.0289 | 99.8% | +0.65 | 0.2761 |  | +0.01 | 0.4944 |  | 0.8671 |
| amazon | bwgnn | `opd_action_strict` | 5 | 0.8656±0.0311 | 99.8% | +0.16 | 0.4399 |  | -0.21 | 0.5787 |  | 0.8671 |
| amazon | bwgnn | `opd_action_strict_mh` | 5 | 0.8643±0.0314 | 99.7% | -0.49 | 0.6745 |  | -1.88 | 0.9337 |  | 0.8671 |
| amazon | sage | `off_policy` | 5 | 0.8392±0.0185 | 98.6% |  |  |  | -1.58 | 0.9054 |  | 0.8511 |
| amazon | sage | `all_node_mh` | 5 | 0.8438±0.0138 | 99.1% | +0.96 | 0.1960 |  | -3.01 | 0.9803 |  | 0.8511 |
| amazon | sage | `det_mask` | 5 | 0.8489±0.0121 | 99.7% | +1.58 | 0.0946 |  |  |  |  | 0.8511 |
| amazon | sage | `det_mask_mh` | 5 | 0.8474±0.0132 | 99.6% | +1.39 | 0.1184 |  | -1.53 | 0.9001 |  | 0.8511 |
| amazon | sage | `det_mask_no_rel` | 5 | 0.8475±0.0128 | 99.6% | +1.57 | 0.0962 |  | -1.33 | 0.8727 |  | 0.8511 |
| amazon | sage | `det_mask_fixed_bce` | 5 | 0.8477±0.0125 | 99.6% | +1.68 | 0.0839 |  | -0.94 | 0.8006 |  | 0.8511 |
| amazon | sage | `det_mask_rev_only` | 5 | 0.8483±0.0124 | 99.7% | +1.44 | 0.1118 |  | -1.88 | 0.9334 |  | 0.8511 |
| amazon | sage | `det_mask_single_denom` | 5 | 0.8466±0.0137 | 99.5% | +1.83 | 0.0703 |  | -1.01 | 0.8154 |  | 0.8511 |
| amazon | sage | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.8511 |
| amazon | sage | `det_mask_cbr` | 5 | 0.8490±0.0132 | 99.7% | +1.60 | 0.0923 |  | +0.15 | 0.4456 |  | 0.8511 |
| amazon | sage | `g_opd_flash` | 5 | 0.8423±0.0148 | 99.0% | +0.61 | 0.2889 |  | -3.60 | 0.9886 |  | 0.8511 |
| amazon | sage | `opd_action_strict` | 5 | 0.8396±0.0178 | 98.6% | +0.13 | 0.4520 |  | -2.42 | 0.9638 |  | 0.8511 |
| amazon | sage | `opd_action_strict_mh` | 5 | 0.8377±0.0187 | 98.4% | -0.52 | 0.6861 |  | -2.10 | 0.9483 |  | 0.8511 |
| amazon | gcn | `off_policy` | 5 | 0.6039±0.2033 | 86.2% |  |  |  | -1.89 | 0.9342 |  | 0.7006 |
| amazon | gcn | `all_node_mh` | 5 | 0.6733±0.2507 | 96.1% | +1.79 | 0.0740 |  | -1.57 | 0.9047 |  | 0.7006 |
| amazon | gcn | `det_mask` | 5 | 0.6856±0.2577 | 97.9% | +1.89 | 0.0658 |  |  |  |  | 0.7006 |
| amazon | gcn | `det_mask_mh` | 5 | 0.6786±0.2553 | 96.9% | +1.69 | 0.0830 |  | -1.64 | 0.9115 |  | 0.7006 |
| amazon | gcn | `det_mask_no_rel` | 5 | 0.6941±0.2595 | 99.1% | +2.32 | 0.0406 | ★ | +0.84 | 0.2238 |  | 0.7006 |
| amazon | gcn | `det_mask_fixed_bce` | 5 | 0.6927±0.2588 | 98.9% | +2.29 | 0.0422 | ★ | +0.76 | 0.2447 |  | 0.7006 |
| amazon | gcn | `det_mask_rev_only` | 5 | 0.6856±0.2570 | 97.9% | +1.96 | 0.0607 |  | -0.03 | 0.5126 |  | 0.7006 |
| amazon | gcn | `det_mask_single_denom` | 5 | 0.6912±0.2574 | 98.7% | +2.39 | 0.0376 | ★ | +0.40 | 0.3556 |  | 0.7006 |
| amazon | gcn | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.7006 |
| amazon | gcn | `det_mask_cbr` | 5 | 0.6938±0.2604 | 99.0% | +2.16 | 0.0484 | ★ | +1.42 | 0.1148 |  | 0.7006 |
| amazon | gcn | `g_opd_flash` | 5 | 0.6726±0.2502 | 96.0% | +1.77 | 0.0753 |  | -1.70 | 0.9182 |  | 0.7006 |
| amazon | gcn | `opd_action_strict` | 5 | 0.6673±0.2478 | 95.2% | +1.59 | 0.0932 |  | -3.26 | 0.9845 |  | 0.7006 |
| amazon | gcn | `opd_action_strict_mh` | 5 | 0.6377±0.2502 | 91.0% | +0.57 | 0.3007 |  | -1.63 | 0.9109 |  | 0.7006 |
| amazon | gat | `off_policy` | 5 | 0.5164±0.3806 | 92.6% |  |  |  | -1.48 | 0.8938 |  | 0.5575 |
| amazon | gat | `all_node_mh` | 5 | 0.5299±0.3912 | 95.0% | +1.67 | 0.0853 |  | +0.00 | 0.4983 |  | 0.5575 |
| amazon | gat | `det_mask` | 5 | 0.5298±0.3924 | 95.0% | +1.48 | 0.1062 |  |  |  |  | 0.5575 |
| amazon | gat | `det_mask_mh` | 5 | 0.5271±0.3897 | 94.5% | +1.68 | 0.0841 |  | -0.96 | 0.8042 |  | 0.5575 |
| amazon | gat | `det_mask_no_rel` | 5 | 0.5359±0.3972 | 96.1% | +1.22 | 0.1448 |  | +0.73 | 0.2518 |  | 0.5575 |
| amazon | gat | `det_mask_fixed_bce` | 5 | 0.5367±0.3978 | 96.3% | +1.23 | 0.1438 |  | +0.75 | 0.2486 |  | 0.5575 |
| amazon | gat | `det_mask_rev_only` | 5 | 0.5304±0.3929 | 95.1% | +1.54 | 0.0995 |  | +1.34 | 0.1250 |  | 0.5575 |
| amazon | gat | `det_mask_single_denom` | 5 | 0.5371±0.3983 | 96.3% | +1.30 | 0.1313 |  | +0.83 | 0.2254 |  | 0.5575 |
| amazon | gat | `det_mask_crd` | 0 | n/a | n/a |  |  |  |  |  |  | 0.5575 |
| amazon | gat | `det_mask_cbr` | 5 | 0.5316±0.3941 | 95.3% | +1.62 | 0.0905 |  | +1.76 | 0.0762 |  | 0.5575 |
| amazon | gat | `g_opd_flash` | 5 | 0.5264±0.3899 | 94.4% | +1.56 | 0.0964 |  | -1.20 | 0.8510 |  | 0.5575 |
| amazon | gat | `opd_action_strict` | 5 | 0.5277±0.3898 | 94.7% | +1.11 | 0.1652 |  | -0.86 | 0.7800 |  | 0.5575 |
| amazon | gat | `opd_action_strict_mh` | 5 | 0.5176±0.3820 | 92.8% | +0.19 | 0.4285 |  | -1.46 | 0.8903 |  | 0.5575 |

## Cross-cell summary (per mode)

**Statistical convention**: per-cell paired-t only (5-seed, one-sided H1: mode > baseline).  Cells are NOT pooled — pooling treats cells as IID and inflates apparent sample size (Critic round-4 fix).  Teacher AUPRC is cross-seed mean (Critic round-7 CRITICAL fix — was single-seed first-non-NaN value, falsely showing amazon-gcn 0.229).

| Mode | Cells | Mean AUPRC across cells | Directional + vs off_policy | Sig vs off_policy (p<0.05) | Sig vs off_policy (p<0.01) | Directional + vs **det_mask** | Sig vs **det_mask** (p<0.05) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off_policy` | 8 | 0.6407±0.1353 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| `all_node_mh` | 8 | 0.6597±0.1286 | 8/8 | 3/8 | 0/8 | 3/8 | 0/8 |
| `det_mask` | 8 | 0.6714±0.1230 | 8/8 | 3/8 | 2/8 | 0/8 | 0/8 |
| `det_mask_mh` | 8 | 0.6663±0.1249 | 7/8 | 2/8 | 2/8 | 0/8 | 0/8 |
| `det_mask_no_rel` | 8 | 0.6721±0.1229 | 8/8 | 4/8 | 2/8 | 4/8 | 1/8 |
| `det_mask_fixed_bce` | 8 | 0.6704±0.1239 | 8/8 | 4/8 | 2/8 | 3/8 | 0/8 |
| `det_mask_rev_only` | 8 | 0.6707±0.1231 | 8/8 | 2/8 | 2/8 | 2/8 | 0/8 |
| `det_mask_single_denom` | 8 | 0.6710±0.1228 | 8/8 | 4/8 | 2/8 | 4/8 | 0/8 |
| `det_mask_cbr` | 8 | 0.6755±0.1217 | 8/8 | 4/8 | 2/8 | 8/8 | 3/8 |
| `g_opd_flash` | 8 | 0.6585±0.1289 | 8/8 | 4/8 | 1/8 | 2/8 | 0/8 |
| `opd_action_strict` | 8 | 0.6583±0.1280 | 8/8 | 3/8 | 1/8 | 1/8 | 0/8 |
| `opd_action_strict_mh` | 8 | 0.6487±0.1312 | 6/8 | 2/8 | 0/8 | 0/8 | 0/8 |
