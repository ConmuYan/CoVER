# CoVER-FD Experiment Results — Idea 1 + Idea 2B (TPAMI-grade)

**Seeds**: 42, 123, 456, 789, 2026  |  **df** = 4  |  **Sig**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★

---

## 1. Idea-1 CoVER-REL Canonical — 8 cells × 5 seeds (mean ± sd)

| Cell | AUPRC | AUROC | M-F1 | G-Means |
|---|---:|---:|---:|---:|
| yelpchi-bwgnn | 0.6089 ± 0.0083 | 0.8826 ± 0.0043 | 0.7495 ± 0.0059 | 0.7301 ± 0.0236 |
| yelpchi-sage | 0.6001 ± 0.0105 | 0.8787 ± 0.0029 | 0.7466 ± 0.0051 | 0.7485 ± 0.0122 |
| yelpchi-gcn | 0.4946 ± 0.0149 | 0.8541 ± 0.0052 | 0.7150 ± 0.0092 | 0.7209 ± 0.0206 |
| yelpchi-gat | 0.5316 ± 0.0136 | 0.8632 ± 0.0064 | 0.7249 ± 0.0065 | 0.7285 ± 0.0196 |
| amazon-bwgnn | 0.8683 ± 0.0269 | 0.9755 ± 0.0114 | 0.9174 ± 0.0052 | 0.8851 ± 0.0105 |
| amazon-sage | 0.8336 ± 0.0546 | 0.9553 ± 0.0231 | 0.8986 ± 0.0311 | 0.8586 ± 0.0423 |
| amazon-gcn | 0.4668 ± 0.1356 | 0.8646 ± 0.0433 | 0.7267 ± 0.0536 | 0.6781 ± 0.0698 |
| amazon-gat | 0.4592 ± 0.3055 | 0.8048 ± 0.1488 | 0.6900 ± 0.1865 | 0.5034 ± 0.3981 |

---

## 2. Idea-1 Ablation — 8 cells × 3 switches × 5 seeds (paired-t vs canonical)

### 2.1 gate_uniform (softmax → 1/R)

| Cell | Metric | Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6089 ± 0.0083 | 0.6060 ± 0.0111 | -0.0029 | +1.345 | ns |
| yelpchi-bwgnn | AUROC | 0.8826 ± 0.0043 | 0.8799 ± 0.0047 | -0.0027 | +5.619 | ★★ |
| yelpchi-bwgnn | M-F1 | 0.7495 ± 0.0059 | 0.7468 ± 0.0060 | -0.0026 | +2.578 | trend |
| yelpchi-bwgnn | G-Means | 0.7301 ± 0.0236 | 0.7270 ± 0.0136 | -0.0032 | +0.349 | ns |
| yelpchi-sage | AUPRC | 0.6001 ± 0.0105 | 0.5860 ± 0.0097 | -0.0141 | +5.856 | ★★ |
| yelpchi-sage | AUROC | 0.8787 ± 0.0029 | 0.8717 ± 0.0040 | -0.0070 | +8.316 | ★★ |
| yelpchi-sage | M-F1 | 0.7466 ± 0.0051 | 0.7396 ± 0.0060 | -0.0069 | +4.136 | ★ |
| yelpchi-sage | G-Means | 0.7485 ± 0.0122 | 0.7319 ± 0.0109 | -0.0166 | +3.650 | ★ |
| yelpchi-gcn | AUPRC | 0.4946 ± 0.0149 | 0.4947 ± 0.0121 | +0.0000 | -0.008 | ns |
| yelpchi-gcn | AUROC | 0.8541 ± 0.0052 | 0.8493 ± 0.0040 | -0.0048 | +5.520 | ★★ |
| yelpchi-gcn | M-F1 | 0.7150 ± 0.0092 | 0.7120 ± 0.0068 | -0.0030 | +0.786 | ns |
| yelpchi-gcn | G-Means | 0.7209 ± 0.0206 | 0.7180 ± 0.0131 | -0.0028 | +0.371 | ns |
| yelpchi-gat | AUPRC | 0.5316 ± 0.0136 | 0.5072 ± 0.0117 | -0.0244 | +3.609 | ★ |
| yelpchi-gat | AUROC | 0.8632 ± 0.0064 | 0.8482 ± 0.0051 | -0.0150 | +7.788 | ★★ |
| yelpchi-gat | M-F1 | 0.7249 ± 0.0065 | 0.7103 ± 0.0078 | -0.0145 | +6.342 | ★★ |
| yelpchi-gat | G-Means | 0.7285 ± 0.0196 | 0.7170 ± 0.0208 | -0.0114 | +1.067 | ns |
| amazon-bwgnn | AUPRC | 0.8683 ± 0.0269 | 0.8669 ± 0.0268 | -0.0015 | +0.365 | ns |
| amazon-bwgnn | AUROC | 0.9755 ± 0.0114 | 0.9752 ± 0.0120 | -0.0003 | +0.367 | ns |
| amazon-bwgnn | M-F1 | 0.9174 ± 0.0052 | 0.9142 ± 0.0069 | -0.0032 | +2.924 | ★ |
| amazon-bwgnn | G-Means | 0.8851 ± 0.0105 | 0.8798 ± 0.0173 | -0.0053 | +1.223 | ns |
| amazon-sage | AUPRC | 0.8336 ± 0.0546 | 0.8161 ± 0.0808 | -0.0175 | +1.460 | ns |
| amazon-sage | AUROC | 0.9553 ± 0.0231 | 0.9508 ± 0.0245 | -0.0044 | +4.438 | ★ |
| amazon-sage | M-F1 | 0.8986 ± 0.0311 | 0.8885 ± 0.0454 | -0.0102 | +1.449 | ns |
| amazon-sage | G-Means | 0.8586 ± 0.0423 | 0.8500 ± 0.0607 | -0.0086 | +0.954 | ns |
| amazon-gcn | AUPRC | 0.4668 ± 0.1356 | 0.4796 ± 0.1403 | +0.0128 | -1.076 | ns |
| amazon-gcn | AUROC | 0.8646 ± 0.0433 | 0.8716 ± 0.0472 | +0.0070 | -2.169 | trend |
| amazon-gcn | M-F1 | 0.7267 ± 0.0536 | 0.7318 ± 0.0554 | +0.0051 | -0.976 | ns |
| amazon-gcn | G-Means | 0.6781 ± 0.0698 | 0.6942 ± 0.0694 | +0.0161 | -0.700 | ns |
| amazon-gat | AUPRC | 0.4592 ± 0.3055 | 0.4780 ± 0.3094 | +0.0189 | -2.131 | trend |
| amazon-gat | AUROC | 0.8048 ± 0.1488 | 0.8132 ± 0.1344 | +0.0084 | -0.893 | ns |
| amazon-gat | M-F1 | 0.6900 ± 0.1865 | 0.6875 ± 0.1839 | -0.0025 | +0.458 | ns |
| amazon-gat | G-Means | 0.5034 ± 0.3981 | 0.4923 ± 0.3878 | -0.0110 | +0.837 | ns |

### 2.2 shared_expert (R MLPs → 1 shared + one-hot)

| Cell | Metric | Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6089 ± 0.0083 | 0.6064 ± 0.0107 | -0.0026 | +1.193 | ns |
| yelpchi-bwgnn | AUROC | 0.8826 ± 0.0043 | 0.8824 ± 0.0032 | -0.0001 | +0.202 | ns |
| yelpchi-bwgnn | M-F1 | 0.7495 ± 0.0059 | 0.7482 ± 0.0049 | -0.0012 | +1.431 | ns |
| yelpchi-bwgnn | G-Means | 0.7301 ± 0.0236 | 0.7288 ± 0.0211 | -0.0014 | +0.157 | ns |
| yelpchi-sage | AUPRC | 0.6001 ± 0.0105 | 0.5978 ± 0.0038 | -0.0023 | +0.404 | ns |
| yelpchi-sage | AUROC | 0.8787 ± 0.0029 | 0.8785 ± 0.0013 | -0.0002 | +0.250 | ns |
| yelpchi-sage | M-F1 | 0.7466 ± 0.0051 | 0.7454 ± 0.0037 | -0.0012 | +0.425 | ns |
| yelpchi-sage | G-Means | 0.7485 ± 0.0122 | 0.7348 ± 0.0217 | -0.0137 | +1.734 | ns |
| yelpchi-gcn | AUPRC | 0.4946 ± 0.0149 | 0.5002 ± 0.0206 | +0.0056 | -0.771 | ns |
| yelpchi-gcn | AUROC | 0.8541 ± 0.0052 | 0.8588 ± 0.0077 | +0.0047 | -1.726 | ns |
| yelpchi-gcn | M-F1 | 0.7150 ± 0.0092 | 0.7195 ± 0.0091 | +0.0045 | -1.089 | ns |
| yelpchi-gcn | G-Means | 0.7209 ± 0.0206 | 0.7444 ± 0.0144 | +0.0236 | -1.747 | ns |
| yelpchi-gat | AUPRC | 0.5316 ± 0.0136 | 0.5134 ± 0.0181 | -0.0182 | +2.881 | ★ |
| yelpchi-gat | AUROC | 0.8632 ± 0.0064 | 0.8623 ± 0.0046 | -0.0009 | +0.620 | ns |
| yelpchi-gat | M-F1 | 0.7249 ± 0.0065 | 0.7246 ± 0.0061 | -0.0002 | +0.080 | ns |
| yelpchi-gat | G-Means | 0.7285 ± 0.0196 | 0.7400 ± 0.0106 | +0.0115 | -1.144 | ns |
| amazon-bwgnn | AUPRC | 0.8683 ± 0.0269 | 0.8659 ± 0.0268 | -0.0024 | +1.333 | ns |
| amazon-bwgnn | AUROC | 0.9755 ± 0.0114 | 0.9747 ± 0.0118 | -0.0008 | +1.900 | ns |
| amazon-bwgnn | M-F1 | 0.9174 ± 0.0052 | 0.9149 ± 0.0033 | -0.0025 | +1.058 | ns |
| amazon-bwgnn | G-Means | 0.8851 ± 0.0105 | 0.8874 ± 0.0158 | +0.0024 | -0.668 | ns |
| amazon-sage | AUPRC | 0.8336 ± 0.0546 | 0.8351 ± 0.0449 | +0.0015 | -0.307 | ns |
| amazon-sage | AUROC | 0.9553 ± 0.0231 | 0.9571 ± 0.0146 | +0.0019 | -0.416 | ns |
| amazon-sage | M-F1 | 0.8986 ± 0.0311 | 0.8999 ± 0.0273 | +0.0013 | -0.638 | ns |
| amazon-sage | G-Means | 0.8586 ± 0.0423 | 0.8610 ± 0.0323 | +0.0024 | -0.393 | ns |
| amazon-gcn | AUPRC | 0.4668 ± 0.1356 | 0.4614 ± 0.1318 | -0.0055 | +0.501 | ns |
| amazon-gcn | AUROC | 0.8646 ± 0.0433 | 0.8670 ± 0.0454 | +0.0024 | -1.014 | ns |
| amazon-gcn | M-F1 | 0.7267 ± 0.0536 | 0.7250 ± 0.0521 | -0.0017 | +0.263 | ns |
| amazon-gcn | G-Means | 0.6781 ± 0.0698 | 0.6875 ± 0.0650 | +0.0094 | -0.398 | ns |
| amazon-gat | AUPRC | 0.4592 ± 0.3055 | 0.4522 ± 0.3182 | -0.0069 | +0.514 | ns |
| amazon-gat | AUROC | 0.8048 ± 0.1488 | 0.8116 ± 0.1462 | +0.0068 | -2.286 | trend |
| amazon-gat | M-F1 | 0.6900 ± 0.1865 | 0.6894 ± 0.1864 | -0.0006 | +0.092 | ns |
| amazon-gat | G-Means | 0.5034 ± 0.3981 | 0.5148 ± 0.4074 | +0.0114 | -0.807 | ns |

### 2.3 no_proto (drop C sub-space)

| Cell | Metric | Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6089 ± 0.0083 | 0.6092 ± 0.0095 | +0.0002 | -0.233 | ns |
| yelpchi-bwgnn | AUROC | 0.8826 ± 0.0043 | 0.8831 ± 0.0046 | +0.0006 | -1.060 | ns |
| yelpchi-bwgnn | M-F1 | 0.7495 ± 0.0059 | 0.7494 ± 0.0068 | -0.0000 | +0.024 | ns |
| yelpchi-bwgnn | G-Means | 0.7301 ± 0.0236 | 0.7412 ± 0.0254 | +0.0111 | -0.635 | ns |
| yelpchi-sage | AUPRC | 0.6001 ± 0.0105 | 0.5868 ± 0.0079 | -0.0133 | +7.072 | ★★ |
| yelpchi-sage | AUROC | 0.8787 ± 0.0029 | 0.8731 ± 0.0032 | -0.0056 | +7.030 | ★★ |
| yelpchi-sage | M-F1 | 0.7466 ± 0.0051 | 0.7384 ± 0.0065 | -0.0082 | +2.635 | trend |
| yelpchi-sage | G-Means | 0.7485 ± 0.0122 | 0.7344 ± 0.0150 | -0.0140 | +6.691 | ★★ |
| yelpchi-gcn | AUPRC | 0.4946 ± 0.0149 | 0.3992 ± 0.0075 | -0.0954 | +17.001 | ★★★ |
| yelpchi-gcn | AUROC | 0.8541 ± 0.0052 | 0.8018 ± 0.0034 | -0.0523 | +19.260 | ★★★ |
| yelpchi-gcn | M-F1 | 0.7150 ± 0.0092 | 0.6594 ± 0.0053 | -0.0556 | +10.943 | ★★★ |
| yelpchi-gcn | G-Means | 0.7209 ± 0.0206 | 0.6454 ± 0.0187 | -0.0755 | +4.656 | ★★ |
| yelpchi-gat | AUPRC | 0.5316 ± 0.0136 | 0.3976 ± 0.0070 | -0.1340 | +16.024 | ★★★ |
| yelpchi-gat | AUROC | 0.8632 ± 0.0064 | 0.7995 ± 0.0048 | -0.0637 | +14.029 | ★★★ |
| yelpchi-gat | M-F1 | 0.7249 ± 0.0065 | 0.6549 ± 0.0075 | -0.0700 | +12.462 | ★★★ |
| yelpchi-gat | G-Means | 0.7285 ± 0.0196 | 0.6372 ± 0.0212 | -0.0912 | +6.580 | ★★ |
| amazon-bwgnn | AUPRC | 0.8683 ± 0.0269 | 0.8665 ± 0.0272 | -0.0018 | +0.951 | ns |
| amazon-bwgnn | AUROC | 0.9755 ± 0.0114 | 0.9738 ± 0.0136 | -0.0017 | +1.642 | ns |
| amazon-bwgnn | M-F1 | 0.9174 ± 0.0052 | 0.9149 ± 0.0066 | -0.0024 | +0.941 | ns |
| amazon-bwgnn | G-Means | 0.8851 ± 0.0105 | 0.8893 ± 0.0173 | +0.0042 | -0.708 | ns |
| amazon-sage | AUPRC | 0.8336 ± 0.0546 | 0.8335 ± 0.0549 | -0.0001 | +0.085 | ns |
| amazon-sage | AUROC | 0.9553 ± 0.0231 | 0.9540 ± 0.0242 | -0.0012 | +1.745 | ns |
| amazon-sage | M-F1 | 0.8986 ± 0.0311 | 0.9006 ± 0.0305 | +0.0020 | -1.872 | ns |
| amazon-sage | G-Means | 0.8586 ± 0.0423 | 0.8617 ± 0.0336 | +0.0031 | -0.571 | ns |
| amazon-gcn | AUPRC | 0.4668 ± 0.1356 | 0.4565 ± 0.1294 | -0.0104 | +0.889 | ns |
| amazon-gcn | AUROC | 0.8646 ± 0.0433 | 0.8697 ± 0.0461 | +0.0052 | -2.502 | trend |
| amazon-gcn | M-F1 | 0.7267 ± 0.0536 | 0.7221 ± 0.0507 | -0.0045 | +0.809 | ns |
| amazon-gcn | G-Means | 0.6781 ± 0.0698 | 0.6863 ± 0.0648 | +0.0082 | -0.486 | ns |
| amazon-gat | AUPRC | 0.4592 ± 0.3055 | 0.4572 ± 0.3066 | -0.0020 | +0.309 | ns |
| amazon-gat | AUROC | 0.8048 ± 0.1488 | 0.8086 ± 0.1389 | +0.0038 | -0.603 | ns |
| amazon-gat | M-F1 | 0.6900 ± 0.1865 | 0.6853 ± 0.1823 | -0.0047 | +2.161 | trend |
| amazon-gat | G-Means | 0.5034 ± 0.3981 | 0.4917 ± 0.3872 | -0.0117 | +1.840 | ns |

---

## 3. Idea-2B Learned Extractor — 8 cells × 5 seeds (paired-t vs Idea-1 canonical)

### 3.1 Per-cell metrics (mean ± sd)

| Cell | Metric | Idea-1 Canon | 2B Learned | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6089 ± 0.0083 | 0.6489 ± 0.0132 | +0.0399 | -11.486 | ★★★ |
| yelpchi-bwgnn | AUROC | 0.8826 ± 0.0043 | 0.8986 ± 0.0032 | +0.0160 | -16.455 | ★★★ |
| yelpchi-bwgnn | M-F1 | 0.7495 ± 0.0059 | 0.7692 ± 0.0065 | +0.0198 | -7.234 | ★★ |
| yelpchi-bwgnn | G-Means | 0.7301 ± 0.0236 | 0.7621 ± 0.0227 | +0.0319 | -2.392 | trend |
| yelpchi-sage | AUPRC | 0.6001 ± 0.0105 | 0.6520 ± 0.0072 | +0.0519 | -10.015 | ★★★ |
| yelpchi-sage | AUROC | 0.8787 ± 0.0029 | 0.8992 ± 0.0020 | +0.0205 | -13.404 | ★★★ |
| yelpchi-sage | M-F1 | 0.7466 ± 0.0051 | 0.7764 ± 0.0045 | +0.0298 | -15.963 | ★★★ |
| yelpchi-sage | G-Means | 0.7485 ± 0.0122 | 0.7611 ± 0.0083 | +0.0127 | -6.577 | ★★ |
| yelpchi-gcn | AUPRC | 0.4946 ± 0.0149 | 0.5904 ± 0.0102 | +0.0958 | -12.407 | ★★★ |
| yelpchi-gcn | AUROC | 0.8541 ± 0.0052 | 0.8944 ± 0.0030 | +0.0403 | -13.909 | ★★★ |
| yelpchi-gcn | M-F1 | 0.7150 ± 0.0092 | 0.7707 ± 0.0046 | +0.0557 | -14.856 | ★★★ |
| yelpchi-gcn | G-Means | 0.7209 ± 0.0206 | 0.7770 ± 0.0152 | +0.0562 | -6.048 | ★★ |
| yelpchi-gat | AUPRC | 0.5316 ± 0.0136 | 0.6340 ± 0.0122 | +0.1024 | -9.612 | ★★★ |
| yelpchi-gat | AUROC | 0.8632 ± 0.0064 | 0.9017 ± 0.0026 | +0.0385 | -11.242 | ★★★ |
| yelpchi-gat | M-F1 | 0.7249 ± 0.0065 | 0.7742 ± 0.0036 | +0.0493 | -14.762 | ★★★ |
| yelpchi-gat | G-Means | 0.7285 ± 0.0196 | 0.7754 ± 0.0074 | +0.0470 | -5.815 | ★★ |
| amazon-bwgnn | AUPRC | 0.8683 ± 0.0269 | 0.8671 ± 0.0271 | -0.0012 | +0.556 | ns |
| amazon-bwgnn | AUROC | 0.9755 ± 0.0114 | 0.9764 ± 0.0086 | +0.0009 | -0.553 | ns |
| amazon-bwgnn | M-F1 | 0.9174 ± 0.0052 | 0.9175 ± 0.0049 | +0.0002 | -0.152 | ns |
| amazon-bwgnn | G-Means | 0.8851 ± 0.0105 | 0.8834 ± 0.0147 | -0.0016 | +0.694 | ns |
| amazon-sage | AUPRC | 0.8336 ± 0.0546 | 0.8511 ± 0.0146 | +0.0175 | -0.719 | ns |
| amazon-sage | AUROC | 0.9553 ± 0.0231 | 0.9624 ± 0.0066 | +0.0071 | -0.771 | ns |
| amazon-sage | M-F1 | 0.8986 ± 0.0311 | 0.9091 ± 0.0121 | +0.0104 | -1.196 | ns |
| amazon-sage | G-Means | 0.8586 ± 0.0423 | 0.8689 ± 0.0206 | +0.0103 | -0.966 | ns |
| amazon-gcn | AUPRC | 0.4668 ± 0.1356 | 0.7006 ± 0.2638 | +0.2337 | -3.796 | ★ |
| amazon-gcn | AUROC | 0.8646 ± 0.0433 | 0.9221 ± 0.0753 | +0.0575 | -3.627 | ★ |
| amazon-gcn | M-F1 | 0.7267 ± 0.0536 | 0.8419 ± 0.1191 | +0.1152 | -3.752 | ★ |
| amazon-gcn | G-Means | 0.6781 ± 0.0698 | 0.8161 ± 0.1317 | +0.1380 | -3.691 | ★ |
| amazon-gat | AUPRC | 0.4592 ± 0.3055 | 0.5575 ± 0.3810 | +0.0984 | -1.708 | ns |
| amazon-gat | AUROC | 0.8048 ± 0.1488 | 0.8217 ± 0.1751 | +0.0169 | -1.104 | ns |
| amazon-gat | M-F1 | 0.6900 ± 0.1865 | 0.7395 ± 0.2265 | +0.0494 | -1.671 | ns |
| amazon-gat | G-Means | 0.5034 ± 0.3981 | 0.5506 ± 0.4355 | +0.0472 | -1.697 | ns |

### 3.2 Cross-cell sig summary

| Metric | wins(sig) | wins(ns) | losses | total |
|---|:---:|:---:|:---:|:---:|
| AUPRC | 0 | 3 | 5 | 8 |
| AUROC | 0 | 3 | 5 | 8 |
| M-F1 | 0 | 3 | 5 | 8 |
| G-Means | 0 | 4 | 4 | 8 |

---

## 4. Idea-2B Module Ablation — YelpChi × 4 bases × 3 switches × 5 seeds

### 4.1 drop_gcn (remove GCN encoder)

| Cell | Metric | 2B Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6489 ± 0.0132 | 0.6371 ± 0.0092 | -0.0118 | +2.368 | trend |
| yelpchi-bwgnn | AUROC | 0.8986 ± 0.0032 | 0.8945 ± 0.0027 | -0.0041 | +3.110 | ★ |
| yelpchi-bwgnn | M-F1 | 0.7692 ± 0.0065 | 0.7664 ± 0.0045 | -0.0029 | +1.770 | ns |
| yelpchi-bwgnn | G-Means | 0.7621 ± 0.0227 | 0.7612 ± 0.0214 | -0.0009 | +0.060 | ns |
| yelpchi-sage | AUPRC | 0.6520 ± 0.0072 | 0.6454 ± 0.0104 | -0.0066 | +3.489 | ★ |
| yelpchi-sage | AUROC | 0.8992 ± 0.0020 | 0.8967 ± 0.0035 | -0.0026 | +2.018 | ns |
| yelpchi-sage | M-F1 | 0.7764 ± 0.0045 | 0.7725 ± 0.0072 | -0.0038 | +1.601 | ns |
| yelpchi-sage | G-Means | 0.7611 ± 0.0083 | 0.7763 ± 0.0123 | +0.0152 | -4.704 | ★★ |
| yelpchi-gcn | AUPRC | 0.5904 ± 0.0102 | 0.5936 ± 0.0184 | +0.0032 | -0.485 | ns |
| yelpchi-gcn | AUROC | 0.8944 ± 0.0030 | 0.8940 ± 0.0031 | -0.0004 | +0.175 | ns |
| yelpchi-gcn | M-F1 | 0.7707 ± 0.0046 | 0.7688 ± 0.0042 | -0.0019 | +0.547 | ns |
| yelpchi-gcn | G-Means | 0.7770 ± 0.0152 | 0.7669 ± 0.0183 | -0.0101 | +1.338 | ns |
| yelpchi-gat | AUPRC | 0.6340 ± 0.0122 | 0.6297 ± 0.0071 | -0.0043 | +0.694 | ns |
| yelpchi-gat | AUROC | 0.9017 ± 0.0026 | 0.8996 ± 0.0010 | -0.0021 | +2.129 | ns |
| yelpchi-gat | M-F1 | 0.7742 ± 0.0036 | 0.7720 ± 0.0010 | -0.0022 | +1.560 | ns |
| yelpchi-gat | G-Means | 0.7754 ± 0.0074 | 0.7767 ± 0.0081 | +0.0013 | -0.189 | ns |

### 4.2 drop_proto (remove proto features)

| Cell | Metric | 2B Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6489 ± 0.0132 | 0.6561 ± 0.0150 | +0.0072 | -1.050 | ns |
| yelpchi-bwgnn | AUROC | 0.8986 ± 0.0032 | 0.9010 ± 0.0033 | +0.0024 | -1.321 | ns |
| yelpchi-bwgnn | M-F1 | 0.7692 ± 0.0065 | 0.7749 ± 0.0073 | +0.0057 | -2.494 | trend |
| yelpchi-bwgnn | G-Means | 0.7621 ± 0.0227 | 0.7718 ± 0.0115 | +0.0097 | -0.957 | ns |
| yelpchi-sage | AUPRC | 0.6520 ± 0.0072 | 0.6509 ± 0.0051 | -0.0011 | +0.228 | ns |
| yelpchi-sage | AUROC | 0.8992 ± 0.0020 | 0.9004 ± 0.0024 | +0.0011 | -0.625 | ns |
| yelpchi-sage | M-F1 | 0.7764 ± 0.0045 | 0.7752 ± 0.0021 | -0.0012 | +0.493 | ns |
| yelpchi-sage | G-Means | 0.7611 ± 0.0083 | 0.7667 ± 0.0200 | +0.0056 | -0.877 | ns |
| yelpchi-gcn | AUPRC | 0.5904 ± 0.0102 | 0.6106 ± 0.0148 | +0.0202 | -6.462 | ★★ |
| yelpchi-gcn | AUROC | 0.8944 ± 0.0030 | 0.8975 ± 0.0025 | +0.0030 | -2.113 | ns |
| yelpchi-gcn | M-F1 | 0.7707 ± 0.0046 | 0.7738 ± 0.0045 | +0.0031 | -0.984 | ns |
| yelpchi-gcn | G-Means | 0.7770 ± 0.0152 | 0.7687 ± 0.0114 | -0.0083 | +1.039 | ns |
| yelpchi-gat | AUPRC | 0.6340 ± 0.0122 | 0.6469 ± 0.0070 | +0.0129 | -1.645 | ns |
| yelpchi-gat | AUROC | 0.9017 ± 0.0026 | 0.9037 ± 0.0022 | +0.0020 | -1.198 | ns |
| yelpchi-gat | M-F1 | 0.7742 ± 0.0036 | 0.7769 ± 0.0026 | +0.0027 | -1.038 | ns |
| yelpchi-gat | G-Means | 0.7754 ± 0.0074 | 0.7796 ± 0.0113 | +0.0042 | -0.750 | ns |

### 4.3 encoder_shared (shared encoder + one-hot)

| Cell | Metric | 2B Canonical | Ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| yelpchi-bwgnn | AUPRC | 0.6489 ± 0.0132 | 0.6422 ± 0.0067 | -0.0067 | +1.681 | ns |
| yelpchi-bwgnn | AUROC | 0.8986 ± 0.0032 | 0.8960 ± 0.0034 | -0.0026 | +1.772 | ns |
| yelpchi-bwgnn | M-F1 | 0.7692 ± 0.0065 | 0.7692 ± 0.0033 | -0.0001 | +0.036 | ns |
| yelpchi-bwgnn | G-Means | 0.7621 ± 0.0227 | 0.7623 ± 0.0112 | +0.0002 | -0.015 | ns |
| yelpchi-sage | AUPRC | 0.6520 ± 0.0072 | 0.6486 ± 0.0068 | -0.0034 | +0.911 | ns |
| yelpchi-sage | AUROC | 0.8992 ± 0.0020 | 0.8988 ± 0.0038 | -0.0004 | +0.154 | ns |
| yelpchi-sage | M-F1 | 0.7764 ± 0.0045 | 0.7734 ± 0.0024 | -0.0030 | +1.425 | ns |
| yelpchi-sage | G-Means | 0.7611 ± 0.0083 | 0.7876 ± 0.0077 | +0.0265 | -4.227 | ★ |
| yelpchi-gcn | AUPRC | 0.5904 ± 0.0102 | 0.6015 ± 0.0162 | +0.0111 | -3.971 | ★ |
| yelpchi-gcn | AUROC | 0.8944 ± 0.0030 | 0.8949 ± 0.0031 | +0.0004 | -0.203 | ns |
| yelpchi-gcn | M-F1 | 0.7707 ± 0.0046 | 0.7721 ± 0.0049 | +0.0014 | -0.473 | ns |
| yelpchi-gcn | G-Means | 0.7770 ± 0.0152 | 0.7686 ± 0.0141 | -0.0084 | +1.217 | ns |
| yelpchi-gat | AUPRC | 0.6340 ± 0.0122 | 0.6371 ± 0.0167 | +0.0031 | -0.401 | ns |
| yelpchi-gat | AUROC | 0.9017 ± 0.0026 | 0.8993 ± 0.0035 | -0.0024 | +1.124 | ns |
| yelpchi-gat | M-F1 | 0.7742 ± 0.0036 | 0.7731 ± 0.0072 | -0.0011 | +0.457 | ns |
| yelpchi-gat | G-Means | 0.7754 ± 0.0074 | 0.7687 ± 0.0113 | -0.0067 | +1.125 | ns |

### 4.4 Cross-metric summary (AUPRC)

| Switch | sig hurt | sig help | ns | total |
|---|:---:|:---:|:---:|:---:|
| drop_gcn | 1 | 0 | 3 | 4 |
| drop_proto | 0 | 1 | 3 | 4 |
| encoder_shared | 0 | 1 | 3 | 4 |

---

## 5. Key Findings

### 5.1 Base-strength × evidence-type interaction law (Idea-1)

- Proto (C) **load-bearing on weak bases**: YelpChi-GAT no_proto = −0.134 AUPRC ★★★ (t=−16); YelpChi-GCN = −0.095 ★★★. **Inert on saturated bases**: Amazon-BWGNN = −0.002 ns.

- Schema gate load-bearing on YelpChi (4/4 cells AUROC ★★), inert on Amazon strong bases.

- Shared MLP ≡ independent experts (0/32 sig AUROC).

### 5.2 Learned > hand-crafted (Idea-2B)

- **19/32 stat-sig wins**. YelpChi: 15/16 sig. Amazon-GCN: +0.234 AUPRC ★.

- Saturated Amazon bases: learned ≈ hand-crafted (no harm).

### 5.3 GCN encoder absorbs proto function (Idea-2B ablation)

- YelpChi-GCN drop_proto: Idea-1 = **−0.095 ★★★** (proto critical); Idea-2B = **+0.020 ★★** (proto harmful).

- Learned GCN encoder internalizes fraud/benign discrimination → proto becomes redundant noise.

---

*Total: 260 runs (Idea-1: 160 + Idea-2B canonical: 40 + Idea-2B ablation: 60). Generated automatically.*

