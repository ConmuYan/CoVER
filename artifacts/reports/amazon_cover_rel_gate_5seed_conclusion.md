# amazon CoVER-REL Gate 5-Seed Conclusion

- Verdict: **Strong GO**
- Decision reason: anchor_gate matches or exceeds best-single mean Delta AUPRC.
- Best relation: UVU
- Best-single mean Delta AUPRC: 0.003159
- Anchor-gate mean Delta AUPRC: 0.003508
- Base-gate mean Delta AUPRC: 0.003402
- Anchor-gate mean near-cap: 0.912726
- Base-gate mean near-cap: 0.912073
- Conservative anchor-gate mean Delta AUPRC: 0.001785
- Conservative anchor-gate mean near-cap: 0.000000
- Conservative branch accepted: True

## Metrics

| dataset | method | row_type | seed | status | auprc | delta_auprc_vs_base | delta_auprc_vs_best_single | roc_auc | delta_roc_auc_vs_base | macro_f1 | delta_macro_f1_vs_best_single | near_cap_fraction | mean_gate_all_relations | positive_auprc_vs_base_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amazon | base | seed | 42 | complete | 0.868756 | 0.000000 | -0.006095 | 0.971690 | 0.000000 | 0.915169 | 0.000000 |  |  |  |
| amazon | base | seed | 123 | complete | 0.860012 | 0.000000 | -0.000704 | 0.982005 | 0.000000 | 0.916695 | 0.000000 |  |  |  |
| amazon | base | seed | 456 | complete | 0.859061 | 0.000000 | -0.002112 | 0.975258 | 0.000000 | 0.913402 | -0.000531 |  |  |  |
| amazon | base | seed | 789 | complete | 0.812132 | 0.000000 | -0.004003 | 0.915122 | 0.000000 | 0.912445 | -0.001708 |  |  |  |
| amazon | base | seed | 2026 | complete | 0.856040 | 0.000000 | -0.002879 | 0.948515 | 0.000000 | 0.927341 | 0.000000 |  |  |  |
| amazon | base | mean |  | 5/5 complete | 0.851200 | 0.000000 | -0.003159 | 0.958518 | 0.000000 | 0.917010 | -0.000448 |  |  | 0 |
| amazon | base | std |  | 5/5 complete | 0.022346 | 0.000000 | 0.002033 | 0.027334 | 0.000000 | 0.006002 | 0.000741 |  |  | 0 |
| amazon | best_single | seed | 42 | complete | 0.874851 | 0.006095 | 0.000000 | 0.973396 | 0.001706 | 0.915169 | 0.000000 | 0.930258 |  |  |
| amazon | best_single | seed | 123 | complete | 0.860716 | 0.000704 | 0.000000 | 0.982390 | 0.000385 | 0.916695 | 0.000000 | 0.775285 |  |  |
| amazon | best_single | seed | 456 | complete | 0.861173 | 0.002112 | 0.000000 | 0.975958 | 0.000699 | 0.913933 | 0.000000 | 0.937877 |  |  |
| amazon | best_single | seed | 789 | complete | 0.816135 | 0.004003 | 0.000000 | 0.917040 | 0.001918 | 0.914153 | 0.000000 | 0.727227 |  |  |
| amazon | best_single | seed | 2026 | complete | 0.858919 | 0.002879 | 0.000000 | 0.949873 | 0.001358 | 0.927341 | 0.000000 | 0.772019 |  |  |
| amazon | best_single | mean |  | 5/5 complete | 0.854359 | 0.003159 | 0.000000 | 0.959731 | 0.001213 | 0.917458 | 0.000000 | 0.828533 |  | 5 |
| amazon | best_single | std |  | 5/5 complete | 0.022297 | 0.002033 | 0.000000 | 0.026848 | 0.000654 | 0.005631 | 0.000000 | 0.098229 |  | 5 |
| amazon | all_rel | seed | 42 | complete | 0.873167 | 0.004411 | -0.001684 | 0.972983 | 0.001293 | 0.916182 | 0.001013 | 0.944993 |  |  |
| amazon | all_rel | seed | 123 | complete | 0.861424 | 0.001412 | 0.000708 | 0.982305 | 0.000300 | 0.916695 | 0.000000 | 0.921216 |  |  |
| amazon | all_rel | seed | 456 | complete | 0.861570 | 0.002509 | 0.000397 | 0.975986 | 0.000727 | 0.913933 | 0.000000 | 0.949849 |  |  |
| amazon | all_rel | seed | 789 | complete | 0.815072 | 0.002940 | -0.001062 | 0.917065 | 0.001943 | 0.914942 | 0.000789 | 0.918955 |  |  |
| amazon | all_rel | seed | 2026 | complete | 0.859081 | 0.003041 | 0.000162 | 0.950545 | 0.002030 | 0.927341 | 0.000000 | 0.811035 |  |  |
| amazon | all_rel | mean |  | 5/5 complete | 0.854063 | 0.002863 | -0.000296 | 0.959777 | 0.001259 | 0.917819 | 0.000360 | 0.909210 |  | 5 |
| amazon | all_rel | std |  | 5/5 complete | 0.022478 | 0.001080 | 0.001026 | 0.026712 | 0.000753 | 0.005431 | 0.000500 | 0.056589 |  | 5 |
| amazon | anchor_gate | seed | 42 | complete | 0.875109 | 0.006353 | 0.000258 | 0.973676 | 0.001986 | 0.916182 | 0.001013 | 0.921299 | 0.853179 |  |
| amazon | anchor_gate | seed | 123 | complete | 0.860689 | 0.000677 | -0.000027 | 0.982000 | -0.000005 | 0.917698 | 0.001003 | 0.983339 | 0.730905 |  |
| amazon | anchor_gate | seed | 456 | complete | 0.861474 | 0.002413 | 0.000301 | 0.975897 | 0.000639 | 0.913933 | 0.000000 | 0.952026 | 0.992190 |  |
| amazon | anchor_gate | seed | 789 | complete | 0.816736 | 0.004604 | 0.000601 | 0.918125 | 0.003003 | 0.915430 | 0.001277 | 0.805677 | 0.658266 |  |
| amazon | anchor_gate | seed | 2026 | complete | 0.859532 | 0.003492 | 0.000613 | 0.950087 | 0.001572 | 0.927341 | 0.000000 | 0.901289 | 0.939597 |  |
| amazon | anchor_gate | mean |  | 5/5 complete | 0.854708 | 0.003508 | 0.000349 | 0.959957 | 0.001439 | 0.918117 | 0.000659 | 0.912726 | 0.834827 | 5 |
| amazon | anchor_gate | std |  | 5/5 complete | 0.022152 | 0.002151 | 0.000267 | 0.026342 | 0.001171 | 0.005332 | 0.000611 | 0.067444 | 0.139688 | 5 |
| amazon | base_gate | seed | 42 | complete | 0.874126 | 0.005370 | -0.000725 | 0.973437 | 0.001747 | 0.916182 | 0.001013 | 0.915188 | 0.811632 |  |
| amazon | base_gate | seed | 123 | complete | 0.861079 | 0.001067 | 0.000363 | 0.982367 | 0.000362 | 0.916695 | 0.000000 | 0.889568 | 0.876072 |  |
| amazon | base_gate | seed | 456 | complete | 0.862044 | 0.002983 | 0.000871 | 0.976557 | 0.001298 | 0.912921 | -0.001012 | 0.938630 | 0.928395 |  |
| amazon | base_gate | seed | 789 | complete | 0.816558 | 0.004426 | 0.000423 | 0.918764 | 0.003641 | 0.914418 | 0.000265 | 0.860265 | 0.560746 |  |
| amazon | base_gate | seed | 2026 | complete | 0.859202 | 0.003162 | 0.000283 | 0.950184 | 0.001669 | 0.927341 | 0.000000 | 0.956715 | 0.981157 |  |
| amazon | base_gate | mean |  | 5/5 complete | 0.854602 | 0.003402 | 0.000243 | 0.960262 | 0.001744 | 0.917511 | 0.000053 | 0.912073 | 0.831600 | 5 |
| amazon | base_gate | std |  | 5/5 complete | 0.022063 | 0.001629 | 0.000587 | 0.026227 | 0.001195 | 0.005694 | 0.000726 | 0.038401 | 0.163917 | 5 |
| amazon | anchor_gate_conservative | seed | 42 | complete | 0.872194 | 0.003438 | -0.002657 | 0.972624 | 0.000934 | 0.916182 | 0.001013 | 0.000000 | 0.814232 |  |
| amazon | anchor_gate_conservative | seed | 123 | complete | 0.860801 | 0.000789 | 0.000085 | 0.982171 | 0.000167 | 0.917698 | 0.001003 | 0.000000 | 0.568498 |  |
| amazon | anchor_gate_conservative | seed | 456 | complete | 0.860668 | 0.001607 | -0.000505 | 0.975729 | 0.000471 | 0.912921 | -0.001012 | 0.000000 | 0.450387 |  |
| amazon | anchor_gate_conservative | seed | 789 | complete | 0.813765 | 0.001633 | -0.002369 | 0.916008 | 0.000886 | 0.914942 | 0.000789 | 0.000000 | 0.947973 |  |
| amazon | anchor_gate_conservative | seed | 2026 | complete | 0.857498 | 0.001458 | -0.001421 | 0.949341 | 0.000826 | 0.927341 | 0.000000 | 0.000000 | 0.768945 |  |
| amazon | anchor_gate_conservative | mean |  | 5/5 complete | 0.852985 | 0.001785 | -0.001374 | 0.959175 | 0.000657 | 0.917817 | 0.000359 | 0.000000 | 0.710007 | 5 |
| amazon | anchor_gate_conservative | std |  | 5/5 complete | 0.022625 | 0.000986 | 0.001175 | 0.027128 | 0.000329 | 0.005604 | 0.000871 | 0.000000 | 0.198993 | 5 |

