# Amazon CoVER-REL Relation Ablation 3-Seed

- Dataset/model: `amazon/bwgnn`
- Generated git hash: `725bd98`
- Verdict: **Conditional GO**
- Decision reason: AUPRC is flat, but ranking gap improves and Macro-F1 is preserved.
- Best single relation: `UVU`
- Best single relation mean Delta AUPRC: 0.002273
- Best single relation mean Delta Macro-F1: 0.000746
- All-relation mean Delta AUPRC: 0.002287
- Relation utility pattern: distributed_or_weak
- Residual near-cap caution: UPU, ALL mean near-cap >= 0.90

## Interpretation

- YelpChi utility was concentrated in R-U-R; Amazon is evaluated schema-first over UPU/USU/UVU without YelpChi relation-name assumptions.
- Amazon best relation is `UVU`; compare its sign and magnitude against YelpChi RUR-only mean Delta AUPRC +0.027099.
- Qwen latents were not used in this task; Stage3 remains LLM-free.

## Metrics

| relation | row_type | seed | status | delta_auprc | delta_roc_auc | delta_macro_f1_at_val | delta_f1_at_val | ranking_gap_delta | residual_shift_mean | residual_shift_max_abs | near_cap_fraction | positive_auprc_seeds | relations | prototype_labels | test_label_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upu | seed | 123 | complete | 0.000389 | 0.000009 | 0.001003 | 0.001895 | -0.041082 | 0.196884 | 0.200001 | 0.976725 |  | UPU | train_only | False |
| upu | seed | 456 | complete | 0.000929 | 0.000445 | 0.000531 | 0.001067 | 0.054921 | 0.172520 | 0.200000 | 0.853985 |  | UPU | train_only | False |
| upu | seed | 789 | complete | 0.003483 | 0.001337 | 0.003245 | 0.006040 | 0.354841 | 0.123594 | 0.200001 | 0.933272 |  | UPU | train_only | False |
| upu | mean |  | 3/3 complete | 0.001600 | 0.000597 | 0.001593 | 0.003001 | 0.122893 | 0.164333 | 0.200001 | 0.921327 | 3 |  |  |  |
| upu | std |  | 3/3 complete | 0.001652 | 0.000677 | 0.001450 | 0.002664 | 0.206529 | 0.037324 | 0.000000 | 0.062235 | 3 |  |  |  |
| usu | seed | 123 | complete | -0.000020 | -0.000010 | 0.000000 | 0.000000 | -0.097774 | 0.167019 | 0.199621 | 0.474046 |  | USU | train_only | False |
| usu | seed | 456 | complete | 0.002328 | 0.000576 | 0.000531 | 0.001067 | 0.090202 | 0.171194 | 0.200000 | 0.842348 |  | USU | train_only | False |
| usu | seed | 789 | complete | 0.003409 | 0.002091 | 0.001708 | 0.003072 | 0.073325 | 0.063366 | 0.200001 | 0.677662 |  | USU | train_only | False |
| usu | mean |  | 3/3 complete | 0.001906 | 0.000886 | 0.000746 | 0.001380 | 0.021918 | 0.133860 | 0.199874 | 0.664685 | 2 |  |  |  |
| usu | std |  | 3/3 complete | 0.001753 | 0.001084 | 0.000874 | 0.001560 | 0.103999 | 0.061085 | 0.000219 | 0.184494 | 2 |  |  |  |
| uvu | seed | 123 | complete | 0.000704 | 0.000385 | 0.000000 | 0.000000 | 0.076273 | 0.160007 | 0.200001 | 0.775285 |  | UVU | train_only | False |
| uvu | seed | 456 | complete | 0.002112 | 0.000699 | 0.000531 | 0.001067 | 0.246742 | 0.148528 | 0.200000 | 0.937877 |  | UVU | train_only | False |
| uvu | seed | 789 | complete | 0.004003 | 0.001918 | 0.001708 | 0.003072 | 0.124401 | 0.083289 | 0.200001 | 0.727227 |  | UVU | train_only | False |
| uvu | mean |  | 3/3 complete | 0.002273 | 0.001001 | 0.000746 | 0.001380 | 0.149139 | 0.130608 | 0.200001 | 0.813463 | 3 |  |  |  |
| uvu | std |  | 3/3 complete | 0.001655 | 0.000810 | 0.000874 | 0.001560 | 0.087886 | 0.041379 | 0.000000 | 0.110393 | 3 |  |  |  |
| all | seed | 123 | complete | 0.001412 | 0.000300 | 0.000000 | 0.000000 | 0.072692 | 0.177645 | 0.200001 | 0.921216 |  | UPU,USU,UVU | train_only | False |
| all | seed | 456 | complete | 0.002509 | 0.000727 | 0.000531 | 0.001067 | 0.189197 | 0.157343 | 0.200000 | 0.949849 |  | UPU,USU,UVU | train_only | False |
| all | seed | 789 | complete | 0.002940 | 0.001943 | 0.002497 | 0.004657 | 0.361617 | 0.095618 | 0.200001 | 0.918955 |  | UPU,USU,UVU | train_only | False |
| all | mean |  | 3/3 complete | 0.002287 | 0.000990 | 0.001009 | 0.001908 | 0.207835 | 0.143536 | 0.200001 | 0.930007 | 3 |  |  |  |
| all | std |  | 3/3 complete | 0.000788 | 0.000852 | 0.001315 | 0.002440 | 0.145361 | 0.042721 | 0.000000 | 0.017221 | 3 |  |  |  |

