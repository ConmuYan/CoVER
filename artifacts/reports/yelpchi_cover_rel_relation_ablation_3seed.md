# YelpChi CoVER-REL Relation Ablation 3-Seed

- Best relation set by mean Delta AUPRC: `rur` (0.024725)
- All relations mean Delta AUPRC: 0.019772
- RUR mean Delta AUPRC: 0.024725
- RSR mean Delta AUPRC: -0.000259
- RTR mean Delta AUPRC: 0.001064

## Table

| relation_set | stage3_run_name | mean_base_auprc | mean_cover_auprc | mean_delta_auprc | mean_delta_roc_auc | mean_delta_macro_f1 | mean_ranking_gap | mean_ranking_gap_delta | mean_fn_correction_rate | mean_residual_shift_max_abs | go |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | qwen_directional_t200_cover_rel_nollm | 0.467967 | 0.487739 | 0.019772 | 0.005974 | 0.006112 | -0.846405 | 0.26444 | 0.033606 | 0.2 | True |
| rur | qwen_directional_t200_cover_rel_rur_nollm | 0.467967 | 0.492692 | 0.024725 | 0.007059 | 0.011497 | -0.870117 | 0.264213 | 0.038201 | 0.199924 | True |
| rsr | qwen_directional_t200_cover_rel_rsr_nollm | 0.467967 | 0.467708 | -0.000259 | -9.5e-05 | -0.017616 | -0.792889 | 0.266448 | 0.006257 | 0.2 | False |
| rtr | qwen_directional_t200_cover_rel_rtr_nollm | 0.467967 | 0.46903 | 0.001064 | 0.000202 | -0.015836 | -0.798123 | 0.262208 | 0.009393 | 0.2 | False |

