# Amazon CoVER-REL-Judge 3-Seed Conclusion

- Verdict: **Acceptable GO**
- Reason: Amazon judge is weakly non-negative over anchor gate with valid verifier; inspect per-seed alpha saturation before 5-seed expansion.
- Mean Delta AUPRC vs anchor_gate: 0.000173
- Mean Delta AUPRC vs UVU-only: 0.000465
- Judge acceptance rate: 0.933333
- Alpha LLM mean: 0.441609
- Alpha saturation risk seeds: [456]

## Metrics

| seed | status | judge_auprc | delta_auprc_vs_anchor | delta_auprc_vs_uvu | roc_auc | macro_f1 | judge_acceptance_rate | alpha_llm_mean | delta_llm_max_abs | near_cap_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | complete | 0.861427 | 0.000737 | 0.000710 | 0.982087 | 0.917698 | 0.933333 | 0.135488 | 0.512501 | 0.583640 |
| 456 | complete | 0.862168 | 0.000694 | 0.000995 | 0.975529 | 0.913933 | 0.933333 | 0.995988 | 0.985241 | 0.979153 |
| 789 | complete | 0.815825 | -0.000910 | -0.000309 | 0.917628 | 0.914680 | 0.933333 | 0.193350 | 0.624195 | 0.786587 |
| mean | 3/3 complete | 0.846473 | 0.000173 | 0.000465 | 0.958415 | 0.915437 | 0.933333 | 0.441609 | 0.707312 | 0.783127 |
| std | 3/3 complete | 0.026544 | 0.000939 | 0.000686 | 0.035474 | 0.001993 | 0.000000 | 0.480978 | 0.247087 | 0.197779 |

