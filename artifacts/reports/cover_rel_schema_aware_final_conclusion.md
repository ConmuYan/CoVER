# CoVER-REL Schema-Aware Final Conclusion

- YelpChi best relation: RUR.
- Amazon best relation: UVU.
- Anchor/base gates were evaluated without Qwen latents and with dataset-driven relation schemas.
- Strong GO requires gate mean Delta AUPRC to match or exceed best-single; Acceptable GO allows near-cap or stability gains within 0.002 AUPRC.

## Summary

| dataset | verdict | best_relation | best_single_delta_auprc | anchor_gate_delta_auprc | base_gate_delta_auprc | conservative_delta_auprc | anchor_gate_delta_vs_best_single | base_gate_delta_vs_best_single | conservative_delta_vs_best_single | best_single_near_cap | anchor_gate_near_cap | base_gate_near_cap | conservative_near_cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yelpchi | Acceptable GO | RUR | 0.027099 | 0.026585 | 0.016175 |  | -0.000515 | -0.010924 |  | 0.731901 | 0.682957 | 0.838069 |  |
| amazon | Strong GO | UVU | 0.003159 | 0.003508 | 0.003402 | 0.001785 | 0.000349 | 0.000243 | -0.001374 | 0.828533 | 0.912726 | 0.912073 | 0.000000 |


## Interpretation

- CoVER-REL is schema-aware because relation schemas differ across YelpChi and Amazon and are configured rather than hardcoded.
- YelpChi remains RUR-concentrated: anchor_gate stays within 0.001 AUPRC of RUR-only and reduces near-cap, while base_gate is weaker.
- Amazon is UVU-centered but distributed: anchor/base gates improve over UVU-only, while the conservative anchor branch trades AUPRC for zero near-cap.
- Final model selection should keep dataset-specific relation schemas but not hardcode YelpChi relation names.
