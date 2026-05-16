| dataset | final quantitative takeaway | best_phase2_setting | phase2_delta_auprc_vs_base | phase2_delta_auprc_vs_legacy_gate | judge_effect_in_phase2 | safety_positioning |
| --- | --- | --- | --- | --- | --- | --- |
| yelpchi | unified Phase2 Reasoner is the strongest saved result | E2 judge residual | +0.1002 | +0.0678 | E1/E2 add +0.0006/+0.0007 over E0 | accepted judge only; rejected/missing alpha forced to 0 |
| amazon | base is near-saturated; legacy gate remains slightly higher | E1/E2 tied by rounded AUPRC | +0.0001 | -0.0017 | no meaningful AUPRC gain over E0 | accepted judge only; rejected/missing alpha forced to 0 |

Recommended positioning:

- Present the Phase2 CoVER-REL Reasoner as the clean two-phase training framework.
- Do not claim Phase2 dominates all legacy results: YelpChi improves strongly, while Amazon preserves base-level ranking but trails the legacy gate by about 0.0017 AUPRC.
- Keep the LLM judge framed as contract-verified evidence alignment and conservative explanation support, not as a primary predictor.
