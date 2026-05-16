| dataset | method | role | seeds_complete | auprc | delta_auprc_vs_base | delta_auprc_vs_legacy_gate | roc_auc | macro_f1 | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yelpchi | BWGNN deterministic base | frozen structural prior | 5 | 0.4674 ± 0.0151 | 0.0000 | -0.0324 | 0.8076 | 0.6483 | `artifacts/tables/phase2_5seed_summary.md` |
| yelpchi | Legacy CoVER-REL-Gate | legacy relation gate baseline | 5 | 0.4998 ± 0.0210 | +0.0324 | 0.0000 | 0.8177 | 0.6598 | `artifacts/tables/phase2_5seed_summary.md` |
| yelpchi | Legacy CoVER-REL-Judge | legacy LLM-assisted extension | 5 | 0.5006 ± 0.0248 | +0.0332 | +0.0008 | 0.8180 | 0.6650 | `artifacts/tables/phase2_5seed_summary.md` |
| yelpchi | Phase2 CoVER-REL Reasoner E0 | unified relation-only reasoner | 5 | 0.5669 ± 0.0151 | +0.0995 | +0.0671 | 0.8665 | 0.7279 | `artifacts/tables/phase2_5seed_summary.md` |
| yelpchi | Phase2 CoVER-REL Reasoner E1 | judge alignment only | 5 | 0.5675 ± 0.0145 | +0.1001 | +0.0677 | 0.8672 | 0.7278 | `artifacts/tables/phase2_5seed_summary.md` |
| yelpchi | Phase2 CoVER-REL Reasoner E2 | judge alignment + conservative residual | 5 | 0.5676 ± 0.0144 | +0.1002 | +0.0678 | 0.8673 | 0.7279 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | BWGNN deterministic base | frozen structural prior | 5 | 0.8643 ± 0.0190 | 0.0000 | -0.0018 | 0.9747 | 0.9168 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | Legacy CoVER-REL-Gate | legacy relation gate baseline | 5 | 0.8661 ± 0.0185 | +0.0018 | 0.0000 | 0.9752 | 0.9174 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | Legacy CoVER-REL-Judge | legacy LLM-assisted extension | 5 | 0.8663 ± 0.0180 | +0.0020 | +0.0002 | 0.9751 | 0.9168 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | Phase2 CoVER-REL Reasoner E0 | unified relation-only reasoner | 5 | 0.8643 ± 0.0185 | +0.0000 | -0.0018 | 0.9748 | 0.9113 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | Phase2 CoVER-REL Reasoner E1 | judge alignment only | 5 | 0.8644 ± 0.0186 | +0.0001 | -0.0017 | 0.9748 | 0.9133 | `artifacts/tables/phase2_5seed_summary.md` |
| amazon | Phase2 CoVER-REL Reasoner E2 | judge alignment + conservative residual | 5 | 0.8644 ± 0.0186 | +0.0001 | -0.0017 | 0.9748 | 0.9133 | `artifacts/tables/phase2_5seed_summary.md` |

Notes:

- The deterministic base, legacy gate, legacy judge, and Phase2 results are copied from saved artifacts; no new metric is inferred from logs.
- Phase2 E2 is the strongest YelpChi configuration, but it does not recover the small Amazon gain of the legacy gate. Paper claims should therefore separate the YelpChi Phase2 gain from the Amazon near-saturation finding.
