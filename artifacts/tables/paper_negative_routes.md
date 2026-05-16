| route | dataset | seeds | mean_delta_auprc | mean_delta_macro_f1 | latent_cosine | lesson |
| --- | --- | --- | --- | --- | --- | --- |
| CoVER-LIFT canonical ERR hidden | yelpchi | 3 | -0.000291 | -0.006762 | 0.804623 | Latent alignment can be high while canonical ERR hidden is not fraud-discriminative. |
| ERR-only / structure-only evidence | yelpchi |  |  |  |  | Thin structured evidence lacks the relation-aware anonymous feature signal needed for ranking gains. |
| Unrestricted Judge tuning | amazon | 3 | 0.000173 |  |  | Original judge was weakly positive but had seed-level alpha saturation; conservative strength-aware fusion is safer. |
