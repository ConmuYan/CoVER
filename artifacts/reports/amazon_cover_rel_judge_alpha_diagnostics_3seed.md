# Amazon CoVER-REL-Judge Alpha Diagnostics 3-Seed

Overall near-cap is measured against the base-detector residual and can remain high because the anchor relation branch already uses a large residual. LLM near-cap isolates the judge branch effect relative to the relation-only logit.

| variant | alpha_mean | alpha_max | alpha_fake | alpha_real | alpha_uncertain | alpha_weak | alpha_moderate | alpha_strong | llm_near_cap | overall_near_cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_judge | 0.441609 | 0.485134 |  |  |  | 0.000000 | 0.436027 | 0.443633 | 0.333333 | 0.783127 |
| delta05 | 0.688910 | 0.719733 | 0.690760 | 0.682367 | 0.682595 | 0.000000 | 0.689340 | 0.688704 | 0.666667 | 0.790271 |
| alpha05 | 0.344442 | 0.359866 | 0.345366 | 0.341172 | 0.341283 | 0.000000 | 0.344659 | 0.344337 | 0.666667 | 0.790299 |
| conservative | 0.183866 | 0.221170 | 0.185593 | 0.157927 | 0.183835 | 0.000000 | 0.179584 | 0.185452 | 0.267857 | 0.798588 |
| strength_gate | 0.263068 | 0.447026 | 0.301438 | 0.263813 | 0.062640 | 0.000000 | 0.125803 | 0.308805 | 0.232143 | 0.798616 |
