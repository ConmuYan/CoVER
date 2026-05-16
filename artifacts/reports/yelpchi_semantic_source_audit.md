# YelpChi Semantic Source Audit

- Dataset path: `datasets/YelpChi.mat`
- Decision: `use_feature_buckets_relation_summaries`
- Raw review text available: `false`
- Feature count: `32`
- Feature names available: `false`
- Relation keys available: `{"R-S-R": true, "R-T-R": true, "R-U-R": true}`

## Source Availability

| Source | Available | Evidence | Keys | Decision note |
| --- | --- | --- | --- | --- |
| raw_review_text | false | no text-like review/content key in .mat | - | fall back to feature buckets |
| user_id | false | only R-U-R relation is available | - | summarize same-user relation graph |
| product_id | false | only product-derived relations are available | - | summarize R-S-R/R-T-R relation graphs |
| rating_or_star | false | rating appears only through R-S-R relation semantics | - | use R-S-R relation summary |
| timestamp_or_month | false | time appears only through R-T-R relation semantics | - | use R-T-R relation summary |
| relation_R_U_R | true | shape=45954x45954, dtype=float64, nnz=98630, density=4.67049e-05 | net_rur | same-user neighbor summary |
| relation_R_S_R | true | shape=45954x45954, dtype=float64, nnz=6805486, density=0.00322265 | net_rsr | same-product-same-rating neighbor summary |
| relation_R_T_R | true | shape=45954x45954, dtype=float64, nnz=1147232, density=0.000543256 | net_rtr | same-product-same-month neighbor summary |
| homogeneous_graph | true | shape=45954x45954, dtype=float64, nnz=7693958, density=0.00364337 | homo | existing structural graph prior |
| handcrafted_features | true | shape=45954x32, dtype=float64, nnz=1469088, density=0.999021 | features | 32 anonymous numeric features |
| feature_names | false | 32 anonymous numeric features | - | semanticize as feature_00..feature_31 buckets |
| labels | true | shape=1x45954, dtype=int64 | label | labels are for train-only retrieval/prototypes, never target packet fields |

## MAT Key Inventory

| Key | Shape | Dtype | NNZ |
| --- | --- | --- | --- |
| features | 45954x32 | float64 | 1469088 |
| homo | 45954x45954 | float64 | 7693958 |
| label | 1x45954 | int64 | - |
| net_rsr | 45954x45954 | float64 | 6805486 |
| net_rtr | 45954x45954 | float64 | 1147232 |
| net_rur | 45954x45954 | float64 | 98630 |

## CoVER-META-Lite Implication

The local YelpChi artifact does not expose raw text or direct user/product/rating/time columns.
The first evidence-packet version should therefore use anonymous 32-feature buckets plus relation-specific summaries from `net_rur`, `net_rsr`, and `net_rtr`.
Train labels may only be used to build train-only prototype or retrieval summaries; target labels and split identity must not appear in packet fields.
