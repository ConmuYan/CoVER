# configs/ — Layout & Conventions

This directory groups configs by **method generation**, not by dataset.
The canonical method is the **two-phase CoVER-REL Reasoner**:

```
Phase1  : train a fresh base detector (BWGNN / GraphSAGE / GCN / GAT)
          → frozen as structural prior
Phase2  : train a unified CoVER-REL Reasoner over relation-aware evidence
          and optional contract-verified score-blind judge features
          → z_i = b_i + Δ_rel,i + α_i · Δ_llm,i
```

See `docs/cover_main.md` for the canonical method, loss, and E0/E1/E2/E3 ablation taxonomy.

## Layout

```
configs/
├── README.md                                # this file
├── {yelpchi,amazon}_{bwgnn,sage,gcn,gat}.yaml
│         # Phase1 base detector configs (model-level only)
│
├── phase2_reasoner/
│   ├── phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml
│   └── phase2_amazon_yelpstyle_judge_align.yaml
│         # Canonical Phase2 unified CoVER-REL Reasoner (current method).
│         # These are the "confirmed" configs transferred from BWGNN tuning.
│
└── cover-rel-gj/                            # legacy "Gate + Judge" family
    ├── stage3_legacy/                       # Stage3 anchor_gate / judge_train
    │   ├── stage3_cover_rel_gate_nollm.yaml             # YelpChi anchor_gate
    │   ├── stage3_cover_rel_amazon_nollm.yaml           # Amazon anchor_gate
    │   ├── stage3_cover_rel_gcn_gate_nollm.yaml         # GCN  anchor_gate
    │   ├── stage3_cover_rel_yelpchi_sage_nollm.yaml     # SAGE anchor_gate (YelpChi)
    │   ├── stage3_cover_rel_amazon_sage_nollm.yaml      # SAGE anchor_gate (Amazon)
    │   ├── stage3_cover_rel_judge_yelpchi.yaml          # YelpChi judge_train
    │   ├── stage3_cover_rel_judge_amazon.yaml           # Amazon judge_train
    │   └── stage3_cover_rel_nollm.yaml                  # YelpChi legacy nollm
    │
    └── phase2_ablations/                    # Phase2 E0 / E1 / E2 / E3 ablation configs
        ├── phase2_{yelpchi,amazon}_E0_relgate.yaml
        ├── phase2_{yelpchi,amazon}_E1_judge_align.yaml
        ├── phase2_{yelpchi,amazon}_E2_judge_residual.yaml
        ├── phase2_{yelpchi,amazon}_E3_no_trust.yaml
        ├── phase2_{yelpchi,amazon}_{gcn,gat}_E{0,2}*.yaml
        ├── phase2_{yelpchi,amazon}_sage_E{0,1,2}.yaml
        └── phase2_amazon_E{0,1,2,3}_*_v2.yaml
```

## Naming convention

| Family               | Path                                          | Status                                  |
|----------------------|-----------------------------------------------|-----------------------------------------|
| Phase1 base          | `configs/{ds}_{base}.yaml`                    | active                                  |
| **Phase2 canonical** | `configs/phase2_reasoner/*.yaml`              | **canonical (current method)**          |
| Phase2 ablations     | `configs/cover-rel-gj/phase2_ablations/*.yaml`| diagnostic / E0/E1/E2/E3 study           |
| Legacy Stage3 G/J    | `configs/cover-rel-gj/stage3_legacy/*.yaml`   | historical baseline (Gate + Judge)      |

`cover-rel-gj` = legacy **G**ate + **J**udge family. Everything under this
prefix predates the unified two-phase Reasoner and is kept for reproducibility
of older results. New experiments should target `configs/phase2_reasoner/`.

## E0 / E1 / E2 / E3 (ablation taxonomy)

| Variant | `use_judge` | `alpha_max` | `lambda_align` | `lambda_trust` | Final logit                              |
|---------|:-----------:|:-----------:|:--------------:|:--------------:|------------------------------------------|
| E0      | false       | 0           | 0              | >0             | b + Δ_rel                                |
| E1      | true        | 0           | >0             | >0             | b + Δ_rel   (judge → gate alignment)     |
| E2      | true        | >0          | >0             | >0             | b + Δ_rel + α · Δ_llm                    |
| E3      | true        | >0          | >0             | **0**          | b + Δ_rel + α · Δ_llm (no trust)         |

The current canonical "confirmed" config (`configs/phase2_reasoner/...`) is
an E1-flavored Phase2 setting (`alpha_max=0`, `lambda_align=1e-2`,
`lambda_trust=3e-3`); the LLM is used for **alignment / regularization** only,
not as a residual predictor.

## Migration map (old → new)

Old in-repo references look like `configs/<file>.yaml`. The new locations are:

| Old path                                                | New path                                                                       |
|---------------------------------------------------------|--------------------------------------------------------------------------------|
| `configs/stage3_cover_rel_*.yaml`                       | `configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_*.yaml`                   |
| `configs/phase2_{ds}_E{0,1,2,3}_*.yaml`                 | `configs/cover-rel-gj/phase2_ablations/phase2_{ds}_E{0,1,2,3}_*.yaml`          |
| `configs/phase2_{ds}_{gcn,gat,sage}_E{0,1,2}*.yaml`     | `configs/cover-rel-gj/phase2_ablations/phase2_{ds}_{gcn,gat,sage}_E{0,1,2}*.yaml` |
| `configs/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml` | `configs/phase2_reasoner/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`        |
| `configs/phase2_amazon_yelpstyle_judge_align.yaml`      | `configs/phase2_reasoner/phase2_amazon_yelpstyle_judge_align.yaml`             |

Historical reports under `artifacts/reports/` and `artifacts/paper/` keep
the **old** paths verbatim as time-stamped snapshots; do not rewrite them.

## What goes where, going forward

- Add a **new base detector** → `configs/{ds}_{newbase}.yaml`
- Add a **new Phase2 canonical setting** → `configs/phase2_reasoner/<name>.yaml`
- Add a **new E0/E1/E2/E3 ablation** → `configs/cover-rel-gj/phase2_ablations/<name>.yaml`
- Do **not** add new files under `configs/cover-rel-gj/stage3_legacy/` —
  that subdirectory is frozen.
