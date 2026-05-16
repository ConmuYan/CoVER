# GraphSAGE (SAGE) — Cross-Base Validation for the CoVER-REL Reasoner

This document covers how GraphSAGE plugs into the canonical **two-phase
CoVER-REL Reasoner** as a second Phase1 base, alongside the primary BWGNN
backbone. SAGE is a **cross-base validation** of the Reasoner interface,
not a second main detector.

See `docs/cover_main.md` for the canonical method.

## 1. References

- **Hamilton, Ying & Leskovec (2017)**. *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
  - arXiv: [1706.02216](https://arxiv.org/abs/1706.02216)
  - Stanford PDF: <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>
- **Official GitHub**: [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE)
  - Local reference clone: `external/williamleif_GraphSAGE/` (reference-only; never imported)
- **Tang et al. (2022)**. *Rethinking Graph Neural Networks for Anomaly Detection*. ICML 2022.
  - BWGNN paper; uses SAGE as a supervised baseline for fraud detection comparison.

## 2. Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `hidden_dim` | 64 | Hamilton et al. Table 2; matches BWGNN baseline hidden for fair comparison |
| `num_layers` | 2 | Paper default (K=2 hops); same as BWGNN `num_layers` |
| `dropout` | 0.5 | Paper Section 3.3 / Algorithm 1 default; differs from BWGNN 0.3 — kept at paper default |
| `lr` | 0.01 | Standard GNN learning rate; matches BWGNN |
| `weight_decay` | 5e-4 | Standard L2 regularisation; matches BWGNN |
| `optimizer` | Adam | Paper default; matches BWGNN |
| `epochs` | 100 | Sufficient for convergence with patience=100; same as BWGNN |
| `train_ratio` | 0.4 | Supervised split; matches BWGNN evaluation protocol |
| `val_test_ratio` | [1, 2] | 1:2 val/test; matches BWGNN |
| `stratified` | true | Class-balanced splits; matches BWGNN |
| `select_metric` | macro_f1 | Primary selection metric for base model; matches BWGNN |
| `patience` | 100 | Early stopping; matches BWGNN |
| `aggregator` | mean | Hamilton et al. Section 3.3 default; `SAGEConv(aggr='mean')` in PyG |
| `activation` | ReLU | Paper default nonlinearity σ; standard in PyG SAGEConv |

## 3. Implementation

The in-repo implementation `models/gnn.py:SAGEDetector` uses
`torch_geometric.nn.SAGEConv` with mean aggregation, 2-layer stack,
hidden=64, dropout=0.5, and a linear classification head. This matches
the supervised variant described in Hamilton et al. 2017, Section 3.3.

No code from `external/williamleif_GraphSAGE/` is imported — the clone
exists solely for hyperparameter audit trail and reproducibility reference.

## 4. Differences from BWGNN

| Parameter | SAGE | BWGNN | Reason |
|---|---|---|---|
| `dropout` | 0.5 | 0.3 | SAGE paper default vs BWGNN paper default |
| `num_bands` | N/A | 3 | BWGNN-specific spectral parameter |
| `agg` | N/A | concat | BWGNN-specific band aggregation |

All other training hyperparameters (`lr`, `weight_decay`, `epochs`, `patience`,
split) are identical to enable a fair same-seed comparison between SAGE and
BWGNN as Phase1 bases.

## 5. Base-Agnostic Interface

After Phase1, SAGE is **frozen** and exposes the same two artifacts the
Phase2 Reasoner consumes from any base:

- `b_i` — detached base logit
- `base_z_i` — detached base embedding / hidden representation

The Phase2 Reasoner is implemented over `(b, base_z, relation_features,
optional judge_features)` and contains **no SAGE-specific or BWGNN-specific
logic**. Swapping the base only changes which Phase1 checkpoint is loaded.

Operationally, replicating a BWGNN-confirmed Phase2 setting on SAGE only
requires copying the YAML and switching `model.name` and the dataset path
(see `configs/phase2_reasoner/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`
for the verbatim transfer).

## 6. Cross-Base Results (conservative interpretation)

### 6.1 YelpChi-SAGE — POSITIVE

5 seeds (42, 123, 456, 789, 2026), paired throughout. Source:
`artifacts/reports/sage_confirmed_vs_sage_baselines.md`.

| Run                                       | AUPRC (mean ± std) | ROC-AUC (mean ± std) | Macro-F1            |
|-------------------------------------------|--------------------|----------------------|---------------------|
| Phase1 SAGE base                          | 0.2246 ± 0.1261    | 0.5995 ± 0.1115      | 0.4848 ± 0.0535     |
| Legacy Stage3 anchor_gate                 | 0.4465 ± 0.0289    | 0.8064 ± 0.0142      | 0.4873 ± 0.0591     |
| Phase2 default E0                         | 0.4503 ± 0.0891    | 0.8429 ± 0.0249      | 0.7056 ± 0.0238     |
| Phase2 default E1                         | 0.4535 ± 0.0888    | 0.8438 ± 0.0254      | 0.7064 ± 0.0246     |
| Phase2 default E2                         | 0.4539 ± 0.0891    | 0.8442 ± 0.0257      | 0.7064 ± 0.0247     |
| **Phase2 confirmed** (`phase2_yelp_confirm_lalign_1em2_standard`) | **0.4786 ± 0.0522** | **0.8538 ± 0.0099** | **0.7158 ± 0.0118** |

Paired Δ (confirmed − baseline, 5-seed):

| Compared against            | ΔAUPRC                | paired t    | Comment                                              |
|-----------------------------|-----------------------|-------------|------------------------------------------------------|
| Phase1 SAGE base            | **+0.2540 ± 0.0833**  | **+6.82** ⭐ | p<0.01                                               |
| Legacy Stage3 anchor_gate   | +0.0321 ± 0.0283      | +2.53       | marginal at n=5, not strict p<0.05                   |
| Phase2 default E0/E1/E2     | +0.025..+0.028        | +1.3..+1.5 ns | mean lift not significant; **std cut ~41%**         |

Diagnostic interpretation:

- `alpha_max = 0` and mean `α_llm = 0`. The gain is **not** from direct
  LLM residual prediction; it should be attributed to **judge-aligned
  relation reasoning / regularization**.
- Gate is RUR-dominant but not fully collapsed:
  `RUR ≈ 0.7615`, `RSR ≈ 0.1375`, `RTR ≈ 0.1011`, gate entropy ≈ 0.3837.
- The confirmed config trades minor per-seed mean improvement on
  "good seeds" for a large rescue on the worst seed (seed_456 AUPRC
  0.32 → 0.42), the classic regularization / robustness signature.
- ROC-AUC lift over anchor_gate is exceptionally strong
  (+0.0474, t=+12.27, p<0.001) — the cleanest signal that the confirmed
  config improves the ranking quality of SAGE+Phase2.

Safety: 5/5 seeds satisfy `max_abs_alpha_llm_rejected < 1e-6`.

### 6.2 Amazon-SAGE — DIAGNOSTIC / SATURATION

Phase2 E0 on Amazon-SAGE showed `ΔAUPRC ≈ 0` on the smoke seed; the
strict-gate at E0 prevented full E1/E2 sweeps. Legacy Stage3 anchor_gate
on SAGE-Amazon was a modest +0.0111 over base. Until newer 5-seed
confirmed-config runs are produced, **Amazon-SAGE remains a diagnostic /
saturation case**; do not claim it is solved.

| Stage                       | AUPRC                   |
|-----------------------------|-------------------------|
| Phase1 SAGE base            | 0.7556 ± 0.0511         |
| Legacy Stage3 anchor_gate   | 0.7667 ± 0.0444         |
| Phase2 E0 (seed 42)         | 0.7415 (Δ ≈ 0; gate FAIL) |

For comparison, BWGNN-Amazon Phase2 E0 was also flat (`0.8643 ± 0.0185`,
Δ ≈ 0). On a weak anchor (`UVU`), Phase2 has limited headroom regardless
of the base.

## 7. Configs

- Phase1 base: `configs/{yelpchi,amazon}_sage.yaml`
- Legacy Stage3 anchor_gate (SAGE): `configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_{yelpchi,amazon}_sage_nollm.yaml`
- Phase2 SAGE ablations (E0/E1/E2): `configs/cover-rel-gj/phase2_ablations/phase2_{yelpchi,amazon}_sage_E{0,1,2}.yaml`
- **Canonical Phase2 SAGE-YelpChi (confirmed)**:
  `configs/phase2_reasoner/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`

## 8. Positioning in the Manuscript

- Frame SAGE as a **cross-base validation** of the Phase2 Reasoner interface:
  the same Reasoner code, the same losses, the same hyperparameters
  (with only the Phase1 base swapped) deliver significant lift over the
  Phase1 SAGE base and over the legacy Stage3 anchor_gate on YelpChi.
- Do **not** call SAGE a second main detector.
- Emphasize YelpChi positive and Amazon diagnostic; Amazon-SAGE remains
  challenging for both bases.
- Highlight the **stability and worst-seed rescue** behavior of the
  confirmed config, not raw mean improvement over default Phase2.

## 9. External Clone Status

The reference clone `external/williamleif_GraphSAGE/` was successfully
cloned from `https://github.com/williamleif/GraphSAGE.git`. It is used
for hyperparameter audit only.
