# configs/ — Layout & Conventions (cls-only canonical)

Configs are grouped by **training phase**. The canonical method is the
two-phase **CoVER-REL Reasoner** with a single cls-only loss; see
[`AGENTS.md`](../AGENTS.md) §§1–11 for the full TPAMI-style derivation.

```
Phase 1 : train a fresh base detector (BWGNN / GraphSAGE / GCN / GAT)
          → frozen as structural prior (SHA-256 verified)
Phase 2 : train a unified CoVER-REL Reasoner over relation-aware
          score-blind 9-dim evidence statistics
          → z_i = b_i + Δ_rel,i    (no LLM judge term, no α·Δ_llm)
          → L   = L_cls            (no L_int / L_sparse / L_align)
```

## Layout

```
configs/
├── README.md                                          # this file
├── {yelpchi,amazon}_{bwgnn,sage,gcn,gat}.yaml         # Phase 1 base detector (8 files)
└── phase2_reasoner/
    └── ablation/                                      # Idea-1 ablation suite (7 cells)
        ├── idea1_canonical_clsonly.yaml               # cls-only canonical baseline (★)
        ├── idea1_ablate_gate_uniform.yaml             # gate         : softmax → uniform 1/R
        ├── idea1_ablate_evidence_no_structural.yaml   # evidence drop: group A  (dims 0-2)
        ├── idea1_ablate_evidence_no_incoherence.yaml  # evidence drop: group B  (dims 3-5)
        ├── idea1_ablate_evidence_no_proto.yaml        # evidence drop: group C  (dims 6-8)
        ├── idea1_ablate_shared_expert.yaml            # experts      : per-relation → shared
        └── idea1_ablate_unbounded_residual.yaml       # residual     : tanh saturation → identity
```

## What each file is for

| Path | Purpose |
|---|---|
| `{ds}_{base}.yaml` | Phase 1 trainer. Reads `dataset` / `model` / `train` / `eval`. |
| `phase2_reasoner/ablation/idea1_canonical_clsonly.yaml` | Phase 2 cls-only baseline. Anchors all paired-t comparisons. |
| `phase2_reasoner/ablation/idea1_ablate_*.yaml` | Idea-1 ablation cells; only one structural toggle changes per file. |

## Idea-1 ablation toggles (only knobs that vary)

| Knob | Canonical | Ablation value | Carrier file |
|---|---|---|---|
| `gate_mode` | `softmax` | `uniform`  (each `g_{i,r}=1/R`) | `idea1_ablate_gate_uniform.yaml` |
| `evidence_groups` | `[A, B, C]` (all 9 dims) | `[B, C]` (drop **A** structural, dims 0-2) | `idea1_ablate_evidence_no_structural.yaml` |
| `evidence_groups` | `[A, B, C]` | `[A, C]` (drop **B** feature/neighbour, dims 3-5) | `idea1_ablate_evidence_no_incoherence.yaml` |
| `evidence_groups` | `[A, B, C]` | `[A, B]` (drop **C** prototype-relative, dims 6-8) | `idea1_ablate_evidence_no_proto.yaml` |
| `expert_shared` | `false` (per-relation MLPs) | `true`  (single MLP + one-hot relation indicator) | `idea1_ablate_shared_expert.yaml` |
| `residual_activation` | `tanh` (bounded `δ_max · tanh(u)`) | `identity` (unbounded `δ_max · u`) | `idea1_ablate_unbounded_residual.yaml` |

Every other Phase 2 hyperparameter (`tau_gate=0.7`, `delta_rel_max=2.0`,
`rel_hidden_dim=64`, `rel_num_layers=2`, `rel_dropout=0.30`, AdamW
`lr=1e-3 / wd=1e-4`, `epochs=300`, `patience=50`) is held *identical* across
the 7 configs so the paired-t Δ vs canonical isolates the toggle.

## Deprecated noop knobs

These keys are accepted in the Phase 2 config for backward compatibility but
do not change behaviour and emit a one-shot `DeprecationWarning`:

| Key | Status | Falsification |
|---|---|---|
| `use_judge`, `alpha_max`, `judge_*`, `delta_llm_max`, `alpha_bias_init` | Hard-rejected if *on* (raises `NotImplementedError`) | LLM-judge α·Δ_llm: t=+0.03, p=0.976 |
| `lambda_int`, `lambda_trust` | Silently ignored | L_int: t=+0.26, p=0.81 (single); t=+1.02, p=0.36 (with L_sparse) |
| `lambda_sparse` | Silently ignored | L_sparse: t=+2.59, p=0.061 (closest to bar, still fail) |
| `lambda_align` | Hard-rejected if non-zero | L_align: t=−1.14, p=0.32 |
| `eta_llm` | Silently ignored | Tied to dead judge path |

See `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md` and
`artifacts/tables/paper_negative_routes.md` for the full 5-seed paired-t
evidence behind each removal.

## Adding new configs

- **New base detector** → `configs/{ds}_{newbase}.yaml` with only
  `dataset` / `model` / `train` / `eval` blocks.
- **New Phase 2 ablation cell** → `configs/phase2_reasoner/ablation/<name>.yaml`,
  flipping exactly one structural toggle vs `idea1_canonical_clsonly.yaml`
  so the paired-t against it is interpretable. The trainer reads any of
  `gate_mode` / `evidence_groups` / `expert_shared` / `residual_activation`;
  introduce a new toggle only after wiring it through `models/cover_rel_reasoner.py`.

Do *not* re-introduce `stage2:` / `evidence:` / `llm:` / `reasoner:` blocks
or any of the deprecated loss weights — they will be silently no-op'd or
hard-rejected and the run will be indistinguishable from canonical.
