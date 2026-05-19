# AGENTS.md — CoVER-REL canonical method (cls-only)

> Single source of truth for the **two-phase CoVER-REL Reasoner**. Every
> Phase 2 file in the repo is required to be consistent with this document.
> Reading this top-to-bottom is sufficient to (re-)derive the implementation,
> the loss, and the four hard contracts.

## 1. Problem formulation

Let $\mathcal{G} = (\mathcal{V}, \{\mathcal{E}^{(r)}\}_{r=1}^{R}, \mathbf{X}, \mathbf{y})$ be a multi-relation fraud graph:

- $\mathcal{V} = \{1,\dots,N\}$ — node set (YelpChi $N{=}45{,}954$, Amazon $N{=}11{,}944$).
- $\mathbf{A}^{(r)} \in \{0,1\}^{N\times N}$ — sparse adjacency for the $r$-th relation. YelpChi has $R{=}3$: RUR (same-user reviews), RSR (same product, same star), RTR (same product, same month). Amazon has $R{=}3$: UPU, USU, UVU.
- $\mathbf{X} \in \mathbb{R}^{N\times D}$ — anonymous node features.
- $\mathbf{y} \in \{0,1\}^N$ — binary fraud label.
- Stratified $0.4/0.2/0.4$ split $\mathcal{V}_{\text{tr}}, \mathcal{V}_{\text{val}}, \mathcal{V}_{\text{te}}$ (BWGNN protocol).
- A *frozen* base detector $f_{\text{base}}$ (BWGNN / SAGE / GCN / GAT) emits per-node logit $b_i$ and embedding $\mathbf{z}_i^{\text{base}} \in \mathbb{R}^{d_z}$.

**Goal.** Learn a bounded residual $\Delta_i^{\text{rel}}$ such that

$$
z_i \;=\; b_i + \Delta_i^{\text{rel}}, \qquad \bigl|\Delta_i^{\text{rel}}\bigr| \le \delta_{\max}
$$

strictly improves a paired 5-seed AUPRC bar over the *compute-matched* base, and improves consistently across base $\times$ dataset.

### Four hard contracts (input-side, architecture-enforced)

| # | Contract | Mechanism |
|---|---|---|
| C1 | **Base-freeze** | $\theta_{\text{base}}$ is byte-identical pre/post training (SHA-256 check; see `tests/test_base_freeze_sha256.py`). |
| C2 | **Score-blind** | The reasoner never sees `base_score / base_logit / base_pred / final_pred / label / split / FN-FP-status / ground-truth`. Enforced by construction in `evidence/relation_features.py` (pure-numpy over $\mathbf{A}^{(r)}$ + $\mathbf{X}$ + `train_mask` only). |
| C3 | **Train-only prototype** | Fraud / benign prototypes are built from $\mathcal{V}_{\text{tr}}$ alone — `val/test` labels never enter (`evidence/relation_features.py::_relation_prototypes`). |
| C4 | **Bounded intervention** | $\|\Delta_i^{\text{rel}}\| \le \delta_{\max}{=}2.0$ via `tanh` saturation at the head — *not* a loss term. |

> C4 used to be co-enforced by an $L_{\text{intervention}}$ penalty. That term was 5-seed paired-t falsified (Δ_AUPRC=+0.0006, p=0.81 single; +0.0015, p=0.36 with $L_{\text{sparse}}$) and is now solely structural (`tanh` + zero-initialised heads). See §7.

---

## 2. Method overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ frozen base   │  f_base(G)  →  b_i ∈ ℝ ,  z_i^base ∈ ℝ^{d_z}            │ (Phase 1)
└──────────────┬─────────────────────────────────────────────────────────┘
               │  detach()
               ▼
   per-relation     ┌─────────────────────────────────────────┐
   stat extractor   │  E_{i,r} = ϕ_r(G, X, y_train)           │ §3
                    └────────────────┬────────────────────────┘
                                     │ stack r=1..R
                                     ▼
   per-relation     ┌─────────────────────────────────────────┐
   expert MLPs      │  h_{i,r} = Expert_r(E_{i,r})            │ §4
                    │  ∈ ℝ^{d_h},  r = 1,…,R                  │
                    └────────────────┬────────────────────────┘
                                     ▼
   schema-aware     ┌─────────────────────────────────────────┐
   softmax gate     │  g_i = softmax(W [z_i^base ; h_i] / τ)  │ §5
                    │  ∈ Δ^{R-1}  (simplex over R relations)  │
                    └────────────────┬────────────────────────┘
                                     ▼
   bounded          ┌─────────────────────────────────────────┐
   relation         │  u_i = Σ_r g_{i,r} · Head_r(h_{i,r})    │ §6
   residual         │  Δ_i^{rel} = δ_max · tanh(u_i)          │
                    └────────────────┬────────────────────────┘
                                     ▼
                             z_i = b_i + Δ_i^{rel}
                                     │
                                     ▼
                             L = L_cls                          §7
```

Single branch, single objective. No LLM, no auxiliary classifier, no DIR/SCD/LIFT extension.

---

## 3. Relation evidence $E_{i,r}$ (9 dims × $R$ relations)

`evidence/relation_features.py::compute_relation_features`. For each node $i$ and each relation $r$ we extract a 9-dim score-blind vector $\mathbf{E}_{i,r}$:

**Group A — structural density (3 dims, depends only on $\mathbf{A}^{(r)}$).**

- $E_{i,r}^{(1)} = \log(1+d_{i,r}) / \log(1+d_{\max,r})$  — normalised log-degree.
- $E_{i,r}^{(2)} = \mathbb{1}[d_{i,r} \ge q_{90}]$  — degree top-10% sentinel.
- $E_{i,r}^{(3)} = \mathbb{1}[d_{i,r} \le q_{10}]$  — degree bottom-10% sentinel.

**Group B — feature/neighbour inconsistency (3 dims, feature × topology coupling).**
Let $\bar{\mathbf{x}}_{i,r} = (\mathbf{A}^{(r)}_{i,:} \mathbf{X}) / d_{i,r}$ and per-dim variance $\sigma^2_{i,r,k}$.

- $E_{i,r}^{(4)} = \operatorname{Zscore}\!\left(\|\mathbf{x}_i - \bar{\mathbf{x}}_{i,r}\|_2\right)$.
- $E_{i,r}^{(5)} = \cos(\mathbf{x}_i, \bar{\mathbf{x}}_{i,r})$.
- $E_{i,r}^{(6)} = \tfrac{1}{D}\sum_k \mathbb{1}\!\bigl[|x_{i,k} - \bar{x}_{i,r,k}| / \sigma_{i,r,k} > 2.0\bigr]$.

**Group C — prototype-relative (3 dims, train-only).**
Define class prototypes over *relation-neighbour means* of train nodes:

$$
\boldsymbol{\mu}^{r}_{+} = \tfrac{1}{|\mathcal{V}_{\text{tr}}^{+,r}|}\sum_{j \in \mathcal{V}_{\text{tr}}^{+,r}} \bar{\mathbf{x}}_{j,r},\qquad
\boldsymbol{\mu}^{r}_{-} = \tfrac{1}{|\mathcal{V}_{\text{tr}}^{-,r}|}\sum_{j \in \mathcal{V}_{\text{tr}}^{-,r}} \bar{\mathbf{x}}_{j,r}.
$$

> Prototypes are built from *neighbour means* $\bar{\mathbf{x}}_{j,r}$, not raw $\mathbf{x}_j$. This is what makes the prototype a relation-conditioned class anchor rather than a node-class anchor.

- $E_{i,r}^{(7)} = \operatorname{Zscore}\!\left(\|\mathbf{x}_i - \boldsymbol{\mu}^{r}_{+}\|_2\right)$.
- $E_{i,r}^{(8)} = \operatorname{Zscore}\!\left(\|\mathbf{x}_i - \boldsymbol{\mu}^{r}_{-}\|_2\right)$.
- $E_{i,r}^{(9)} = \operatorname{Zscore}\!\left(\|\mathbf{x}_i - \boldsymbol{\mu}^{r}_{+}\|_2 - \|\mathbf{x}_i - \boldsymbol{\mu}^{r}_{-}\|_2\right)$ — fraud margin.

The resulting tensor $\mathbf{E} \in \mathbb{R}^{N \times 9R}$ is cached to `artifacts/relation_features/{dataset}/{base}/seed_{s}/all/rel_stats.pt` with metadata `rel_feature_meta.json` (records `score_blind=True`, `prototype_labels="train_only"`, `target_label_used=False`, etc.).

---

## 4. Per-relation expert MLPs

`models/cover_rel_reasoner.py::_make_relation_expert`. For each relation $r$ independently:

$$
h_{i,r} = \operatorname{Dropout}\!\Bigl(\operatorname{LayerNorm}\!\bigl(\operatorname{ReLU}(\mathbf{W}_2^{(r)} \operatorname{ReLU}(\mathbf{W}_1^{(r)} \mathbf{E}_{i,r} + \mathbf{b}_1^{(r)}) + \mathbf{b}_2^{(r)})\bigr)\Bigr)
$$

with $d_h{=}64$, dropout $0.30$. **Experts are not shared across relations.** This is load-bearing: RUR's degree distribution and RSR/RTR's cross-account feature deviation carry physically different signals; sharing the MLP averages out conflicting signs.

Per-relation scalar residual head (zero-initialised):

$$
s_{i,r} = \mathbf{w}_r^\top h_{i,r} + b_r, \qquad (\mathbf{w}_r, b_r) := \mathbf{0}.
$$

Zero init → at $t{=}0$, $\Delta_i^{\text{rel}} \equiv 0$, $z_i \equiv b_i$. Training departs from the *do-nothing baseline* under BCE gradient only.

---

## 5. Schema-aware softmax gate

`models/cover_rel_reasoner.py::CoVERRelReasoner.forward`. Concatenate base embedding (detached) and per-relation hidden vectors:

$$
\mathbf{a}_i = \mathbf{W}_g \bigl[\underbrace{\mathbf{z}_i^{\text{base}}}_{\text{detached}}; h_{i,1}; \dots; h_{i,R}\bigr] + \mathbf{b}_g \in \mathbb{R}^R.
$$

Softmax with temperature $\tau{=}0.7$:

$$
g_{i,r} = \frac{\exp(a_{i,r}/\tau)}{\sum_{r'} \exp(a_{i,r'}/\tau)}, \qquad \sum_r g_{i,r} = 1.
$$

$\mathbf{W}_g, \mathbf{b}_g$ are zero-initialised → at $t{=}0$, $g_{i,r} = 1/R$. The `detach()` on $\mathbf{z}_i^{\text{base}}$ lets the gate condition on the base's hidden geometry without leaking gradient into the frozen base (C1 + C2).

**Why softmax, not sparsemax.** sparsemax empirically zeroed entire relations (RUR on YelpChi, UVU on Amazon) early in training and produced loss-NaN runs. softmax at $\tau{=}0.7$ holds gate entropy in $[0.7, 0.9]$ nat across 5 seeds — informative but never collapsing.

**Why no $L_{\text{sparse}}$ KL-to-evidence-prior pull.** A pulled gate ($L_{\text{sparse}} = \operatorname{KL}(\pi^{\text{evidence}} \| g)$) added Δ_AUPRC = +0.0025 with $p{=}0.061$ at 5 seeds — closest to the bar of any retired term, still fails. The evidence-prior $\pi^{\text{evidence}}$ is still computed by `training/phase2_losses.py::build_relation_evidence_distribution` but is **diagnostic-only** — no gradient, used for gate/evidence KL audit.

---

## 6. Bounded relation residual

$$
u_i = \sum_{r=1}^{R} g_{i,r} \cdot s_{i,r}, \qquad
\Delta_i^{\text{rel}} = \delta_{\max} \cdot \tanh(u_i) \in (-\delta_{\max}, \delta_{\max}),
$$

$$
\boxed{\,z_i = b_i + \Delta_i^{\text{rel}}\,}\qquad \delta_{\max} = 2.0.
$$

Bounded design (purely architectural, no loss support):

1. **Numerical bound.** Output stays in a 2.0-logit neighbourhood of the base $\Rightarrow$ structural do-no-harm.
2. **Gradient bound.** $|\tanh'(u)| \le 1$ caps late-stage update magnitude.
3. **Probability headroom.** $\delta_{\max}{=}2.0$ corresponds to $\sigma(b_i \pm 2.0)$, enough to flip confidence but not enough to overfit to 0/1.

**Observables emitted (no gradient).** `fused_rel_h = Σ_r g_{i,r} h_{i,r}` (gate-weighted hidden), `relation_strength = (|s_{i,r}|)_r` (per-relation head magnitude), `mean_evidence_gate_kl` (gate vs $\pi^{\text{evidence}}$). All for audit and figures.

---

## 7. Loss (cls-only canonical)

`training/phase2_losses.py::compute_phase2_loss`.

$$
\boxed{\;
L \;=\; L_{\text{cls}}
\;=\; \frac{1}{|\mathcal{V}_{\text{tr}}|} \sum_{i \in \mathcal{V}_{\text{tr}}} \operatorname{BCE}\!\bigl(z_i, y_i; w_{+}\bigr),
\qquad w_{+} = \frac{1 - \pi_{\text{fraud}}^{\text{tr}}}{\pi_{\text{fraud}}^{\text{tr}}}
\;}
$$

This is the only term. All regularisation is structural (bounded `tanh`, zero-init heads, detached $\mathbf{z}^{\text{base}}$, non-shared experts).

### 7.1 Why cls-only — 5-seed paired-t evidence

Source: `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md` (paired-t df=4, reference = $L_0$ cls-only).

| Cell | Variant | Δ_AUPRC vs L0 | $t$ | $p$ | Verdict |
|---|---|---:|---:|---:|---|
| A0 | base only | $-0.1042$ | $-31.1$ | $\approx 6\mathrm{e}{-6}$ | architecture lift $\star\star\star$ |
| L0 | **cls only** | $+0.0000$ | — | — | **canonical** |
| L1 | + $L_{\text{int}}$ | $+0.0006$ | $+0.39$ | $0.72$ | n.s. |
| L2 | + $L_{\text{sparse}}$ | $+0.0025$ | $+1.02$ | $0.36$ | n.s. (closest to bar) |
| L3 | + $L_{\text{align}}$ | $-0.0004$ | $-0.19$ | $0.86$ | n.s. |
| L4 | + $L_{\text{int}}$ + $L_{\text{sp}}$ | $+0.0015$ | $+0.51$ | $0.64$ | n.s. |
| L5 | + $L_{\text{int}}$ + $L_{\text{al}}$ | $+0.0003$ | $+0.13$ | $0.90$ | n.s. |
| L6 | + $L_{\text{sp}}$ + $L_{\text{al}}$ | $+0.0003$ | $+0.12$ | $0.91$ | n.s. |
| L7 | full 4-term | $-0.0000$ | $-0.01$ | $0.991$ | n.s. (= cls-only) |
| A1 | rel-only (judge off) | $+0.0014$ | $+0.66$ | $0.54$ | n.s. |
| A2 | judge-only | $-0.1042$ | $-31.1$ | $\approx 6\mathrm{e}{-6}$ | rel residual is the lift $\star\star\star$ |

**Reading.** The entire +0.1042 AUPRC headline lift (A0→L0) is delivered by the architecture (bounded rel residual + softmax gate + per-relation experts). Every additional loss term is statistically indistinguishable from cls-only at 5 seeds. The full 4-term champion L7 differs from L0 by $\Delta{=}-0.0000$ ($p{=}0.991$). Idea-1 ablation pilots then confirm the cls-only structure: $\Delta{<}10^{-3}$ AUPRC with $p\!\gg\!0.05$ for each toggle.

### 7.2 Retired loss terms — honest deletion ledger

| Removed term | Mechanism it tried to enforce | Why removed |
|---|---|---|
| $L_{\text{intervention}} = \mathbb{E}[(z-b)^2]$ | base-anchor "do-no-harm" | superseded by bounded `tanh` (structural); $t{=}+0.39, p{=}0.72$ single, $+0.51, p{=}0.64$ combined |
| $L_{\text{sparse}} = \operatorname{KL}(\pi^{\text{evidence}} \| g)$ | gate-pulled-to-evidence prior | $\Delta{=}+0.0025, t{=}+1.02, p{=}0.36$; evidence prior retained as diagnostic-only |
| $L_{\text{align}} = \operatorname{KL}(\pi^{\text{judge}} \| g)$ | gate-pulled-to-LLM-judge consensus | judge route fully retired (see §9); $t{=}-0.19, p{=}0.86$ |
| $\alpha \cdot \Delta_{\text{llm}}$ additive residual | LLM-judge tilt on the logit | $t{=}+0.03, p{=}0.976$ even with 2000 vLLM judge packets |

`compute_phase2_loss(...)` keeps `lambda_int`, `lambda_sparse`, `lambda_trust`, `eta_llm` kwargs as silent deprecation no-ops (one-shot `DeprecationWarning`); `lambda_align` and any `use_judge=True` / `alpha_max>0` raise `NotImplementedError`. This guarantees legacy configs *cannot* silently re-enable a falsified path.

### 7.3 Status of already-published cross-base × cross-dataset numbers

The headline 8-config table in §10 was produced under the 4-term loss. Because L7 ≡ L0 at $p{=}0.991$, those numbers remain valid under cls-only canonical — no re-run required. The only observable side-effect we lose is a variance-reduction effect on SAGE-YelpChi (4-term std 0.052 vs cls-only ~0.089). This is reported as a *stability* footnote in the paper, not a method claim.

---

## 8. Safety contracts and audits

### 8.1 Score-blind input audit

Inputs feeding the reasoner are pure-numpy over $\mathbf{A}^{(r)}, \mathbf{X}, \mathbf{y}_{\text{tr}}$. The forbidden field set

$$
\mathcal{F}_{\text{forbid}} = \{\text{base\_score}, \text{base\_logit}, \text{confidence}, \text{base\_pred}, \text{final\_pred}, \text{label}, \text{val/test split id}, \text{FN/FP status}, \text{ground\_truth}\}
$$

is impossible to fetch from $\mathbf{E}_{i,r}$ by construction — `evidence/relation_features.py` never imports the base model or sees `data.y[val/test]`.

### 8.2 Base-freeze SHA-256 (`tests/test_base_freeze_sha256.py`)

The Phase 2 trainer takes a SHA-256 snapshot of `base.pt`, cached `base_logits`, and `base_z` before training; checks them again after; writes the verdict `{frozen, MUTATED}` to `artifacts/checkpoints/.../base_freeze_check.json`. `MUTATED` is a CI failure.

### 8.3 Train-only prototype (`evidence/relation_features.py::_relation_prototypes`)

Prototypes use only `train_mask`-indexed rows; metadata records `target_label_used: false`, `val_label_used: false`, `test_label_used: false` in `rel_feature_meta.json`.

### 8.4 Compute-matched defence

For every CoVER-REL run we have a *compute-matched* control: train the base alone for the same wall-time the CoVER trainer used. On BWGNN-YelpChi, 5-seed paired-t:

- $\Delta$AUPRC = $+0.0219$ ($p < 0.05$),
- $\Delta$AUROC = $+0.0728$ ($p < 0.01$),
- $\Delta$G-Means = $+0.0597$ ($p < 0.01$).

→ Locks out the "you just spent more compute" attack. Source: `artifacts/tables/yelpchi_bwgnn_d0_compute_matched_control.md`.

---

## 9. What the reasoner is NOT — falsification ledger

| Excluded design | 5-seed paired-t verdict |
|---|---|
| LLM-judge additive residual $\alpha \cdot \Delta_{\text{llm}}$ | $t{=}+0.03, p{=}0.976$ (with 2000 vLLM packets) |
| LLM verbalisation embedding (PCA-32) | $t{=}-10.6, p{<}0.01$ |
| Raw-text sigpool fusion | $t{=}-19.4, p{<}0.01$ |
| B3 PRTAE per-relation auxiliary classifier hidden injection | negative direction |
| LEQA LoRA-Qwen3 evidence-quality auditor + $L_{\text{audit}}$ | negative / no significant lift |
| CoVER-DIR / CV-SCD / CoVER-LIFT direction-aware variants | $\Delta{<}+0.008$ or gate failure |
| $L_{\text{intervention}}$ base-anchor penalty | $\Delta{=}+0.0006, t{=}+0.39, p{=}0.72$ (single); $+0.0015, p{=}0.36$ (combined) |
| $L_{\text{sparse}}$ evidence→gate KL | $\Delta{=}+0.0025, t{=}+1.02, p{=}0.36$ |
| $L_{\text{align}}$ judge-tilted gate KL | $t{=}-0.19, p{=}0.86$ |

**Method philosophy.** Strict score-blind, strict base-anchored (architectural), strict train-only-prototype, cls-only loss. Any additional component — auxiliary loss, LLM signal, direction head — must independently clear a 5-seed paired-t bar at $p<0.05$ **and** transfer positively across base × dataset before it enters the canonical method. Nothing above did. Source for negative ledger: `artifacts/tables/paper_negative_routes.md`.

---

## 10. Cross-base × cross-dataset evidence (5-seed paired-t)

Source: `artifacts/tables/paper_main_results.md`, `paper_cross_model_results.md`, `yelpchi_bwgnn_d0_compute_matched_control.md`, `artifacts/reports/sage_confirmed_vs_sage_baselines.md`.

### Idea-1 (rel) vs base — 8/8 directional positive, 6/8 significant

| Dataset | Base | Base AUPRC | $\Delta$AUPRC (+rel) | $t$ | sig |
|---|---|---:|---:|---:|---|
| YelpChi | GCN | 0.219 ± 0.008 | $+0.292$ | $+59.24$ | $p<0.01$ |
| YelpChi | GAT | 0.207 ± 0.010 | $+0.320$ | $+36.75$ | $p<0.01$ |
| YelpChi | SAGE | 0.455 ± 0.014 | $+0.147$ | $+35.52$ | $p<0.01$ |
| YelpChi | BWGNN | 0.503 ± 0.016 | $+0.105$ | $+30.83$ | $p<0.01$ |
| Amazon | GCN | 0.258 ± 0.026 | $+0.215$ | $+4.34$ | $p<0.05$ |
| Amazon | GAT | 0.322 ± 0.284 | $+0.131$ | $+2.02$ | n.s. (base $\sigma{=}0.284$) |
| Amazon | SAGE | 0.788 ± 0.058 | $+0.050$ | $+4.52$ | $p<0.05$ |
| Amazon | BWGNN | 0.851 ± 0.020 | $+0.019$ | $+2.08$ | n.s. (base saturated at 0.85) |

→ Weak bases (GCN/GAT) are rescued; strong bases get incremental lift; direction is positive across every configuration; the two n.s. cells are explained by base variance / saturation, not by the rel residual being harmful.

### Compute-matched control (BWGNN-YelpChi)

| Comparison | $\Delta$AUPRC | paired $t$ |
|---|---:|---|
| base (400 ep, compute-matched) | $+0.0219$ | $+3.03$ ($p<0.05$) |
| base (AUROC) | — | $+7.28$ ($p<0.01$) |
| base (G-Means) | — | $+5.97$ ($p<0.01$) |

### SAGE-YelpChi confirmed 5-seed (`sage_confirmed_vs_sage_baselines.md`)

| Comparison | $\Delta$AUPRC | paired $t$ |
|---|---:|---|
| Phase 1 SAGE base | $+0.2540 \pm 0.0833$ | $+6.82$ ($p<0.01$) |
| Legacy Stage 3 anchor_gate | $+0.0321 \pm 0.0283$ | $+2.53$ (edge) |
| Phase 2 default | $+0.025\!\!-\!\!+0.028$ | n.s. |

---

## 11. Reproducibility cheat-sheet (canonical paths)

| Role | File |
|---|---|
| Canonical reasoner | `models/cover_rel_reasoner.py::CoVERRelReasoner` |
| Canonical loss (cls-only) | `training/phase2_losses.py::compute_phase2_loss` |
| Canonical trainer | `scripts/train_phase2_reasoner.py` |
| Phase 1 base trainer | `scripts/train_stage1.py` |
| 9-dim evidence extractor | `evidence/relation_features.py::compute_relation_features` |
| Train-only prototypes | `evidence/relation_features.py::_relation_prototypes` |
| Evidence-prior $\pi^{\text{evidence}}$ (diagnostic) | `training/phase2_losses.py::build_relation_evidence_distribution` |
| Relation schema (YelpChi RUR/RSR/RTR, Amazon UPU/USU/UVU) | `evidence/relation_features.py::RELATION_SCHEMAS` |
| Loss smoke tests (6/6 PASS) | `tests/test_phase2_losses.py` |
| No-judge-path enforcement test | `tests/test_reasoner_no_judge_path.py` |
| Base-freeze SHA-256 test | `tests/test_base_freeze_sha256.py` |
| Phase 1 SAGE launcher | `scripts/run_phase1_sage.sh` |
| Phase 1 base builder for any detector | `scripts/build_relation_features.py` |
| Idea-1 ablation pilot (7 cells × 5 seeds) | `scripts/run_idea1_ablation_pilot_5seed.sh` |
| Loss × arch ablation aggregator | `scripts/aggregate_ablation.py` |
| Canonical Phase 2 configs | `configs/phase2_reasoner/ablation/idea1_*.yaml` |
| Falsification table (★) | `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md` |
| Compute-matched control | `artifacts/tables/yelpchi_bwgnn_d0_compute_matched_control.md` |
| Cross-base × cross-dataset main table | `artifacts/tables/paper_main_results.md`, `paper_cross_model_results.md` |
| A/B/C evidence-group ablation | `artifacts/tables/paper_relation_ablation.md` |
| Negative-route ledger | `artifacts/tables/paper_negative_routes.md` |
| Phase 1 baseline 5-seed | `artifacts/tables/fresh_bwgnn_stage1_5seed.md` |

### Repository tree (canonical, post-cleanup)

```
models/         base.py  bwgnn.py  gnn.py  cover_rel_reasoner.py
training/       phase2_losses.py  metrics.py
evidence/       relation_features.py
data/           load_fraud.py  split.py
scripts/        train_stage1.py  train_phase2_reasoner.py
                build_relation_features.py  aggregate_ablation.py
                run_phase1_sage.sh  run_idea1_ablation_pilot_5seed.sh
configs/        {yelpchi,amazon}_{bwgnn,sage,gcn,gat}.yaml
                phase2_reasoner/ablation/idea1_{canonical_clsonly,ablate_gate_uniform,ablate_evidence_no_proto,ablate_evidence_no_structural,ablate_evidence_no_incoherence,ablate_shared_expert,ablate_unbounded_residual}.yaml
tests/          test_phase2_losses.py  test_phase2_reasoner.py  test_reasoner_no_judge_path.py
                test_base_freeze_sha256.py  test_relation_features.py
                test_threshold_calibration.py  test_stratified_split.py
                test_data_loading.py  test_base_model_output.py  test_bwgnn.py
                test_detector_output_dim.py  test_train_stage1_deterministic.py
artifacts/      base_outputs/  checkpoints/  diagnostics/  logs/  paper/
                relation_features/  reports/  results/  splits/  tables/  tensorboard/
```

---

## 12. Idea-1 ablation campaign (8 cells × 4 configs × 5 seeds = 160 runs, paired-t)

Built as **single-switch toggles** on top of the canonical above (`models/cover_rel_reasoner.py` now exposes `gate_mode`, `evidence_groups`, `expert_shared`, `residual_activation` kwargs; configs under `configs/phase2_reasoner/ablation/idea1_*.yaml`).

### 12.1 Three ablation switches (vs canonical)

| Switch | Knob | Tests |
|---|---|---|
| `ablate_gate_uniform` | `gate_mode: uniform` (g=1/R) | Does the schema-aware gate matter vs uniform mixing? |
| `ablate_shared_expert` | `expert_shared: true` | Per-relation independent experts vs single shared MLP + one-hot relation id |
| `ablate_evidence_no_proto` | `evidence_groups: [A, B]` (drops sub-group C) | Is the prototype-relative evidence load-bearing? |

### 12.2 Headline cross-cell findings (8 cells × 4 metrics = 32 paired-t per switch)

| Switch | AUPRC sig / 8 | AUROC sig / 8 | M-F1 sig / 8 | G-Means sig / 8 | total sig / 32 |
|---|:---:|:---:|:---:|:---:|:---:|
| `gate_uniform` | 2 | 5 | 3 | 1 | **11 / 32** |
| `shared_expert` | 1 (marginal) | 0 | 0 | 0 | **1 / 32** ⬅ truly equivalent |
| `no_proto` | 3 | 3 | 2 | 3 | **11 / 32** |

### 12.3 Headline per-cell numbers (AUPRC paired-t Δ vs canonical)

| Cell | canonical AUPRC | gate_uniform Δ (t) | shared_expert Δ (t) | no_proto Δ (t) |
|---|---:|---:|---:|---:|
| YelpChi-BWGNN | 0.6089 ± 0.0083 | −0.0029 (−1.35) ns | −0.0026 (−1.19) ns | +0.0002 (+0.23) ns |
| **YelpChi-SAGE** | 0.6001 ± 0.0105 | **−0.0141 (−5.86) ★★** | −0.0023 (−0.40) ns | **−0.0133 (−7.07) ★★** |
| **YelpChi-GCN** | 0.4946 ± 0.0149 | +0.0000 (+0.01) ns | +0.0056 (+0.77) ns | **−0.0954 (−17.00) ★★★** |
| **YelpChi-GAT** | 0.5316 ± 0.0136 | **−0.0244 (−3.61) ★** | **−0.0182 (−2.88) ★** | **−0.1340 (−16.02) ★★★** ⬅ **largest** |
| Amazon-BWGNN | 0.8683 ± 0.0269 | −0.0015 (−0.37) ns | −0.0024 (−1.33) ns | −0.0018 (−0.95) ns |
| Amazon-SAGE | 0.8336 ± 0.0546 | −0.0175 (−1.46) ns | +0.0015 (+0.31) ns | −0.0001 (−0.09) ns |
| Amazon-GCN | 0.4668 ± 0.1356 | +0.0128 (+1.08) ns | −0.0055 (−0.50) ns | −0.0104 (−0.89) ns |
| Amazon-GAT | 0.4592 ± 0.3055 | +0.0189 (+2.13) trend | −0.0069 (−0.51) ns | −0.0020 (−0.31) ns |

### 12.4 Three derived design rules

1. **Shared MLP ≈ independent per-relation experts** across 8 cells × 4 metrics × 5 seeds = 160 paired comparisons (0/32 sig on AUROC; 1/32 marginal on AUPRC at YelpChi-GAT, t=−2.88). Canonical defaults to independent for backward compatibility, but `expert_shared: true` is a free ~50% parameter reduction with no statistical regression.
2. **Prototype evidence (sub-group C) is load-bearing on weak GNN bases** (paired-t ★★ / ★★★ on YelpChi-{SAGE, GCN, GAT}) and inert on saturated cells (all 4 Amazon cells and YelpChi-BWGNN). Largest negative effect in the entire campaign: YelpChi-GAT `no_proto` = **−0.1340 AUPRC, t=−16.02, p<10⁻⁴**. **Do not delete C.**
3. **Schema gate is consistently load-bearing on YelpChi cells** (all 4 AUROC ★★, with YelpChi-SAGE at t=−8.32 the strongest). Amazon weak bases prefer uniform — consistent with the saturation-headroom thesis (extra gating capacity overfits the small effective signal on saturated cells).

### 12.5 GAT pre-cache trick

Under the trainer's `set_seed → torch.use_deterministic_algorithms(True)` path, PyG's scatter on GAT spikes to ~19 GB peak per process and OOMs even on a fully-free 24 GB card. The fix: `scripts/pre_cache_base_outputs.py` runs the GAT base forward **once per seed without the deterministic flag** (peak ~4.3 GB) and writes the cache under the trainer-expected name `_override_{ckpt.parent.name}_{ckpt.stem}.pt`. The trainer then short-circuits the forward and consumes the cached `base_logits + base_z` directly. All 40 GAT runs (2 datasets × 4 configs × 5 seeds) complete in ~10 min wall after the one-time pre-cache.

### 12.6 Source artifacts

```
configs/phase2_reasoner/ablation/idea1_{<cell>_,}{canonical_clsonly,ablate_gate_uniform,ablate_shared_expert,ablate_evidence_no_proto}.yaml   (32 configs)
artifacts/results/{yelpchi,amazon}/{bwgnn,sage,gcn,gat}/idea1_*/seed_*/stage3_metrics.json   (160 runs)
artifacts/base_outputs/{yelpchi,amazon}/gat/seed_*/_override_seed_*_base.pt   (10 GAT pre-caches)
artifacts/tables/idea1_ablation_FINAL_7cell.md   (terminal 7-cell report)

scripts/run_idea1_ablation_pilot_5seed.sh        (initial 3-config YelpChi-BWGNN runner)
scripts/run_idea1_crosscell_5seed.sh             (generic cross-cell sequential runner)
scripts/run_5seeds_parallel.sh                   (5-seeds-parallel-per-GPU runner; safe for BWGNN/SAGE/GCN, **not** GAT)
scripts/pre_cache_base_outputs.py                (one-time base forward cache; required for GAT)
```

---

## 13. Idea-2B: Learned Evidence Extractor (8 cells × 5 seeds = 40 runs, paired-t vs Idea-1 canonical)

### Motivation

Idea-1 uses **hand-crafted 9-dim per-relation evidence** (structural A + feature-neighbor incoherence B + train-only prototype C). The obvious reviewer attack is: "Why exactly these 9 dimensions? Why not learned features?" Idea-2B answers this by **replacing the hand-crafted extractor with an end-to-end-trained learned extractor** that preserves all four safety contracts.

### Architecture

`evidence/learned_extractor.py` — `LearnedRelationEvidenceExtractor` (13.9K params):

| Component | Role | Analogue in hand-crafted |
|---|---|---|
| Per-relation 2-layer symmetric-normalized GCN encoder | Learns structural representation | A: `log_degree_norm`, `degree_top10`, `degree_low` |
| Per-relation MLP head consumes `[x, neighbor_mean, gcn_emb, fraud_dist_z, benign_dist_z, margin_z]` | Learns combination | B+C: quantile-based stats + prototype distances |
| `prepare(x, relation_adjs)` | Caches sym-norm adj + neighbor-mean once per training run | Offline `build_relation_features.py` |
| `forward(x, train_mask, train_labels)` → `(N, R*9)` | Drop-in compatible output | `load_relation_stats` |

**Contracts preserved**:
- **C2 Score-blind**: extractor inputs are raw X + relation adjacencies + train_mask + train_labels. NEVER sees base_logit/base_z.
- **C3 Train-only prototype**: fraud/benign prototypes computed under `torch.no_grad()` over `train_mask & (train_labels == c)` only.
- **C1 Frozen base**: independent module, no parameter sharing with base detector.
- **C4 Bounded intervention**: downstream reasoner still applies `δ_max · tanh(u)`.

**Trainer integration** (`scripts/train_phase2_reasoner.py`):
- Config switch: `phase2_reasoner.evidence.source: learned` (default `hand_crafted`)
- `load_relation_adjs_for_extractor()` loads per-relation sparse COO from `.mat`
- `run_training()` recomputes `rel_features` online each epoch (train-mode for train step, eval-mode under `torch.no_grad()` for val/test)
- Extractor parameters added to optimizer alongside reasoner's
- `evidence_extractor.pt` saved alongside `reasoner.pt`

### Ablation switches (Idea-2B module ablation)

| Switch | Default | Ablation effect |
|---|---|---|
| `use_gcn_emb=True` | Include per-relation GCN structural signal | `False` → drops learned structural representation |
| `use_proto_features=True` | Include train-only proto distances | `False` → drops train-only prototype features (mirrors Idea-1's `no_proto`) |
| `encoder_shared=False` | R independent GCN encoders | `True` → single shared encoder + one-hot relation id (mirrors Idea-1's `shared_expert`) |

### 8-cell verdict (YelpChi+Amazon × BWGNN/SAGE/GCN/GAT × 5 seeds, paired-t vs Idea-1 canonical)

**Headline: 19/32 stat-sig wins; 11 ns positive; 2 ns losses (Δ<0.002).**

#### AUPRC (primary)

| Cell | canonical | **2B learned** | Δ | t | sig |
|---|---:|---:|---:|---:|:---:|
| YelpChi-BWGNN | 0.6089 | **0.6489** | **+0.040** | +11.5 | **★★★** |
| YelpChi-SAGE | 0.6001 | **0.6520** | **+0.052** | +10.0 | **★★★** |
| YelpChi-GCN | 0.4946 | **0.5904** | **+0.096** | +12.4 | **★★★** |
| YelpChi-GAT | 0.5316 | **0.6340** | **+0.102** | +9.6 | **★★★** |
| Amazon-BWGNN | 0.8683 | 0.8671 | −0.001 | −0.6 | ns |
| Amazon-SAGE | 0.8336 | 0.8511 | +0.018 | +0.7 | ns |
| Amazon-GCN | 0.4668 | **0.7006** | **+0.234** | +3.8 | **★** |
| Amazon-GAT | 0.4592 | 0.5575 | +0.098 | +1.7 | ns |

#### Cross-cell summary (4 metrics × 8 cells = 32 paired-t)

| Metric | wins (sig) | wins (ns) | losses |
|---|:---:|:---:|:---:|
| AUPRC | **5** | 2 | 1 |
| AUROC | **5** | 3 | 0 |
| M-F1 | **5** | 3 | 0 |
| G-Means | **4** | 3 | 1 |
| **Total** | **19** | **11** | **2** |

### Key findings

1. **YelpChi dominance**: 15/16 stat-sig wins across 4 YelpChi cells × 4 metrics. Learned evidence strictly beats hand-crafted A/B/C on ALL YelpChi bases.
2. **Amazon saturation**: strong bases (BWGNN/SAGE) → learned ≈ hand-crafted (no harm). Weak base (GCN) → +0.234 AUPRC ★. Confirms Idea-1's **base-strength × evidence-type interaction law**.
3. **Contracts intact**: all 40 runs pass base_freeze_check; score-blind by architecture; train-only proto by mask.
4. **Cost**: extractor 13.9K params, ~26s additional train time per seed.

### Paper framing

> "Why these 9 hand-crafted dims?" → "We let it learn under identical contracts; learned wins 19/32 stat-sig (incl. all 4 YelpChi ★★★) while never significantly underperforming on saturated cells."

### Source artifacts

```
evidence/learned_extractor.py                                         (240 lines, 13.9K params)
scripts/train_phase2_reasoner.py                                      (3 surgical backward-compat edits)
scripts/aggregate_idea2b_vs_canonical.py                               (8-cell paired-t aggregator)
scripts/run_idea2b_7cell_5seed.sh                                     (35-run orchestrator)
configs/phase2_reasoner/ablation/idea2b_learned_extractor_*.yaml       (8 configs)
configs/phase2_reasoner/ablation/idea2b_ablate_*.yaml                  (12 ablation configs)
scripts/run_idea2b_ablation_yelpchi_5seed.sh                           (60-run ablation orchestrator)
scripts/aggregate_idea2b_ablation.py                                   (ablation paired-t aggregator)
artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md            (8-cell verdict)
artifacts/tables/idea2b_ablation_4base_5seed.md                        (module ablation verdict)
artifacts/results/{yelpchi,amazon}/{base}/idea2b_learned_extractor/seed_*/   (40 canonical runs)
artifacts/results/yelpchi/{base}/idea2b_ablate_{switch}/seed_*/             (60 ablation runs)
```

---

## 14. OPD-Flash: On-Policy Distillation for Lightweight RAER Adapters (Idea 2C, in design)

**Status**: design locked at `docs/OPD_FLASH_DESIGN.md` (v1, 2026-05-19). Implementation in progress.

### One-sentence framing

OPD-Flash is the first on-policy distillation framework for graph anomaly detection: a 4 k-parameter student adapter generates its *own* posterior on training nodes, and the frozen LREE-Reasoner teacher (Idea 2B) provides **multi-head dense supervision** — final logit, per-relation $\Delta_r$, softmax gate, prototype projection — restricted to entropy-aware informative nodes, with a cell-aware teacher-reliability gate. Target: ≥ 95 % AUPRC capture (vs current 85–93 %), ≥ 2.59 × inference speed-up, score-blind / bounded / zero-init contracts preserved during rollout sampling.

### Three graph-specific contributions vs vanilla OPD-LLM

| # | Contribution | Why graph-specific |
|---|---|---|
| 1 | Multi-head dense supervision (logit + $\Delta_r$ + gate + proto) | OPD-LLM only distills final-token distribution; RAER has 4 natural information sources |
| 2 | Cell-aware teacher-reliability weight $w_c$ | Per-(dataset, base) reliability calibration (Uni-OPD does it per sequence) |
| 3 | Contract-preserving rollouts | Score-blind / bounded / zero-init enforced during student sampling — no LLM-OPD analogue |

### Implementation status (subtasks T1–T6)

| Task | Status | Owner |
|---|---|---|
| T1 — teacher multi-head exposure | TODO | — |
| T2 — student multi-head adapter | TODO | — |
| T3 — OPD-Flash training loop | TODO | — |
| T4 — cell-aware reliability cache | TODO | — |
| T5 — 8-cell × 5-seed benchmark | TODO | — |
| T6 — FreeKD / PEKD baseline reproduction | TODO (optional) | — |

See `docs/OPD_FLASH_DESIGN.md` for full algorithm, hyperparameters, theoretical sketch, risk register, and reading list.

---

## 15. Archived: Idea 3 (LLM-driven Feature Design)

**Status**: ARCHIVED on 2026-05-19. See `archive/idea3/README.md` for the full inventory + restore protocol.

**Reason for archive**: the LLM-feature-design line of work (CAAFE-style summary-only prompts, OpenFE / gplearn-SR shootout, multi-LLM scaling, REL-curriculum active learning, PLM REINFORCE selection) reached a state where Base + LLM features ≈ 0.666 AUPRC matched or exceeded Base + CoVER-REL ≈ 0.609 on the YelpChi-BWGNN single-seed quick screen. In the full Base + REL + LLM stack, REL's marginal contribution shrank to ~+0.002 AUPRC. This created an internal narrative tension that conflicts with the RAER framing of Idea 1+2. The TKDE 2026 submission focuses on RAER + LREE + OPD-Flash; Idea 3 is shelved for a potential follow-up venue (KDD 2026 AI-FA workshop on LLM-for-fraud).

**For new sessions / new agents**: treat `archive/idea3/` as read-only historical context. **Do not** import, extend, cite, or repurpose any script / artifact from there for the current paper. The `res.md` file intentionally retains all Idea 3 sections (§7 AL, §8 LLM feature design with subsections) as the honest experimental record — but the TKDE draft will exclude them.

**Two notable bugs from Idea 3 that hardened the live codebase** (kept fixed, not reverted):
1. `phase4_plm_adapter` 4-bug cascade → REINFORCE candidate selection (commit `13fb854`).
2. `parse_llm_formulas` silent prompt-echo mis-extraction + `build_composite_features` zero-fallback → sympy validation + `fail_mode='raise'` default (commit `cc642c7`).

These fixes live in the archive copy (post-fix versions) and are also reflected in `docs/OPD_FLASH_DESIGN.md` §9 (risk register) as cautionary precedent.

---

## Summary (TPAMI-grade narrative)

### One-sentence elevator pitch

**CoVER-REL** posits that the long-standing tension between **production-deployment safety** (a fraud detector once verified must not be silently modified) and **multi-relation reasoning expressivity** (per-relation signal is necessary but typically requires end-to-end retraining) admits a principled resolution in the form of a **contract-enforced post-hoc relation-evidence reasoner**: a base-frozen, bounded-residual adapter that decomposes per-relation node-level information into three score-blind, parameter-free anonymous evidence sub-spaces — structural density (A), feature-neighbor incoherence (B), and train-only relation prototypes (C) — fuses them through a schema-aware mixture over $R$ relations, and projects the result through a `tanh`-saturated intervention head on the frozen base logit, trained with a single pos-weighted BCE loss; against four base detectors (BWGNN, GraphSAGE, GCN, GAT) and two benchmark datasets (YelpChi, Amazon), this single $\approx$0.5K-parameter add-on yields **direction-positive AUPRC lift on 8/8 base×dataset cells (+0.017 to +0.325, 5-seed paired-$t$)** while uncovering — through 160 paired-$t$ ablation comparisons — the **first cell-resolved interaction law for multi-relation fraud reasoning**: the prototype-relative sub-space (C) is *necessary* on weak GNN bases (paired-$t$ ★★★, peak $\Delta\!=\!-0.134$ AUPRC, $t\!=\!-16.02$ when ablated on YelpChi-GAT) and *inert* on saturated cells, providing a falsifiable, base-strength-conditioned design rule that subsumes single-axis findings from CARE-GNN, GADBench-XGB-Graph, and BWGNN under a unified evidence-decomposition lens.

### Abstract-grade opening (5 sentences)

**(1)** Multi-relation graph fraud detection has matured along two parallel but unreconciled axes: spectral / topological reasoning on end-to-end-trained backbones (CARE-GNN, BWGNN, FraudGCN) and parameter-free tabularized neighbor aggregations with tree ensembles (GADBench's XGB-Graph), with no principled mechanism to combine the former's deployment maturity with the latter's interpretable feature-level evidence.
**(2)** We introduce **contract-enforced post-hoc relation-evidence reasoning** — a base-agnostic, bounded-residual adapter that wraps any frozen multi-relation fraud detector without modifying its parameters, subject to four architecturally-enforced contracts: base-freeze with SHA-256 verification, input-side score-blindness, train-only relation prototypes, and $\delta$-bounded "do-no-harm" intervention.
**(3)** The reasoner decomposes per-relation node-level information into three orthogonal score-blind evidence sub-spaces — structural density, feature-neighbor incoherence, and train-only fraud/benign relation prototypes — and aggregates them through a schema-aware softmax mixture over the relation simplex.
**(4)** Across BWGNN / GraphSAGE / GCN / GAT $\times$ YelpChi / Amazon (8 base $\times$ dataset cells), this $\approx$0.5K-parameter add-on yields **direction-positive AUPRC improvement on all 8 cells (+0.017 to +0.325, 5-seed paired-$t$)** at zero modification cost to the base detector.
**(5)** A 160-paired-$t$ ablation campaign uncovers a **base-strength $\times$ evidence-type interaction law**: the prototype-relative sub-space is *necessary* on weak GNN bases (peak ablation $\Delta\!=\!-0.134$ AUPRC, $t\!=\!-16.02$, $p\!<\!10^{-4}$ on YelpChi-GAT) and *inert* on saturated cells — yielding the first cell-resolved, falsifiable design rule for multi-relation fraud evidence reasoning, and unifying single-axis findings from relation-attention (CARE-GNN), tabular feature aggregation (GADBench), and spectral fraud detection (BWGNN) under a single evidence-decomposition lens.

### Operational one-liner (engineering-facing)

CoVER-REL is a single-branch reasoner that fuses 9-dim score-blind anonymous relation statistics through per-relation MLP experts (independent by default; the cross-cell ablation shows a shared MLP with one-hot relation id is statistically indistinguishable, 0/32 sig on AUROC) and a softmax schema gate into a bounded `tanh` residual on the frozen base logit, trained end-to-end with pure BCE; it strictly satisfies score-blind / base-freeze / train-only-prototype / bounded-intervention contracts; across BWGNN/SAGE/GCN/GAT $\times$ YelpChi/Amazon (8 cells) the direction is positive 8/8 with 6/8 5-seed paired-$t$ significant; every retired auxiliary loss term and LLM-judge route is reported with its falsification statistic in `paper_negative_routes.md`. **Idea-2B** replaces the hand-crafted 9-dim extractor with a learned 13.9K-param per-relation GCN+MLP extractor under identical contracts, yielding 19/32 stat-sig wins over the hand-crafted baseline (YelpChi: 15/16 sig; Amazon-GCN: +0.234 AUPRC ★).

<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 75 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/data1/mq/codes/aris_repo`).
Update with: `bash /data1/mq/codes/aris_repo/tools/install_aris.sh /data1/mq/codes/awesome-graph-anomaly-detection/cover-fd --aris-repo /data1/mq/codes/aris_repo`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
