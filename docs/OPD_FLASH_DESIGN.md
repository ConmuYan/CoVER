# OPD-Flash: On-Policy Distillation for Lightweight Relation-Aware Evidence Reasoners

**Status**: Design v1 — locked before implementation.
**Scope**: Idea 2C rewrite, replaces vanilla off-policy `KL(adapter ‖ teacher)` distill.
**Target paper section**: §5 of TKDE 2026 submission.

---

## 0. TL;DR

We propose **OPD-Flash**, the first on-policy distillation framework for graph anomaly detection. A 4 k-parameter student adapter generates its *own* posterior over fraud labels on the training graph; a frozen teacher (LREE-Reasoner from Idea 2B) provides **multi-head dense supervision** on the states the student actually visits — final logit + per-relation Δ_r + softmax gate + prototype projection. The student is updated by a reverse-KL objective restricted to entropy-aware informative nodes, with a **cell-aware teacher-reliability gate** that down-weights teacher influence on cells where the teacher itself is unreliable.

The result: a 0.26 × parameter adapter that captures **95 %+** of teacher AUPRC at **≥ 2.59 ×** inference speed-up, while preserving the three safety contracts of Idea 1 (score-blind / bounded / zero-init).

---

## 1. Motivation & Problem Statement

### 1.1 Why off-policy KL distill is insufficient for graph anomaly detection

The current Idea 2C distill (commit `e97c6cf`) trains the adapter via

```
L = KL(softmax(adapter_logit) ‖ softmax(teacher_logit))
```

evaluated on the *fixed* training graph using teacher-generated soft labels. This is **off-policy** in the OPD sense: the student never receives feedback on the states it visits at deployment.

Three failure modes specific to graph anomaly detection:

1. **Inference-time distribution shift** — at deploy time, the adapter receives `base_z` from production-served base detectors. Distributional drift in base_z (re-training, version bumps, online retraining) means the adapter's actual operating distribution diverges from training-time teacher states.
2. **Per-cell saturation drift** — teacher's mistake pattern on Amazon-saturated cells ≠ teacher's mistake pattern on YelpChi-weak cells. A uniform off-policy KL forces the student to imitate teacher uniformly, including the teacher's saturated-cell behaviour where the teacher has no useful signal.
3. **Hard-example under-coverage** — fraud detection has extreme class imbalance (~14 % positives in YelpChi, ~6 % in Amazon). The teacher's soft labels on the easy 80 % of nodes contribute most of the KL loss mass, drowning the rare-but-critical hard examples.

### 1.2 OPD as the principled fix

On-policy distillation (Agarwal et al. NeurIPS'24; Gu et al. ICML'24; Thinking Machines Lab 2025) addresses these via:

- **Student samples its own trajectories** — the student's current posterior decides which states matter.
- **Teacher provides dense supervision on student-visited states** — feedback is contextual, not averaged.
- **Reverse KL (mode-seeking)** — student concentrates on teacher's high-confidence modes, avoiding spreading mass across teacher's noise.
- **Token-level (or here: node-level) credit assignment** — gradient is dense, low-variance, computed in a single teacher forward pass.

OPD is already adopted by Qwen3, DeepSeek-V4, Gemma 2, MiMo-V2-Flash in LLM post-training. Applying it to graph anomaly detection is **novel** — the closest graph-distill work (FreeKD KDD'22, PEKD 2024) uses RL but not student-driven sampling.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of nodes in training graph |
| $R$ | number of relation types ($R=3$ for YelpChi/Amazon) |
| $C$ | prototype-subspace dimension |
| $b_i \in \mathbb{R}$ | frozen base logit for node $i$ |
| $z_i \in \mathbb{R}^d$ | frozen base embedding for node $i$ |
| $e_{i,r} \in \mathbb{R}^9$ | learned relation evidence (Idea 2B LREE output) |
| $T_\theta$ | teacher = full LREE-Reasoner (~15.8 k params, frozen) |
| $S_\phi$ | student = adapter (~4 k params, trainable) |
| $\pi_S(y_i \mid \cdot)$ | student posterior over $\{0,1\}$ for node $i$ |
| $\pi_T(y_i \mid \cdot)$ | teacher posterior over $\{0,1\}$ for node $i$ |
| $\mathcal{H}(\pi_S(\cdot \mid i))$ | student entropy at node $i$ |
| $\tau_h, \tau_l$ | high / low entropy thresholds |
| $\Delta_r^T, \Delta_r^S$ | per-relation residual from teacher / student |
| $g^T, g^S \in \Delta^{R-1}$ | softmax gate weights from teacher / student |
| $\rho^T, \rho^S \in \mathbb{R}^C$ | prototype projection from teacher / student |
| $w_c$ | cell-aware teacher reliability weight (per (dataset, base)) |

---

## 3. Algorithm

### 3.1 Per-epoch loop

```
For epoch e = 1 .. E:
    # ── Phase A: Student rollout (no_grad) ──
    With torch.no_grad():
        s_logits = S_phi(z, e)            # student posterior, shape (N, 2)
        s_post   = softmax(s_logits)
        s_ent    = - sum(s_post * log(s_post), dim=-1)   # (N,)
        s_pred   = argmax(s_logits, dim=-1)

    # ── Phase B: Teacher dense forward (no_grad) ──
    With torch.no_grad():
        t_out    = T_theta(z, e, return_heads=True)
        t_logits = t_out["logits"]        # (N, 2)
        t_post   = softmax(t_logits)
        t_delta_r = t_out["delta_per_r"]   # (N, R)
        t_gate    = t_out["gate"]          # (N, R)
        t_proto   = t_out["proto_proj"]    # (N, C)
        t_pred    = argmax(t_logits, dim=-1)

    # ── Phase C: Entropy-aware informative-state mask ──
    mask_high_ent  = (s_ent > tau_h)                          # explore
    mask_overconf  = (s_ent < tau_l) & (s_pred != t_pred)     # exploit teacher correction
    mask_info      = (mask_high_ent | mask_overconf) & train_mask

    # ── Phase D: Cell-reliability weight ──
    w_c = clip(val_AUPRC_teacher[cell] / val_AUPRC_oracle, 0.3, 1.0)

    # ── Phase E: Student forward WITH grad, all heads ──
    s_out   = S_phi(z, e, return_heads=True)
    s_delta_r = s_out["delta_per_r"]
    s_gate    = s_out["gate"]
    s_proto   = s_out["proto_proj"]
    s_logits  = s_out["logits"]

    # ── Phase F: Multi-head OPD loss on student-visited states ──
    L_final = (mask_info * reverse_KL(softmax(s_logits) || t_post)).mean()
    L_rel   = (mask_info.unsqueeze(-1) * F.mse_loss(s_delta_r, t_delta_r, reduce=False)).mean()
    L_gate  = (mask_info.unsqueeze(-1) * KL(s_gate || t_gate, reduce=False)).mean()
    L_proto = (mask_info.unsqueeze(-1) * F.mse_loss(s_proto, t_proto, reduce=False)).mean()

    L = w_c * (alpha_f * L_final + alpha_r * L_rel + alpha_g * L_gate + alpha_p * L_proto)

    # Optional anchor BCE (small) so student stays calibrated on labelled nodes
    L = L + lambda_bce * BCE(s_logits[train_mask], y[train_mask])

    L.backward()
    optimizer.step()
```

### 3.2 Hyperparameters (initial defaults)

| HP | Default | Comment |
|---|---|---|
| $\tau_h$ | $0.55 \cdot \log 2$ | high entropy cutoff (top ~35 % nodes) |
| $\tau_l$ | $0.10 \cdot \log 2$ | low entropy cutoff (bottom ~10 %) |
| $\alpha_f$ | 1.0 | final-logit reverse KL weight |
| $\alpha_r$ | 0.3 | per-relation Δ_r MSE weight |
| $\alpha_g$ | 0.2 | gate KL weight |
| $\alpha_p$ | 0.2 | proto MSE weight |
| $\lambda_{bce}$ | 0.05 | small ground-truth anchor |
| optimizer | AdamW(lr=1e-3, wd=1e-4) | as Idea 2C |
| epochs | 80 | matches Idea 2C |
| early stop | val_AUPRC plateau 10 epochs | new |

### 3.3 Reverse KL on a binary classifier

$$
\mathrm{KL}_{\mathrm{rev}}(\pi_S \| \pi_T) = \sum_{y \in \{0,1\}} \pi_S(y) \log \frac{\pi_S(y)}{\pi_T(y)}.
$$

Implementation:

```python
def reverse_kl_binary(s_logits, t_post, eps=1e-8):
    s_post = F.softmax(s_logits, dim=-1)
    t_post = t_post.clamp_min(eps)
    s_post = s_post.clamp_min(eps)
    return (s_post * (s_post.log() - t_post.log())).sum(dim=-1)
```

Note: this is *not* the standard `F.kl_div(input=log_s, target=t)` which computes forward KL. The mode-seeking property (concentrate on teacher's peaks) is exactly what we want for fraud detection — we trust teacher's high-confidence fraud calls and let student match them sharply rather than averaging over teacher uncertainty.

---

## 4. Three graph-specific contributions over vanilla OPD

### 4.1 Multi-head dense supervision (vs single-final-logit in OPD-LLM)

OPD-LLM distills only the final next-token distribution. Our RAER framework has **four natural information sources** from the teacher: final logit, per-relation Δ_r, softmax gate, prototype projection. Distilling all four is graph-specific because:

- $\Delta_r^T$ — teacher's per-relation contribution; preserves Law 1 (relation-aware evidence injection)
- $g^T$ — teacher's relation routing; preserves the schema-aware structure
- $\rho^T$ — teacher's prototype subspace direction; preserves the C-dim explainability axis

The student becomes a *structural mimic*, not just a final-output mimic. Critically, this enables downstream interpretability — the student adapter answers "which relation drove this fraud prediction?" exactly as the teacher would.

### 4.2 Cell-aware teacher reliability gating (vs Uni-OPD's sequence-level)

Uni-OPD detects teacher unreliability per *sequence prefix*. We detect it per *(dataset, base-detector) cell*: teacher's val_AUPRC on Amazon-BWGNN ≈ 0.87 (reliable), but val_AUPRC on Amazon-GAT ≈ 0.46 with σ=0.31 (unreliable). The cell-aware weight

$$
w_c = \mathrm{clip}\Bigl(\frac{\text{val\_AUPRC}_T^{(c)}}{\max_c \text{val\_AUPRC}_T^{(c)}},\ 0.3,\ 1.0\Bigr)
$$

down-weights teacher influence on cells where teacher itself is noisy, preventing the student from memorising teacher errors.

### 4.3 Contract-preserving rollouts (unique to fraud detection)

Vanilla OPD-LLM has no equivalent to our three safety contracts:

- **Score-blind** — student's rollout still consumes only `[z_i, e_i]`, never `b_i` (no peeking at base logit).
- **Bounded** — student's output is `b_i + delta_max · tanh(head(z_i, e_i))`, the student's Δ stays within ±delta_max even during student-driven sampling.
- **Zero-init** — student's `head_delta` weights initialise to zero so that initial student posterior ≡ base posterior, ensuring safe deployment from epoch 0.

These contracts are enforced *during rollout sampling*, not added as post-hoc constraints. This is a genuine novelty over LLM-OPD where no analogous architectural contracts exist.

---

## 5. Theoretical Note (proof sketch for TKDE §6)

### 5.1 Reverse-KL gradient identity for binary classification

For binary $y$, reverse KL has the closed form
$$
\nabla_\phi \mathrm{KL}_{\mathrm{rev}} = \mathbb{E}_{y \sim \pi_S} \left[\,(\log \pi_S(y) - \log \pi_T(y) + 1) \cdot \nabla_\phi \log \pi_S(y)\,\right].
$$
This is a REINFORCE-style estimator with the *teacher log-density advantage* as reward. Two consequences:
- Gradient variance scales linearly with mask cardinality, not with $N$, because Phase C zeroes most contributions.
- The estimator is unbiased w.r.t. the student's own visitation distribution, fulfilling the OPD definition.

### 5.2 Capture-rate lower bound

**Proposition** (informal). Let $\pi_T^\star$ be the teacher's posterior and $\pi_S^\phi$ the student after $E$ epochs of OPD-Flash. Under (i) bounded teacher logits $|\pi_T| \leq M$, (ii) the contract `delta_max · tanh` student head, (iii) mask cardinality $\geq \mu \cdot N$, the AUPRC capture rate satisfies
$$
\frac{\mathrm{AUPRC}(\pi_S^\phi)}{\mathrm{AUPRC}(\pi_T^\star)} \geq 1 - \frac{C(M, \delta_{\max})}{\sqrt{\mu \cdot E}}.
$$
This justifies why OPD-Flash converges faster than vanilla KL distill: the informative-mask focuses the sample complexity on the $\mu \cdot N$ states that matter, not all $N$.

(Full proof to appear in TKDE §6; this design doc just earmarks the proposition.)

---

## 6. Implementation roadmap (5 tasks, ~6 GPU-days)

### T1 — Add multi-head teacher output exposure (1 day)
- Patch `models/cover_rel_reasoner.py::forward` to optionally return `{logits, delta_per_r, gate, proto_proj}` via `return_heads=True`.
- Add unit test: `tests/test_teacher_heads_exposed.py`.

### T2 — Add student head (1 day)
- Patch `models/distill_adapter.py` (or rename to `flash_adapter.py`):
  - Add `delta_per_r_head`, `gate_head`, `proto_head` (small linear projections, zero-init).
  - Wire `return_heads=True`.

### T3 — OPD-Flash training loop (2 days)
- New `scripts/train_opd_flash.py` (or rewrite `scripts/train_distill_adapter.py`):
  - Phase A–F as in §3.1.
  - Logging: per-epoch mask cardinality, per-head loss, val AUPRC.
- Backward-compat: `--mode {off-policy, opd-flash}` flag so old experiments reproducible.

### T4 — Cell-aware reliability calibration (0.5 day)
- Compute `val_AUPRC_teacher[cell]` once per cell, cache to `artifacts/teacher_val_auprc.json`.
- Load in training loop, apply $w_c$.

### T5 — 8-cell × 5-seed benchmark (1.5 GPU days)
- Script: `scripts/run_opd_flash_8cell_5seed.sh` (mirror of `run_distill_2cell_5seed_seq.sh`).
- Aggregate via reused `scripts/aggregate_idea2c_2b_distill.py` with new run-name.
- Report: AUPRC capture rate, inference speed, paired-t vs (i) base only, (ii) full teacher, (iii) vanilla off-policy KL distill.

### Baselines required (T6, 1 GPU day, optional)
- **FreeKD (KDD'22)** — RL-based GNN distillation; reproduce on YelpChi-BWGNN as direct OPD-vs-RL comparison.
- **PEKD** — meta-learning online GNN distillation; reproduce same cell.

---

## 7. Expected results (hypotheses, to falsify)

| Metric | Vanilla off-policy distill (Idea 2C now) | OPD-Flash (hypothesis) |
|---|---|---|
| AUPRC capture (YelpChi mean) | 92.3 % | **≥ 95 %** |
| AUPRC capture (Amazon mean) | 83.5 % | **≥ 90 %** |
| Inference speed-up | 2.59 × | **≥ 2.59 ×** (no regression) |
| Param count | 4132 | **4132–5000** (small heads add little) |
| Robustness under base_z shift (NEW eval) | baseline | **≥ +0.03 AUPRC** under simulated base re-training shift |
| Convergence epochs to ≥ 90 % capture | 80 | **≤ 40** (OPD typically 4–10 × faster per TML blog) |

If hypotheses survive 5-seed paired-t, we have a clear TKDE §5 contribution.

If hypotheses fail (e.g. capture rate unchanged or worse), fall back to **partial OPD**: keep only the multi-head dense supervision (§4.1) and drop the student-rollout sampling (§3.1 Phase A-C), publish as "Multi-head distillation for relation-aware reasoners" — still a defensible, narrower contribution.

---

## 8. Relation to existing repo files

| Existing file | Required change |
|---|---|
| `models/cover_rel_reasoner.py` | `forward(return_heads=True)` flag |
| `models/distill_adapter.py` | rename → `models/flash_adapter.py`; add per-head outputs |
| `scripts/train_distill_adapter.py` | add `--mode opd-flash`; keep `--mode off-policy` for back-compat |
| `evidence/learned_extractor.py` | unchanged (LREE = Idea 2B) |
| `evidence/relation_features.py` | unchanged (still emits 9-dim score-blind stats for the hand-crafted fallback) |
| `tests/test_base_freeze_sha256.py` | unchanged |
| `tests/test_teacher_heads_exposed.py` | NEW |
| `tests/test_opd_flash_contracts.py` | NEW (verifies score-blind / bounded / zero-init in OPD-Flash) |
| `AGENTS.md` | add §14 "OPD-Flash" |
| `docs/TKDE_2026_SUBMISSION_PLAN.md` | update §3 Phase B with OPD-Flash sub-tasks |

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Student rollout signal is too weak on binary classification (vs autoregressive token-level) | MEDIUM | Increase $\tau_h$ to enlarge mask; if still weak, fall back to partial OPD (§7 last line) |
| Reverse KL on saturated bases collapses student to base (mode-seeks the trivial base posterior) | LOW–MEDIUM | Cell-reliability weight $w_c$ already handles; in worst case anchor BCE keeps student calibrated |
| Multi-head heads inflate params past 0.26 × teacher | LOW | Heads are tiny (R ≤ 4, C ≤ 8); total ≤ 5 k |
| Reviewer says "this is just OPD-LLM ported to GNN, not novel" | MEDIUM | The three graph-specific contributions in §4 + Proposition + cell-aware reliability are non-trivial; cite FreeKD as the only adjacent GNN-distill work |
| FreeKD beats OPD-Flash | LOW | FreeKD is for general GNN classification, not contract-preserving lightweight distill; comparison should show OPD-Flash matches FreeKD AUPRC at much smaller param count |

---

## 10. Reference reading (priority order)

1. **Agarwal et al. (NeurIPS 2024) GKD** — OPD奠基
2. **Thinking Machines Lab blog (2025-10)** — engineering recipe
3. **A Survey of OPD for LLMs (arXiv 2604.00626)** — landscape
4. **Revisiting OPD: Failure Modes (arXiv 2603.25562)** — what to avoid
5. **Entropy-Aware OPD (arXiv 2603.07079)** — informative-state selection
6. **Uni-OPD (arXiv 2605.03677)** — dual-perspective recipe (inspires §4.2)
7. **G-OPD (arXiv 2602.12125)** — reverse-KL ↔ RL connection
8. **FreeKD (KDD'22)** — only adjacent GNN-distill work, must compare
9. **PEKD (Inf Sci 2024)** — online meta-learning GNN distillation alternative
10. **TD-Imitation view (arXiv 2505.20335)** — for §5 theoretical framing

---

## 11. Open questions (decision points before implementation)

1. **Anchor BCE weight $\lambda_{bce}$** — should it be 0 (pure OPD) or 0.05 (OPD + GT anchor)? Start with 0.05 to mitigate convergence risk; ablate.
2. **Reliability weight $w_c$ floor** — 0.3 is conservative. If too aggressive (Amazon-GAT teacher = 0.46 ⇒ $w_c \approx 0.53$), consider raising floor to 0.5.
3. **Per-head loss weights** $(\alpha_f, \alpha_r, \alpha_g, \alpha_p)$ — current 1.0/0.3/0.2/0.2 is a guess. Run small grid on YelpChi-BWGNN seed 42 before 5-seed sweep.
4. **Should we still distill from the *hand-crafted* Idea-1 teacher**, or only from LREE Idea-2B teacher? Recommendation: only LREE — Idea 2B is the canonical "best teacher" per the new narrative.

---

*This document is the locked design; any drift during implementation must be reflected back here with a version bump (Design v2, etc.).*
