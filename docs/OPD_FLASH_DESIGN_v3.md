# G-OPD-Flash: Graph On-Policy Distillation for Contract-Preserving Lightweight Reasoners

**Status**: Design v3 — supersedes v1 (`docs/OPD_FLASH_DESIGN.md`, locked 2026-05-19).
**Scope**: C3 of TKDE 2026 submission; rewrite of vanilla off-policy KL distill (Idea 2C).
**Target paper section**: §5 of TKDE 2026 submission.
**Audit basis**: Codex `gpt-5.5` xhigh proof-checker (`PROOF_AUDIT.md`, FAIL/critical_gap on v1) + main-thread Opus 4.7 Q1–Q10 review.

---

## 0. TL;DR

We propose **G-OPD-Flash**, a *graph* on-policy distillation framework for contract-preserving lightweight RAER/LREE student adapters. Unlike LLM-OPD where the policy is over autoregressive token trajectories, G-OPD-Flash defines the policy **over student-selected node states**: each epoch the student induces a sampling distribution $q_\phi(i)$ over training nodes from its own posterior entropy, predicted-fraud probability, and residual magnitude; we draw a mini-batch from $q_\phi$ (gradient detached, GKD-style) and query the frozen teacher only on these student-selected nodes. The distillation loss is a **scalar Bernoulli entropy-aware mixed KL** (reverse-KL when teacher is confident, forward-KL when teacher is uncertain) over three auxiliary heads (final logit, per-relation contribution $c_r = g_r s_r$, gate $g_r$), summed and normalised by selected-node weight mass, with a **node-level reliability weight** and **adaptive BCE anchor** that takes over when teacher reliability is low.

**Targets** (5-seed paired-$t$ falsifiable, NOT proven achievements): $\geq 95\,\%$ AUPRC capture of the LREE teacher at $\geq 2.59\times$ inference speed-up under the four hard contracts of [§1 Operating Principles](../AGENTS.md#1-operating-principles): base-freeze SHA-256, score-blind input, train-only prototype, $\delta$-bounded residual.

---

## 0.1 Revision log (v1 → v3)

| # | v1 | v3 | Trigger |
|---|---|---|---|
| R1 | "first OPD for GAD" | **"first graph on-policy distillation: policy over student-selected node states, not token prefixes"** | Codex I1 FATAL + Opus Q1 + Q7 |
| R2 | deterministic entropy mask | **stochastic student-policy node sampling** $i \sim q_\phi$ (detached, GKD-style) | Codex I1 FATAL + counter-example CE-2 |
| R3 | 2-class softmax reverse-KL | **scalar Bernoulli mixed KL** $(1-\eta) \mathrm{KL}_{\mathrm{rev}} + \eta\,\mathrm{KL}_{\mathrm{fwd}}$ | Codex I2 CRITICAL + entropy-aware OPD lit |
| R4 | `.mean()` over all $N$ nodes | $\sum_i w_i \mathcal{L}_i / \max(\sum_i w_i, 1)$ over selected nodes | Codex I4 CRITICAL |
| R5 | REINFORCE framing of reverse-KL identity | **closed-form algebra**; REINFORCE only mentioned as `strict-OPD` ablation | Codex I3 MAJOR + Opus Q2 |
| R6 | AUPRC capture-rate lower bound $1 - C/\sqrt{\mu E}$ | **DELETED** (counter-example CE-1 in `PROOF_AUDIT.md`); replaced by **3 small provable propositions** | Codex I5 FATAL + CE-1 |
| R7 | "three safety contracts" (score-blind / bounded / zero-init) | **four hard contracts** (base-freeze SHA-256 / score-blind input / train-only proto / $\delta$-bounded residual) | Codex I9 CRITICAL |
| R8 | cell-aware scalar reliability "prevents memorising teacher errors" | **cross-cell curriculum prior** (normalized lift over prevalence) × **node-level reliability** × **adaptive $\lambda_{\mathrm{bce}}$** | Codex I7 MAJOR + I8 + CE-3 |
| R9 | 4 heads "independent supervision signals" (logit / $\Delta_r$ / gate / proto) | **3 auxiliary heads** (logit / $c_r = g_r s_r$ / gate); proto dropped pending canonical-teacher definition | Codex I11 MAJOR + I12 |
| R10 | C3 default uses LREE teacher (C2 prerequisite) | **teacher-agnostic recipe**; ablation runs over hand-crafted CoVER-REL teacher AND LREE teacher | Codex I13 CRITICAL |
| R11 | "achieves $\geq 95\,\%$ capture" | **"targets $\geq 95\,\%$ capture (5-seed paired-$t$ falsifiable)"** | Codex I6 CRITICAL + Opus Q8 |
| R12 | T1–T6 (5–6 GPU days) | **T1–T7 (9–10 GPU days)**: adds `strict-OPD` ablation mode (T6) + deployment-shift eval (T7) | Codex experiment-matrix + Opus Q6 |
| R13 | name `OPD-Flash` | **`G-OPD-Flash`** (graph-level OPD, paper-title friendly) | Codex naming recommendation |

---

## 1. Motivation & Problem Statement

### 1.1 Why vanilla off-policy KL distill is insufficient

The Idea-2C baseline (commit `e97c6cf`) trains the 4 k-parameter student adapter via
$$
\mathcal{L}_{\mathrm{off}} = \frac{1}{N_{\mathrm{train}}} \sum_{i \in \mathrm{train}} \mathrm{KL}\!\left(\sigma(s^S_i) \,\big\|\, \sigma(s^T_i)\right),
$$
evaluated on the **fixed** training graph using teacher-generated soft labels. Three concrete failure modes specific to graph fraud detection:

1. **Inference-time distribution shift over `base_z`.** At deployment, the student receives `base_z` from the production-served base detector; whenever the base is re-trained, version-bumped, or online-retrained, the student's actual operating distribution diverges from the training-time teacher cache. Off-policy distill has no feedback channel to adapt.
2. **Cell-saturation drift.** Teacher's residual pattern on Amazon-saturated cells $\neq$ teacher's residual pattern on YelpChi-weak cells. Uniform off-policy KL forces the student to mimic teacher uniformly, including saturated-cell behaviour where the teacher itself has no useful signal.
3. **Hard-example under-coverage.** GFD has extreme class imbalance ($\sim$14 % positives in YelpChi, $\sim$6 % in Amazon). Teacher soft labels on the easy 80 % of nodes contribute most of the KL mass, drowning rare-but-critical hard examples.

### 1.2 Why naive LLM-OPD does not transfer

LLM-OPD (Agarwal et al. NeurIPS'24 GKD; Gu et al. ICML'24 minILM; Thinking Machines Lab 2025-10) sets the policy over **autoregressive token trajectories**: $\pi_\phi(y_t \mid x, y_{<t})$. The training distribution shifts each epoch because the student samples its own next tokens; teacher provides token-level feedback on those samples. This depends on:

- (i) a **multi-step** action space generating non-trivial trajectory variation;
- (ii) **exposure bias** between teacher-forcing training and free-running inference.

GFD has **neither**: each node is a single-step binary classification; there is no autoregressive prefix; inference is teacher-forcing-equivalent. Directly porting LLM-OPD pseudocode yields either (a) a degenerate single-step REINFORCE with high variance (our `strict-OPD` ablation, §3.5) or (b) a fake-OPD deterministic-mask procedure that pretends to be on-policy (v1's mistake, surfaced by Codex CE-2).

### 1.3 G-OPD: the right abstraction for graph fraud detection

The clean translation of "on-policy" to GFD is to keep the **sampling distribution itself** policy-dependent, while letting the action space stay single-step. Concretely:

> **G-OPD principle.** The student induces a *node-sampling distribution* $q_\phi(i)$ each epoch from its current posterior; the teacher is queried only on nodes drawn from $q_\phi$; the gradient flows through the distillation loss on those samples, *not* through $q_\phi$ itself (detached, à la GKD §3 stop-gradient on sampling).

This preserves the on-policy spirit (training data depends on $\pi_\phi^e$) without requiring multi-step trajectories or high-variance REINFORCE. It also recovers LLM-OPD's two structural benefits in the GFD setting: (i) the student focuses teacher feedback on the states it actually visits at deployment-relevant distributions, and (ii) hard examples that the student's current posterior is uncertain about receive proportionally more teacher supervision.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of nodes in training graph |
| $R$ | number of relation types ($R=3$ for YelpChi/Amazon) |
| $b_i \in \mathbb{R}$ | frozen scalar base logit for node $i$ (NOT a 2-vector) |
| $z_i \in \mathbb{R}^d$ | frozen base embedding for node $i$ |
| $\phi_{i,r} \in \mathbb{R}^9$ | hand-crafted score-blind relation evidence (Idea 1 canonical) |
| $e_{i,r} \in \mathbb{R}^{d_e}$ | learned relation evidence (Idea 2B LREE output, $d_e = 9$) |
| $T_\theta$ | teacher RAER reasoner (hand-crafted CoVER-REL OR LREE-Reasoner), frozen |
| $S_\phi$ | student adapter ($\sim$4 k–5 k params, trainable) |
| $s^T_i, s^S_i \in \mathbb{R}$ | **scalar** teacher / student raw logit for node $i$ |
| $p^T_i = \sigma(s^T_i),\ p^S_i = \sigma(s^S_i) \in [0,1]$ | Bernoulli fraud-probability |
| $H(p) = -p\log p - (1-p)\log(1-p)$ | scalar Bernoulli entropy |
| $\delta^S_i \in [-\delta_{\max}, \delta_{\max}]$ | student residual on base logit; $s^S_i = b_i + \delta^S_i$ |
| $g^T_r, g^S_r \in \Delta^{R-1}$ | softmax gate over relations (teacher / student) |
| $s^T_{i,r}, s^S_{i,r} \in \mathbb{R}$ | per-relation pre-gate scalar (teacher / student) |
| $c^T_{i,r} = g^T_{i,r} \cdot s^T_{i,r}$ | **per-relation contribution** (gate-weighted) — well-defined for any RAER teacher |
| $q_\phi(i)$ | student-induced node-sampling distribution (§3.1) |
| $r^c \in [0,1]$ | cross-cell curriculum prior for cell $c$ (normalized lift, §3.4) |
| $r^{\mathrm{node}}_i \in [0,1]$ | node-level reliability for node $i$ |
| $\lambda_{\mathrm{bce}}(i)$ | adaptive BCE anchor weight, $\lambda_{\min} + (1 - r^{\mathrm{node}}_i)\lambda_{\mathrm{extra}}$ |
| $\eta_i = H(p^T_i) / \log 2 \in [0,1]$ | teacher entropy normalised to $[0,1]$ |
| $\tau$ | sampling temperature for $q_\phi$ |
| $K$ | mini-batch size sampled per epoch from $q_\phi$ |

**Convention.** All KL/log operations clamp probabilities to $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 10^{-8}$. "Scalar Bernoulli" means we operate on $p \in [0,1]$ rather than 2-vector softmax over $\{0,1\}$, fixing Codex I2 type mismatch.

---

## 3. Algorithm

### 3.1 Phase A — student forward; build $q_\phi$

With grad disabled:
$$
s^S_i = b_i + \delta^S_i,\quad p^S_i = \sigma(s^S_i),\quad H^S_i = H(p^S_i).
$$
Build unnormalised sampling weights:
$$
\tilde q_\phi(i) \;=\; \varepsilon \;+\; H^S_i \;+\; \alpha \cdot p^S_i \;+\; \beta \cdot |\delta^S_i|, \qquad i \in \mathrm{train}.
$$
Normalise: $q_\phi(i) = \tilde q_\phi(i)\,/\,\sum_{j} \tilde q_\phi(j)$. Defaults $\alpha = 0.5,\ \beta = 0.3,\ \varepsilon = 10^{-3}$.

Intuition: high-entropy nodes (student uncertain), predicted-positive nodes (focus on fraud calls), and large-residual nodes (student is actively intervening) get proportionally more teacher feedback.

### 3.2 Phase B — sample $K$ nodes; teacher sparse forward

Sample $\mathcal{B} = \{i_1, \ldots, i_K\}$ without replacement from $q_\phi$ (default $K = \min(2048,\ N_{\mathrm{train}})$). **Detach $q_\phi$** — no gradient flows back through sampling weights (GKD §3 stop-gradient convention).

Teacher forward only on $\mathcal{B}$ (saves $\sim 70\,\%$ of teacher FLOPs vs full-graph):
$$
\{s^T_i,\ \{s^T_{i,r}\}_{r=1}^R,\ g^T_i\}_{i \in \mathcal{B}} \;\leftarrow\; T_\theta\!\left(z, \{\phi_{i,r}\}\,\text{or}\,\{e_{i,r}\}\right).
$$
Derive $p^T_i = \sigma(s^T_i)$ and per-relation contributions $c^T_{i,r} = g^T_{i,r} s^T_{i,r}$.

### 3.3 Phase C — student forward with grad on $\mathcal{B}$, three heads exposed

The student adapter, **enabled with `return_heads=True`**, returns three quantities on $\mathcal{B}$:

1. **Final scalar logit head**: $s^S_i = b_i + \delta_{\max} \tanh(W_\delta [z_i, \overline{e_i}])$ where $W_\delta$ is zero-initialised (bounded + zero-init contracts).
2. **Per-relation contribution head**: $c^S_{i,r} = g^S_{i,r} s^S_{i,r}$ with $g^S, s^S$ produced by tiny dedicated heads ($\leq 300$ params each).
3. **Gate head**: $g^S_{i,\cdot}$ already produced as a byproduct of head (2).

Total student size: 4132 (base) + 3 \(\times\) (head) $\approx$ 4500–5000 params, $\leq 0.32 \times$ teacher.

### 3.4 Phase D — entropy-aware mixed KL on $\mathcal{B}$

For each $i \in \mathcal{B}$, define teacher-entropy weight $\eta_i = H(p^T_i) / \log 2$, then
$$
\mathcal{L}^{\mathrm{logit}}_i \;=\; (1 - \eta_i) \cdot \mathrm{KL}_{\mathrm{rev}}(p^S_i \| p^T_i) \;+\; \eta_i \cdot \mathrm{KL}_{\mathrm{fwd}}(p^S_i \| p^T_i),
$$
where each Bernoulli KL is the scalar closed form
$$
\mathrm{KL}_{\mathrm{rev}}(p^S \| p^T) = p^S \log\tfrac{p^S}{p^T} + (1 - p^S)\log\tfrac{1 - p^S}{1 - p^T},
\qquad
\mathrm{KL}_{\mathrm{fwd}} = \mathrm{KL}_{\mathrm{rev}}(p^T \| p^S).
$$
Auxiliary head losses (matched on $\mathcal{B}$ only):
$$
\mathcal{L}^{\mathrm{rel}}_i = \tfrac{1}{R}\sum_{r} \big(c^S_{i,r} - c^T_{i,r}\big)^2,
\qquad
\mathcal{L}^{\mathrm{gate}}_i = \mathrm{KL}(g^S_i \| g^T_i).
$$

### 3.5 Phase E — node-level reliability + adaptive BCE anchor

**Cross-cell curriculum prior** (Codex I8 normalisation fix): precompute for each cell $c$
$$
r^c = \mathrm{clip}\!\left(\frac{\mathrm{AUPRC}^{c}_T - \pi^c}{1 - \pi^c},\ r_{\min},\ 1\right),\qquad r_{\min} = 0.1,
$$
where $\pi^c$ is the cell's class prevalence. This is *lift over chance* and is comparable across datasets; cached to `artifacts/teacher_curriculum_prior.json` once per cell.

**Node-level reliability** combines cell prior, teacher confidence, and ECE-style calibration-bin reliability:
$$
r^{\mathrm{node}}_i = r^c \cdot \mathrm{conf}(p^T_i) \cdot \mathrm{cal}_{\mathrm{bin}(p^T_i)},
$$
where $\mathrm{conf}(p) = |2p - 1|$ and $\mathrm{cal}_b$ is the empirical accuracy in confidence bin $b$ on val (10 equal-width bins).

**Adaptive BCE anchor**:
$$
\lambda_{\mathrm{bce}}(i) = \lambda_{\min} + (1 - r^{\mathrm{node}}_i)\,\lambda_{\mathrm{extra}},\qquad \lambda_{\min}=0.05,\ \lambda_{\mathrm{extra}}=0.50.
$$
When teacher is unreliable on a node, $\lambda_{\mathrm{bce}}$ rises toward $0.55$ and GT effectively dominates; when teacher is reliable, $\lambda_{\mathrm{bce}} \approx 0.05$ and distillation dominates. **This actually re-routes supervision**, unlike v1's scalar cell-weight that only re-scaled the same distillation gradient.

### 3.6 Phase F — sum-normalised total loss

$$
\boxed{\;
\mathcal{L} \;=\; \frac{\sum_{i \in \mathcal{B}} r^{\mathrm{node}}_i \cdot \big(\alpha_f \mathcal{L}^{\mathrm{logit}}_i + \alpha_r \mathcal{L}^{\mathrm{rel}}_i + \alpha_g \mathcal{L}^{\mathrm{gate}}_i\big) \;+\; \sum_{i \in \mathcal{B}} \lambda_{\mathrm{bce}}(i)\cdot \mathrm{BCE}(s^S_i, y_i \mid y_i \in \mathrm{train})}{\max\!\big(\sum_{i \in \mathcal{B}} r^{\mathrm{node}}_i,\ 1\big)}
\;}
$$
**Normaliser is the selected-node weight mass**, not full $N$ — fixes Codex I4 (objective scale invariant to $|\mathcal{B}|$). Defaults $\alpha_f = 1.0,\ \alpha_r = 0.3,\ \alpha_g = 0.2$.

### 3.7 Per-epoch pseudocode

```python
for epoch in range(E):
    # --- Phase A: student-policy q_phi ---
    with torch.no_grad():
        s_S, delta_S, _ = student(z, feats, return_heads=False)
        p_S = torch.sigmoid(s_S)
        H_S = bernoulli_entropy(p_S)
        q   = eps + H_S + alpha_q * p_S + beta_q * delta_S.abs()
        q   = q / q.sum()                            # detached

    # --- Phase B: sample B and teacher sparse forward ---
    idx = torch.multinomial(q, num_samples=K, replacement=False)
    with torch.no_grad():
        t_out = teacher(z[idx], feats[idx], return_heads=True)
        p_T, c_T_r, g_T = t_out["p"], t_out["c_per_r"], t_out["gate"]
        eta = bernoulli_entropy(p_T) / math.log(2)

    # --- Phase C: student forward WITH grad on idx, three heads ---
    s_out = student(z[idx], feats[idx], return_heads=True)
    p_S_b, c_S_r, g_S = sigmoid(s_out["logit"]), s_out["c_per_r"], s_out["gate"]

    # --- Phase D: entropy-aware mixed Bernoulli KL ---
    L_logit = (1 - eta) * bern_kl_rev(p_S_b, p_T) + eta * bern_kl_fwd(p_S_b, p_T)
    L_rel   = ((c_S_r - c_T_r) ** 2).mean(dim=-1)
    L_gate  = kl_categorical(g_S, g_T)

    # --- Phase E: node-level reliability + adaptive BCE ---
    r_node    = r_c * confidence(p_T) * cal_bin_reliability(p_T)
    lam_bce_i = lam_min + (1 - r_node) * lam_extra
    mask_lab  = is_labelled_train[idx]
    L_bce_per = lam_bce_i * F.binary_cross_entropy_with_logits(
        s_out["logit"], y[idx].float(), reduction="none")
    L_bce_per = L_bce_per * mask_lab.float()

    # --- Phase F: sum-normalised total loss ---
    L_distill = r_node * (alpha_f*L_logit + alpha_r*L_rel + alpha_g*L_gate)
    denom     = r_node.sum().clamp_min(1.0)
    L_total   = (L_distill.sum() + L_bce_per.sum()) / denom

    L_total.backward()
    optimizer.step()
```

### 3.8 Hyperparameter defaults

| HP | Default | Rationale |
|---|---|---|
| $K$ | $\min(2048,\ N_{\mathrm{train}})$ | $\geq 40\,\%$ of YelpChi train; $\leq 30\,\%$ Amazon |
| $\alpha_q,\ \beta_q,\ \varepsilon$ | $0.5,\ 0.3,\ 10^{-3}$ | entropy first; positive bias mild; residual bias mild |
| $\tau$ (softmax over $\tilde q$) | $1.0$ | flat sampling; ablate $\{0.5, 1.0, 2.0\}$ |
| $\alpha_f, \alpha_r, \alpha_g$ | $1.0,\ 0.3,\ 0.2$ | final dominates; aux 1/3 of final |
| $\lambda_{\min},\ \lambda_{\mathrm{extra}}$ | $0.05,\ 0.50$ | anchor floor 0.05; rises to 0.55 when teacher unreliable |
| $r_{\min}$ | $0.1$ | never fully zero teacher cell |
| optimizer | AdamW(lr=1e-3, wd=1e-4) | matches Idea 2C |
| epochs $E$ | $80$ | matches Idea 2C |
| early stop | val_AUPRC plateau 10 ep | new |

---

## 4. Three graph-specific contributions (over vanilla off-policy KD)

### 4.1 Student-policy node sampling (vs deterministic mask / full-graph KD)

This is the **core G-OPD contribution**. The sampling distribution $q_\phi$ depends on the *current* student parameters $\phi$. Each epoch the training data the loss is computed on is a fresh sample from $q_\phi^e$. Teacher is queried only on $\mathcal{B}$, giving $\sim 3\times$ teacher-forward speed-up per epoch.

Contrast with LLM-OPD: there the trajectory is a sequence of tokens, and on-policy means the entire prefix the teacher scores comes from the student. Here the "trajectory" is a single node, but the *selection* of which nodes the teacher scores is policy-dependent — this is the legitimate graph analogue per `GKD ICLR 2024 §3`.

Detached sampling avoids high-variance score-function gradients (compare to `strict-OPD` ablation in §6, mode `opd_action_strict`).

### 4.2 Entropy-aware mixed KL (vs pure reverse-KL)

Pure reverse-KL is mode-seeking and concentrates student on teacher's confident peaks — desirable when teacher is confident, but pathological when teacher itself has high entropy (saturated bases, near-boundary nodes). We linearly interpolate to forward-KL via $\eta_i = H(p^T_i)/\log 2$: teacher-confident region behaves like reverse-KL (mass-concentration), teacher-uncertain region behaves like forward-KL (mass-covering, preserves recall on rare fraud calls).

This is consistent with the `Entropy-Aware OPD (arXiv 2603.07079)` empirical recipe: hybrid divergences outperform pure reverse-KL on high-entropy support.

### 4.3 Contract-preserving rollouts under four hard contracts

The student's sampling, forward, and loss MUST respect all four `AGENTS.md §1` Operating Principles contracts during every epoch (not just at deployment):

| Contract | Where enforced in G-OPD-Flash |
|---|---|
| **C1 base-freeze SHA-256** | base detector parameters never updated; SHA-256 hash logged at epoch 0 and 80; assertion failure aborts run |
| **C2 score-blind input** | student head takes only $[z_i, \overline{\phi_{i,\cdot}}]$ OR $[z_i, \overline{e_{i,\cdot}}]$; $b_i$ added *outside* the head as $s^S_i = b_i + \delta_{\max}\tanh(\cdot)$; static analysis pass in `tests/test_opd_flash_contracts.py` rejects any read of `b_i` inside `S_\phi.forward` |
| **C3 train-only prototype** | prototype tensors (if used by teacher) computed exclusively from labelled train indices; teacher forward asserted train-only at construction time |
| **C4 $\delta$-bounded residual** | $\delta^S_i = \delta_{\max} \tanh(\cdot) \in [-\delta_{\max}, \delta_{\max}]$ by architecture, not by post-hoc clipping |

**Score-blind boundary clarification** (Codex I10): the contract is on **student input**. Teacher logit $s^T_i$ entering the distillation loss is *training supervision*, not student input; this is consistent with how SHA-256 verified base detectors interoperate with any RAER teacher in C1.

### 4.4 Multi-head auxiliary internal-state matching (NOT four independent signals)

We explicitly downgrade v1's "four independent supervision signals" claim (Codex I11). The three heads match teacher's **internal computation graph** (final logit; per-relation contribution; gate) — they are coupled by construction (final logit is a function of $c_r$ summed under $g$). Head ablation experiments (§6) measure each head's *marginal* utility:

- `final-only` — pure distillation baseline.
- `final + rel` — does per-relation supervision help disentanglement?
- `final + gate` — does relation-routing supervision help?
- `final + rel + gate` — full G-OPD-Flash.

Justification for retaining all three: they enable downstream interpretability (the student can answer "which relation drove this fraud prediction" exactly as the teacher would), even if their AUPRC effect is small.

**Why proto-head is dropped** (Codex I12): the canonical teacher (`models/cover_rel_reasoner.py`) does not currently expose a well-typed `proto_proj` tensor. We will revisit in T1; if the teacher MLP's penultimate hidden vector qualifies, a fourth proto-head will be added in v3.1.

---

## 5. Theoretical Notes (three small provable propositions)

**Replaces** v1's false capture-rate bound (Codex CE-1: $p^T = 0.5001, p^- = 0.5000 \Rightarrow$ AUPRC $= 1$ but $p^S = 0.4999, p^- = 0.5000 \Rightarrow$ AUPRC $= 0.5$, score-distance $\to 0$).

### Proposition P1 — Unbiased sampled objective (proof sketch in TKDE §5.1)

Let $\mathcal{L}^*(\phi) = \mathbb{E}_{i \sim q_\phi}[\ell(\phi; i)]$ be the true G-OPD population risk, where $\ell$ is the per-node distillation+BCE term. Then the mini-batch estimator
$$
\hat{\mathcal{L}}(\phi) \;=\; \frac{1}{K}\sum_{i_k \sim q_\phi}\ell(\phi; i_k)
$$
satisfies $\mathbb{E}_{i_k \sim q_\phi}[\hat{\mathcal{L}}] = \mathcal{L}^*(\phi)$ **conditioned on the detached sampling distribution**. Sampling-induced score-function gradients are zero (stop-gradient by construction), so $\nabla_\phi \hat{\mathcal{L}}$ is unbiased w.r.t. $\nabla_\phi \mathcal{L}^*$. (This matches GKD §3 Proposition 1.)

### Proposition P2 — Ranking-stability lemma (proof in TKDE §5.2)

Let $\mathcal{P} \subset \mathrm{train}$ be positive nodes and $\mathcal{N}$ be negatives. Define teacher margin
$$
\gamma_T = \min_{(i,j) \in \mathcal{P}\times\mathcal{N},\ s^T_i > s^T_j}\big(s^T_i - s^T_j\big).
$$
If $\sup_i |s^S_i - s^T_i| < \gamma_T / 2$, then for every $(i, j) \in \mathcal{P} \times \mathcal{N}$ with $s^T_i > s^T_j$, we also have $s^S_i > s^S_j$; equivalently, the student-induced ranking on $\mathcal{P} \times \mathcal{N}$ matches teacher's. Hence
$$
\mathrm{AUPRC}(s^S) \;\geq\; \mathrm{AUPRC}(s^T) \;-\; \mathrm{drop}(\gamma_T, \mathcal{P}\times\mathcal{N}\setminus \mathcal{M}),
$$
where the residual drop is bounded by the mass of low-margin pairs $|\mathcal{M}| = \{(i,j) : s^T_i - s^T_j < \gamma_T\}$. **Margin-conditional**, not unconditional — directly responds to CE-1.

### Proposition P3 — Contract preservation (proof by static-analysis pass)

For any G-OPD-Flash trained student $S_\phi$ initialised with $W_\delta = 0$:

1. **C1 base-freeze**: base detector parameters are never in the optimiser's parameter list (asserted by `tests/test_opd_flash_contracts.py`).
2. **C2 score-blind input**: a syntactic AST pass on `S_\phi.forward` rejects any expression that reads `b_i` before $\delta^S_i$ is computed. Final logit `s_S = b + delta_max * tanh(head(z, e))` adds $b$ outside the head — verified.
3. **C3 train-only prototype**: teacher forward uses prototype constructed from train indices only; G-OPD-Flash never modifies teacher.
4. **C4 $\delta$-bounded residual**: $\delta^S_i \in [-\delta_{\max}, \delta_{\max}]$ holds at every epoch since $\tanh \in [-1, 1]$.
5. **Zero-init implies base-equivalent posterior at epoch 0**: $W_\delta = 0 \Rightarrow \delta^S_i = 0 \Rightarrow s^S_i = b_i \Rightarrow p^S_i = \sigma(b_i)$ matches frozen base. Hence safe deployment from epoch 0.

All three propositions can be stated formally and proved in 2–4 lines each — well within TKDE §5 budget.

---

## 6. Implementation roadmap (T1–T7, 9–10 GPU days)

### T1 — Teacher multi-head exposure ($\sim$1 day)

- Patch `models/cover_rel_reasoner.py::forward` with `return_heads=True` to emit `{logit, c_per_r, gate}` (drop `proto` for v3; revisit in v3.1 if penultimate hidden qualifies).
- Apply same patch to LREE-Reasoner construct path (`models/cover_rel_reasoner.py` with `--use_learned_extractor`).
- Add `tests/test_teacher_heads_exposed.py`: assert shape `(N,)`, `(N, R)`, `(N, R)` and that $\sum_r g^T_{i,r} \cdot s^T_{i,r}$ reconstructs the final residual.

### T2 — Student multi-head adapter ($\sim$1 day)

- Rename `models/distill_adapter.py` → `models/flash_adapter.py` (back-compat alias retained).
- Add `c_per_r_head` (per-relation pre-gate scalar) and `gate_head` (softmax over $R$); both zero-init.
- Wire `return_heads=True` to match teacher interface.

### T3 — G-OPD-Flash training loop ($\sim$2 days)

- New `scripts/train_g_opd_flash.py` (or rewrite `scripts/train_distill_adapter.py`):
  - Implement Phase A–F per §3.7.
  - Args: `--mode {off_policy, all_node_mh, det_mask, g_opd_flash, opd_action_strict}` (5 ablation arms in §6 matrix).
  - Logging: per-epoch $|\mathcal{B}|$, $\sum r^{\mathrm{node}}$, per-head loss, $\lambda_{\mathrm{bce}}$ mean, val AUPRC.
- Back-compat: existing `train_distill_adapter.py` retained as `--mode off_policy` baseline.

### T4 — Curriculum prior + node-level reliability cache ($\sim$0.5 day)

- Compute per-cell `(AUPRC_T^c - prevalence^c) / (1 - prevalence^c)` once, write to `artifacts/teacher_curriculum_prior.json`.
- Fit 10-bin ECE calibration tables per cell on val, write to `artifacts/teacher_calibration_bins.json`.
- Load both in training loop.

### T5 — 8-cell × 5-seed full benchmark ($\sim$2 GPU days)

- Script `scripts/run_g_opd_flash_8cell_5seed.sh` (mirror of `run_distill_2cell_5seed_seq.sh`) running both default mode `g_opd_flash` and the 4 baseline modes (T6).
- Aggregate via `scripts/aggregate_g_opd_flash.py`:
  - AUPRC capture rate vs full LREE teacher
  - inference speed-up vs full LREE teacher
  - paired-$t$ vs (i) base only, (ii) full LREE teacher, (iii) vanilla `off_policy` distill, (iv) deterministic-mask v1.

### T6 — `strict-OPD` ablation mode ($\sim$1 day)

- Implement `--mode opd_action_strict`:
  - Sample $y_i \sim \mathrm{Bernoulli}(p^S_i)$ per node $i \in \mathcal{B}$.
  - Reward $r_i = \log p^T_i(y_i \mid \cdot)$; baseline $\hat b = \bar r$ over batch.
  - Policy loss: $-\sum_i (r_i - \hat b)\log p^S_i(y_i)$.
- Purpose: reviewer-defence ablation — establishes that GKD-style detached sampling beats high-variance single-step REINFORCE in GFD, as predicted by Codex.

### T7 — Deployment-shift evaluation ($\sim$0.5–1 day)

- Re-train base detectors with seed-shifted initialisation $\Rightarrow$ produce `base_v1.pt` and `base_v2.pt` per cell.
- Train student under each mode using `base_v1` cache; **evaluate on `base_v2` cache** (simulates production base-version bump).
- Hypothesis: $g\_opd\_flash$ degrades less than $off\_policy$ KD, providing direct GFD evidence for OPD's exposure-bias claim from `OPD survey arXiv 2604.00626`.

### T8 (optional, $\sim$1–1.5 GPU days) — FreeKD / PEKD baselines

- Reproduce FreeKD (KDD'22, RL-based GNN distillation) on YelpChi-BWGNN.
- Reproduce PEKD on the same cell.
- Direct head-to-head comparison: AUPRC vs param count.

**Total**: T1–T7 = 9–10 GPU days; T8 optional adds 1–1.5 more.

---

## 7. Expected results (hypotheses to falsify, NOT proven targets)

| Metric | Vanilla off-policy (Idea 2C now) | **G-OPD-Flash target** | Falsification protocol |
|---|---|---|---|
| AUPRC capture (YelpChi mean) | 92.3 % | **$\geq 95\,\%$** | 5-seed paired-$t$ per cell, drop if YelpChi mean $<$ 93 % |
| AUPRC capture (Amazon mean) | 83.5 % | **$\geq 90\,\%$** | same |
| Inference speed-up | 2.59 × | **$\geq 2.59\times$ (no regression)** | $H_0$: speed-up unchanged; reject if measured $<$ 2.4 × |
| Param count | 4132 | **$\leq$ 5000** | hard cap; 3 heads $\leq 600$ extra |
| Deployment-shift AUPRC delta (T7) | baseline | **$\geq +0.03$ AUPRC vs `off_policy`** | $H_0$: shift-eval AUPRC equal; reject by 5-seed paired-$t$ ★ |
| Convergence epochs to $\geq 90\,\%$ capture | 80 | **$\leq 40$** | TML 2025-10 blog reports 4–10 × faster OPD convergence; if $>$ 60, drop convergence claim |

**Fallback if hypotheses fail.** If G-OPD-Flash $\not\succ$ `off_policy` on 5-seed paired-$t$, we publish under the narrower frame "**Flash-RAER: contract-preserving multi-head distillation with student-policy node sampling**" — drop the OPD-novelty claim, keep the engineering contribution (4 k-param adapter, $\geq$ 95 % capture, 4 contracts preserved). This is Codex's `Option A` retreat path, kept warm.

---

## 8. Experiment matrix (8 ablation arms)

Per Codex `gpt-5.5` audit recommendation:

| # | Mode flag | Description | Defends against reviewer attack |
|---|---|---|---|
| 1 | `off_policy` | vanilla KL distill on full graph | "is OPD even needed?" |
| 2 | `all_node_mh` | multi-head distill, but full-graph (no sampling) | "multi-head alone enough?" |
| 3 | `det_mask` | v1 deterministic entropy mask (rev-KL final-only) | "stochastic > deterministic mask?" |
| 4 | **`g_opd_flash`** | full G-OPD-Flash (main method) | — |
| 5 | `opd_action_strict` | Bernoulli action + REINFORCE | "why not classic OPD-RL?" |
| 6 | `g_opd_no_reliability` | $r^{\mathrm{node}} \equiv 1$, $\lambda_{\mathrm{bce}} \equiv 0.05$ | "reliability really matters?" |
| 7 | `g_opd_final_only` / `+rel` / `+gate` | head ablation (3 sub-arms) | "which head carries the weight?" |
| 8 | `g_opd_teacher_handcrafted` vs `g_opd_teacher_lree` | teacher swap | **C3 independence from C2** (Codex I13) |

**Plus**: T7 deployment-shift split across {`off_policy`, `det_mask`, `g_opd_flash`} — single dedicated comparison.

---

## 9. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Student sampling signal degenerates on tiny $|\mathcal{B}|$ in Amazon (small fraud counts) | MEDIUM | adapt $K = \min(2048, N_{\mathrm{train}})$; floor $K \geq 5 \cdot $ positive count |
| R2 | Reverse-KL collapses student to base on saturated cells | LOW–MED | mixed KL ($\eta$-weighted forward) + adaptive $\lambda_{\mathrm{bce}}$ rises when teacher unreliable |
| R3 | Multi-head params exceed 5 k cap | LOW | heads tiny ($R \leq 3$); total $\sim$ 4500–5000 |
| R4 | Reviewer: "this is just GKD with mask" | MED | T6 `strict-OPD` ablation + T7 deployment-shift give direct GFD evidence beyond GKD |
| R5 | FreeKD outperforms G-OPD-Flash on raw AUPRC | LOW | FreeKD is full-model distill (not lightweight); we win on param ratio + 4 contracts preservation |
| R6 | T7 deployment shift shows no advantage | MEDIUM | fall back to Flash-RAER framing; deployment-shift becomes "robust to base version" not "OPD necessity" |
| R7 | Calibration bins (T4) unreliable on small Amazon val | MED | smooth with Laplace prior; or fall back to confidence-only $r^{\mathrm{node}}_i$ |
| R8 | C2 teacher swap (T6 arm 8) shows hand-crafted teacher → G-OPD = LREE teacher → G-OPD within noise | LOW | this would actually *strengthen* C3 independence claim — report honestly |

---

## 10. Reference reading (priority order)

1. **Agarwal et al. (NeurIPS 2024) GKD** — On-policy KD for LLMs; §3 detached sampling = G-OPD-Flash mechanism. https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf
2. **OPD survey arXiv 2604.00626 (2026)** — landscape; G-OPD-Flash positioned as graph-state-policy variant. https://arxiv.org/abs/2604.00626
3. **Thinking Machines Lab blog (2025-10)** — engineering recipe; informs convergence-epoch hypothesis.
4. **Entropy-Aware OPD (arXiv 2603.07079)** — directly motivates §3.4 mixed KL; cited for empirical recipe. https://papers.cool/arxiv/2603.07079
5. **OPD Failure Modes (arXiv 2603.25562)** — reverse-KL instability conditions; informs $\eta$-weighting. https://huggingface.co/papers/2603.25562
6. **G-OPD reverse-KL ↔ RL (arXiv 2602.12125)** — theoretical link; used for §5 P1 proof.
7. **Uni-OPD (arXiv 2605.03677)** — per-sequence reliability; §3.5 node-level reliability is the per-node analogue.
8. **FreeKD (KDD'22)** — RL-based GNN distillation; T8 baseline. https://arxiv.org/abs/2206.06561
9. **PEKD (Inf Sci 2024)** — online meta-learning GNN distillation; T8 baseline.
10. **TD-Imitation view (arXiv 2505.20335)** — alternative §5 theoretical framing; cited for P2 ranking-stability flavour.

---

## 11. Open questions

1. **Sampling temperature $\tau$**: should the $\tilde q_\phi$ pre-softmax temperature be a learnable scalar? Default 1.0 fixed; ablate $\{0.5, 1.0, 2.0\}$ in T5.
2. **Mini-batch $K$**: trade-off teacher-forward speed-up vs gradient noise. Default $\min(2048, N_{\mathrm{train}})$ but worth ablation.
3. **Calibration-bin $\mathrm{cal}_b$**: should we use isotonic regression instead of equal-width bins for Amazon's skewed posterior? Decide at T4 after empirical inspection.
4. **Proto head revival** (v3.1): if `cover_rel_reasoner.py`'s penultimate hidden vector exposes interpretable axes, add 4th aux head; otherwise leave at 3.
5. **C3 paper-side independence claim**: enough to ablate `teacher = {hand-crafted, LREE}`, or do we also need `teacher = vanilla CARE-GNN` to make C3 truly recipe-agnostic? Recommendation: 2 teachers is enough for TKDE space; add 3rd in journal revision.

---

## 12. Cheat-sheet for paper writing (TKDE §5)

**One-sentence paper claim** (lock):

> We introduce **G-OPD-Flash**, a graph on-policy distillation procedure that adapts the on-policy principle from autoregressive trajectories to graph fraud detection by sampling student-selected node states from a stop-gradient student-induced distribution and querying a frozen RAER/LREE teacher only on those states; the loss is an entropy-aware mixed Bernoulli KL over three auxiliary heads with node-level reliability weighting and adaptive BCE anchor, preserving the four hard contracts (base-freeze SHA-256, score-blind input, train-only prototype, $\delta$-bounded residual) and targeting $\geq 95\,\%$ AUPRC capture of the teacher at $\geq 2.59\times$ inference speed-up under a 4 k-parameter student.

**One-sentence novelty contour** (lock):

> Unlike LLM-OPD where the policy is over token prefixes, G-OPD-Flash's policy is over **student-selected node states**; unlike vanilla off-policy KD (Idea 2C baseline), the training distribution is **policy-dependent and stop-gradient**; unlike FreeKD's full-model RL distillation, G-OPD-Flash preserves four architectural safety contracts and operates at $\leq 5$ k parameters.

**Three crisp findings** (Laws — populate after T5 evidence):
- **Finding G-1** (hypothesis): student-policy node sampling adds $+\Delta_1$ AUPRC capture over deterministic mask on weak bases.
- **Finding G-2** (hypothesis): entropy-aware mixed KL prevents reverse-KL collapse on saturated bases; quantified by $\Delta_2$ on Amazon-BWGNN.
- **Finding G-3** (hypothesis): adaptive $\lambda_{\mathrm{bce}}$ + node-level reliability $> $ scalar cell weight on cells where teacher AUPRC $<$ 0.6, by $\Delta_3$.

---

*Living document. Any drift during implementation must be reflected back here with a version bump (v3.1, v3.2, …). v1 (`docs/OPD_FLASH_DESIGN.md`) is retained as historical record; v3 supersedes for all future work.*
