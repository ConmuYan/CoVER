# TKDE 2026 投稿计划 — CoVER-FD

**Target venue**: IEEE Transactions on Knowledge and Data Engineering (TKDE)
**Target score**: 8.0/10 (clear-accept territory)
**Current estimated score**: 6.0–6.5/10 (single-domain benchmark, light baselines, no theory)
**Required uplift**: +1.5–2.0 points across 3 axes (scalability, baselines, theory)

---

## 1. Submission framing (rewritten for TKDE)

### 1.1 Headline
> *"Contract-bounded reasoning for graph fraud detection: how relation-aware evidence, lightweight distillation, and privacy-preserving LLM feature design interact with base-detector strength."*

### 1.2 Three contributions
1. **CoVER-REL Reasoner (Idea 1)** — softmax-gated, per-relation expert MLPs over a prototype subspace, injected as a bounded `tanh`-clipped residual onto a frozen base detector under three safety contracts (score-blind / bounded-delta / zero-init).
2. **Distill Adapter (Idea 2C)** — KL-distilled lightweight adapter (4 k params, 0.26 × teacher) recovering 85–93 % of the teacher AUPRC at **2.59 × inference speed-up**, with the score-blind contract preserved end-to-end.
3. **LLM Feature Design Protocol (Idea 3)** — *summary-only* prompts (no raw rows shown to the LLM) drive an instruction-tuned 4 B model to brainstorm a 20-formula candidate pool; a BERT/RoBERTa REINFORCE selector picks the val-optimal 5-subset. The protocol significantly beats SOTA AutoFE (OpenFE NeurIPS-23, gplearn-SR) on YelpChi while ceding to OpenFE on Amazon — characterised by a **base-strength × evidence-type interaction law**.

### 1.3 Empirical laws (the TKDE-worthy findings)
- **Law 1 (evidence × strength)** — weak base + rich relations ⇒ all evidence injection paths work; strong base + simple structure ⇒ injection saturates (~0 marginal).
- **Law 2 (pool > selection)** — LLM's irreducible contribution is candidate-pool brainstorming (+16 pp); the selection-within-pool step adds only ~3 pp.
- **Law 3 (encoder absorbs proto)** — learned GCN encoder internalises prototype function on weak bases, making explicit prototypes redundant.

---

## 2. Critical prior-art comparison (must be in §7 Related Work)

Each entry must be reproduced or, at minimum, head-to-head compared using published numbers on overlapping cells.

| Prior work | Venue / year | Overlap with our work | Delta we must claim |
|---|---|---|---|
| **PromoGuardian** (arXiv 2510.12652) | KDD 2026 (likely accept) | Idea 1: multi-relation fused GNN, 5 M nodes @ Meituan | Different: contract-bounded residual (not full GNN); transferable to YelpChi/Amazon (PromoGuardian is Meituan-only) |
| **H²IDE** (TKDE 2024) | TKDE | Idea 1: disentangled homophily/heterophily on multi-relation | Different: our prototype subspace is C-dim explicit; H²IDE is disentangled representation |
| **Multi-AD** (Expert Syst Appl 2025) | ESWA | Idea 3: LM-based multi-evidence coherence GAD | Different: Multi-AD uses LM to *judge coherence*; we use LLM to *design features* |
| **Mitigating Tail Effect** (TKDE 2025) | TKDE | Idea 1: community-enhanced multi-relation GNN | Different: we don't address tail; orthogonal |
| **Risk-aware Graph Repr** (TKDE 2025) | TKDE | Idea 1: attribute-driven fraud | Different: ours is relation-aware, not attribute-aware |
| **DualKD** (Entropy 2025) | Entropy | Idea 2C: dual-level KD for few-shot GAD | Different: ours preserves contracts; theirs is few-shot |
| **GraphDART** (arXiv 2501.02796) | arXiv | Idea 2C: graph distillation for APT detection | Different domain (APT vs e-commerce); ours has score-blind contract |
| **CAAFE** (Hollmann et al. 2023) | arXiv 2305.03403 | Idea 3: context-aware LLM feature engineering | Different: we use summary-only (privacy); CAAFE shows raw rows |
| **DeepFeature / SMARTFEAT** | 2024 | Idea 3: multi-source LLM feature generation | Different: ours has REINFORCE selector + LLM brainstormer two-stage |
| **SCFCRC** (arXiv 2501.12430) | 2025 | Idea 1: 4-expert MoE + gating multi-relation | Different: bounded residual contract + per-relation expert; SCFCRC is direct classifier |

---

## 3. Score-uplift roadmap (6.5 → 8.0+)

### Phase A — Must do (+1.2 pts, ~2 weeks GPU)

| ID | Action | Score gain | GPU cost | ETA |
|---|---|---|---|---|
| **A1** | Add **T-Finance + T-Social** benchmark (4 bases × 5 seeds each — extends current 8-cell to 16-cell) | **+0.4** | 1–2 days | Week 1 |
| **A2** | Add **DGraph-Fin** (3 M nodes, 4 M edges) — TKDE-grade scalability; required by reviewers | **+0.4** | 2–3 days (memory tuning) | Week 2 |
| **A3** | Reproduce **H²IDE + PromoGuardian + Mitigating Tail Effect + Risk-aware GR + Multi-AD** head-to-head on at least YelpChi/Amazon (5 new baselines) | **+0.4** | 3–5 days (implementations vary) | Week 1–2 |

### Phase B — Should do (+0.4 pts, ~3 days)

| ID | Action | Score gain | GPU cost | ETA |
|---|---|---|---|---|
| **B1** | Write **Proposition 1** (Lipschitz bound on bounded residual: `‖z_full − z_base‖ ≤ delta_max · R · ‖W_g‖`) + **Proposition 2** (base-strength saturation lemma: when base AUPRC > τ_sat, any evidence injection collapses to ε marginal). Include 1-page proof sketch. | **+0.3** | 0 GPU; 1–2 days writing | Week 2 |
| **B2** | Upload anonymised code + processed data to **IEEE DataPort** and **Code Ocean** (TKDE explicitly encourages) | **+0.1** | 0 GPU; 0.5 day | Week 3 |

### Phase C — Nice to have (+0.3 pts, ~2 days)

| ID | Action | Score gain | GPU cost | ETA |
|---|---|---|---|---|
| **C1** | Reposition Idea 3 as **"Privacy-preserving LLM-based feature design protocol"** — emphasise summary-only ≠ CAAFE (raw rows). Targets TKDE's privacy-aware audience. | **+0.2** | 0 GPU; 0.5 day writing | Week 3 |
| **C2** | **5-seed × {qwen3-4b-base, qwen3-4b-instruct, llama3.1-8b-instruct}** scaling sweep on YelpChi-BWGNN to robustify the reverse-scaling finding | **+0.1** | 1 day GPU | Week 3 |

**Projected uplift** : 6.5 → 7.0 (A1) → 7.4 (A2) → 7.7 (A3) → 8.0 (B1) → 8.1 (B2) → 8.3 (C1 + C2) ✅

---

## 4. Paper section rewrite plan (TKDE format)

| Section | Old structure | New TKDE structure |
|---|---|---|
| §1 Intro | "we propose CoVER-REL" | Lead with empirical Law 1 + privacy/safety angle |
| §2 Preliminaries | Skipped | Formalise problem, define score-blind / bounded contracts |
| §3 Method (Idea 1+2) | Direct architecture | Architecture + 3 contracts + Proposition 1 (bounded) |
| §4 LLM protocol (Idea 3) | "we ask the LLM…" | Two-stage decomposition (brainstorm + select) framed as privacy-preserving AutoFE |
| §5 Theoretical analysis | None | Proposition 1+2 with proofs; spectral / Lipschitz bound |
| §6 Experiments | 8-cell × 5-seed YelpChi/Amazon | 5 datasets × 4-cell × 5-seed + DGraph-Fin scalability; 10+ baselines; OOM/OOT report |
| §7 Industry insight | None | Why score-blind matters in regulated fraud detection (subsection) |
| §8 Related work | Light | Detailed table vs 10 prior works in §2 above |
| §9 Conclusion | Standard | Limitations + future work |

---

## 5. Reproducibility checklist (TKDE requirement)

- [ ] 5 seeds × 4 bases × 5 datasets = 100 base detectors archived under SHA-256
- [ ] Random seeds documented per script (`SEEDS = [42, 123, 456, 789, 2026]`)
- [ ] OOM / OOT for every cell × method reported in supplementary
- [ ] Leakage audit passes preregistered (val/test-shuffle ⇒ no formula change ✅; train-shuffle ⇒ change ✅; deterministic input → output ✅)
- [ ] LLM `temperature=0.6, top_p=0.9, max_new_tokens=2048, do_sample=True` logged with rl_seed for every model
- [ ] IEEE DataPort link for processed YelpChi/Amazon/T-Finance/T-Social splits
- [ ] Code Ocean capsule for end-to-end reproduction of Table 3 (main result)
- [ ] All 5 silent-failure bug post-mortems (parser, builder, phase-parsing, max_new_tokens, PLM 4-bug) documented in `audit/` and referenced in supplementary

---

## 6. Risk register (reviewer 2 simulation)

| Reviewer concern | Likelihood | Mitigation |
|---|---|---|
| "Method overlap with SCFCRC / H²IDE / PromoGuardian — what's new?" | **HIGH** | §7 table + §3 contract framing + Proposition 1+2 (no prior provides theoretical bound) |
| "YelpChi/Amazon (~46K / ~11K) are tiny — does it scale?" | **HIGH** | A1 + A2 (T-Social 5.7 M, DGraph-Fin 3 M) |
| "LLM design wins on YelpChi but loses on Amazon — why?" | **HIGH** | §6.X split-by-dataset (already in res.md §8.5.1) + Law 1 + Proposition 2 |
| "Single-seed multi-LLM scaling is not enough" | **HIGH** | C2 (5-seed × multi-family scaling) |
| "Privacy claim is hand-wavy" | **MEDIUM** | C1 (formal privacy framing: summary-only protocol vs CAAFE raw-row exposure) |
| "Where is the theoretical contribution?" | **MEDIUM** | B1 (Proposition 1 + 2 with proofs) |
| "Reproducibility — random seeds? OOM?" | **MEDIUM** | B2 (DataPort + Code Ocean) |

---

## 7. Submission timeline (proposed)

| Week | Activities | Deliverable |
|---|---|---|
| 1 | A1 (T-Finance/T-Social) + A3 (H²IDE/Mitigating Tail/Risk-aware reproduce) | 16-cell table + 3 baseline rows |
| 2 | A2 (DGraph-Fin) + B1 (Proposition 1+2 writing) | million-node row + theory section |
| 3 | A3 remainder (PromoGuardian/Multi-AD) + B2 + C1 + C2 | All artefacts uploaded; final benchmark table |
| 4 | Paper writing (sections 1–9) | Full PDF draft |
| 5 | Internal review + kill-argument adversarial pass | Polished draft |
| 6 | Submission to TKDE | Submitted |

**Total ETA**: ~6 weeks of focused work (assuming current 4-RTX-3090 setup).

---

## 8. Open questions

1. **Should we add a 5th dataset (e.g., a synthetic million-node graph)** to flex scalability further?
2. **Should Idea 3 be split into a separate paper** (workshop-style at AI-FA @ KDD 2026)? Pros: cleaner TKDE narrative for Idea 1+2C. Cons: loses the three-way interaction story.
3. **Theoretical depth**: do we need a third proposition (e.g., distill adapter consistency bound)? Recommendation: yes if A2 leaves theory thin.
4. **GPU budget**: Phase A requires ~6-day continuous GPU. Confirm 4× 3090 availability.

---

*Living document. Updated as Phase A/B/C complete.*
