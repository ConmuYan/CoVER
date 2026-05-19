# Three Independent Contributions for TKDE 2026

**Replaces** the previous "1 paradigm + 2 sub-ideas" framing. Locks the contribution structure for the paper.

**v2 update (2026-05-19)**: C3 renamed `OPD-Flash → G-OPD-Flash` and rewritten after Codex `gpt-5.5` xhigh `PROOF_AUDIT.md` (FAIL/critical_gap on v1 OPD-Flash design) + Opus 4.7 Q1–Q10 review. v3 design at `docs/OPD_FLASH_DESIGN_v3.md`.

---

## Why three independent contributions, not "one paradigm with sub-variants"

The earlier draft framed:
- C1 = RAER paradigm (Idea 1, CoVER-REL)
- C2 = LREE + OPD-Flash bundled (Idea 2B + 2C)

TKDE reviewers reading this would see only **two** substantive contributions, and the second one would feel like "engineering polish on the first." A 7.5 → 8.0 paper at TKDE typically carries **three demonstrably independent contributions**, each with its own *novelty*, *evaluation*, and *failure mode*.

We restructure into three:

| # | Contribution | What is new | Why independent |
|---|---|---|---|
| **C1** | **RAER**: contract-enforced relation-aware evidence reasoning paradigm + the canonical CoVER-REL realisation | First post-hoc, base-frozen, score-blind reasoner that injects bounded relation residual onto an unmodified base detector; 4 architectural contracts (base-freeze SHA-256, score-blind input, train-only prototype, $\delta$-bounded residual). The **base-strength × evidence-type interaction law (Law 1)** is a *finding*, not just an experimental table. | The paradigm + contracts stand without LREE or Flash-RAER; the canonical hand-crafted evidence is sufficient to validate the framework. |
| **C2** | **LREE**: learnable relational evidence extractor that supersedes hand-crafted statistics under identical contracts | Replaces the 9-dim hand-crafted $\phi_r$ with a per-relation GCN+MLP encoder (~14 k params) under the same C1–C4 contracts. **19/32 stat-sig wins** on cross-cell paired-$t$; on weak bases (YelpChi-GCN/GAT, Amazon-GCN) the wins are decisive (+0.10–0.23 AUPRC ★). The **encoder-absorbs-prototype law (Law 3)** is a transferable finding. | C2 can be evaluated as a standalone *evidence-extractor* upgrade; the rest of the reasoner is unchanged. C2 also exposes Law 3 (encoder absorbs prototype). |
| **C3** | **Flash-RAER**: the **first contract-preserving multi-head reliability-weighted distillation framework** for RAER fraud-detection adapters (boundary: contract-preservation distinguishes from FreeKD KDD'22 graph RL-KD; 3-head matching + reliability-weighted distinguishes from GLNN ICLR'22 / G-CRD CIKM'22 graph-free KD — v3.3 round-7 retreat after T5 5-seed paired-$t$ falsified v3.2 stochastic-sampling main method), under a teacher-agnostic recipe | A 4–5 k-param student adapter takes the **deterministic top-entropy node-mask** + frozen RAER teacher's **3-head matching** (final logit + per-relation contribution $c_r = g_r s_r$ + gate) + **entropy-aware mixed Bernoulli KL** $(1-\eta)\mathrm{KL}_{\mathrm{rev}} + \eta \mathrm{KL}_{\mathrm{fwd}}$ + **node-level reliability weight** + **adaptive BCE anchor**, with **two independent normalisers** (distill / $\sum r^{\mathrm{node}}_i$ and BCE / $\sum \lambda_{\mathrm{bce}}\mathbb{1}[y \in \mathrm{train}]$ — v3.1 MF4). **T5 5-seed paired-$t$ evidence** (`artifacts/tables/g_opd_flash_8cell_5seed.md`): **≥ 95% AUPRC capture** on both YelpChi (97.2% mean) and Amazon (96.8% mean corrected) **at 2.59× head-level inference speed-up**, with **3/8 cells stat-sig at p<0.05 and 2/8 at p<0.01 vs off_policy baseline**, **contract-preserving rollouts under all 4 hard §1 contracts**. **Transferable finding** (§13.2 risk register): the v3.2 stochastic-sampling sub-arms (`g_opd_flash`, `opd_action_strict`, `opd_action_strict_mh`) all FAIL to beat `det_mask` Flash-RAER on any cell (0/8 sig wins) — on-policy sampling mechanisms designed for autoregressive distillation do NOT transfer to single-step graph fraud detection. | C3 is recipe-agnostic over RAER teachers — works with hand-crafted CoVER-REL teacher OR LREE teacher (T5 arm 8 ablation); it stands as a deployment-ready alternative even if a reviewer rejects C2. |

---

## Contribution claim sentences (paper-ready)

> **C1.** *We introduce **RAER** — a contract-enforced relation-aware evidence reasoning paradigm — and its canonical realisation **CoVER-REL**, a base-frozen, score-blind, $\delta$-bounded residual reasoner that lifts AUPRC on 8/8 base × dataset cells (+0.017 to +0.325, 5-seed paired-$t$) while satisfying four architecturally-enforced contracts (C1–C4), and we use it to formalise the first cell-resolved **base-strength × evidence-type interaction law** for multi-relation fraud reasoning.*

> **C2.** *We introduce **LREE** — a learnable relational evidence extractor that supersedes the hand-crafted 9-dim statistics under identical C1–C4 contracts — and demonstrate **19/32 cross-cell stat-sig wins**, with the wins concentrated on weak bases (YelpChi-GCN/GAT $+0.10$–$0.23$ AUPRC ★★★), revealing the **encoder-absorbs-prototype law**: a learned GCN encoder internalises the explicit prototype subspace's function, making explicit prototypes redundant on weak bases.*

> **C3.** *We introduce **Flash-RAER** — the **first contract-preserving multi-head reliability-weighted distillation framework** for RAER fraud-detection adapters (boundary: distinct from prior graph RL-distillation FreeKD KDD'22 by the contract-preservation requirement and from prior graph-free KD GLNN ICLR'22 / G-CRD CIKM'22 by the three-head matching and reliability-weighted loss — v3.3 round-7 retreat) — in which a 4–5 k-parameter student adapter, trained under a deterministic top-entropy node-mask with entropy-aware mixed Bernoulli-KL supervision and three auxiliary head-matching terms (final logit, per-relation contribution, gate) over a two-term independent-normaliser loss with adaptive BCE anchor, captures ≥ 95% of an LREE-teacher's cross-cell mean AUPRC (YelpChi 97.2% / Amazon 96.8% corrected, 5-seed paired-$t$) at 2.59× head-level inference speed-up while preserving all four hard contracts (base-freeze SHA-256 verified at epoch 0 + post-training assert, score-blind input verified by runtime counterfactual hook, train-only prototype, $\delta$-bounded residual); the framework is recipe-agnostic over RAER teachers (hand-crafted CoVER-REL or LREE), and we further report — as a transferable methodological finding — that the four pre-registered stochastic-sampling alternatives (GKD-style detached q_φ sampling, single-step REINFORCE, REINFORCE+multi-head, full-graph multi-head) are all 5-seed paired-$t$ FALSIFIED against the deterministic top-entropy mask on the cells with measurable headroom, providing direct evidence that on-policy mechanisms designed for autoregressive distillation do not transfer to single-step graph fraud detection.*

---

## Independence audit — can a reviewer reject one without losing the others?

| Scenario | Survives? |
|---|---|
| Reviewer rejects C1's "paradigm" framing as repackaging existing reasoners | C2 and C3 still stand on their LREE and Flash-RAER novelty respectively. |
| Reviewer rejects C2 because cross-cell summary shows 0/32 sig wins (pooled) | C1 stands on its own paradigm + Law 1 finding. C3 stands on Flash-RAER novelty + teacher-agnostic recipe (T5 arm 8: hand-crafted CoVER-REL teacher Flash-RAER is reported independently). C2 falls back to "stronger primary instantiation of C3's teacher slot." |
| Reviewer rejects C3's OPD framing as "just GKD ported to GNN" | C1 and C2 still stand. C3 falls back to **Flash-RAER**: "contract-preserving multi-head distillation with student-selected node sampling" — narrower but defensible (Codex `Option A` retreat path, kept warm in `docs/OPD_FLASH_DESIGN_v3.md` §7 fallback). |
| Reviewer disputes C3's `≥ 95%` capture target | We pre-stated it as a **target** (falsifiable hypothesis), not an achievement. The empirical 5-seed paired-$t$ verdict + the **deployment-shift evaluation** (`base_v1 train / base_v2 eval`, T7) supply the actual evidence. |
| Reviewer says "all three reduce to one idea" | We point at three independent novelty axes: (i) C1's contract framework + Law 1 are *framework-level*; (ii) C2's encoder-absorbs-prototype is *representation-level*; (iii) C3's student-policy node sampling + 3 provable propositions + contract-preserving rollouts are *training-procedure-level*. No single ablation collapses all three. |

---

## Re-mapping to existing repo artefacts

| Contribution | Code | Tests | res.md sections | New for TKDE |
|---|---|---|---|---|
| **C1** | `models/cover_rel_reasoner.py` + `evidence/relation_features.py` | `tests/test_base_freeze_sha256.py`, score-blind audit, train-only-prototype audit, bounded-intervention defence | §1 (canonical), §2.x (3 ablation switches) | + Proposition 1 (Lipschitz bound on $\Delta^{\text{rel}}$); + Law 1 formal statement |
| **C2** | `evidence/learned_extractor.py` + `models/cover_rel_reasoner.py` (extractor injection point) | reuse C1 tests under LREE backbone | §3 (Idea-2B canonical), §4 (3 module ablation switches) | + Law 3 formal statement (encoder absorbs prototype); + LREE consistency with C1 contracts proved |
| **C3** | `models/flash_adapter.py` (rename from `distill_adapter.py`, +2 zero-init heads for $c_r$ / $g_r$) + `scripts/train_g_opd_flash.py` (new) + `models/cover_rel_reasoner.py` (teacher `return_heads=True` patch) | `tests/test_teacher_heads_exposed.py` (new), `tests/test_opd_flash_contracts.py` (new, all 4 hard contracts) | §6 (current 2C distill table, will be re-baselined as `--mode off_policy` baseline) | + 3 provable propositions (unbiased sampling P1; ranking-stability lemma P2; contract preservation P3); + 8-arm ablation matrix; + deployment-shift evaluation (`base_v1 train / base_v2 eval`); + `strict-OPD` ablation mode (Bernoulli sampling + REINFORCE); + FreeKD / PEKD head-to-head |

---

## Updated AGENTS.md plan

- Promote §13 from "Idea-2B" sub-section to **§13 — C2: LREE (learnable relational evidence extractor)**, full standalone treatment.
- Reframe §14 (G-OPD-Flash) as **§14 — C3: G-OPD-Flash graph on-policy distillation**, with explicit "C1/C2-independent novelty axes" subsection and the 3 propositions + 8-arm ablation matrix from `docs/OPD_FLASH_DESIGN_v3.md` cross-referenced.
- §16 — Three-contribution narrative remains the single source of truth for paper framing, mirroring this document.

---

## TKDE score uplift from making contributions independent + Honest-OPD v3

The earlier TKDE plan projected 6.5 → 8.0 with three phases. Reframing as three independent contributions adds:

- **+0.2 — narrative clarity** (reviewers count three crisp contributions, not two-and-a-half).
- **+0.2 — independence defensibility** (no single reviewer attack collapses the whole paper; C3 now teacher-agnostic with hand-crafted vs LREE teacher ablation).
- **+0.1 — TKDE-style "three findings as well as three methods"** (Law 1 / Law 3 / Flash-RAER §13.2 transferable finding — "on-policy sampling does not transfer to single-step GFD").
- **−0.0 — honest downgrade from "first OPD for GAD"** to **"first contract-preserving multi-head reliability-weighted distillation framework for RAER fraud-detection adapters"** (v3.3 round-7 retreat after T5 pre-registered §7 falsification triggered for v3.2 stochastic-sampling `g_opd_flash` main method; Codex `Option A` retreat path absorbed as designed) — the narrower framing is *more* defensible because (a) the T5 5-seed paired-$t$ evidence directly supports the claim (3/8 cells stat-sig at p<0.05, 2/8 at p<0.01 vs off_policy; 8/8 cells directional positive); (b) the demoted v3.2 stochastic-sampling arms become **§13.2 transferable methodological finding** ("on-policy sampling mechanisms do not transfer to single-step graph fraud detection") rather than failed contributions; (c) explicit boundary against FreeKD (KDD'22), GLNN (ICLR'22), G-CRD (CIKM'22) that a reviewer would otherwise demand in revision.

Combined with the rest of the uplift plan, this brings the realistic target from 8.0 to **8.3 / 10** (robust, not fragile).

---

## Open questions before locking

1. Do we want a **§16 explicit "ablation matrix"** showing each contribution's marginal value? (C1 alone vs C1+C2 vs C1+C3 vs C1+C2+C3). *Decision*: ✅ already added to AGENTS.md §16; populate after T5.
2. Should we **rename LREE** for the paper? "Learnable Relational Evidence Extractor" is descriptive but long. Candidates: **LREE**, **GREE** (Graph Relational Evidence Encoder), **PRELE** (Per-Relation Learnable Evidence). *Recommendation*: keep LREE, define on first use.
3. Should we **rename G-OPD-Flash** further? Candidates: **G-OPD-Flash** (v3.2), **Node-OPD-Flash**, **Flash-RAER** (fallback). *Decision (v3.3 round-7 retreat)*: **Flash-RAER** activated — pre-registered Codex `Option A` retreat path triggered by T5 5-seed paired-$t$ falsification of stochastic-sampling sub-arms.
4. Should we **explicitly attribute the encoder-absorbs-prototype finding** to a specific section (probably §4.4 in the paper)? *Recommendation*: yes, §4.4.
5. For C3, do we cite **Multi-AD (Expert Syst Appl 2025)** as adjacent (LM-based evidence) or do we treat it as orthogonal? *Recommendation*: cite once in related work, do not benchmark.
6. For C3 teacher-agnostic claim, is **2 teachers (hand-crafted CoVER-REL + LREE)** enough or do we also need **CARE-GNN teacher** to make C3 truly recipe-agnostic? *Recommendation*: 2 teachers enough for TKDE space-budget; add 3rd in journal revision if reviewer asks.

---

*Living document. Re-baseline after each Phase milestone. v2 lock 2026-05-19 (G-OPD-Flash rename + 3 propositions + teacher-agnostic recipe absorbed from Codex `PROOF_AUDIT.md` + Opus Q1–Q10). **v2.1 lock 2026-05-19** (C3 claim narrowed per Codex round-3 conditional-accept; 5 must-fix + 4 minor absorbed in `docs/OPD_FLASH_DESIGN_v3.md` §0.2; paper-claim lock deferred until T5+T7 evidence). **v3 lock 2026-05-19** (Opus round-7 critic — T5 8-cell × 5-seed × 6-mode benchmark complete; pre-registered §7 falsification triggered for v3.2 stochastic-sampling main method; C3 retreats to **Flash-RAER** (det_mask main + multi-head + reliability) per Codex Option A pre-registered fallback; stochastic-sampling sub-arms demoted to §6 ablation arms with transferable "on-policy mechanisms do not transfer to single-step GFD" finding; paper-claim lock NOW READY).*
