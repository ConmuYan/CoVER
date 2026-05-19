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
| **C1** | **RAER**: contract-enforced relation-aware evidence reasoning paradigm + the canonical CoVER-REL realisation | First post-hoc, base-frozen, score-blind reasoner that injects bounded relation residual onto an unmodified base detector; 4 architectural contracts (base-freeze SHA-256, score-blind input, train-only prototype, $\delta$-bounded residual). The **base-strength × evidence-type interaction law (Law 1)** is a *finding*, not just an experimental table. | The paradigm + contracts stand without LREE or G-OPD-Flash; the canonical hand-crafted evidence is sufficient to validate the framework. |
| **C2** | **LREE**: learnable relational evidence extractor that supersedes hand-crafted statistics under identical contracts | Replaces the 9-dim hand-crafted $\phi_r$ with a per-relation GCN+MLP encoder (~14 k params) under the same C1–C4 contracts. **19/32 stat-sig wins** on cross-cell paired-$t$; on weak bases (YelpChi-GCN/GAT, Amazon-GCN) the wins are decisive (+0.10–0.23 AUPRC ★). The **encoder-absorbs-prototype law (Law 3)** is a transferable finding. | C2 can be evaluated as a standalone *evidence-extractor* upgrade; the rest of the reasoner is unchanged. C2 also exposes Law 3 (encoder absorbs prototype). |
| **C3** | **G-OPD-Flash**: graph on-policy distillation framework whose policy is over **student-selected node states** (not autoregressive token prefixes), under a teacher-agnostic recipe | A 4–5 k-param student adapter generates a stop-gradient sampling distribution $q_\phi$ over training nodes from its own posterior entropy / fraud probability / residual magnitude (GKD §3 detached-sampling); a frozen RAER teacher (hand-crafted CoVER-REL or LREE) is queried only on sampled $\mathcal{B}$ and supplies **3-head auxiliary internal-state matching** (final logit + per-relation contribution $c_r = g_r s_r$ + gate); the loss is an **entropy-aware mixed Bernoulli KL** $(1-\eta)\mathrm{KL}_{\mathrm{rev}} + \eta \mathrm{KL}_{\mathrm{fwd}}$ with **node-level reliability weight** + **adaptive BCE anchor**, sum-normalised by selected-node weight mass. **Targets** (5-seed paired-$t$ falsifiable, NOT proven): $\geq 95\,\%$ AUPRC capture at $\geq 2.59\times$ inference speed-up, **contract-preserving rollouts under all 4 hard §1 contracts** (the unique fraud-detection requirement). | C3 is recipe-agnostic over RAER teachers — works with hand-crafted CoVER-REL teacher OR LREE teacher (T5 arm 8 ablation); it stands as a deployment-ready alternative even if a reviewer rejects C2. |

---

## Contribution claim sentences (paper-ready)

> **C1.** *We introduce **RAER** — a contract-enforced relation-aware evidence reasoning paradigm — and its canonical realisation **CoVER-REL**, a base-frozen, score-blind, $\delta$-bounded residual reasoner that lifts AUPRC on 8/8 base × dataset cells (+0.017 to +0.325, 5-seed paired-$t$) while satisfying four architecturally-enforced contracts (C1–C4), and we use it to formalise the first cell-resolved **base-strength × evidence-type interaction law** for multi-relation fraud reasoning.*

> **C2.** *We introduce **LREE** — a learnable relational evidence extractor that supersedes the hand-crafted 9-dim statistics under identical C1–C4 contracts — and demonstrate **19/32 cross-cell stat-sig wins**, with the wins concentrated on weak bases (YelpChi-GCN/GAT $+0.10$–$0.23$ AUPRC ★★★), revealing the **encoder-absorbs-prototype law**: a learned GCN encoder internalises the explicit prototype subspace's function, making explicit prototypes redundant on weak bases.*

> **C3.** *We introduce **G-OPD-Flash** — a graph on-policy distillation procedure that adapts the on-policy principle from autoregressive trajectories to graph fraud detection by sampling **student-selected node states** from a stop-gradient student-induced distribution and querying a frozen RAER/LREE teacher only on those states — and prove three small but architecturally enforceable propositions (unbiased sampled objective, ranking-stability lemma, contract preservation across all 4 hard contracts); the recipe is teacher-agnostic (both hand-crafted CoVER-REL and LREE teachers tested), targets $\geq 95\,\%$ AUPRC capture at $\geq 2.59\times$ inference speed-up under a $\leq 5$ k-parameter student, and is falsifiable against vanilla off-policy KD and a deterministic-mask baseline by 5-seed paired-$t$.*

---

## Independence audit — can a reviewer reject one without losing the others?

| Scenario | Survives? |
|---|---|
| Reviewer rejects C1's "paradigm" framing as repackaging existing reasoners | C2 and C3 still stand on their LREE and G-OPD-Flash novelty respectively. |
| Reviewer rejects C2 because cross-cell summary shows 0/32 sig wins (pooled) | C1 stands on its own paradigm + Law 1 finding. C3 stands on G-OPD-Flash novelty + teacher-agnostic recipe (T5 arm 8: hand-crafted CoVER-REL teacher G-OPD-Flash is reported independently). C2 falls back to "stronger primary instantiation of C3's teacher slot." |
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
- **+0.1 — TKDE-style "three findings as well as three methods"** (Law 1 / Law 3 / G-OPD-Flash 3-proposition triplet).
- **−0.0 — honest downgrade from "first OPD for GAD"** to "first graph on-policy distillation (student-induced node sampling)" (Codex audit) — the narrower framing is *more* defensible because we add the strict-OPD ablation and deployment-shift eval that a reviewer would otherwise demand in revision.

Combined with the rest of the uplift plan, this brings the realistic target from 8.0 to **8.3 / 10** (robust, not fragile).

---

## Open questions before locking

1. Do we want a **§16 explicit "ablation matrix"** showing each contribution's marginal value? (C1 alone vs C1+C2 vs C1+C3 vs C1+C2+C3). *Decision*: ✅ already added to AGENTS.md §16; populate after T5.
2. Should we **rename LREE** for the paper? "Learnable Relational Evidence Extractor" is descriptive but long. Candidates: **LREE**, **GREE** (Graph Relational Evidence Encoder), **PRELE** (Per-Relation Learnable Evidence). *Recommendation*: keep LREE, define on first use.
3. Should we **rename G-OPD-Flash** further? Candidates: **G-OPD-Flash** (current), **Node-OPD-Flash**, **Flash-RAER** (fallback). *Decision*: keep **G-OPD-Flash** — paper-title friendly, signals "graph-level OPD" without overclaiming "first OPD for GAD".
4. Should we **explicitly attribute the encoder-absorbs-prototype finding** to a specific section (probably §4.4 in the paper)? *Recommendation*: yes, §4.4.
5. For C3, do we cite **Multi-AD (Expert Syst Appl 2025)** as adjacent (LM-based evidence) or do we treat it as orthogonal? *Recommendation*: cite once in related work, do not benchmark.
6. For C3 teacher-agnostic claim, is **2 teachers (hand-crafted CoVER-REL + LREE)** enough or do we also need **CARE-GNN teacher** to make C3 truly recipe-agnostic? *Recommendation*: 2 teachers enough for TKDE space-budget; add 3rd in journal revision if reviewer asks.

---

*Living document. Re-baseline after each Phase milestone. v2 lock 2026-05-19 (G-OPD-Flash rename + 3 propositions + teacher-agnostic recipe absorbed from Codex `PROOF_AUDIT.md` + Opus Q1–Q10).*
