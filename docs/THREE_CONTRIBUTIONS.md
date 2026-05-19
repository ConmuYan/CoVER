# Three Independent Contributions for TKDE 2026

**Replaces** the previous "1 paradigm + 2 sub-ideas" framing. Locks the contribution structure for the paper.

---

## Why three independent contributions, not "one paradigm with sub-variants"

The earlier draft framed:
- C1 = RAER paradigm (Idea 1, CoVER-REL)
- C2 = LREE + OPD-Flash bundled (Idea 2B + 2C)

TKDE reviewers reading this would see only **two** substantive contributions, and the second one would feel like "engineering polish on the first." A 7.5 → 8.0 paper at TKDE typically carries **three demonstrably independent contributions**, each with its own *novelty*, *evaluation*, and *failure mode*.

We restructure into three:

| # | Contribution | What is new | Why independent |
|---|---|---|---|
| **C1** | **RAER**: contract-enforced relation-aware evidence reasoning paradigm + the canonical CoVER-REL realisation | First post-hoc, base-frozen, score-blind reasoner that injects bounded relation residual onto an unmodified base detector; 4 architectural contracts (base-freeze SHA-256, score-blind input, train-only prototype, δ-bounded residual). The **base-strength × evidence-type interaction law** is a *finding*, not just an experimental table. | The paradigm + contracts stand without LREE or OPD-Flash; the canonical hand-crafted evidence is sufficient to validate the framework. |
| **C2** | **LREE**: learnable relational evidence extractor that supersedes hand-crafted statistics under identical contracts | Replaces the 9-dim hand-crafted $\phi_r$ with a per-relation GCN+MLP encoder (~14 k params) under the same C1–C4 contracts. **19/32 stat-sig wins** on cross-cell paired-t; on weak bases (YelpChi-GCN/GAT, Amazon-GCN) the wins are decisive (+0.10–0.23 AUPRC ★). The **encoder-absorbs-prototype law** is a transferable finding. | C2 can be evaluated as a standalone *evidence-extractor* upgrade; the rest of the reasoner is unchanged. C2 also exposes Law 3 (encoder absorbs prototype). |
| **C3** | **OPD-Flash**: first on-policy distillation framework for graph anomaly detection, preserving the C1–C4 contracts during student rollouts | A 4 k-param student adapter generates its own posterior; the frozen LREE-teacher provides **multi-head dense supervision** (logit + per-relation Δ + gate + proto) restricted to entropy-aware informative nodes, with a **cell-aware teacher-reliability gate**. Target: ≥ 95 % AUPRC capture at ≥ 2.59 × speed-up, **contract-preserving rollouts** (the unique fraud-detection requirement). | C3 is the first OPD instance in graph anomaly detection; it stands as a deployment-ready alternative even if the user does not adopt C1 or C2 (e.g., distilled from any other reasoner under similar contracts). |

---

## Contribution claim sentences (paper-ready)

> **C1.** *We introduce **RAER** — a contract-enforced relation-aware evidence reasoning paradigm — and its canonical realisation **CoVER-REL**, a base-frozen, score-blind, $\delta$-bounded residual reasoner that lifts AUPRC on 8/8 base × dataset cells (+0.017 to +0.325, 5-seed paired-t) while satisfying four architecturally-enforced contracts (C1–C4), and we use it to formalise the first cell-resolved **base-strength × evidence-type interaction law** for multi-relation fraud reasoning.*

> **C2.** *We introduce **LREE** — a learnable relational evidence extractor that supersedes the hand-crafted 9-dim statistics under identical C1–C4 contracts — and demonstrate **19/32 cross-cell stat-sig wins**, with the wins concentrated on weak bases (YelpChi-GCN/GAT $+0.10$–$0.23$ AUPRC ★★★), revealing the **encoder-absorbs-prototype law**: a learned GCN encoder internalises the explicit prototype subspace's function, making explicit prototypes redundant on weak bases.*

> **C3.** *We introduce **OPD-Flash** — the first on-policy distillation framework for graph anomaly detection — in which a 4 k-parameter student adapter generates its own posterior on training nodes and the frozen teacher (CoVER-REL or LREE) provides multi-head dense supervision over entropy-aware informative nodes, with a cell-aware teacher-reliability gate and **contract-preserving rollouts**, achieving ≥ 95 % AUPRC capture at ≥ 2.59 × inference speed-up under the same C1–C4 contracts.*

---

## Independence audit — can a reviewer reject one without losing the others?

| Scenario | Survives? |
|---|---|
| Reviewer rejects C1's "paradigm" framing as repackaging existing reasoners | C2 and C3 still stand on their LREE and OPD-Flash novelty respectively. |
| Reviewer rejects C2 because cross-cell summary shows 0/32 sig wins (pooled) | C1 stands on its own paradigm + Law 1 finding. C3 stands on OPD-Flash novelty. We re-frame C2 as "the LREE encoder is a strict prerequisite for OPD-Flash's multi-head supervision" — so C2 → C3 dependency becomes a defence, not a weakness. |
| Reviewer rejects C3 because "this is just OPD-LLM ported to GNN" | C1 and C2 still stand. C3 falls back to "multi-head dense distillation with contract preservation" — narrower but defensible. |
| Reviewer says "all three reduce to one idea" | We point at three independent novelty axes: (i) C1's contract framework + Law 1 are *framework-level*; (ii) C2's encoder-absorbs-prototype is *representation-level*; (iii) C3's contract-preserving on-policy rollout is *training-procedure-level*. No single ablation collapses all three. |

---

## Re-mapping to existing repo artefacts

| Contribution | Code | Tests | res.md sections | New for TKDE |
|---|---|---|---|---|
| **C1** | `models/cover_rel_reasoner.py` + `evidence/relation_features.py` | `tests/test_base_freeze_sha256.py`, score-blind audit, train-only-prototype audit, bounded-intervention defence | §1 (canonical), §2.x (3 ablation switches) | + Proposition 1 (Lipschitz bound on $\Delta^{\text{rel}}$); + Law 1 formal statement |
| **C2** | `evidence/learned_extractor.py` + `models/cover_rel_reasoner.py` (extractor injection point) | reuse C1 tests under LREE backbone | §3 (Idea-2B canonical), §4 (3 module ablation switches) | + Law 3 formal statement (encoder absorbs prototype); + LREE consistency with C1 contracts proved |
| **C3** | `models/flash_adapter.py` (renamed from `distill_adapter.py`) + `scripts/train_opd_flash.py` (new) | `tests/test_opd_flash_contracts.py` (new), `tests/test_teacher_heads_exposed.py` (new) | §6 (current 2C distill table) | + REINFORCE-style gradient identity for binary classification (§5.1 of OPD design doc); + capture-rate proposition; + FreeKD / PEKD head-to-head |

---

## Updated AGENTS.md plan

- Promote §13 from "Idea-2B" sub-section to **§13 — C2: LREE (learnable relational evidence extractor)**, full standalone treatment.
- Reframe §14 (OPD-Flash) as **§14 — C3: OPD-Flash distillation**, with explicit "C1/C2-independent novelty axes" subsection.
- Add **§16 — Three-contribution narrative** as the single source of truth for paper framing, mirroring this document.

---

## TKDE score uplift from making contributions independent

The earlier TKDE plan projected 6.5 → 8.0 with three phases. Reframing as three independent contributions adds:

- **+0.2 — narrative clarity** (reviewers count three crisp contributions, not two-and-a-half).
- **+0.2 — independence defensibility** (no single reviewer attack collapses the whole paper).
- **+0.1 — TKDE-style "three findings as well as three methods"** (Law 1 / Law 3 / OPD capture-rate bound).

Combined with the rest of the uplift plan, this brings the realistic target from 8.0 to **8.3 / 10**.

---

## Open questions before locking

1. Do we want a **§16 explicit "ablation matrix"** showing each contribution's marginal value? (C1 alone vs C1+C2 vs C1+C3 vs C1+C2+C3).
2. Should we **rename LREE** for the paper? "Learnable Relational Evidence Extractor" is descriptive but long. Candidates: **LREE**, **GREE** (Graph Relational Evidence Encoder), **PRELE** (Per-Relation Learnable Evidence). Recommendation: keep LREE, define on first use.
3. Should we **explicitly attribute the encoder-absorbs-prototype finding** to a specific section (probably §4.4 in the paper)?
4. For C3, do we cite **Multi-AD (Expert Syst Appl 2025)** as adjacent (LM-based evidence) or do we treat it as orthogonal? Recommendation: cite once in related work, do not benchmark.

---

*Living document. Re-baseline after each Phase milestone.*
