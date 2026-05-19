# Proof Audit: OPD-Flash / Three-Contribution Structure

**Verdict**: FAIL  
**Reason code**: critical_gap  
**Reviewer**: Codex `gpt-5.5`, reasoning `xhigh`  
**Audited inputs**:

- `docs/OPD_FLASH_DESIGN.md`
- `docs/THREE_CONTRIBUTIONS.md`
- `AGENTS.md`

## Executive Verdict

The C1/C2 parts of the three-contribution structure remain broadly defensible
as empirical/methodological contributions. The C3 OPD-Flash contribution is
not defensible as currently written.

The core failure is not implementation detail. The current algorithm is a
deterministic full-graph masked reverse-KL distillation procedure. It does not
define a student-induced state/action distribution, does not sample student
actions or trajectories, and therefore does not satisfy the usual OPD claim.

The Opus 4.7 review's "major revision needed" conclusion is about right, and
possibly slightly soft for C3. The whole paper idea is not broken, but the
current C3 theory and novelty wording would not survive a severe methodology
review.

## Issue Ledger

| ID | Severity | Status | Impact | Location | Finding | Required action |
|---|---|---|---|---|---|---|
| I1 | FATAL | INVALID | GLOBAL | `docs/OPD_FLASH_DESIGN.md:11,75-122` | Deterministic full-graph entropy masking is not on-policy distillation. | Rename C3 or implement a real student-induced sampling/policy objective. |
| I2 | CRITICAL | INVALID | GLOBAL | `docs/OPD_FLASH_DESIGN.md:55-61,141-152`; `AGENTS.md:162-164` | Binary outputs are treated as two-class softmax logits, but RAER emits scalar logits. | Use Bernoulli sigmoid KL or explicitly map scalar logits to `[0,z]`. |
| I3 | MAJOR | OVERSTATED | GLOBAL | `docs/OPD_FLASH_DESIGN.md:155,197-203` | Reverse KL identity is algebraic, but the REINFORCE/OPD/mode-seeking interpretation is overstated. | Keep closed-form KL; remove estimator language unless actual sampling exists. |
| I4 | CRITICAL | INVALID | GLOBAL | `docs/OPD_FLASH_DESIGN.md:110-115,201-203` | `.mean()` over all nodes changes loss scale with mask size; no unbiased estimator exists in the pseudocode. | Normalize by selected nodes and delete variance claims. |
| I5 | FATAL | INVALID | GLOBAL | `docs/OPD_FLASH_DESIGN.md:205-213` | AUPRC capture bound is false under stated assumptions. | Replace with empirical target or a margin-conditional ranking lemma. |
| I6 | CRITICAL | OVERSTATED | GLOBAL | `docs/THREE_CONTRIBUTIONS.md:31`; `AGENTS.md:524-547` | C3 says "achieves" before implementation/results exist. | Change to "targets" until 8-cell 5-seed evidence exists. |
| I7 | MAJOR | UNJUSTIFIED | GLOBAL | `docs/OPD_FLASH_DESIGN.md:100,173-179` | Cell reliability scalar cannot by itself prevent memorizing teacher errors. | Reframe as heuristic weighting or add node-level reliability/BCE dominance. |
| I8 | MAJOR | UNCLEAR | LOCAL | `docs/OPD_FLASH_DESIGN.md:100,175-177` | Reliability denominator is inconsistent and raw AUPRC is not cross-dataset comparable. | Use a single normalized lift metric over prevalence or base. |
| I9 | CRITICAL | INVALID | GLOBAL | `docs/OPD_FLASH_DESIGN.md:13,183-189`; `AGENTS.md:27-34` | "Three contracts" / "same C1-C4" is inconsistent with the four hard contracts. | Restate base-freeze, score-blind, train-only prototype, bounded residual. |
| I10 | MAJOR | UNJUSTIFIED | GLOBAL | `docs/OPD_FLASH_DESIGN.md:85-97,110,185-187` | Score-blind boundary is ambiguous because teacher logits/predictions drive training mask/loss. | Define input/inference score-blindness separately from teacher-supervised training. |
| I11 | MAJOR | OVERSTATED | LOCAL | `docs/OPD_FLASH_DESIGN.md:161-169` | Four heads are not four independent supervision signals; several are coupled. | Reframe as auxiliary internal-state matching; validate with head ablations. |
| I12 | MAJOR | UNCLEAR | LOCAL | `docs/OPD_FLASH_DESIGN.md:54,64-66,90-91` | `proto_proj` / `rho` is not defined by the canonical teacher outputs. | Define the tensor mathematically and in code. |
| I13 | CRITICAL | OVERSTATED | GLOBAL | `docs/THREE_CONTRIBUTIONS.md:21,39-42`; `AGENTS.md:575-584` | C3 is not logically independent of C1/C2 as written. | Frame independence as evaluation-axis independence, or add C1+C3 / non-LREE teacher ablations. |

## Counterexample Red Team

### CE-1: AUPRC bound cannot hold without rank-margin assumptions

Take one positive node and one negative node. Let the teacher scores be
`0.5001` for the positive and `0.5000` for the negative, so teacher AUPRC is
1. Let the student scores be `0.4999` for the positive and `0.5000` for the
negative. The student is arbitrarily close in score space, but the ranking is
reversed and AUPRC drops to 0.5. Therefore bounded residuals, bounded logits,
and mask cardinality do not imply an AUPRC capture bound.

### CE-2: Entropy mask is not a student visitation distribution

Two students can induce the same entropy/disagreement mask over the fixed
training graph while assigning different sampled fraud labels. The current
loss sees only the deterministic mask and teacher soft targets, so it cannot
distinguish those student policies. This is adaptive reweighting, not a
policy-rollout objective.

### CE-3: Cell-level reliability scalar does not identify teacher errors

If a teacher systematically flips a subset of hard labels in a low-reliability
cell, multiplying the distillation loss by `w_c=0.3` still pulls the student
toward the wrong target unless the BCE anchor dominates. A scalar cell weight
changes gradient scale or tradeoff; it does not make teacher supervision
selectively reliable.

## Comparison With Opus 4.7 Review

| Opus point | This audit | Verdict |
|---|---|---|
| Q1: current design is not OPD | Confirmed, with FATAL issue I1. | Correct. |
| Q2: reverse-KL identity should be closed-form, not REINFORCE | Confirmed; also flags scalar-vs-two-logit mismatch. | Correct and incomplete. |
| Q3: AUPRC proposition conditions insufficient | Confirmed as FATAL; explicit two-node counterexample added. | Correct. |
| Q4: cell-aware reliability partially contradictory | Confirmed; scalar weight does not identify errors. | Correct. |
| Q5: multi-head signals partly redundant | Confirmed; should be "internal-state matching" not independent supervision. | Correct. |
| Q6: timeline underestimated | Not re-audited deeply here; reviewer agrees implementation/results are TODO. | Plausible. |
| Q7: novelty depends on true OPD | Confirmed; if OPD claim is dropped, novelty narrows. | Correct. |

## Recovery Options

### Option A: Honest rename, strongest near-term route

Rename C3 from OPD-Flash to a non-OPD compression contribution, e.g.
`Flash-RAER` or `Student-Adaptive Multi-Head Distillation`.

Defensible claim:

> We introduce Flash-RAER, a contract-preserving lightweight distillation
> procedure for RAER/LREE teachers. Flash-RAER trains a 4-5K parameter student
> adapter with entropy/disagreement-selected training nodes and auxiliary
> matching of teacher final logits, relation contributions, gates, and
> prototype-state projections, while preserving inference-time base-freeze,
> score-blind evidence inputs, train-only prototype construction, and bounded
> residual intervention.

This keeps C3 as a training-procedure/deployment contribution, but removes the
fragile "first OPD" and theorem claims.

### Option B: Make it real OPD

Define a student-induced distribution and optimize under it. Minimal binary
version:

1. Student defines Bernoulli action `a_i ~ pi_S(. | i)`.
2. Teacher returns `log pi_T(a_i | i)` and optional internal-head rewards on
   the selected nodes/actions.
3. Optimize a policy-gradient or exactly enumerated policy objective with an
   explicit baseline and selected-node normalization.

This is mathematically cleaner but likely lower signal in binary node
classification and adds variance. It also still needs an exposure-bias
experiment showing why OPD helps over masked off-policy KL.

### Option C: Keep OPD wording only as analogy

Use "OPD-inspired" and state explicitly that the graph setting lacks
autoregressive trajectory exposure bias. This is weaker but safer than
claiming first OPD in graph anomaly detection.

## Required Paper/Design Edits Before Submission

1. Replace all "first on-policy distillation framework" claims unless Option B
   is implemented and tested.
2. Replace C3 "achieving >=95% capture" with "targeting" until results exist.
3. Convert all binary KL notation to scalar Bernoulli form.
4. Delete the AUPRC capture-rate proposition or rewrite it as a conditional
   ranking-stability lemma with margin assumptions.
5. Rewrite the contract section around all four hard contracts from
   `AGENTS.md`.
6. Define `proto_proj` / `rho` exactly, or drop the proto-head until such a
   teacher tensor exists.
7. Add ablations: final-only vs multi-head, no gate-head, no proto-head,
   no reliability weight, mask normalized vs unnormalized, and
   `lambda_bce in {0, 0.05, 1.0}`.

## Acceptance Gate

The proof-checker acceptance gate fails:

- FATAL/CRITICAL issues remain open.
- The main C3 theorem-style claim is false under stated assumptions.
- Several hypotheses are undefined or inconsistent with canonical RAER.
- Counterexample pass found a concrete AUPRC counterexample.

No target design files were edited in this audit.

## Report Compile Check

`proof_audit_report.tex` was generated, but PDF compilation failed in the
current environment because `/data1/anaconda3/bin/pdflatex` could not find
`pdflatex.fmt` during format initialization. The `.tex` source is present and
the JSON/Markdown audit artifacts validated successfully.
