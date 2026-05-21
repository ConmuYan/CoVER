# Chapter 3 Methodology Review

Date: 2026-05-21

Reviewer route: Codex subagent, `gpt-5.5`, xhigh reasoning.

Reviewer agent id: `019e499c-f08d-7ac2-9441-b84937eb6d91`

## Context

Reviewed file: `docs/chapter3_methodology.md`.

The review evaluated the revised RAER-FD methodology chapter after removing unsupported Teacher auxiliary losses and simplifying the CBR-Flash student narrative.

User correction during review: E7 loss ablation was done incorrectly and must **not** support the methodology narrative.

## Scores

| Dimension | Score | Meaning |
|---|---:|---|
| Standalone methodology quality | 8.0 / 10 | Strong workshop / solid journal-revision / borderline top-venue methodology section |
| Implementation/results consistency | 8.3 / 10 | Mostly aligned with current implementation; main risk is over-interpreting CBR-Flash simplification |

## Main Judgment

The chapter is substantially improved: the RAER Teacher story is clear, the base-freeze / score-blind / train-only / bounded-residual contracts are well stated, and the LREE role is cleaner than before.

The remaining weakness is CBR-Flash wording. Without valid E7 evidence, the chapter should not imply that the simplified student objective is empirically proven superior to multi-head Teacher matching. It should be framed as a conservative, implementation-consistent objective: supervised anchor plus Teacher residual-budget guidance.

## Top Issues

1. CBR-Flash is closer to a budget-aware supervised residual learner than a full distillation student unless it explicitly matches Teacher logits/residuals.
2. The term “distillation” should be used carefully; “Teacher-guided residual budgeting” may better describe the current objective.
3. LREE should specify the feature space of prototypes and clarify that prototypes are fixed train-split statistics.
4. The PriorF-GNN saturation story needs boundary conditions: near-identity residuals are meaningful only when strong-base evidence supports saturation and non-degradation.
5. Diagnostic quantities should not be implied to be faithful explanations.

## Recommended Edits

1. In the opening, change CBR-Flash from “compresses the residual interface” to “learns a lightweight same-contract residual interface using Teacher residual magnitude as budget guidance.”
2. In Section 3.4, explicitly state the budget term is not a Teacher residual matching loss.
3. After the student objective, add that the objective is chosen for conservative implementation consistency, not because invalid or insufficient loss ablations prove it optimal.
4. After prototype equations, add that prototypes are computed on the training split and then fixed for all nodes.
5. In Stage 2, state that only final-logit residual magnitude enters the student budget; gate and per-relation contribution are diagnostics only.

## Verdict

One more revision pass is recommended before treating the chapter as strong-journal ready. The pass should focus on CBR-Flash terminology and evidence discipline, not on adding more technical detail.
