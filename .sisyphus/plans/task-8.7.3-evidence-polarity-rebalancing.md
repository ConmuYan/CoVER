# Task 8.7.3: Evidence Polarity Rebalancing for CoVER-DIR

## TL;DR

> **Quick Summary**: Rebalance EvidenceCard/graph evidence token generation so that score-blind payloads can represent fraud-dominant, benign-dominant, and mixed evidence. Fix the upstream evidence polarity issue that causes all payloads to be fraud-dominant.
>
> **Deliverables**:
> - Token polarity system in `evidence/vocab.py`
> - Token polarity learning from train-only statistics in `evidence/prototypes.py`
> - Balanced graph evidence token generation in `evidence/adapter.py`
> - Fixed prototype-relative fields using distinctive tokens
> - Payload polarity audit script
> - Tests for evidence polarity
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Step A → Step B → Step C → Step D → Step E → Step F

---

## Context

### Original Request

Task 8.7.2 completed direction calibration and payload diagnostics. It found that the lack of `decrease_risk` is not primarily a prompt problem. The evidence payloads are overwhelmingly fraud-dominant:
- benign_signal_available_rate = 11.76%
- fraud_signal_available_rate = 100%
- benign_dominant_payload_count = 0
- fraud_dominant_payload_count = 15
- mixed_payload_count = 2

**Conclusion**: Do not run `qwen_directional_t200` yet. The Stage2 evidence generation must be fixed first.

### Research Findings

1. **Root Cause**: The evidence adapter generates fraud-like signals for ALL nodes because:
   - Evidence fields are bucketed (low/medium/high)
   - Fraud prototype has "high" for most fields
   - Benign nodes may also have "high" values, causing them to match fraud prototype

2. **Current Implementation**:
   - `evidence/vocab.py`: Has `GRAPH_EVIDENCE_TOKENS` but no polarity classification
   - `evidence/prototypes.py`: Computes distinctive tokens but doesn't use them for similarity
   - `evidence/adapter.py`: Uses raw field matching for prototype similarity, not distinctive tokens

3. **Key Issue**: `compute_prototype_similarity()` compares raw field values (high/medium/low) instead of using distinctive tokens from train-only statistics.

---

## Work Objectives

### Core Objective

Rebalance evidence generation so that:
1. Fraud-dominant payloads contain fraud-like tokens
2. Benign-dominant payloads contain benign-like tokens
3. Mixed payloads contain both types
4. Prototype similarity uses distinctive tokens, not raw field matching

### Concrete Deliverables

1. **Token polarity system**: `TOKEN_POLARITY_FRAUD`, `TOKEN_POLARITY_BENIGN`, `TOKEN_POLARITY_NEUTRAL` sets in `evidence/vocab.py`
2. **Token polarity learning**: `compute_token_polarity_stats()` in `evidence/prototypes.py`
3. **Balanced token generation**: Add benign-like tokens to `evidence/adapter.py`
4. **Fixed prototype similarity**: Use distinctive tokens for similarity computation
5. **Payload polarity audit**: `scripts/audit_payload_polarity.py`
6. **Tests**: `tests/test_evidence_polarity.py`

### Definition of Done

- [ ] `pytest -q` passes
- [ ] Payload polarity audit shows:
  - `benign_dominant_payload_count >= 5`
  - `fraud_dominant_payload_count >= 5`
  - `mixed_payload_count >= 3`
  - `benign_signal_available_rate >= 30%`
  - `fraud_signal_available_rate >= 50%`

### Must Have

- Token polarity classification (fraud-like, benign-like, neutral)
- Benign-like token generation (HF_RATIO_LOW, FEAT_NEIGH_COS_TOP20, etc.)
- Token polarity learning from train-only statistics
- Prototype similarity using distinctive tokens
- Payload polarity audit script

### Must NOT Have (Guardrails)

- No base_score/score/logit/prob/confidence values to LLM
- No base prediction to LLM
- No target label to LLM
- No FN/FP/base_error terminology to LLM
- No test labels in prototypes/retrieval/trace/prompt/verifier/loss
- No artificial forcing of LLM outputs

---

## Verification Strategy

### Test Decision

- **Infrastructure exists**: YES
- **Automated tests**: YES (tests-after)
- **Framework**: pytest
- **QA Policy**: Every task includes agent-executed QA scenarios

### QA Policy

Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
├── Task 1: Add token polarity system to evidence/vocab.py [quick]
├── Task 2: Add token polarity learning to evidence/prototypes.py [unspecified-high]
└── Task 3: Add benign-like token generation to evidence/adapter.py [unspecified-high]

Wave 2 (After Wave 1 - integration):
├── Task 4: Fix prototype-relative fields in evidence/adapter.py [deep]
├── Task 5: Add payload polarity summary to EvidenceCard [unspecified-high]
└── Task 6: Create payload polarity audit script [quick]

Wave 3 (After Wave 2 - testing):
├── Task 7: Create tests/test_evidence_polarity.py [unspecified-high]
└── Task 8: Run payload polarity audit and verify [quick]

Wave FINAL (After ALL tasks):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

- **1**: None → 2, 3, 4
- **2**: 1 → 4, 6
- **3**: 1 → 4, 5
- **4**: 2, 3 → 5, 6
- **5**: 3, 4 → 7
- **6**: 2, 4 → 8
- **7**: 5 → 8
- **8**: 6, 7 → F1-F4

---

## TODOs

- [ ] 1. Add token polarity system to evidence/vocab.py

  **What to do**:
  - Add `TOKEN_POLARITY_FRAUD` set with fraud-like tokens
  - Add `TOKEN_POLARITY_BENIGN` set with benign-like tokens
  - Add `TOKEN_POLARITY_NEUTRAL` set with neutral/mixed tokens
  - Add `TOKEN_POLARITY_MAP` dict for fallback polarity
  - Update `GRAPH_EVIDENCE_TOKENS` to include all polarity types

  **Must NOT do**:
  - Do not include base_score/score/logit/prob/confidence
  - Do not include target label or base prediction

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References**:
  - `evidence/vocab.py:30-63` - Current GRAPH_EVIDENCE_TOKENS definition
  - `evidence/vocab.py:65-72` - REASON_TYPES for reference

  **Acceptance Criteria**:
  - [ ] `TOKEN_POLARITY_FRAUD` contains 13 tokens
  - [ ] `TOKEN_POLARITY_BENIGN` contains 12 tokens
  - [ ] `TOKEN_POLARITY_NEUTRAL` contains 3 tokens
  - [ ] `TOKEN_POLARITY_MAP` maps each token to polarity
  - [ ] `pytest tests/test_evidence_polarity.py::TestTokenPolarity -v` passes

  **QA Scenarios**:

  ```
  Scenario: Token polarity sets are correctly defined
    Tool: Bash (python)
    Preconditions: evidence/vocab.py updated
    Steps:
      1. python -c "from evidence.vocab import TOKEN_POLARITY_FRAUD, TOKEN_POLARITY_BENIGN, TOKEN_POLARITY_NEUTRAL; print(f'Fraud: {len(TOKEN_POLARITY_FRAUD)}, Benign: {len(TOKEN_POLARITY_BENIGN)}, Neutral: {len(TOKEN_POLARITY_NEUTRAL)}')"
      2. Assert output shows correct counts
    Expected Result: Fraud: 13, Benign: 12, Neutral: 3
    Evidence: .sisyphus/evidence/task-1-polarity-sets.txt
  ```

  **Commit**: YES
  - Message: `feat(vocab): add token polarity classification system`
  - Files: `evidence/vocab.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py::TestTokenPolarity -v`

---

- [ ] 2. Add token polarity learning to evidence/prototypes.py

  **What to do**:
  - Add `compute_token_polarity_stats()` function
  - Compute for each token: count in fraud/benign, P(token|fraud), P(token|benign), log_odds, class_contrast_score
  - Use train nodes only
  - Save to `prototype_stats.json`
  - If learned polarity conflicts with hand-coded polarity, log warning and prefer learned

  **Must NOT do**:
  - Do not use val/test labels
  - Do not use base_score/prob/logit/confidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 6
  - **Blocked By**: Task 1

  **References**:
  - `evidence/prototypes.py:51-95` - `_compute_distinctive_tokens()` function
  - `evidence/prototypes.py:130-241` - `PrototypeBuilder.build()` method
  - `evidence/vocab.py` - TOKEN_POLARITY_FRAUD, TOKEN_POLARITY_BENIGN (from Task 1)

  **Acceptance Criteria**:
  - [ ] `compute_token_polarity_stats()` function exists
  - [ ] Returns dict with `token_polarity_stats`, `fraud_distinctive_tokens`, `benign_distinctive_tokens`, `neutral_tokens`
  - [ ] `test_label_used=false` in output
  - [ ] `pytest tests/test_evidence_polarity.py::TestTokenPolarityLearning -v` passes

  **QA Scenarios**:

  ```
  Scenario: Token polarity stats use train nodes only
    Tool: Bash (python)
    Preconditions: evidence/prototypes.py updated
    Steps:
      1. python -c "
         import torch
         from evidence.prototypes import PrototypeBuilder
         builder = PrototypeBuilder()
         train_mask = torch.tensor([True, True, False, False])
         y = torch.tensor([1, 0, 1, 0])
         evidence_tokens = {0: {'degree_level': 'high'}, 1: {'degree_level': 'low'}}
         base_preds = torch.tensor([1, 0, 1, 0])
         result = builder.build(train_mask, y, evidence_tokens, base_preds)
         print('test_label_used:', result.get('token_polarity_stats', {}).get('test_label_used', 'missing'))
         "
      2. Assert output shows test_label_used: False
    Expected Result: test_label_used: False
    Evidence: .sisyphus/evidence/task-2-polarity-stats.txt
  ```

  **Commit**: YES
  - Message: `feat(prototypes): add token polarity learning from train-only statistics`
  - Files: `evidence/prototypes.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py::TestTokenPolarityLearning -v`

---

- [ ] 3. Add benign-like token generation to evidence/adapter.py

  **What to do**:
  - Add benign-like token generation:
    - high feature-neighbor cosine → FEAT_NEIGH_COS_TOP20
    - high embedding-neighbor cosine → EMB_NEIGH_COS_TOP20
    - low high-frequency ratio → HF_RATIO_LOW
    - stable band energies → BAND_ENERGY_STABLE
    - low normal-structure distance → NORMAL_STRUCTURE_DIST_LOW
    - low interference / stable clean view → CLEAN_VIEW_STABLE or LOW_INTERFERENCE_EDGE_RATIO
    - high neighbor consistency → NEIGHBOR_CONSISTENCY_HIGH
  - Keep existing fraud-like tokens
  - Update `generate_graph_evidence_tokens()` and `generate_graph_evidence_tokens_vectorized()`

  **Must NOT do**:
  - Do not include raw base score, probability, logit, confidence
  - Do not include target label or base prediction

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Task 1

  **References**:
  - `evidence/adapter.py:546-666` - `generate_graph_evidence_tokens()` method
  - `evidence/adapter.py:762-858` - `generate_graph_evidence_tokens_vectorized()` method
  - `evidence/vocab.py` - TOKEN_POLARITY_BENIGN (from Task 1)

  **Acceptance Criteria**:
  - [ ] Benign-like tokens can be generated (FEAT_NEIGH_COS_TOP20, EMB_NEIGH_COS_TOP20, etc.)
  - [ ] `pytest tests/test_evidence_polarity.py::TestBenignTokenGeneration -v` passes

  **QA Scenarios**:

  ```
  Scenario: Benign-like tokens can be generated
    Tool: Bash (python)
    Preconditions: evidence/adapter.py updated
    Steps:
      1. python -c "
         import torch
         from evidence.adapter import EvidenceAdapter
         adapter = EvidenceAdapter('bwgnn', torch.randn(10, 5), torch.randint(0, 10, (2, 20)))
         tokens = adapter.generate_graph_evidence_tokens(0, torch.randn(10), torch.randn(10, 5), None, None)
         print('Tokens:', tokens)
         has_benign = any(t in TOKEN_POLARITY_BENIGN for t in tokens)
         print('Has benign:', has_benign)
         "
      2. Assert output shows benign tokens present
    Expected Result: Has benign: True
    Evidence: .sisyphus/evidence/task-3-benign-tokens.txt
  ```

  **Commit**: YES
  - Message: `feat(adapter): add benign-like token generation`
  - Files: `evidence/adapter.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py::TestBenignTokenGeneration -v`

---

- [ ] 4. Fix prototype-relative fields in evidence/adapter.py

  **What to do**:
  - Update `_compute_prototype_relative_fields()` to use distinctive tokens
  - Compute `fraud_similarity` = similarity to fraud_distinctive_tokens
  - Compute `benign_similarity` = similarity to benign_distinctive_tokens
  - Update `closer_to_fraud_prototype` based on fraud_similarity
  - Update `closer_to_benign_prototype` based on benign_similarity
  - Update `prototype_conflict_level` based on both similarities
  - Use train-only IDF-weighted token similarity

  **Must NOT do**:
  - Do not use raw high/medium/low field matching alone
  - Do not use test labels

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `evidence/adapter.py:380-415` - `_compute_prototype_relative_fields()` method
  - `evidence/adapter.py:77-103` - `compute_prototype_similarity()` function
  - `evidence/prototypes.py:51-95` - `_compute_distinctive_tokens()` function

  **Acceptance Criteria**:
  - [ ] Prototype similarity uses distinctive tokens, not raw field matching
  - [ ] `pytest tests/test_evidence_polarity.py::TestPrototypeSimilarity -v` passes

  **QA Scenarios**:

  ```
  Scenario: Prototype similarity uses distinctive tokens
    Tool: Bash (python)
    Preconditions: evidence/adapter.py updated
    Steps:
      1. python -c "
         from evidence.adapter import compute_prototype_similarity
         from evidence.schema import ReasoningChannel
         reasoning = ReasoningChannel(degree_level='high', neighbor_consistency='low', feature_neighbor_discrepancy='high', detector_signal='strong', detector_signal_strength='strong', counter_signal='weak')
         fraud_proto = {'degree_level': 'high', 'neighbor_consistency': 'low'}
         benign_proto = {'degree_level': 'low', 'neighbor_consistency': 'high'}
         fraud_match, _, _ = compute_prototype_similarity(reasoning, fraud_proto)
         benign_match, _, _ = compute_prototype_similarity(reasoning, benign_proto)
         print(f'Fraud match: {fraud_match}, Benign match: {benign_match}')
         "
      2. Assert output shows higher fraud_match for fraud-like reasoning
    Expected Result: Fraud match > Benign match
    Evidence: .sisyphus/evidence/task-4-prototype-similarity.txt
  ```

  **Commit**: YES
  - Message: `fix(adapter): use distinctive tokens for prototype similarity`
  - Files: `evidence/adapter.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py::TestPrototypeSimilarity -v`

---

- [ ] 5. Add payload polarity summary to EvidenceCard

  **What to do**:
  - Add `evidence_polarity` field to ReasoningChannel or EvidenceCard
  - Compute: fraud_token_count, benign_token_count, neutral_token_count
  - Compute: fraud_token_score, benign_token_score
  - Compute: evidence_polarity (fraud_dominant / benign_dominant / mixed / weak)
  - Compute: evidence_polarity_margin (high / medium / low)
  - All fields must be categorical or bucketed
  - Include in teacher payload

  **Must NOT do**:
  - Do not include raw base score, probability, logit, confidence
  - Do not include target label or base prediction

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 3, 4

  **References**:
  - `evidence/schema.py:13-46` - ReasoningChannel dataclass
  - `evidence/schema.py:48-85` - EvidenceCard dataclass
  - `evidence/adapter.py:306-315` - Prototype-relative fields integration

  **Acceptance Criteria**:
  - [ ] `evidence_polarity` field exists in ReasoningChannel or EvidenceCard
  - [ ] Teacher payload contains `evidence_polarity`
  - [ ] `pytest tests/test_evidence_polarity.py::TestPayloadPolarity -v` passes

  **QA Scenarios**:

  ```
  Scenario: Teacher payload contains evidence_polarity
    Tool: Bash (python)
    Preconditions: evidence/schema.py and evidence/adapter.py updated
    Steps:
      1. python -c "
         import torch
         from evidence.adapter import EvidenceAdapter
         adapter = EvidenceAdapter('bwgnn', torch.randn(10, 5), torch.randint(0, 10, (2, 20)))
         card = adapter._extract_single(0, torch.randn(10), torch.randn(10, 5), None, None)
         payload = card.to_teacher_payload()
         print('evidence_polarity:', payload.get('reasoning', {}).get('evidence_polarity', 'missing'))
         "
      2. Assert output shows evidence_polarity present
    Expected Result: evidence_polarity: fraud_dominant or benign_dominant or mixed or weak
    Evidence: .sisyphus/evidence/task-5-payload-polarity.txt
  ```

  **Commit**: YES
  - Message: `feat(adapter): add payload polarity summary to EvidenceCard`
  - Files: `evidence/schema.py`, `evidence/adapter.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py::TestPayloadPolarity -v`

---

- [ ] 6. Create payload polarity audit script

  **What to do**:
  - Create `scripts/audit_payload_polarity.py`
  - Sample 30 nodes (10 FN, 10 FP, 5 high-loss, 5 val boundary)
  - Report: benign_signal_available_rate, fraud_signal_available_rate
  - Report: benign_dominant_payload_count, fraud_dominant_payload_count, mixed_payload_count, weak_payload_count
  - Report: evidence_polarity distribution, token polarity distribution
  - Report: benign token coverage by node category, fraud token coverage by node category
  - Run without LLM first

  **Must NOT do**:
  - Do not use test labels in prototypes/retrieval/trace

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `scripts/run_stage2_microbenchmark.py:54-105` - `sample_microbenchmark_nodes()` function
  - `scripts/run_stage2_microbenchmark.py:148-233` - `compute_payload_diagnostics()` function

  **Acceptance Criteria**:
  - [ ] `scripts/audit_payload_polarity.py` exists
  - [ ] Script runs without LLM
  - [ ] Reports all required metrics

  **QA Scenarios**:

  ```
  Scenario: Audit script runs without LLM
    Tool: Bash (python)
    Preconditions: scripts/audit_payload_polarity.py created
    Steps:
      1. python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --seed 123 --debug
      2. Assert output shows payload diagnostics
    Expected Result: Script completes and shows benign_dominant_payload_count, fraud_dominant_payload_count, etc.
    Evidence: .sisyphus/evidence/task-6-audit-script.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add payload polarity audit script`
  - Files: `scripts/audit_payload_polarity.py`
  - Pre-commit: `python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --seed 123 --debug`

---

- [ ] 7. Create tests/test_evidence_polarity.py

  **What to do**:
  - Create `tests/test_evidence_polarity.py`
  - Test 1: benign-like tokens can be generated
  - Test 2: fraud-like tokens can be generated
  - Test 3: token polarity stats use train nodes only
  - Test 4: prototype similarity uses distinctive tokens
  - Test 5: payload polarity distinguishes fraud_dominant / benign_dominant / mixed
  - Test 6: teacher payload contains evidence_polarity but not labels/base predictions/scores
  - Test 7: score-blind check still passes

  **Must NOT do**:
  - Do not use test labels in tests

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 8)
  - **Blocks**: Task 8
  - **Blocked By**: Task 5

  **References**:
  - `tests/test_prototypes.py` - Existing prototype tests
  - `tests/test_retrieval.py` - Existing retrieval tests
  - `tests/test_directional_prompt.py` - Existing prompt tests

  **Acceptance Criteria**:
  - [ ] `tests/test_evidence_polarity.py` exists
  - [ ] All 7 tests pass
  - [ ] `pytest tests/test_evidence_polarity.py -v` shows all PASS

  **QA Scenarios**:

  ```
  Scenario: All polarity tests pass
    Tool: Bash (pytest)
    Preconditions: tests/test_evidence_polarity.py created
    Steps:
      1. pytest tests/test_evidence_polarity.py -v
      2. Assert all tests pass
    Expected Result: 7 passed
    Evidence: .sisyphus/evidence/task-7-polarity-tests.txt
  ```

  **Commit**: YES
  - Message: `test(evidence): add evidence polarity tests`
  - Files: `tests/test_evidence_polarity.py`
  - Pre-commit: `pytest tests/test_evidence_polarity.py -v`

---

- [ ] 8. Run payload polarity audit and verify

  **What to do**:
  - Run `scripts/audit_payload_polarity.py` with real data
  - Verify passing conditions:
    - `benign_dominant_payload_count >= 5`
    - `fraud_dominant_payload_count >= 5`
    - `mixed_payload_count >= 3`
    - `benign_signal_available_rate >= 30%`
    - `fraud_signal_available_rate >= 50%`
  - If audit fails, output token/prototype diagnosis

  **Must NOT do**:
  - Do not run Qwen if audit fails

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 7)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 6, 7

  **References**:
  - `scripts/audit_payload_polarity.py` - Audit script (from Task 6)
  - `configs/yelpchi_bwgnn.yaml` - Config file

  **Acceptance Criteria**:
  - [ ] Audit script runs successfully
  - [ ] All passing conditions met
  - [ ] Report saved to `artifacts/reports/payload_polarity_audit.md`

  **QA Scenarios**:

  ```
  Scenario: Payload polarity audit passes
    Tool: Bash (python)
    Preconditions: All previous tasks completed
    Steps:
      1. python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --seed 123
      2. Check output for passing conditions
    Expected Result: benign_dominant_payload_count >= 5, fraud_dominant_payload_count >= 5, mixed_payload_count >= 3
    Evidence: .sisyphus/evidence/task-8-audit-results.txt
  ```

  **Commit**: YES
  - Message: `chore: run payload polarity audit`
  - Files: `artifacts/reports/payload_polarity_audit.md`
  - Pre-commit: None

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest -q`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run payload polarity audit with real data. Verify all passing conditions met.

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built.

---

## Commit Strategy

- **1**: `feat(vocab): add token polarity classification system` - evidence/vocab.py
- **2**: `feat(prototypes): add token polarity learning from train-only statistics` - evidence/prototypes.py
- **3**: `feat(adapter): add benign-like token generation` - evidence/adapter.py
- **4**: `fix(adapter): use distinctive tokens for prototype similarity` - evidence/adapter.py
- **5**: `feat(adapter): add payload polarity summary to EvidenceCard` - evidence/schema.py, evidence/adapter.py
- **6**: `feat(scripts): add payload polarity audit script` - scripts/audit_payload_polarity.py
- **7**: `test(evidence): add evidence polarity tests` - tests/test_evidence_polarity.py
- **8**: `chore: run payload polarity audit` - artifacts/reports/payload_polarity_audit.md

---

## Success Criteria

### Verification Commands

```bash
pytest -q  # Expected: All tests pass
python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --seed 123  # Expected: All passing conditions met
```

### Final Checklist

- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Payload polarity audit passes
- [ ] benign_dominant_payload_count >= 5
- [ ] fraud_dominant_payload_count >= 5
- [ ] mixed_payload_count >= 3
- [ ] benign_signal_available_rate >= 30%
- [ ] fraud_signal_available_rate >= 50%
