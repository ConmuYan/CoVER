# CoVER-FD Project Memory

> Last updated: 2026-05-12
> Current phase: Task 2 complete, ready for Task 3

---

## Project Overview

**CoVER-FD**: Contract-Verified Evidence Distillation for LLM-Free Fake Review Detection

Three-stage pipeline:
1. Train base GNN detector
2. Generate score-blind evidence cards → rule/LLM teacher → ERR → verifier → cache
3. Train evidence-conditioned student reasoner (LLM-free inference)

---

## Completion Status

### ✅ Task 1: Data Loading + Stage 1 Baseline

| Component | File | Status |
|-----------|------|--------|
| Data loader | `data/load_fraud.py` | ✅ .pt/.pkl/.npz/.mat + synthetic |
| Split utility | `data/split.py` | ✅ deterministic + scarcity |
| GNN models | `models/gnn.py` | ✅ GCN/SAGE/GAT |
| Metrics | `training/metrics.py` | ✅ AUC/AUPRC/F1/P@K/R@K |
| Training script | `scripts/train_stage1.py` | ✅ --debug support |
| Tests | `tests/test_data_loading.py`, `tests/test_reasoner_shapes.py` | ✅ 16 pass |

**Verification**: `pytest -q` = 16 passed, `train_stage1.py --debug` = success

### ✅ Task 2: Evidence Schema + Rule ERR

| Component | File | Status |
|-----------|------|--------|
| Schema | `evidence/schema.py` | ✅ CalibrationChannel, ReasoningChannel, EvidenceCard, ERR |
| Score-blind check | `evidence/prompt.py` | ✅ SCORE_LEAKAGE_KEYS, assert_score_blind_payload |
| Evidence adapter | `evidence/adapter.py` | ✅ degree, neighbor consistency, discrepancy |
| Rule teacher | `evidence/rule_teacher.py` | ✅ deterministic ERR generation |
| Stage 2 script | `scripts/generate_stage2_err.py` | ✅ --teacher rule --debug |
| Tests | `tests/test_score_blind.py`, `tests/test_evidence_adapter.py` | ✅ 28 pass total |

**Verification**: `pytest -q` = 28 passed, `generate_stage2_err.py --debug` = 35 nodes, score-blind PASSED

### ⬜ Task 3: Verifier (Next)

- `evidence/contracts.yaml` - fraud reason type definitions
- `evidence/verifier.py` - schema/availability/role/contract/score-blind/label checks
- `tests/test_verifier.py`
- Update `generate_stage2_err.py` to save accepted/rejected ERR

### ⬜ Task 4: Reasoner + Stage 3

- `models/reasoner.py` - EvidenceEncoder, type/pos/neg heads, residual readout
- `training/losses.py` - supervised + evidence distillation loss
- `scripts/train_stage3.py`
- `tests/test_reasoner_shapes.py` - rho=0 test

### ⬜ Task 5: LLM Teacher

- `evidence/llm_teacher.py` - openai-compatible, temperature=0, retry=3
- Update `generate_stage2_err.py` --teacher llm

### ⬜ Task 6: Experiments

- `scripts/run_scarcity.py` - 5/10/20/40/100% train ratios
- `scripts/run_ablation.py` - full/no_verifier/no_counter/etc.
- `scripts/aggregate_results.py` - mean ± std

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model API | `forward(x, edge_index)` → all-node logits | Cleaner than masked forward |
| Schema | dataclasses | Simpler than Pydantic |
| Teacher | Rule first, LLM later | Offline only, cache replay |
| Verifier | Deterministic contracts | No LLM-as-judge |
| Training | Stage 1→2(cache)→3(distill) | Decoupled, reproducible |

---

## Artifacts Structure

```
artifacts/
├── checkpoints/{dataset}/{model}/seed_{seed}/base.pt
├── logs/{dataset}/{model}/seed_{seed}/stage1.json
├── err_cache/{dataset}/{model}/seed_{seed}/
│   ├── evidence_cards.jsonl
│   ├── teacher_payloads.jsonl
│   ├── rule_err.jsonl
│   └── stage2_stats.json
└── results/  (Stage 3 metrics - not yet created)
```

---

## Score-Blind Boundary

**CalibrationChannel** (our use only):
- base_score, uncertainty

**ReasoningChannel** (LLM visible):
- degree_level, neighbor_consistency, feature_neighbor_discrepancy
- detector_signal, detector_signal_strength, counter_signal
- allowed_support_ids, allowed_counter_ids

**Forbidden in LLM prompt**: base_score, score, logit, prob, probability, confidence, label, y, target

---

## Validation Commands

```bash
# Current (Task 1-2)
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug

# After Task 3
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

---

## External References

- `external/gread-core/` - GReaD-Core: full implementation reference
- `external/LinguGKD/` - LinguGKD: distillation pattern reference
- `docs/repo_understanding_report.md` - analysis of both repos

---

## Debug Checklist

- [x] rho=0 时 CoVER-FD 输出必须等于 base detector (Task 4)
- [x] prompt 文件中不得出现 score / logit / confidence
- [ ] accepted_mask 为 0 时训练不能报错 (Task 4)
- [ ] test nodes 不允许使用 label compatibility (Task 3)
- [x] summary 不进入任何 loss
- [ ] LLM 输出必须 cache，训练阶段不能联网 (Task 5)
- [ ] 每个 seed 保存 split index、checkpoint、ERR cache hash

---

## Next Action

**Task 3: Implement Evidence Contract Verifier**

Files to create:
1. `evidence/contracts.yaml`
2. `evidence/verifier.py`
3. `tests/test_verifier.py`

Files to modify:
1. `scripts/generate_stage2_err.py` - add verifier, save accepted/rejected

Acceptance:
```bash
pytest -q
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
# Should see: verifier_stats.json, accepted.jsonl, rejected.jsonl
```
