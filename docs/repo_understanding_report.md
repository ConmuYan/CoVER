# Repository Understanding Report

## external/gread-core

### Project Entry
- `src/gread_core/` - Main package
- `scripts/` - Training and evaluation scripts
- `tests/` - Unit and paper alignment tests

### Key Modules
| Module | Purpose |
|--------|---------|
| `data/loaders.py` | Dataset loading (yelpchi, amazon, tfinance, tsocial, synthetic) |
| `data/splits.py` | Deterministic train/val/test mask generation |
| `detectors/pyg_gnn.py` | GCN, GAT, SAGE, GIN detectors |
| `detectors/bwgnn.py` | Beta Wavelet GNN detector |
| `models/reasoner.py` | Evidence-conditioned residual reasoner |
| `evidence/` | Evidence schema, adapter, verifier |
| `losses/` | Supervised and reasoning losses |

### Reusable Ideas
1. **forward_with_embedding pattern** - Returns (logit, embedding) for downstream use
2. **Deterministic splits** - Seed-based mask generation with stratified option
3. **Score-blind evidence** - CalibrationChannel for model scores, ReasoningChannel for LLM
4. **Evidence Contract Verifier** - Deterministic, no LLM-as-judge
5. **Residual readout** - `final_logit = base_logit + rho * residual_logit`

### Reusable Interfaces
- `Data` object structure: x, edge_index, y, train_mask, val_mask, test_mask
- `forward(x, edge_index) -> (logit, embedding)` for all detectors
- `EvidenceCard` with calibration + reasoning channels
- `ERR` with risk_type, supporting/counter evidence

### Should NOT Reuse
- Complex abstractions (DetectorProtocol, adapter registry)
- LLM integration code (we implement our own)
- PyG GraphGym dependency (too heavy)

### Implementation Suggestions
1. Keep models simple - direct `forward(x, edge_index)` not graph-level methods
2. Use dataclasses for schema (not Pydantic)
3. Rule teacher first, LLM teacher later
4. Verifier should be pure Python, no external deps

---

## external/LinguGKD

### Project Entry
- `distillation/main.py` - Training entry point
- `distillation/man_model_builder.py` - Model creation
- `distillation/custom_graphgym/` - Custom loss and training

### Key Ideas
1. **Teacher-student distillation** - Cache teacher outputs, train student separately
2. **Loss weight learning** - Softmax-weighted multi-task loss
3. **PyTorch Lightning** - Clean training loop

### Reusable Ideas
1. **Decoupled stages** - Get teacher, cache outputs, train student
2. **Teacher output caching** - Save to disk, load during student training
3. **Loss weight management** - Adaptive weighting for multi-task

### Should NOT Reuse
- PyG GraphGym dependency (too complex for research)
- WandB integration (optional)
- Heavy config system

---

## Relationship to cover-fd

| gread-core Component | cover-fd Equivalent |
|---------------------|---------------------|
| `data/loaders.py` | `data/load_fraud.py` |
| `data/splits.py` | `data/split.py` |
| `detectors/pyg_gnn.py` | `models/gnn.py` |
| `models/reasoner.py` | `models/reasoner.py` (Task 4) |
| `evidence/schema.py` | `evidence/schema.py` (Task 2) |
| `evidence/adapter.py` | `evidence/adapter.py` (Task 2) |
| `evidence/verifier.py` | `evidence/verifier.py` (Task 3) |

| LinguGKD Component | cover-fd Equivalent |
|-------------------|---------------------|
| Teacher caching | ERR caching (Stage 2) |
| Student training | Reasoner training (Stage 3) |
| Loss weighting | `training/losses.py` |

---

## Key Design Decisions

1. **Model API**: `forward(x, edge_index)` returning all-node logits (not masked)
2. **Schema**: Use dataclasses (not Pydantic) for simplicity
3. **Teacher**: Rule first, LLM later (offline only)
4. **Verifier**: Deterministic contracts, no LLM-as-judge
5. **Training**: Stage 1 → Stage 2 (cache) → Stage 3 (distill)
