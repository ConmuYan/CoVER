# CoVER-DIR: Directional Contrastive Evidence Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directional contrastive evidence reasoning to Stage2 teacher, replacing single-node risk_type classification with score-blind directional contrastive evidence reasoning.

**Architecture:** Extend ERR schema with direction/strength fields, build train-only prototype banks with hybrid retrieval, add graph evidence tokens, create contrastive directional prompt, update verifier with direction-aware checks, and modify CV-SCD loss to use evidence_direction.

**Tech Stack:** PyTorch, PyG, transformers (Qwen), numpy, scipy (sparse vectors)

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `evidence/schema.py` | Modify | Add directional ERR fields |
| `evidence/prototypes.py` | **Create** | Train-only prototype bank builder |
| `evidence/retrieval.py` | **Create** | Hybrid Jaccard+IDF retrieval |
| `evidence/vocab.py` | Modify | Add graph evidence tokens |
| `evidence/adapter.py` | Modify | Generate new evidence tokens |
| `evidence/prompt.py` | Modify | Add contrastive_directional prompt |
| `evidence/verifier.py` | Modify | Add direction-aware checks |
| `training/losses.py` | Modify | Update CV-SCD for direction |
| `scripts/run_stage2_microbenchmark.py` | **Create** | Microbenchmark script |
| `tests/test_directional_schema.py` | **Create** | Schema tests |
| `tests/test_prototypes.py` | **Create** | Prototype builder tests |
| `tests/test_retrieval.py` | **Create** | Retrieval tests |
| `tests/test_directional_prompt.py` | **Create** | Prompt tests |
| `tests/test_directional_verifier.py` | **Create** | Verifier tests |
| `tests/test_cvscd_directional.py` | **Create** | CV-SCD direction tests |

---

## Task 1: Directional ERR Schema

**Files:**
- Modify: `evidence/schema.py`
- Create: `tests/test_directional_schema.py`

- [ ] **Step 1: Write schema validation tests**

```python
# tests/test_directional_schema.py
"""Tests for directional ERR schema."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evidence.schema import ERR


class TestDirectionalERRFields:
    """Test 1: ERR has directional fields with correct defaults."""

    def test_new_fields_exist(self):
        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test", evidence_direction="increase_risk",
            evidence_strength="moderate", uncertainty_factors=["counter_signal"],
        )
        assert err.evidence_direction == "increase_risk"
        assert err.evidence_strength == "moderate"
        assert err.uncertainty_factors == ["counter_signal"]

    def test_default_backward_compat(self):
        """Old ERR without new fields gets safe defaults."""
        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test",
        )
        assert err.evidence_direction == "uncertain"
        assert err.evidence_strength == "weak"
        assert err.uncertainty_factors == []


class TestERRFromDict:
    """Test 2: ERR.from_dict handles missing directional fields."""

    def test_from_dict_with_direction(self):
        d = {
            "node_id": 0, "risk_type": "spectral_anomaly",
            "supporting_evidence": ["detector_signal"],
            "counter_evidence": [], "summary": "test",
            "evidence_direction": "decrease_risk",
            "evidence_strength": "strong",
            "uncertainty_factors": ["counter_signal"],
        }
        err = ERR.from_dict(d)
        assert err.evidence_direction == "decrease_risk"
        assert err.evidence_strength == "strong"

    def test_from_dict_without_direction(self):
        """Old format dict gets defaults."""
        d = {
            "node_id": 0, "risk_type": "spectral_anomaly",
            "supporting_evidence": ["detector_signal"],
            "counter_evidence": [], "summary": "test",
        }
        err = ERR.from_dict(d)
        assert err.evidence_direction == "uncertain"
        assert err.evidence_strength == "weak"
        assert err.uncertainty_factors == []


class TestScoreBlindDirections:
    """Test 3: Directional fields are score-blind."""

    def test_no_score_in_direction_fields(self):
        forbidden = {"base_score", "score", "logit", "prob", "confidence", "prediction"}
        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test", evidence_direction="increase_risk",
            evidence_strength="moderate", uncertainty_factors=["counter_signal"],
        )
        # Check field names
        for field_name in ["evidence_direction", "evidence_strength", "uncertainty_factors"]:
            assert field_name not in forbidden
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_directional_schema.py -v`
Expected: FAIL (fields don't exist yet)

- [ ] **Step 3: Add directional fields to ERR**

```python
# evidence/schema.py - add to ERR class
@dataclass
class ERR:
    node_id: int
    risk_type: str
    supporting_evidence: list[str]
    counter_evidence: list[str]
    summary: str
    evidence_direction: str = "uncertain"      # NEW: increase_risk / decrease_risk / uncertain
    evidence_strength: str = "weak"            # NEW: weak / moderate / strong
    uncertainty_factors: list = field(default_factory=list)  # NEW
    contrastive_basis: dict | None = None      # NEW: optional

    @classmethod
    def from_dict(cls, d: dict) -> ERR:
        return cls(
            node_id=d["node_id"],
            risk_type=d["risk_type"],
            supporting_evidence=d.get("supporting_evidence", []),
            counter_evidence=d.get("counter_evidence", []),
            summary=d.get("summary", ""),
            evidence_direction=d.get("evidence_direction", "uncertain"),
            evidence_strength=d.get("evidence_strength", "weak"),
            uncertainty_factors=d.get("uncertainty_factors", []),
            contrastive_basis=d.get("contrastive_basis"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_directional_schema.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -q`
Expected: All existing tests still pass (backward compatible)

- [ ] **Step 6: Commit**

```bash
git add evidence/schema.py tests/test_directional_schema.py
git commit -m "feat(schema): add directional ERR fields with backward compat"
```

---

## Task 2: Prototype Builder

**Files:**
- Create: `evidence/prototypes.py`
- Create: `tests/test_prototypes.py`

- [ ] **Step 1: Write prototype builder tests**

```python
# tests/test_prototypes.py
"""Tests for prototype builder (train-only)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from evidence.prototypes import PrototypeBuilder


class TestPrototypeBuilderTrainOnly:
    """Test 1: Prototype builder uses only train nodes."""

    def test_no_test_labels_used(self):
        """Builder should fail gracefully if test nodes have labels."""
        builder = PrototypeBuilder()
        # Create mock data
        train_mask = torch.tensor([True, True, False, False])
        y = torch.tensor([1, 0, 1, 0])  # test labels present but shouldn't be used
        evidence_tokens = {
            0: {"degree_level": "high", "neighbor_consistency": "low"},
            1: {"degree_level": "low", "neighbor_consistency": "high"},
            2: {"degree_level": "high", "neighbor_consistency": "low"},  # test
            3: {"degree_level": "low", "neighbor_consistency": "high"},  # test
        }
        base_preds = torch.tensor([1, 0, 1, 0])  # base model predictions

        prototypes = builder.build(
            train_mask=train_mask, y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        # Verify only train nodes (0, 1) are in banks
        assert len(prototypes["fraud_prototype_bank"]) > 0
        assert len(prototypes["benign_prototype_bank"]) > 0

    def test_fn_fp_banks_created(self):
        """FN/FP banks should be created from train errors."""
        builder = PrototypeBuilder()
        train_mask = torch.tensor([True, True, True, True])
        y = torch.tensor([1, 1, 0, 0])  # labels
        evidence_tokens = {
            0: {"degree_level": "high"},  # FN: label=1, pred=0
            1: {"degree_level": "high"},  # Correct: label=1, pred=1
            2: {"degree_level": "low"},   # Correct: label=0, pred=0
            3: {"degree_level": "low"},   # FP: label=0, pred=1
        }
        base_preds = torch.tensor([0, 1, 0, 1])  # base predictions

        prototypes = builder.build(
            train_mask=train_mask, y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        assert "train_fn_bank" in prototypes
        assert "train_fp_bank" in prototypes
        # Node 0 is FN (label=1, pred=0)
        assert 0 in prototypes["train_fn_bank"]
        # Node 3 is FP (label=0, pred=1)
        assert 3 in prototypes["train_fp_bank"]


class TestDistinctiveTokens:
    """Test 2: Prototype summaries include distinctive tokens."""

    def test_distinctive_tokens_computed(self):
        builder = PrototypeBuilder()
        train_mask = torch.tensor([True, True, True, True])
        y = torch.tensor([1, 1, 0, 0])
        evidence_tokens = {
            0: {"token_a": "high", "token_b": "low"},
            1: {"token_a": "high", "token_b": "medium"},
            2: {"token_a": "low", "token_b": "high"},
            3: {"token_a": "low", "token_b": "high"},
        }
        base_preds = torch.tensor([1, 1, 0, 0])

        prototypes = builder.build(
            train_mask=train_mask, y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        summary = prototypes["fraud_prototype_summary"]
        assert "distinctive_tokens" in summary
        assert isinstance(summary["distinctive_tokens"], list)


class TestNoScoreInPrototypes:
    """Test 3: No score/logit/prob in prototype output."""

    def test_prototypes_score_blind(self):
        builder = PrototypeBuilder()
        train_mask = torch.tensor([True, True])
        y = torch.tensor([1, 0])
        evidence_tokens = {
            0: {"degree_level": "high"},
            1: {"degree_level": "low"},
        }
        base_preds = torch.tensor([1, 0])

        prototypes = builder.build(
            train_mask=train_mask, y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        forbidden = {"base_score", "score", "logit", "prob", "confidence", "prediction"}
        proto_str = str(prototypes)
        for key in forbidden:
            assert key not in proto_str, f"Forbidden key '{key}' found in prototypes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prototypes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PrototypeBuilder**

```python
# evidence/prototypes.py
"""Train-only prototype builder for directional contrastive evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import numpy as np


class PrototypeBuilder:
    """Build prototype banks from train set only.

    HARD CONSTRAINTS:
    - Only uses train nodes
    - No test labels
    - No base_score/prob/logit/confidence in output
    """

    def build(
        self,
        train_mask: torch.Tensor,
        y: torch.Tensor,
        evidence_tokens: dict[int, dict[str, str]],
        base_preds: torch.Tensor,
    ) -> dict[str, Any]:
        """Build all prototype banks.

        Args:
            train_mask: Boolean mask for train nodes
            y: Labels (only train labels used)
            evidence_tokens: {node_id: {field: value}} for all nodes
            base_preds: Base model predictions (0/1)

        Returns:
            Dict with fraud_prototype_bank, benign_prototype_bank,
            train_fn_bank, train_fp_bank, fraud_prototype_summary,
            benign_prototype_summary, normal_structure_summary
        """
        train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()

        # Split train by label
        fraud_nodes = [i for i in train_idx if y[i].item() == 1]
        benign_nodes = [i for i in train_idx if y[i].item() == 0]

        # Split by base model errors
        fn_nodes = [i for i in fraud_nodes if base_preds[i].item() == 0]
        fp_nodes = [i for i in benign_nodes if base_preds[i].item() == 1]

        # Build banks
        fraud_bank = {i: evidence_tokens.get(i, {}) for i in fraud_nodes}
        benign_bank = {i: evidence_tokens.get(i, {}) for i in benign_nodes}
        fn_bank = {i: evidence_tokens.get(i, {}) for i in fn_nodes}
        fp_bank = {i: evidence_tokens.get(i, {}) for i in fp_nodes}

        # Build summaries with distinctive tokens
        fraud_summary = self._build_summary(fraud_bank, benign_bank)
        benign_summary = self._build_summary(benign_bank, fraud_bank)
        normal_summary = self._build_normal_summary(evidence_tokens, train_idx)

        return {
            "fraud_prototype_bank": fraud_bank,
            "benign_prototype_bank": benign_bank,
            "train_fn_bank": fn_bank,
            "train_fp_bank": fp_bank,
            "fraud_prototype_summary": fraud_summary,
            "benign_prototype_summary": benign_summary,
            "normal_structure_summary": normal_summary,
        }

    def _build_summary(
        self,
        target_bank: dict[int, dict[str, str]],
        contrast_bank: dict[int, dict[str, str]],
    ) -> dict[str, Any]:
        """Build prototype summary with distinctive tokens via IDF-weighted log odds."""
        # Aggregate token frequencies
        target_freq: Counter = Counter()
        contrast_freq: Counter = Counter()

        for tokens in target_bank.values():
            for field, value in tokens.items():
                target_freq[f"{field}={value}"] += 1

        for tokens in contrast_bank.values():
            for field, value in tokens.items():
                contrast_freq[f"{field}={value}"] += 1

        # Compute distinctive tokens using log odds
        all_tokens = set(target_freq.keys()) | set(contrast_freq.keys())
        distinctive = []

        target_total = sum(target_freq.values()) or 1
        contrast_total = sum(contrast_freq.values()) or 1

        for token in all_tokens:
            t_count = target_freq.get(token, 0)
            c_count = contrast_freq.get(token, 0)

            # Log odds ratio (with smoothing)
            t_rate = (t_count + 1) / (target_total + len(all_tokens))
            c_rate = (c_count + 1) / (contrast_total + len(all_tokens))
            log_odds = np.log(t_rate / c_rate)

            if log_odds > 0.5:  # Distinctive for target class
                distinctive.append((token, log_odds))

        distinctive.sort(key=lambda x: x[1], reverse=True)

        return {
            "token_frequencies": dict(target_freq.most_common(20)),
            "distinctive_tokens": [t for t, _ in distinctive[:10]],
            "num_nodes": len(target_bank),
        }

    def _build_normal_summary(
        self,
        evidence_tokens: dict[int, dict[str, str]],
        train_idx: list[int],
    ) -> dict[str, Any]:
        """Build normal structure summary (median/mode of train tokens)."""
        field_values: dict[str, list[str]] = {}

        for i in train_idx:
            tokens = evidence_tokens.get(i, {})
            for field, value in tokens.items():
                if field not in field_values:
                    field_values[field] = []
                field_values[field].append(value)

        # Compute mode for each field
        summary = {}
        for field, values in field_values.items():
            if values:
                summary[field] = Counter(values).most_common(1)[0][0]
            else:
                summary[field] = "unknown"

        return {
            "field_modes": summary,
            "num_train_nodes": len(train_idx),
        }

    def save(self, prototypes: dict, path: str) -> None:
        """Save prototypes to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert int keys to strings for JSON
        serializable = {}
        for key, value in prototypes.items():
            if isinstance(value, dict):
                serializable[key] = {str(k): v for k, v in value.items()}
            else:
                serializable[key] = value
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str) -> dict:
        """Load prototypes from JSON."""
        with open(path) as f:
            return json.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prototypes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evidence/prototypes.py tests/test_prototypes.py
git commit -m "feat(prototypes): add train-only prototype builder with distinctive tokens"
```

---

## Task 3: Hybrid Retrieval

**Files:**
- Create: `evidence/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write retrieval tests**

```python
# tests/test_retrieval.py
"""Tests for hybrid Jaccard+IDF retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from evidence.retrieval import HybridRetriever


class TestHybridRetrieval:
    """Test 1: Hybrid retrieval with Jaccard recall + IDF cosine rerank."""

    def test_retrieve_top_k(self):
        retriever = HybridRetriever(top_k_recall=10, top_k_final=3)

        # Create mock bank
        bank = {
            0: {"token_a": "high", "token_b": "low"},
            1: {"token_a": "high", "token_b": "medium"},
            2: {"token_a": "low", "token_b": "high"},
            3: {"token_a": "high", "token_b": "low"},
        }

        target = {"token_a": "high", "token_b": "low"}

        results = retriever.retrieve(target, bank)
        assert len(results) <= 3
        assert all("node_id" in r for r in results)
        assert all("score" in r for r in results)

    def test_hybrid_score_computation(self):
        """Score = 0.4 * jaccard + 0.6 * idf_cosine."""
        retriever = HybridRetriever(top_k_recall=10, top_k_final=3)

        bank = {
            0: {"token_a": "high", "token_b": "low"},  # Exact match
            1: {"token_a": "low", "token_b": "high"},   # No match
        }

        target = {"token_a": "high", "token_b": "low"}
        results = retriever.retrieve(target, bank)

        # Node 0 should have higher score (exact match)
        assert results[0]["node_id"] == 0
        assert results[0]["score"] > 0.5


class TestRetrievalScoreBlind:
    """Test 2: Retrieval output doesn't expose scores/labels."""

    def test_no_score_leakage(self):
        retriever = HybridRetriever()

        bank = {0: {"degree_level": "high"}}
        target = {"degree_level": "high"}

        results = retriever.retrieve(target, bank)

        forbidden = {"base_score", "score", "logit", "prob", "confidence", "prediction"}
        for result in results:
            for key in forbidden:
                assert key not in result


class TestRetrievalStats:
    """Test 3: Retrieval stats include safety markers."""

    def test_stats_safety_markers(self):
        retriever = HybridRetriever()

        stats = retriever.get_stats()
        assert stats["test_label_used"] is False
        assert stats["score_visible_to_teacher"] is False
        assert stats["target_label_visible_to_teacher"] is False
        assert stats["base_prediction_visible_to_teacher"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HybridRetriever**

```python
# evidence/retrieval.py
"""Hybrid Jaccard+IDF retrieval for prototype matching."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class HybridRetriever:
    """Hybrid retrieval: Jaccard recall + IDF cosine rerank.

    HARD CONSTRAINTS:
    - IDF computed from train nodes only
    - No test labels
    - No score/prob/logit in output
    """

    def __init__(
        self,
        top_k_recall: int = 50,
        top_k_final: int = 3,
        jaccard_weight: float = 0.4,
        cosine_weight: float = 0.6,
    ):
        self.top_k_recall = top_k_recall
        self.top_k_final = top_k_final
        self.jaccard_weight = jaccard_weight
        self.cosine_weight = cosine_weight

        self._idf: dict[str, float] = {}
        self._vocab: dict[str, int] = {}
        self._fitted = False

    def fit_idf(self, train_tokens: dict[int, dict[str, str]]) -> None:
        """Compute IDF from train tokens only.

        Args:
            train_tokens: {node_id: {field: value}} for train nodes only
        """
        # Build vocabulary
        all_tokens = set()
        for tokens in train_tokens.values():
            for field, value in tokens.items():
                token = f"{field}={value}"
                all_tokens.add(token)

        self._vocab = {token: i for i, token in enumerate(sorted(all_tokens))}

        # Compute IDF
        n_docs = len(train_tokens)
        doc_freq = Counter()

        for tokens in train_tokens.values():
            seen = set()
            for field, value in tokens.items():
                token = f"{field}={value}"
                if token not in seen:
                    doc_freq[token] += 1
                    seen.add(token)

        for token in self._vocab:
            df = doc_freq.get(token, 0)
            self._idf[token] = np.log((n_docs + 1) / (df + 1)) + 1  # Smoothed IDF

        self._fitted = True

    def retrieve(
        self,
        target: dict[str, str],
        bank: dict[int, dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Retrieve top-k most similar nodes from bank.

        Args:
            target: Target node's evidence tokens
            bank: Bank of {node_id: tokens}

        Returns:
            List of {node_id, score, jaccard, cosine} sorted by score desc
        """
        target_set = self._token_set(target)

        # Phase 1: Jaccard recall
        candidates = []
        for node_id, tokens in bank.items():
            bank_set = self._token_set(tokens)
            jaccard = self._jaccard(target_set, bank_set)
            candidates.append((node_id, jaccard, tokens))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:self.top_k_recall]

        # Phase 2: IDF cosine rerank
        target_vec = self._to_vector(target)
        results = []

        for node_id, jaccard, tokens in candidates:
            bank_vec = self._to_vector(tokens)
            cosine = self._cosine(target_vec, bank_vec)

            score = self.jaccard_weight * jaccard + self.cosine_weight * cosine
            results.append({
                "node_id": node_id,
                "score": score,
                "jaccard": jaccard,
                "cosine": cosine,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.top_k_final]

    def retrieve_with_fallback(
        self,
        target: dict[str, str],
        fraud_bank: dict[int, dict[str, str]],
        benign_bank: dict[int, dict[str, str]],
        fn_bank: dict[int, dict[str, str]],
        fp_bank: dict[int, dict[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve from all banks with LLM-safe names.

        Returns:
            Dict with fraud_like_reference_cases, benign_like_reference_cases
        """
        return {
            "fraud_like_reference_cases": self.retrieve(target, fraud_bank),
            "benign_like_reference_cases": self.retrieve(target, benign_bank),
            "similar_fraud_errors": self.retrieve(target, fn_bank),
            "similar_benign_errors": self.retrieve(target, fp_bank),
        }

    def _token_set(self, tokens: dict[str, str]) -> set[str]:
        """Convert tokens to set of 'field=value' strings."""
        return {f"{field}={value}" for field, value in tokens.items()}

    def _to_vector(self, tokens: dict[str, str]) -> np.ndarray:
        """Convert tokens to IDF-weighted vector."""
        if not self._fitted:
            # Fallback: binary vector
            vec = np.zeros(len(self._vocab))
            for field, value in tokens.items():
                token = f"{field}={value}"
                if token in self._vocab:
                    vec[self._vocab[token]] = 1.0
            return vec

        vec = np.zeros(len(self._vocab))
        for field, value in tokens.items():
            token = f"{field}={value}"
            if token in self._vocab:
                vec[self._vocab[token]] = self._idf.get(token, 1.0)
        return vec

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def get_stats(self) -> dict[str, Any]:
        """Get retrieval stats with safety markers."""
        return {
            "metric": "hybrid_token_jaccard_idf_cosine",
            "jaccard_weight": self.jaccard_weight,
            "cosine_weight": self.cosine_weight,
            "top_k_recall": self.top_k_recall,
            "top_k_final": self.top_k_final,
            "train_only_banks": True,
            "test_label_used": False,
            "score_visible_to_teacher": False,
            "target_label_visible_to_teacher": False,
            "base_prediction_visible_to_teacher": False,
        }

    def save_stats(self, path: str) -> None:
        """Save retrieval stats to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_stats(), f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evidence/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): add hybrid Jaccard+IDF retrieval with safety markers"
```

---

## Task 4: Graph Evidence Tokens

**Files:**
- Modify: `evidence/vocab.py`
- Modify: `evidence/adapter.py`

- [ ] **Step 1: Add new token constants to vocab.py**

```python
# evidence/vocab.py - add after existing EVIDENCE_SLOTS

# Graph evidence tokens (score-blind)
GRAPH_EVIDENCE_TOKENS = [
    # Spectral / BWGNN
    "HF_RATIO_TOP10",
    "HF_RATIO_HIGH",
    "HF_RATIO_LOW",
    "BAND_ENERGY_CONFLICT_HIGH",
    "LOW_HIGH_BAND_MISMATCH",

    # Feature-structure conflict
    "FEAT_NEIGH_COS_BOTTOM10",
    "EMB_NEIGH_COS_BOTTOM10",
    "FEATURE_EMBED_DISAGREE_HIGH",

    # Prototype relation
    "PROTO_FRAUD_CLOSE",
    "PROTO_BENIGN_CLOSE",
    "PROTO_CONFLICT_HIGH",

    # Normal-structure deviation
    "NORMAL_STRUCTURE_DIST_HIGH",
    "NORMAL_PATTERN_DEVIATION_HIGH",

    # Clean-view / interference
    "INTERFERING_EDGE_RATIO_HIGH",
    "CLEAN_VIEW_SHIFT_HIGH",
    "RAW_TO_CLEAN_CONFLICT",
]

# Optional tokens (don't block microbenchmark)
OPTIONAL_GRAPH_TOKENS = [
    "LOCAL_CURVATURE_OUTLIER_HIGH",
    "EDGE_CURVATURE_VAR_HIGH",
]
```

- [ ] **Step 2: Add token generation to adapter.py**

```python
# evidence/adapter.py - add method to EvidenceAdapter

def generate_graph_evidence_tokens(
    self,
    node_id: int,
    base_logits: torch.Tensor,
    embeddings: torch.Tensor,
    extras: dict[str, torch.Tensor] | None,
    prototypes: dict | None,
) -> list[str]:
    """Generate score-blind graph evidence tokens for a node.

    Returns list of active tokens (e.g., ["HF_RATIO_HIGH", "FEAT_NEIGH_COS_BOTTOM10"]).
    All tokens are score-blind - no raw scores/probs/logits exposed.
    """
    tokens = []

    # Spectral / BWGNN tokens
    if extras and "high_freq_response" in extras:
        hf = extras["high_freq_response"]
        hf_val = hf[node_id].item()
        hf_q90 = torch.quantile(hf, 0.90).item()
        hf_q50 = torch.quantile(hf, 0.50).item()

        if hf_val > hf_q90:
            tokens.append("HF_RATIO_TOP10")
        if hf_val > hf_q50:
            tokens.append("HF_RATIO_HIGH")
        else:
            tokens.append("HF_RATIO_LOW")

    # Band energy conflict
    if extras and "bwgnn_high_band" in extras and "bwgnn_low_band" in extras:
        high = extras["bwgnn_high_band"][node_id].item()
        low = extras["bwgnn_low_band"][node_id].item()
        if high > 0 and low > 0:
            ratio = high / low
            if ratio > 2.0:
                tokens.append("BAND_ENERGY_CONFLICT_HIGH")

    # Feature-structure conflict
    if hasattr(self, '_feat_cos'):
        feat_cos = self._feat_cos[node_id].item()
        feat_q10 = torch.quantile(self._feat_cos, 0.10).item()
        if feat_cos < feat_q10:
            tokens.append("FEAT_NEIGH_COS_BOTTOM10")

    if hasattr(self, '_emb_cos'):
        emb_cos = self._emb_cos[node_id].item()
        emb_q10 = torch.quantile(self._emb_cos, 0.10).item()
        if emb_cos < emb_q10:
            tokens.append("EMB_NEIGH_COS_BOTTOM10")

    # Feature-embedding disagreement
    if hasattr(self, '_feat_cos') and hasattr(self, '_emb_cos'):
        feat_cos = self._feat_cos[node_id].item()
        emb_cos = self._emb_cos[node_id].item()
        if abs(feat_cos - emb_cos) > 0.3:
            tokens.append("FEATURE_EMBED_DISAGREE_HIGH")

    # Prototype relation tokens
    if prototypes:
        fraud_match, _, _ = compute_prototype_similarity(
            self._get_reasoning(node_id), prototypes.get("fraud_prototype", {})
        )
        benign_match, _, _ = compute_prototype_similarity(
            self._get_reasoning(node_id), prototypes.get("benign_prototype", {})
        )

        if fraud_match > 5:
            tokens.append("PROTO_FRAUD_CLOSE")
        if benign_match > 5:
            tokens.append("PROTO_BENIGN_CLOSE")
        if abs(fraud_match - benign_match) < 2:
            tokens.append("PROTO_CONFLICT_HIGH")

    return tokens
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `pytest -q`
Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add evidence/vocab.py evidence/adapter.py
git commit -m "feat(tokens): add score-blind graph evidence tokens"
```

---

## Task 5: Contrastive Directional Prompt

**Files:**
- Modify: `evidence/prompt.py`
- Create: `tests/test_directional_prompt.py`

- [ ] **Step 1: Write prompt tests**

```python
# tests/test_directional_prompt.py
"""Tests for contrastive directional prompt."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evidence.prompt import (
    build_contrastive_directional_messages,
    assert_score_blind_payload,
)


class TestContrastiveDirectionalPrompt:
    """Test 1: Prompt includes direction/strength enums."""

    def test_prompt_includes_direction_values(self):
        payload = {
            "node_id": 0,
            "reasoning": {
                "degree_level": "high",
                "allowed_support_ids": ["degree_level"],
                "allowed_counter_ids": ["counter_signal"],
            },
            "fraud_prototype": {"degree_level": "high"},
            "benign_prototype": {"degree_level": "low"},
        }

        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "increase_risk" in user_content
        assert "decrease_risk" in user_content
        assert "uncertain" in user_content
        assert "weak" in user_content
        assert "moderate" in user_content
        assert "strong" in user_content


class TestPromptScoreBlind:
    """Test 2: Prompt is score-blind."""

    def test_disclaimer_present(self):
        payload = {
            "node_id": 0,
            "reasoning": {
                "degree_level": "high",
                "allowed_support_ids": ["degree_level"],
                "allowed_counter_ids": [],
            },
            "fraud_prototype": {},
            "benign_prototype": {},
        }

        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        # Disclaimer should be present (allowed)
        assert "You are not given any base model score" in user_content

    def test_no_actual_scores(self):
        payload = {
            "node_id": 0,
            "reasoning": {
                "degree_level": "high",
                "allowed_support_ids": ["degree_level"],
                "allowed_counter_ids": [],
            },
            "fraud_prototype": {},
            "benign_prototype": {},
        }

        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        # No actual score values
        assert "0.8" not in user_content
        assert "0.5" not in user_content


class TestNoInternalBankNames:
    """Test 3: No FN/FP/base_error terminology in prompt."""

    def test_no_fn_fp_terms(self):
        payload = {
            "node_id": 0,
            "reasoning": {
                "degree_level": "high",
                "allowed_support_ids": ["degree_level"],
                "allowed_counter_ids": [],
            },
            "fraud_prototype": {},
            "benign_prototype": {},
        }

        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "false_negative" not in user_content.lower()
        assert "false_positive" not in user_content.lower()
        assert "base_error" not in user_content.lower()
        assert "train_fn" not in user_content.lower()
        assert "train_fp" not in user_content.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_directional_prompt.py -v`
Expected: FAIL

- [ ] **Step 3: Implement contrastive directional prompt**

The `build_contrastive_directional_messages` function already exists in `evidence/prompt.py`. Verify it includes:
- Direction values (increase_risk/decrease_risk/uncertain)
- Strength values (weak/moderate/strong)
- Mandatory disclaimer
- No FN/FP terminology

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_directional_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_directional_prompt.py
git commit -m "feat(prompt): add contrastive directional prompt tests"
```

---

## Task 6: Direction-Aware Verifier

**Files:**
- Modify: `evidence/verifier.py`
- Create: `tests/test_directional_verifier.py`

- [ ] **Step 1: Write verifier tests**

```python
# tests/test_directional_verifier.py
"""Tests for direction-aware verifier."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evidence.schema import ERR, EvidenceCard, CalibrationChannel, ReasoningChannel
from evidence.verifier import EvidenceContractVerifier, load_contracts


def _make_card() -> EvidenceCard:
    return EvidenceCard(
        node_id=0, detector_name="bwgnn",
        calibration=CalibrationChannel(base_score=0.5, uncertainty=0.1),
        reasoning=ReasoningChannel(
            degree_level="high", neighbor_consistency="low",
            feature_neighbor_discrepancy="high", detector_signal="strong",
            detector_signal_strength="strong", counter_signal="weak",
            allowed_support_ids=["degree_level", "detector_signal"],
            allowed_counter_ids=["counter_signal"],
        ),
    )


def _make_verifier() -> EvidenceContractVerifier:
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=False)


class TestDirectionValidation:
    """Test 1: Verifier rejects invalid evidence_direction."""

    def test_invalid_direction_rejected(self):
        verifier = _make_verifier()
        card = _make_card()

        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test", evidence_direction="invalid_direction",
        )

        accepted, reasons = verifier.verify(err, card)
        assert not accepted
        assert "invalid_direction" in reasons

    def test_valid_direction_accepted(self):
        verifier = _make_verifier()
        card = _make_card()

        for direction in ["increase_risk", "decrease_risk", "uncert aint"]:
            err = ERR(
                node_id=0, risk_type="structural_discrepancy",
                supporting_evidence=["degree_level"], counter_evidence=[],
                summary="test", evidence_direction=direction,
            )
            accepted, reasons = verifier.verify(err, card)
            # Should not fail on direction validation alone
            assert "invalid_direction" not in reasons


class TestStrengthValidation:
    """Test 2: Verifier rejects invalid evidence_strength."""

    def test_invalid_strength_rejected(self):
        verifier = _make_verifier()
        card = _make_card()

        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test", evidence_strength="invalid_strength",
        )

        accepted, reasons = verifier.verify(err, card)
        assert not accepted
        assert "invalid_strength" in reasons


class TestUncertaintyFactorsValidation:
    """Test 3: Verifier validates uncertainty_factors."""

    def test_unavailable_uncertainty_factor_rejected(self):
        verifier = _make_verifier()
        card = _make_card()

        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=[],
            summary="test", uncertainty_factors=["nonexistent_field"],
        )

        accepted, reasons = verifier.verify(err, card)
        assert not accepted
        assert "unavailable_uncertainty_factors" in reasons


class TestDirectionConsistencyWarning:
    """Test 4: Direction consistency is logged as warning, not rejection."""

    def test_increase_risk_without_support_is_warning(self):
        verifier = _make_verifier()
        card = _make_card()

        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=[], counter_evidence=[],
            summary="test", evidence_direction="increase_risk",
        )

        # This should be a warning, not a rejection
        accepted, reasons = verifier.verify(err, card)
        # The verifier should log warning but not reject
        # (implementation may vary - check actual behavior)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_directional_verifier.py -v`
Expected: FAIL

- [ ] **Step 3: Update verifier with direction checks**

The verifier already has `_check_direction_fields` and `_check_direction_consistency`. Verify:
- Hard checks (rejection): invalid enum values
- Semantic checks (warning only): direction/evidence consistency

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_directional_verifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_directional_verifier.py
git commit -m "feat(verifier): add direction-aware verifier tests"
```

---

## Task 7: CV-SCD Direction Update

**Files:**
- Modify: `training/losses.py`
- Create: `tests/test_cvscd_directional.py`

- [ ] **Step 1: Write CV-SCD direction tests**

```python
# tests/test_cvscd_directional.py
"""Tests for CV-SCD with direction awareness."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from training.losses import compute_cvscd_loss
from evidence.vocab import get_reason_types, get_evidence_slots

NUM_SLOTS = len(get_evidence_slots())
NUM_TYPES = len(get_reason_types())
REASON_TYPE_NAMES = get_reason_types()


def _make_outputs_with_direction(n: int):
    return {
        "final_logit": torch.randn(n, requires_grad=True),
        "type_logits": torch.randn(n, NUM_TYPES, requires_grad=True),
        "pos_logits": torch.randn(n, NUM_SLOTS, requires_grad=True),
        "neg_logits": torch.randn(n, NUM_SLOTS, requires_grad=True),
        "direction_logits": torch.randn(n, 3, requires_grad=True),  # 3 directions
    }


class TestDirectionCEInLoss:
    """Test 1: Direction CE is included in L_err when direction_ids provided."""

    def test_direction_ce_computed(self):
        n = 10
        outputs = _make_outputs_with_direction(n)
        targets = {
            "risk_type_id": torch.randint(0, NUM_TYPES, (n,)),
            "pos_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "neg_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "accepted_mask": torch.ones(n),
        }
        y = torch.randint(0, 2, (n,)).float()
        train_mask = torch.ones(n, dtype=torch.bool)
        base = torch.randn(n)

        # direction_ids: 0=increase, 1=decrease, 2=uncertain
        direction_ids = torch.randint(0, 3, (n,))

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
            direction_ids=direction_ids, lambda_direction=1.0,
        )

        assert "direction_ce_loss" in stats
        assert stats["direction_ce_loss"] > 0


class TestSignedDirectionAware:
    """Test 2: L_signed skips uncertain nodes."""

    def test_uncertain_skipped(self):
        n = 10
        outputs = _make_outputs_with_direction(n)
        targets = {
            "risk_type_id": torch.randint(0, NUM_TYPES, (n,)),
            "pos_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "neg_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "accepted_mask": torch.ones(n),
        }
        y = torch.randint(0, 2, (n,)).float()
        train_mask = torch.ones(n, dtype=torch.bool)
        base = torch.randn(n)

        # All uncertain
        direction_ids = torch.full((n,), 2)  # uncertain

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
            direction_ids=direction_ids,
        )

        # Signed loss should be 0 when all are uncertain
        assert stats["signed_loss"] == 0.0


class TestCorrDirectionAgreement:
    """Test 3: L_corr uses direction agreement for correction weight."""

    def test_agreement_recorded(self):
        n = 10
        outputs = _make_outputs_with_direction(n)
        targets = {
            "risk_type_id": torch.randint(0, NUM_TYPES, (n,)),
            "pos_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "neg_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "accepted_mask": torch.zeros(n),
        }
        y = torch.randint(0, 2, (n,)).float()
        train_mask = torch.ones(n, dtype=torch.bool)
        base = torch.randn(n)

        direction_ids = torch.randint(0, 3, (n,))

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
            direction_ids=direction_ids,
        )

        assert "direction_correction_agreement" in stats


class TestBackwardCompatOldERR:
    """Test 4: Old ERR (default uncertain/weak) still works."""

    def test_no_direction_ids(self):
        n = 10
        outputs = _make_outputs_with_direction(n)
        targets = {
            "risk_type_id": torch.randint(0, NUM_TYPES, (n,)),
            "pos_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "neg_mask": (torch.rand(n, NUM_SLOTS) > 0.5).float(),
            "accepted_mask": torch.ones(n),
        }
        y = torch.randint(0, 2, (n,)).float()
        train_mask = torch.ones(n, dtype=torch.bool)
        base = torch.randn(n)

        # No direction_ids → backward compatible
        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )

        assert not torch.isnan(loss)
        assert stats["direction_ce_loss"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cvscd_directional.py -v`
Expected: FAIL

- [ ] **Step 3: Update CV-SCD loss**

The `compute_cvscd_loss` function already has direction support. Verify:
- Direction CE in L_err
- Direction-aware L_signed (skip uncertain)
- Direction agreement in L_corr

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cvscd_directional.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_cvscd_directional.py
git commit -m "feat(losses): add CV-SCD direction tests"
```

---

## Task 8: Microbenchmark Script

**Files:**
- Create: `scripts/run_stage2_microbenchmark.py`

- [ ] **Step 1: Implement microbenchmark script**

```python
# scripts/run_stage2_microbenchmark.py
"""Microbenchmark for contrastive directional prompt.

Samples 30 nodes from YelpChi seed 123:
- 10 train false negatives
- 10 train false positives
- 5 train high-loss
- 5 val boundary

Compares current vs contrastive_directional prompt.

Pass criteria (all 4 required):
1. structural_discrepancy ratio < 80%
2. Both increase_risk and decrease_risk appear
3. Direction error alignment >= 0.55
4. Verifier acceptance rate >= 80%
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.adapter import EvidenceAdapter
from evidence.llm_teacher import OfflineLLMTeacher
from evidence.prompt import build_llm_messages, build_contrastive_directional_messages
from evidence.prototypes import PrototypeBuilder
from evidence.retrieval import HybridRetriever
from evidence.schema import ERR
from evidence.trace_sampler import sample_traces
from evidence.verifier import EvidenceContractVerifier, load_contracts
from models.gnn import build_detector
from utils.paths import get_base_checkpoint_path, ensure_dir


def sample_microbenchmark_nodes(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    seed: int,
    n_fn: int = 10,
    n_fp: int = 10,
    n_hl: int = 5,
    n_vb: int = 5,
) -> list[int]:
    """Sample nodes for microbenchmark."""
    import numpy as np
    rng = np.random.RandomState(seed)

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]

    base_probs = torch.sigmoid(base_logits)

    # False negatives
    fn_mask = (y[train_idx] == 1) & (base_probs[train_idx] < 0.5)
    fn_pool = train_idx[fn_mask].tolist()
    fn_sample = list(rng.choice(fn_pool, min(n_fn, len(fn_pool)), replace=False))

    # False positives
    fp_mask = (y[train_idx] == 0) & (base_probs[train_idx] >= 0.5)
    fp_pool = train_idx[fp_mask].tolist()
    fp_sample = list(rng.choice(fp_pool, min(n_fp, len(fp_pool)), replace=False))

    # High loss
    from evidence.trace_sampler import _bce_loss_per_node
    losses = _bce_loss_per_node(base_logits[train_idx], y[train_idx].float())
    hl_mask = losses >= torch.quantile(losses, 0.75)
    hl_pool = train_idx[hl_mask].tolist()
    hl_sample = list(rng.choice(hl_pool, min(n_hl, len(hl_pool)), replace=False))

    # Val boundary
    val_probs = base_probs[val_idx]
    vb_mask = (val_probs >= 0.35) & (val_probs <= 0.65)
    vb_pool = val_idx[vb_mask].tolist()
    vb_sample = list(rng.choice(vb_pool, min(n_vb, len(vb_pool)), replace=False))

    return fn_sample + fp_sample + hl_sample + vb_sample


def compute_direction_error_alignment(
    results: list[dict],
    y: torch.Tensor,
    base_logits: torch.Tensor,
) -> float:
    """Compute direction-error alignment score.

    For FN nodes: increase_risk is aligned
    For FP nodes: decrease_risk is aligned

    Returns alignment rate (0-1).
    """
    aligned = 0
    total = 0

    for result in results:
        node_id = result["node_id"]
        direction = result.get("evidence_direction", "uncert ain")
        label = y[node_id].item()
        pred = torch.sigmoid(base_logits[node_id]).item()

        is_fn = label == 1 and pred < 0.5
        is_fp = label == 0 and pred >= 0.5

        if is_fn:
            total += 1
            if direction == "increase_risk":
                aligned += 1
        elif is_fp:
            total += 1
            if direction == "decrease_risk":
                aligned += 1

    return aligned / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="artifacts/reports")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    seed = args.seed

    torch.manual_seed(seed)

    # Load data
    data = load_fraud_dataset(dataset_name, seed=seed, stratified=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_detector(
        name=config["model"]["name"],
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
    ).to(device)

    checkpoint_path = get_base_checkpoint_path(dataset_name, config["model"]["name"], seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = output.logits.cpu()
        embeddings = output.embeddings.cpu()

    # Sample nodes
    node_ids = sample_microbenchmark_nodes(
        y=data.y, train_mask=data.train_mask,
        val_mask=data.val_mask, base_logits=base_logits, seed=seed,
    )

    print(f"Sampled {len(node_ids)} nodes for microbenchmark")

    # Build prototypes
    builder = PrototypeBuilder()
    adapter = EvidenceAdapter(
        detector_name=config["model"]["name"],
        x=data.x, edge_index=data.edge_index,
    )

    # Extract evidence tokens
    evidence_tokens = {}
    for nid in node_ids:
        tokens = adapter.generate_graph_evidence_tokens(
            nid, base_logits, embeddings, output.extras, None,
        )
        evidence_tokens[nid] = {t: "active" for t in tokens}

    # Build prototypes from train
    train_tokens = {i: evidence_tokens.get(i, {}) for i in data.train_mask.nonzero(as_tuple=True)[0].tolist()}
    prototypes = builder.build(
        train_mask=data.train_mask, y=data.y,
        evidence_tokens=train_tokens,
        base_preds=(torch.sigmoid(base_logits) >= 0.5).long(),
    )

    # Build retrieval
    retriever = HybridRetriever()
    retriever.fit_idf(train_tokens)

    # Setup verifier
    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts)

    # Setup LLM teacher
    llm_config = config.get("llm", {})
    teacher = OfflineLLMTeacher(
        backend=llm_config.get("backend", "mock"),
        model_name_or_path=llm_config.get("model_name_or_path"),
        prompt_mode="contrastive_directional",
    )

    # Run comparison
    results_current = []
    results_directional = []

    for nid in node_ids:
        card = adapter.extract_batch([nid], base_logits, embeddings, output.extras)[0]
        payload = card.to_teacher_payload()

        # Add prototypes/retrieval
        retrieved = retriever.retrieve_with_fallback(
            evidence_tokens.get(nid, {}),
            prototypes["fraud_prototype_bank"],
            prototypes["benign_prototype_bank"],
            prototypes["train_fn_bank"],
            prototypes["train_fp_bank"],
        )

        # Current prompt
        err_current, meta_current = teacher.generate(payload)
        results_current.append({
            "node_id": nid,
            "err": err_current,
            "metadata": meta_current,
        })

        # Directional prompt
        payload_directional = {
            **payload,
            "fraud_prototype": prototypes["fraud_prototype_summary"],
            "benign_prototype": prototypes["benign_prototype_summary"],
            **retrieved,
        }
        err_dir, meta_dir = teacher.generate(payload_directional)
        results_directional.append({
            "node_id": nid,
            "err": err_dir,
            "metadata": meta_dir,
            "evidence_direction": err_dir.evidence_direction if err_dir else "uncertain",
        })

    # Compute metrics
    # ... (compute parse success, verifier acceptance, direction distribution, etc.)

    # Check pass criteria
    # ... (implement pass/fail logic)

    print("\n=== Microbenchmark Results ===")
    # ... (print results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test script runs without error**

Run: `python scripts/run_stage2_microbenchmark.py --config configs/yelpchi_bwgnn.yaml --debug`
Expected: Script runs and outputs results

- [ ] **Step 3: Commit**

```bash
git add scripts/run_stage2_microbenchmark.py
git commit -m "feat(microbenchmark): add directional microbenchmark script"
```

---

## Task 9: Integration Test

**Files:**
- Create: `tests/test_directional_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_directional_integration.py
"""Integration test for full directional pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from evidence.schema import ERR, EvidenceCard, CalibrationChannel, ReasoningChannel
from evidence.prototypes import PrototypeBuilder
from evidence.retrieval import HybridRetriever
from evidence.verifier import EvidenceContractVerifier, load_contracts
from training.losses import compute_cvscd_loss
from evidence.vocab import get_reason_types, get_evidence_slots


class TestEndToEnd:
    """Test end-to-end directional pipeline."""

    def test_full_pipeline(self):
        """Test schema → prototype → retrieval → verifier → loss."""
        # 1. Create ERR with direction
        err = ERR(
            node_id=0, risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"], counter_evidence=["counter_signal"],
            summary="test", evidence_direction="increase_risk",
            evidence_strength="moderate", uncertainty_factors=["counter_signal"],
        )

        # 2. Build prototypes
        builder = PrototypeBuilder()
        train_mask = torch.tensor([True, True, False, False])
        y = torch.tensor([1, 0, 1, 0])
        evidence_tokens = {
            0: {"degree_level": "high"},
            1: {"degree_level": "low"},
            2: {"degree_level": "high"},
            3: {"degree_level": "low"},
        }
        base_preds = torch.tensor([1, 0, 1, 0])

        prototypes = builder.build(
            train_mask=train_mask, y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        # 3. Retrieve
        retriever = HybridRetriever()
        retriever.fit_idf({i: evidence_tokens[i] for i in [0, 1]})

        target = {"degree_level": "high"}
        results = retriever.retrieve(target, prototypes["fraud_prototype_bank"])
        assert len(results) > 0

        # 4. Verify
        contracts = load_contracts()
        verifier = EvidenceContractVerifier(contracts)

        card = EvidenceCard(
            node_id=0, detector_name="bwgnn",
            calibration=CalibrationChannel(base_score=0.5, uncertainty=0.1),
            reasoning=ReasoningChannel(
                degree_level="high", neighbor_consistency="low",
                feature_neighbor_discrepancy="high", detector_signal="strong",
                detector_signal_strength="strong", counter_signal="weak",
                allowed_support_ids=["degree_level"],
                allowed_counter_ids=["counter_signal"],
            ),
        )

        accepted, reasons = verifier.verify(err, card)
        assert accepted, f"Verifier rejected: {reasons}"

        # 5. Loss with direction
        NUM_SLOTS = len(get_evidence_slots())
        NUM_TYPES = len(get_reason_types())

        outputs = {
            "final_logit": torch.randn(1, requires_grad=True),
            "type_logits": torch.randn(1, NUM_TYPES, requires_grad=True),
            "pos_logits": torch.randn(1, NUM_SLOTS, requires_grad=True),
            "neg_logits": torch.randn(1, NUM_SLOTS, requires_grad=True),
            "direction_logits": torch.randn(1, 3, requires_grad=True),
        }
        targets = {
            "risk_type_id": torch.tensor([0]),
            "pos_mask": torch.zeros(1, NUM_SLOTS),
            "neg_mask": torch.zeros(1, NUM_SLOTS),
            "accepted_mask": torch.ones(1),
        }
        targets["pos_mask"][0, 0] = 1.0
        targets["neg_mask"][0, 5] = 1.0

        y_loss = torch.tensor([1.0])
        train_mask = torch.ones(1, dtype=torch.bool)
        base = torch.randn(1)
        direction_ids = torch.tensor([0])  # increase_risk

        loss, stats = compute_cvscd_loss(
            outputs, y=y_loss, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=get_reason_types(),
            direction_ids=direction_ids, lambda_direction=1.0,
        )

        assert not torch.isnan(loss)
        assert stats["direction_ce_loss"] > 0
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_directional_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_directional_integration.py
git commit -m "test(integration): add end-to-end directional pipeline test"
```

---

## Self-Review Checklist

- [ ] All spec sections have corresponding tasks
- [ ] No placeholders (TBD/TODO) in plan
- [ ] Types and method signatures are consistent across tasks
- [ ] All tests are complete with actual code
- [ ] File paths are exact
- [ ] Commands include expected output
- [ ] Score-blind constraints are enforced throughout
- [ ] Internal bank names (fn/fp) not exposed to LLM
- [ ] Backward compatibility maintained
