import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.adapter import EvidenceAdapter
from evidence.prompt import build_teacher_payload, assert_score_blind_payload


def _make_extras_with_high_freq(num_nodes=20):
    return {"high_freq_response": torch.randn(num_nodes).abs()}


def test_adapter_with_high_freq_extras():
    x = torch.randn(20, 8)
    edge_index = torch.randint(0, 20, (2, 50))
    adapter = EvidenceAdapter("bwgnn", x, edge_index)

    extras = _make_extras_with_high_freq()
    base_logits = torch.randn(20)
    embeddings = torch.randn(20, 16)

    cards = adapter.extract([0, 1, 2], base_logits, embeddings, extras=extras)

    assert len(cards) == 3
    for card in cards:
        assert "high_frequency_response" in card.reasoning.detector_signal


def test_adapter_without_extras_fallback():
    x = torch.randn(20, 8)
    edge_index = torch.randint(0, 20, (2, 50))
    adapter = EvidenceAdapter("gcn", x, edge_index)

    base_logits = torch.randn(20)
    embeddings = torch.randn(20, 16)

    cards = adapter.extract([0, 1, 2], base_logits, embeddings)

    assert len(cards) == 3
    for card in cards:
        assert "high_frequency_response" not in card.reasoning.detector_signal


def test_adapter_bwgnn_payload_score_blind():
    x = torch.randn(20, 8)
    edge_index = torch.randint(0, 20, (2, 50))
    adapter = EvidenceAdapter("bwgnn", x, edge_index)

    extras = _make_extras_with_high_freq()
    base_logits = torch.randn(20)
    embeddings = torch.randn(20, 16)

    cards = adapter.extract([0, 1], base_logits, embeddings, extras=extras)

    for card in cards:
        payload = build_teacher_payload(card)
        assert_score_blind_payload(payload)
