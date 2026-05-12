import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.adapter import EvidenceAdapter
from evidence.prompt import build_teacher_payload, assert_score_blind_payload


def _make_dummy_graph(num_nodes=50, num_features=16, num_edges=100):
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return x, edge_index


def test_adapter_output_count():
    x, edge_index = _make_dummy_graph()
    adapter = EvidenceAdapter("gcn", x, edge_index)

    node_ids = [0, 1, 2, 3, 4]
    base_logits = torch.randn(50)
    embeddings = torch.randn(50, 32)

    cards = adapter.extract(node_ids, base_logits, embeddings)
    assert len(cards) == 5


def test_adapter_card_has_channels():
    x, edge_index = _make_dummy_graph()
    adapter = EvidenceAdapter("gcn", x, edge_index)

    cards = adapter.extract([0], torch.randn(50), torch.randn(50, 32))
    card = cards[0]

    assert card.calibration is not None
    assert card.reasoning is not None
    assert card.node_id == 0
    assert card.detector_name == "gcn"


def test_adapter_payload_score_blind():
    x, edge_index = _make_dummy_graph()
    adapter = EvidenceAdapter("gcn", x, edge_index)

    cards = adapter.extract([0, 1], torch.randn(50), torch.randn(50, 32))

    for card in cards:
        payload = build_teacher_payload(card)
        assert_score_blind_payload(payload)


def test_adapter_allowed_ids_nonempty():
    x = torch.randn(10, 8)
    edge_index = torch.tensor([
        [0, 0, 0, 1, 2],
        [1, 2, 3, 4, 5],
    ])
    adapter = EvidenceAdapter("gcn", x, edge_index)

    cards = adapter.extract([0], torch.randn(10), torch.randn(10, 16))
    card = cards[0]

    assert len(card.reasoning.allowed_support_ids) > 0


def test_adapter_degree_levels():
    x = torch.randn(10, 8)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 0, 0, 1, 1, 2, 3],
        [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
    ])
    adapter = EvidenceAdapter("gcn", x, edge_index)

    cards = adapter.extract([0, 1, 9], torch.randn(10), torch.randn(10, 16))

    assert cards[0].reasoning.degree_level == "medium"
    assert cards[1].reasoning.degree_level == "low"
    assert cards[2].reasoning.degree_level == "low"
