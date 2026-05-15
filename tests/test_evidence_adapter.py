import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.adapter import EvidenceAdapter
from evidence.prompt import build_teacher_payload, assert_score_blind_payload
from evidence.schema import ReasoningChannel


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


def test_adapter_matches_graph_token_distinctive_prototypes():
    x = torch.randn(12, 8)
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3, 4, 5, 6],
        [1, 2, 2, 3, 4, 5, 6, 7],
    ])
    adapter = EvidenceAdapter("bwgnn", x, edge_index)
    prototypes = {
        "fraud_prototype": {},
        "benign_prototype": {},
        "fraud_prototype_summary": {
            "distinctive_tokens": [
                {"field": "FEATURE_EMBED_DISAGREE_HIGH", "value": "active", "log_odds": 2.0}
            ]
        },
        "benign_prototype_summary": {
            "distinctive_tokens": [
                {"field": "FEATURE_EMBED_AGREE_HIGH", "value": "active", "log_odds": 2.0}
            ]
        },
    }
    reasoning = ReasoningChannel(
        degree_level="high",
        neighbor_consistency="low",
        feature_neighbor_discrepancy="high",
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
        counter_signal="benign_neighbor_signal_low",
    )
    fields = adapter._compute_prototype_relative_fields(
        reasoning,
        prototypes,
        graph_tokens=["FEATURE_EMBED_DISAGREE_HIGH"],
    )

    assert fields["closer_to_fraud_prototype"] == "high"
    assert fields["closer_to_benign_prototype"] == "low"
    assert fields["fraud_prototype_matching_fields"] == ["FEATURE_EMBED_DISAGREE_HIGH"]
