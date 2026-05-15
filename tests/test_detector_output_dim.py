"""Sanity tests for detector output contract.

All detectors registered in models.gnn.DETECTOR_REGISTRY must:
1. Return a Tensor when return_output=False.
2. Return a BaseModelOutput when return_output=True.
3. The embedding dim must equal hidden_dim (so phase2 reasoner base_z_dim
   inferred from z.shape[1] matches hidden_dim).
4. Logits must be 1D after BaseModelOutput.__post_init__.
"""

from __future__ import annotations

import pytest
import torch

from models.base import BaseModelOutput
from models.gnn import DETECTOR_REGISTRY, build_detector


@pytest.fixture
def tiny_graph():
    num_nodes, num_features = 32, 16
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor(
        [[i for i in range(num_nodes - 1)] + [i + 1 for i in range(num_nodes - 1)],
         [i + 1 for i in range(num_nodes - 1)] + [i for i in range(num_nodes - 1)]],
        dtype=torch.long,
    )
    return x, edge_index, num_nodes


@pytest.mark.parametrize("name", ["gcn", "sage", "gat", "bwgnn"])
def test_detector_logits_shape(name, tiny_graph):
    x, edge_index, num_nodes = tiny_graph
    model = build_detector(name=name, in_channels=x.shape[1], hidden_channels=64)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    assert logits.shape == (num_nodes,), f"{name} logits shape {logits.shape} != ({num_nodes},)"


@pytest.mark.parametrize("name", ["gcn", "sage", "gat", "bwgnn"])
def test_detector_returns_base_model_output(name, tiny_graph):
    x, edge_index, _ = tiny_graph
    model = build_detector(name=name, in_channels=x.shape[1], hidden_channels=64)
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, return_output=True)
    assert isinstance(output, BaseModelOutput)
    assert output.logits.ndim == 1
    assert output.embeddings.ndim == 2
    assert isinstance(output.extras, dict)


@pytest.mark.parametrize("name,expected_dim", [
    ("gcn", 64),
    ("sage", 64),
    ("gat", 64),
    ("bwgnn", 64),
])
def test_detector_embedding_dim_matches_hidden(name, expected_dim, tiny_graph):
    """Phase2 CoVERRelReasoner uses base_z_dim = z.shape[1]; this must equal hidden_dim."""
    x, edge_index, num_nodes = tiny_graph
    model = build_detector(name=name, in_channels=x.shape[1], hidden_channels=expected_dim)
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, return_output=True)
    assert output.embeddings.shape == (num_nodes, expected_dim), (
        f"{name} embedding shape {output.embeddings.shape} != ({num_nodes}, {expected_dim})"
    )


def test_bwgnn_provides_high_freq_response(tiny_graph):
    x, edge_index, num_nodes = tiny_graph
    model = build_detector(name="bwgnn", in_channels=x.shape[1], hidden_channels=64)
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, return_output=True)
    assert "high_freq_response" in output.extras
    assert output.extras["high_freq_response"].shape == (num_nodes,)


@pytest.mark.parametrize("name", ["gcn", "sage", "gat"])
def test_non_bwgnn_extras_empty(name, tiny_graph):
    """GCN/SAGE/GAT extras must be an empty dict; adapter guards on its presence."""
    x, edge_index, _ = tiny_graph
    model = build_detector(name=name, in_channels=x.shape[1], hidden_channels=64)
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, return_output=True)
    assert output.extras == {}, f"{name} should have empty extras, got {output.extras}"


def test_registry_has_four_models():
    assert set(DETECTOR_REGISTRY.keys()) == {"gcn", "sage", "gat", "bwgnn"}
