import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import BaseModelOutput
from models.gnn import GCNDetector, SAGEDetector, GATDetector


def _make_input():
    x = torch.randn(20, 16)
    edge_index = torch.randint(0, 20, (2, 50))
    return x, edge_index


def test_gcn_default_returns_tensor():
    x, edge_index = _make_input()
    model = GCNDetector(in_channels=16, hidden_channels=32)
    result = model(x, edge_index)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (20,)


def test_sage_default_returns_tensor():
    x, edge_index = _make_input()
    model = SAGEDetector(in_channels=16, hidden_channels=32)
    result = model(x, edge_index)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (20,)


def test_gat_default_returns_tensor():
    x, edge_index = _make_input()
    model = GATDetector(in_channels=16, hidden_channels=32, attention_heads=4)
    result = model(x, edge_index)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (20,)


def test_gcn_return_output():
    x, edge_index = _make_input()
    model = GCNDetector(in_channels=16, hidden_channels=32)
    output = model(x, edge_index, return_output=True)
    assert isinstance(output, BaseModelOutput)
    assert output.logits.shape == (20,)
    assert output.embeddings.shape == (20, 32)
    assert isinstance(output.extras, dict)


def test_sage_return_output():
    x, edge_index = _make_input()
    model = SAGEDetector(in_channels=16, hidden_channels=32)
    output = model(x, edge_index, return_output=True)
    assert isinstance(output, BaseModelOutput)
    assert output.logits.shape == (20,)
    assert output.embeddings.shape == (20, 32)


def test_gat_return_output():
    x, edge_index = _make_input()
    model = GATDetector(in_channels=16, hidden_channels=32, attention_heads=4)
    output = model(x, edge_index, return_output=True)
    assert isinstance(output, BaseModelOutput)
    assert output.logits.shape == (20,)
    assert output.embeddings.shape == (20, 32)


def test_logits_consistency():
    x, edge_index = _make_input()
    model = GCNDetector(in_channels=16, hidden_channels=32)
    model.eval()

    logits = model(x, edge_index)
    output = model(x, edge_index, return_output=True)

    assert torch.allclose(logits, output.logits)


def test_extras_empty_by_default():
    x, edge_index = _make_input()
    model = GCNDetector(in_channels=16, hidden_channels=32)
    output = model(x, edge_index, return_output=True)
    assert output.extras == {}
