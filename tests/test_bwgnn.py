import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bwgnn import BWGNNDetector
from models.base import BaseModelOutput


def _make_input():
    x = torch.randn(20, 16)
    edge_index = torch.randint(0, 20, (2, 50))
    return x, edge_index


def test_bwgnn_default_returns_tensor():
    x, edge_index = _make_input()
    model = BWGNNDetector(in_channels=16, hidden_channels=32)
    result = model(x, edge_index)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (20,)


def test_bwgnn_return_output():
    x, edge_index = _make_input()
    model = BWGNNDetector(in_channels=16, hidden_channels=32)
    output = model(x, edge_index, return_output=True)
    assert isinstance(output, BaseModelOutput)
    assert output.logits.shape == (20,)
    assert output.embeddings.shape == (20, 32)
    assert isinstance(output.extras, dict)


def test_bwgnn_extras_has_high_freq_response():
    x, edge_index = _make_input()
    model = BWGNNDetector(in_channels=16, hidden_channels=32)
    output = model(x, edge_index, return_output=True)
    assert "high_freq_response" in output.extras
    assert output.extras["high_freq_response"].shape == (20,)


def test_bwgnn_logits_consistency():
    x, edge_index = _make_input()
    model = BWGNNDetector(in_channels=16, hidden_channels=32)
    model.eval()

    logits = model(x, edge_index)
    output = model(x, edge_index, return_output=True)

    assert torch.allclose(logits, output.logits)


def test_bwgnn_build_detector():
    from models.gnn import build_detector

    model = build_detector("bwgnn", in_channels=16, hidden_channels=32)
    assert isinstance(model, BWGNNDetector)
