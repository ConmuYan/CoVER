import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.data import Data
from models.gnn import GCNDetector, SAGEDetector, GATDetector, build_detector


def make_dummy_graph(num_nodes=50, num_features=16, num_edges=100):
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:30] = True
    val_mask[30:40] = True
    test_mask[40:] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def test_gcn_shapes():
    graph = make_dummy_graph()
    model = GCNDetector(in_channels=16, hidden_channels=32, num_layers=2)
    logit, emb = model(graph.x, graph.edge_index)

    assert logit.shape == (50,)
    assert emb.shape == (50, 32)


def test_sage_shapes():
    graph = make_dummy_graph()
    model = SAGEDetector(in_channels=16, hidden_channels=32, num_layers=2)
    logit, emb = model(graph.x, graph.edge_index)

    assert logit.shape == (50,)
    assert emb.shape == (50, 32)


def test_gat_shapes():
    graph = make_dummy_graph()
    model = GATDetector(in_channels=16, hidden_channels=32, num_layers=2, attention_heads=4)
    logit, emb = model(graph.x, graph.edge_index)

    assert logit.shape == (50,)
    assert emb.shape == (50, 32)


def test_build_detector():
    model = build_detector("gcn", in_channels=16, hidden_channels=32)
    assert isinstance(model, GCNDetector)

    model = build_detector("sage", in_channels=16, hidden_channels=32)
    assert isinstance(model, SAGEDetector)

    model = build_detector("gat", in_channels=16, hidden_channels=32)
    assert isinstance(model, GATDetector)


def test_build_detector_invalid():
    with pytest.raises(ValueError):
        build_detector("invalid", in_channels=16)


def test_mask_selection():
    graph = make_dummy_graph()
    model = GCNDetector(in_channels=16, hidden_channels=32)

    logit_all, emb_all = model(graph.x, graph.edge_index)
    assert logit_all.shape == (50,)

    train_logit = logit_all[graph.train_mask]
    assert train_logit.shape == (30,)

    test_logit = logit_all[graph.test_mask]
    assert test_logit.shape == (10,)
