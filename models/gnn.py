from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except ImportError:
    GCNConv = None
    GATConv = None
    SAGEConv = None

from models.base import BaseModelOutput
from models.bwgnn import BWGNNDetector
from models.priorfgnn import PriorFGNNDetector


class GCNDetector(nn.Module):
    detector_name: str = "gcn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, return_output: bool = False,
    ) -> Tensor | BaseModelOutput:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        logits = self.head(embedding).squeeze(-1)

        if return_output:
            return BaseModelOutput(logits=logits, embeddings=embedding, extras={})
        return logits


class SAGEDetector(nn.Module):
    detector_name: str = "sage"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, return_output: bool = False,
    ) -> Tensor | BaseModelOutput:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        logits = self.head(embedding).squeeze(-1)

        if return_output:
            return BaseModelOutput(logits=logits, embeddings=embedding, extras={})
        return logits


class GATDetector(nn.Module):
    detector_name: str = "gat"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is required")

        self.dropout = dropout
        self.attention_heads = attention_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=attention_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * attention_heads, hidden_channels, heads=1, dropout=dropout))
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, return_output: bool = False,
    ) -> Tensor | BaseModelOutput:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        logits = self.head(embedding).squeeze(-1)

        if return_output:
            return BaseModelOutput(logits=logits, embeddings=embedding, extras={})
        return logits


DETECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "gcn": GCNDetector,
    "sage": SAGEDetector,
    "gat": GATDetector,
    "bwgnn": BWGNNDetector,
    "priorfgnn": PriorFGNNDetector,
}


def build_detector(name: str, in_channels: int, **kwargs) -> nn.Module:
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTOR_REGISTRY.keys())}")
    return DETECTOR_REGISTRY[name](in_channels=in_channels, **kwargs)
