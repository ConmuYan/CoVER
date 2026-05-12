from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.utils import degree
except ImportError:
    degree = None

from models.base import BaseModelOutput


def calculate_theta(d: int = 2) -> list[list[float]]:
    if d == 2:
        return [
            [1.0, -1.0, 0.25],
            [0.0, 1.0, -0.5],
            [0.0, 0.0, 0.25],
        ]
    elif d == 3:
        return [
            [1.0, -1.5, 0.75, -0.125],
            [0.0, 1.5, -1.5, 0.375],
            [0.0, 0.0, 0.75, -0.375],
            [0.0, 0.0, 0.0, 0.125],
        ]
    else:
        raise ValueError(f"Unsupported order d={d}. Use d=2 or d=3.")


class PolyConv(nn.Module):
    def __init__(self, theta: list[float]):
        super().__init__()
        self.theta = theta
        self.k = len(theta)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        num_nodes = x.size(0)

        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = self.theta[0] * x
        x_prop = x

        for i in range(1, self.k):
            out = torch.zeros_like(x_prop)
            weighted = norm.unsqueeze(-1) * x_prop[col]
            out.index_add_(0, row, weighted)
            x_prop = x_prop - out
            h = h + self.theta[i] * x_prop

        return h


class BWGNNDetector(nn.Module):
    detector_name: str = "bwgnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_bands: int = 3,
    ) -> None:
        super().__init__()
        if degree is None:
            raise ImportError("torch_geometric is required")

        self.dropout = dropout
        self.num_bands = num_bands

        thetas = calculate_theta(d=num_bands - 1)
        self.convs = nn.ModuleList([PolyConv(theta) for theta in thetas])

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )

        self.band_proj = nn.Linear(hidden_channels * len(self.convs), hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, return_output: bool = False,
    ) -> Tensor | BaseModelOutput:
        h = self.input_proj(x)

        band_outputs = []
        for conv in self.convs:
            band_outputs.append(conv(h, edge_index))

        h_cat = torch.cat(band_outputs, dim=-1)
        embedding = F.relu(self.band_proj(h_cat))
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)

        logits = self.head(embedding).squeeze(-1)

        if return_output:
            high_freq_response = band_outputs[-1].norm(dim=-1)
            extras = {"high_freq_response": high_freq_response}
            return BaseModelOutput(logits=logits, embeddings=embedding, extras=extras)

        return logits


DETECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "gcn": None,
    "sage": None,
    "gat": None,
    "bwgnn": BWGNNDetector,
}
