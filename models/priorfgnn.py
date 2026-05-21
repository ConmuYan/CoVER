"""PriorF-GNN integration for RAER-FD.

Ports the ASDA-enhanced LG-HGCL v2 model from
/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-GNN/lghgcl/models/
into RAER-FD's model registry. Code is a faithful copy with only the
import paths and BaseModelOutput wrapping adjusted; semantics match the
PriorF-GNN reference exactly so the seed-42 result reproduces.

Architecture (yelpchi_full.yaml config):
    MLP branch:  X (33-dim, with HSD) -> MLP(128) -> z_mlp(128)
    GNN branch:  ASDA(X, hsd) -> RGCN(128, R=3) -> z_gnn(128)
    Fusion:      [z_mlp || z_gnn] -> Linear -> sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import softmax as pyg_softmax

from models.base import BaseModelOutput


class ASDALayer(nn.Module):
    """Adaptive Structural Discrepancy Attention (ASDA).

    Edge-level discrepancy coefficient + node-level frequency switch +
    dual-channel low/high-pass fusion driven by HSD.

    Faithful port of PriorF-GNN/lghgcl/models/asda_layer.py. Adds
    optional edge chunking (``edge_chunk_size``) so that the (E, D)
    intermediate tensors are materialised in slices rather than all at
    once — required for YelpChi where ``E ≈ 8M`` would otherwise OOM.
    """

    def __init__(
        self,
        in_dim: int,
        tau: float = 0.1,
        edge_hidden: int = 32,
        node_hidden: int = 16,
        edge_chunk_size: int = 500_000,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.edge_chunk_size = edge_chunk_size

        edge_in_dim = in_dim + 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )

        self.node_switch = nn.Sequential(
            nn.Linear(1, node_hidden),
            nn.ReLU(),
            nn.Linear(node_hidden, 1),
            nn.Sigmoid(),
        )

    def _chunked_edge_score(self, x: Tensor, hsd_norm: Tensor, row: Tensor, col: Tensor) -> Tensor:
        """Compute edge MLP score in chunks to bound peak memory."""
        E = row.size(0)
        chunk = self.edge_chunk_size if self.edge_chunk_size > 0 else E
        scores = []
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            r = row[s:e]; c = col[s:e]
            edge_in = torch.cat(
                [
                    hsd_norm[r].unsqueeze(1),
                    hsd_norm[c].unsqueeze(1),
                    torch.abs(x[r] - x[c]),
                ],
                dim=1,
            )
            scores.append(self.edge_mlp(edge_in).squeeze(-1))
        return torch.cat(scores, dim=0)

    def _chunked_high_pass(self, x: Tensor, alpha_uv: Tensor, row: Tensor, col: Tensor, N: int) -> Tensor:
        """Compute the attention-weighted residual aggregation in chunks."""
        E = row.size(0)
        chunk = self.edge_chunk_size if self.edge_chunk_size > 0 else E
        high_pass = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            r = row[s:e]; c = col[s:e]
            msg = alpha_uv[s:e] * (x[c] - x[r])
            high_pass = high_pass.index_add(0, c, msg)
        return high_pass

    def _chunked_low_pass(self, x: Tensor, edge_w: Tensor, row: Tensor, col: Tensor, N: int) -> Tensor:
        E = row.size(0)
        chunk = self.edge_chunk_size if self.edge_chunk_size > 0 else E
        low_pass = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            r = row[s:e]; c = col[s:e]
            msg = edge_w[s:e].unsqueeze(1) * x[r]
            low_pass = low_pass.index_add(0, c, msg)
        return low_pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        hsd: Tensor,
    ) -> Tensor:
        N = x.size(0)
        row, col = edge_index

        hsd_norm = (hsd - hsd.mean()) / (hsd.std() + 1e-8)

        e_uv = self._chunked_edge_score(x, hsd_norm, row, col)
        alpha_uv = pyg_softmax(e_uv / self.tau, col, dim=0, num_nodes=N).unsqueeze(-1)

        alpha_bar = self.node_switch(hsd_norm.unsqueeze(1))

        src_deg = torch.zeros(N, device=x.device)
        dst_deg = torch.zeros(N, device=x.device)
        src_deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        dst_deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        src_deg = src_deg.clamp(min=1).sqrt()
        dst_deg = dst_deg.clamp(min=1).sqrt()
        edge_w = 1.0 / (src_deg[row] * dst_deg[col])

        low_pass = self._chunked_low_pass(x, edge_w, row, col, N)
        high_pass = self._chunked_high_pass(x, alpha_uv, row, col, N)

        h_new = (1 - alpha_bar) * low_pass + alpha_bar * high_pass
        return h_new


class PriorFGNNDetector(nn.Module):
    """ASDA-enhanced LG-HGCL detector (PriorF-GNN v2).

    Faithful port of PriorF-GNN/lghgcl/models/lg_hgcl_v2.py:LGHGCLNetV2.

    Forward signature: model(x, edge_index, edge_type=None, hsd=None,
    return_output=False). The RAER-FD training loop will pass edge_type
    and hsd via keyword args after switching to the multi-relation pipeline.
    """

    detector_name: str = "priorfgnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_relations: int = 3,
        dropout: float = 0.3,
        use_asda: bool = True,
        use_mlp_branch: bool = True,
        use_gnn_branch: bool = True,
        asda_tau: float = 0.1,
        proj_dim: int = 64,
        # RAER-FD compatibility: ignored but accepted to match build_detector
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        mlp_hidden = hidden_channels
        gnn_hidden = hidden_channels
        out_hidden = hidden_channels

        self.use_mlp = use_mlp_branch
        self.use_gnn = use_gnn_branch
        self.use_asda = use_asda
        self.dropout = dropout
        self.num_relations = num_relations

        if not use_mlp_branch and not use_gnn_branch:
            raise ValueError("At least one branch must be enabled")

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, out_hidden),
                nn.BatchNorm1d(out_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if self.use_gnn:
            self.asda = ASDALayer(in_dim=in_channels, tau=asda_tau) if use_asda else None
            self.rgcn1 = RGCNConv(in_channels, gnn_hidden, num_relations=num_relations)
            self.bn1 = nn.BatchNorm1d(gnn_hidden)
            self.rgcn2 = RGCNConv(gnn_hidden, out_hidden, num_relations=num_relations)
            self.bn2 = nn.BatchNorm1d(out_hidden)

        head_dim = 0
        if self.use_mlp:
            head_dim += out_hidden
        if self.use_gnn:
            head_dim += out_hidden
        self.head = nn.Linear(head_dim, 1)

        self.proj_head = nn.Sequential(
            nn.Linear(head_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.z_proj: Tensor | None = None
        self.z_fused: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor | None = None,
        hsd: Tensor | None = None,
        return_output: bool = False,
    ) -> Tensor | BaseModelOutput:
        z_mlp = None
        z_gnn = None

        if self.use_mlp:
            z_mlp = self.mlp(x)

        if self.use_gnn:
            if edge_type is None:
                raise ValueError("PriorFGNNDetector.forward requires edge_type for RGCN")
            if self.asda is not None:
                if hsd is None:
                    raise ValueError("ASDA branch requires hsd tensor")
                h_input = self.asda(x, edge_index, hsd)
            else:
                h_input = x
            z = self.rgcn1(h_input, edge_index, edge_type)
            z = self.bn1(z)
            z = F.relu(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = self.rgcn2(z, edge_index, edge_type)
            z_gnn = self.bn2(z)
            z_gnn = F.relu(z_gnn)
            z_gnn = F.dropout(z_gnn, p=self.dropout, training=self.training)

        parts = [p for p in (z_mlp, z_gnn) if p is not None]
        z = torch.cat(parts, dim=-1)
        self.z_fused = z
        z_proj = self.proj_head(z)
        self.z_proj = z_proj
        logits = self.head(z).squeeze(-1)

        if return_output:
            extras = {"z_proj": z_proj.detach()}
            return BaseModelOutput(logits=logits, embeddings=z, extras=extras)
        return logits
