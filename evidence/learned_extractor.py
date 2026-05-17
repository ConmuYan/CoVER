"""Learned per-relation evidence extractor for Idea 2B.

Drop-in replacement for the 9-dim hand-crafted A/B/C evidence in
``evidence/relation_features.py``. Preserves all four CoVER-REL safety contracts:

1. **Score-blind**: extractor inputs are raw features ``X``, per-relation
   adjacency matrices ``A_r``, a train mask, and train labels. The extractor
   NEVER sees base logits or base embeddings.
2. **Train-only prototype**: class-anchored prototype features are computed
   inside ``with torch.no_grad()`` from ``train_mask & (train_labels == c)``
   only. No val/test labels touched. No gradient flows back through labels.
3. **Frozen base**: this module is independent of the base detector; no
   parameter sharing, no logit/embedding access.
4. **Bounded intervention**: downstream ``CoVERRelReasoner`` still applies
   ``δ_max · tanh(u)`` so the framework-level intervention bound is unchanged.

Output shape: ``(N, R * out_dim_per_rel)`` — drop-in compatible with the
existing ``load_relation_features_for_phase2()`` interface. Default
``out_dim_per_rel=9`` matches ``evidence.relation_features.RELATION_STAT_NAMES``.

Training: gradient flows from ``L_cls`` through the reasoner's
``relation_features`` argument back into this extractor's parameters. The
hand-crafted prototype features that the trainer used to load from disk are
now produced online per epoch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coalesced_indices_values(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not adj.is_sparse:
        raise ValueError(f"adj must be a sparse tensor, got dense {tuple(adj.shape)}")
    a = adj.coalesce()
    return a.indices(), a.values(), int(a.size(0))


def symmetric_normalize_sparse_adj(adj: torch.Tensor) -> torch.Tensor:
    """Compute D^{-1/2} A D^{-1/2} for a sparse COO tensor; returns sparse COO."""
    indices, values, n = _coalesced_indices_values(adj)
    deg = torch.zeros(n, device=adj.device, dtype=values.dtype)
    deg.scatter_add_(0, indices[0], values)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    norm_values = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, norm_values, (n, n)).coalesce()


def row_normalize_sparse_adj(adj: torch.Tensor) -> torch.Tensor:
    """Compute D^{-1} A (row-stochastic mean aggregator); returns sparse COO."""
    indices, values, n = _coalesced_indices_values(adj)
    deg = torch.zeros(n, device=adj.device, dtype=values.dtype)
    deg.scatter_add_(0, indices[0], values)
    row_vals = values / deg[indices[0]].clamp(min=1.0)
    return torch.sparse_coo_tensor(indices, row_vals, (n, n)).coalesce()


class RelationGCNEncoder(nn.Module):
    """Small 2-layer symmetric-normalized GCN for a single relation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj_sym_norm: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(adj_sym_norm, self.lin1(x))
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = torch.sparse.mm(adj_sym_norm, self.lin2(h))
        return h


class LearnedRelationEvidenceExtractor(nn.Module):
    """Score-blind, train-only-prototype learned per-relation evidence extractor.

    For each relation ``r`` and node ``i``, produces an evidence vector
    ``E_{i,r} ∈ R^{out_dim_per_rel}``. The vector composition (per design):

    - **Structural** signal: 2-layer per-relation GCN embedding (analogue of
      hand-crafted A: ``log_degree_norm``, ``degree_top10``, ``degree_low``).
    - **Feature-neighbor** signal: raw ``x_i`` and mean of relation-neighbor
      features (analogue of B: ``feature_l2_deviation_z``, ``neighbor_cosine``,
      ``zscore_high_fraction``).
    - **Prototype-relative** signal: z-scored Euclidean distances to
      train-only fraud / benign prototypes plus their margin (analogue of C:
      ``fraud_proto_dist_z``, ``benign_proto_dist_z``,
      ``fraud_minus_benign_margin_z``).

    A per-relation MLP head learns the combination, replacing the hand-crafted
    quantile-based stats used in ``evidence/relation_features.py``.
    """

    def __init__(
        self,
        x_dim: int,
        num_relations: int,
        hidden_dim: int = 32,
        encoder_out_dim: int = 16,
        out_dim_per_rel: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_relations <= 0:
            raise ValueError("num_relations must be > 0")
        self.x_dim = int(x_dim)
        self.num_relations = int(num_relations)
        self.hidden_dim = int(hidden_dim)
        self.encoder_out_dim = int(encoder_out_dim)
        self.out_dim_per_rel = int(out_dim_per_rel)
        self.dropout = float(dropout)

        self.relation_encoders = nn.ModuleList([
            RelationGCNEncoder(
                in_dim=x_dim, hidden_dim=hidden_dim,
                out_dim=encoder_out_dim, dropout=dropout,
            )
            for _ in range(num_relations)
        ])

        # Per-relation pooling head input dim:
        #   x (x_dim) + neighbor_mean (x_dim) + gcn_emb (encoder_out_dim)
        #   + fraud_dist_z (1) + benign_dist_z (1) + margin_z (1)
        pool_in_dim = 2 * x_dim + encoder_out_dim + 3
        head_modules: list[nn.Module] = []
        for _ in range(num_relations):
            layers: list[nn.Module] = [
                nn.Linear(pool_in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim_per_rel))
            head_modules.append(nn.Sequential(*layers))
        self.relation_heads = nn.ModuleList(head_modules)

        # Cached fixed-across-training tensors (set by ``prepare()``):
        self._sym_norm_adjs: list[torch.Tensor] = []
        self._neighbor_means: list[torch.Tensor] = []
        self._prepared = False

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, relation_adjs: list[torch.Tensor]) -> None:
        """Cache normalized adj and neighbor-mean per relation.

        Called once after the extractor is moved to its device. ``x`` and each
        adjacency matrix must already be on the same device as the extractor.
        """
        if len(relation_adjs) != self.num_relations:
            raise ValueError(
                f"expected {self.num_relations} relation adjacency matrices, "
                f"got {len(relation_adjs)}"
            )
        if x.shape[1] != self.x_dim:
            raise ValueError(f"x feature dim {x.shape[1]} != extractor x_dim {self.x_dim}")
        sym_norm_adjs: list[torch.Tensor] = []
        neighbor_means: list[torch.Tensor] = []
        for adj in relation_adjs:
            adj_sym = symmetric_normalize_sparse_adj(adj)
            adj_row = row_normalize_sparse_adj(adj)
            sym_norm_adjs.append(adj_sym)
            neighbor_means.append(torch.sparse.mm(adj_row, x))
        self._sym_norm_adjs = sym_norm_adjs
        self._neighbor_means = neighbor_means
        self._prepared = True

    def forward(
        self,
        x: torch.Tensor,
        train_mask: torch.Tensor,
        train_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing per-relation evidence.

        Args:
            x: ``(N, x_dim)`` raw node features.
            train_mask: ``(N,)`` boolean. Only nodes where this is True
                contribute to prototype computation.
            train_labels: ``(N,)`` int64. Only values at ``train_mask=True``
                positions are read; others may be arbitrary.

        Returns:
            ``(N, R * out_dim_per_rel)`` tensor — drop-in compatible with the
            hand-crafted ``rel_features`` tensor consumed by
            ``CoVERRelReasoner.forward``.
        """
        if not self._prepared:
            raise RuntimeError(
                "Call extractor.prepare(x, relation_adjs) before forward()"
            )

        # ----- Train-only prototype features (no gradient through labels) -----
        with torch.no_grad():
            fraud_idx = train_mask & (train_labels == 1)
            benign_idx = train_mask & (train_labels == 0)
            if fraud_idx.any():
                fraud_proto = x[fraud_idx].mean(dim=0)
            else:
                fraud_proto = x.new_zeros(self.x_dim)
            if benign_idx.any():
                benign_proto = x[benign_idx].mean(dim=0)
            else:
                benign_proto = x.new_zeros(self.x_dim)

        # Distances + z-score (gradient flows through x but proto is detached)
        fraud_dist = (x - fraud_proto.detach()).norm(dim=1, keepdim=True)
        benign_dist = (x - benign_proto.detach()).norm(dim=1, keepdim=True)
        margin = fraud_dist - benign_dist

        def _zscore(t: torch.Tensor) -> torch.Tensor:
            std = t.std().clamp(min=1e-6)
            return (t - t.mean()) / std

        fraud_dist_z = _zscore(fraud_dist)
        benign_dist_z = _zscore(benign_dist)
        margin_z = _zscore(margin)

        # ----- Per-relation evidence -----
        per_rel: list[torch.Tensor] = []
        for r in range(self.num_relations):
            adj_sym = self._sym_norm_adjs[r]
            neighbor_mean = self._neighbor_means[r]
            gcn_emb = self.relation_encoders[r](x, adj_sym)
            pool_in = torch.cat(
                [x, neighbor_mean, gcn_emb, fraud_dist_z, benign_dist_z, margin_z],
                dim=1,
            )
            evidence_r = self.relation_heads[r](pool_in)
            per_rel.append(evidence_r)

        return torch.cat(per_rel, dim=1)


def build_learned_extractor(
    x_dim: int,
    num_relations: int,
    cfg: dict | None = None,
) -> LearnedRelationEvidenceExtractor:
    """Construct a LearnedRelationEvidenceExtractor from a config dict.

    Recognised keys (all optional, with documented defaults):
        hidden_dim: int = 32
        encoder_out_dim: int = 16
        out_dim_per_rel: int = 9
        dropout: float = 0.3
    """
    cfg = cfg or {}
    return LearnedRelationEvidenceExtractor(
        x_dim=x_dim,
        num_relations=num_relations,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        encoder_out_dim=int(cfg.get("encoder_out_dim", 16)),
        out_dim_per_rel=int(cfg.get("out_dim_per_rel", 9)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


__all__ = [
    "LearnedRelationEvidenceExtractor",
    "RelationGCNEncoder",
    "build_learned_extractor",
    "symmetric_normalize_sparse_adj",
    "row_normalize_sparse_adj",
]
