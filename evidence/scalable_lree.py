"""Scalable LREE for large RAER-FD datasets.

This variant keeps the LREE contract but avoids full-graph sparse GPU
aggregation. It consumes cached score-blind relation basis features
(``rel_stats.pt``) and learns a per-relation evidence transform.

Contracts:
- Score-blind: inputs are relation basis features only, never base logits.
- Train-only prototypes: inherited from the cached relation basis metadata.
- Base-freeze: independent from base detector parameters.
"""
from __future__ import annotations

import torch
from torch import nn


class ScalableLREE(nn.Module):
    """Cache-backed learnable relation evidence extractor.

    Input and output are both shaped ``(N, R * D)``. The module splits the
    cached basis into relation chunks and applies either independent or shared
    MLP heads to produce learned evidence consumed by ``RAERTeacher``.
    """

    def __init__(
        self,
        num_relations: int,
        in_dim_per_rel: int = 9,
        out_dim_per_rel: int = 9,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        shared_encoder: bool = False,
    ) -> None:
        super().__init__()
        if num_relations <= 0:
            raise ValueError("num_relations must be > 0")
        self.num_relations = int(num_relations)
        self.in_dim_per_rel = int(in_dim_per_rel)
        self.out_dim_per_rel = int(out_dim_per_rel)
        self.hidden_dim = int(hidden_dim)
        self.shared_encoder = bool(shared_encoder)

        def make_head(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(hidden_dim, out_dim_per_rel))
            return nn.Sequential(*layers)

        if self.shared_encoder:
            self.shared_head = make_head(in_dim_per_rel + num_relations)
            self.relation_heads = None
        else:
            self.shared_head = None
            self.relation_heads = nn.ModuleList([
                make_head(in_dim_per_rel) for _ in range(num_relations)
            ])

    def forward(self, relation_basis: torch.Tensor) -> torch.Tensor:
        if relation_basis.ndim != 2:
            raise ValueError(f"relation_basis must be rank-2, got {tuple(relation_basis.shape)}")
        expected = self.num_relations * self.in_dim_per_rel
        if relation_basis.shape[1] != expected:
            raise ValueError(
                f"relation_basis dim mismatch: {relation_basis.shape[1]} != {expected}"
            )

        chunks = torch.split(relation_basis, self.in_dim_per_rel, dim=1)
        outputs: list[torch.Tensor] = []
        if self.shared_encoder:
            eye = torch.eye(
                self.num_relations,
                device=relation_basis.device,
                dtype=relation_basis.dtype,
            )
            assert self.shared_head is not None
            for r, chunk in enumerate(chunks):
                tag = eye[r].unsqueeze(0).expand(chunk.shape[0], -1)
                outputs.append(self.shared_head(torch.cat([chunk, tag], dim=1)))
        else:
            assert self.relation_heads is not None
            for head, chunk in zip(self.relation_heads, chunks):
                outputs.append(head(chunk))
        return torch.cat(outputs, dim=1)


def build_scalable_lree_extractor(
    *,
    num_relations: int,
    cfg: dict | None = None,
    in_dim_per_rel: int = 9,
) -> ScalableLREE:
    cfg = cfg or {}
    return ScalableLREE(
        num_relations=num_relations,
        in_dim_per_rel=int(cfg.get("in_dim_per_rel", in_dim_per_rel)),
        out_dim_per_rel=int(cfg.get("out_dim_per_rel", 9)),
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.3)),
        shared_encoder=bool(cfg.get("shared_encoder", False)),
    )


__all__ = ["ScalableLREE", "build_scalable_lree_extractor"]
