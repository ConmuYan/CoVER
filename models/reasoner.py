from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from evidence.vocab import get_num_values, get_reason_types, get_evidence_slots


class EvidenceEncoder(nn.Module):
    def __init__(self, num_values: int, num_slots: int, emb_dim: int = 16, out_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_values, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(num_slots * emb_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, evidence_token_ids: Tensor) -> Tensor:
        h = self.emb(evidence_token_ids)
        h = h.flatten(start_dim=1)
        return self.proj(h)


class EvidenceReasoner(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_slots: int | None = None,
        num_values: int | None = None,
        num_types: int | None = None,
        hidden_dim: int = 128,
        evidence_emb_dim: int = 16,
        rho: float = 0.3,
    ):
        super().__init__()
        if num_slots is None:
            num_slots = len(get_evidence_slots())
        if num_values is None:
            num_values = get_num_values()
        if num_types is None:
            num_types = len(get_reason_types())

        self.rho = rho
        self.num_slots = num_slots
        self.num_types = num_types

        self.evidence_encoder = EvidenceEncoder(num_values, num_slots, evidence_emb_dim, 64)
        in_dim = z_dim + 64

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.type_head = nn.Linear(hidden_dim, num_types)
        self.pos_head = nn.Linear(hidden_dim, num_slots)
        self.neg_head = nn.Linear(hidden_dim, num_slots)
        self.residual_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: Tensor,
        base_logit: Tensor,
        evidence_token_ids: Tensor,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        """Run the reasoner.

        Args:
            z: Node embeddings from the base detector.
            base_logit: Base detector logits.
            evidence_token_ids: Encoded evidence tokens per node.
            return_debug: If True, include gate/residual diagnostics in the output dict.
        """
        g = self.evidence_encoder(evidence_token_ids)
        h = self.shared(torch.cat([z, g], dim=-1))

        type_logits = self.type_head(h)
        pos_logits = self.pos_head(h)
        neg_logits = self.neg_head(h)

        gate = (
            torch.sigmoid(pos_logits).mean(dim=-1, keepdim=True)
            - torch.sigmoid(neg_logits).mean(dim=-1, keepdim=True)
        )

        residual_raw = self.residual_head(h)
        residual = gate * residual_raw

        final_logit = base_logit.view(-1, 1) + self.rho * residual
        final_logit = final_logit.view(-1)

        outputs = {
            "final_logit": final_logit,
            "type_logits": type_logits,
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
        }

        if return_debug:
            outputs.update(
                {
                    "gate": gate,
                    "residual_raw": residual_raw,
                    "residual": residual,
                    "rho": final_logit.new_tensor(self.rho),
                }
            )

        return outputs
