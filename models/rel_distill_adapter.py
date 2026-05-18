"""RelDistillAdapter — lightweight REL distillation adapter (Idea 2C).

Learns to mimic a frozen CoVER-REL teacher via three-way distillation:

    L = L_bce(y, σ(z_base + Δ_φ))
      + λ_distill · MSE(Δ_φ, sg(Δ_rel_teacher))
      + γ_kl · KL(softmax(π_φ) || sg(softmax(π_rel_teacher)))

At inference, the adapter replaces the full CoVER-REL teacher: it takes the
same inputs (base_z, base_logit, rel_features) but runs a single small MLP
instead of R expert MLPs + gate network + per-relation heads.

Architecture::

    input = [base_z ; base_logit ; rel_features]   (dim = d_z + 1 + R*9)
    → MLP(input → hidden → hidden)
    → Δ_φ = delta_max · tanh(head_delta(h))        (scalar per node)
    → π_φ = head_gate(h)                            (R-dim logits)

Params: ~5K (vs CoVER-REL ~14K).

Safety contracts preserved:
    - Score-blind at inference: same inputs as CoVER-REL (base_z + base_logit
      + rel_features); rel_features are themselves score-blind.
    - Frozen base: adapter never modifies base parameters.
    - Bounded intervention: Δ_φ bounded by delta_max · tanh.
    - Train-only-prototype: teacher outputs (used only during training) are
      pre-cached from a frozen teacher; no label leakage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RelDistillAdapter(nn.Module):
    """Lightweight distillation adapter for CoVER-REL.

    Args:
        base_z_dim: Dimension of base detector embeddings.
        num_relations: Number of relations R.
        rel_stat_dim: Per-relation statistic dimension (default 9).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers (minimum 1).
        dropout: Dropout rate.
        delta_max: Tanh saturation bound for the residual Δ_φ.
    """

    def __init__(
        self,
        base_z_dim: int,
        num_relations: int,
        rel_stat_dim: int = 9,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        delta_max: float = 2.0,
    ):
        super().__init__()
        if num_relations <= 0:
            raise ValueError("num_relations must be > 0")
        if base_z_dim <= 0:
            raise ValueError("base_z_dim must be > 0")

        self.base_z_dim = int(base_z_dim)
        self.num_relations = int(num_relations)
        self.rel_stat_dim = int(rel_stat_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_max = float(delta_max)

        # Score-blind: trunk sees [base_z ; rel_features] only.
        # base_logit is used ONLY for final_logit = base_logit + delta_phi.
        in_dim = self.base_z_dim + self.num_relations * self.rel_stat_dim

        # Shared MLP trunk
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Δ_φ head: scalar residual (zero-init → do-nothing at start)
        self.head_delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

        # π_φ head: R-dim gate logits
        self.head_gate = nn.Linear(hidden_dim, self.num_relations)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)

    def forward(
        self,
        base_z: Tensor,
        base_logit: Tensor,
        relation_features: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            base_z: ``(N, base_z_dim)`` base embeddings.
            base_logit: ``(N,)`` or ``(N, 1)`` base logits.
            relation_features: ``(N, R * rel_stat_dim)`` relation stats.

        Returns:
            Dict with:
                ``delta_phi``  ``(N,)``  bounded residual Δ_φ
                ``gate_logits`` ``(N, R)``  raw gate logits π_φ
                ``gate_probs``  ``(N, R)``  softmax(π_φ)
                ``final_logit`` ``(N,)``  = base_logit + Δ_φ
        """
        n = base_z.shape[0]
        device = base_z.device
        dtype = base_z.dtype

        b_i = base_logit.detach().view(-1).to(device=device, dtype=dtype)

        # Score-blind: trunk input is [base_z ; rel_features] only.
        # base_logit is NOT fed into the MLP — only used for final_logit addition.
        inp = torch.cat([
            base_z.detach(),
            relation_features.to(device=device, dtype=dtype),
        ], dim=-1)

        h = self.trunk(inp)

        # Δ_φ = delta_max · tanh(head_delta(h))
        delta_phi = self.delta_max * torch.tanh(self.head_delta(h).view(-1))

        # π_φ gate logits and softmax
        gate_logits = self.head_gate(h)
        gate_probs = F.softmax(gate_logits, dim=-1)

        return {
            "delta_phi": delta_phi,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "final_logit": b_i + delta_phi,
        }


__all__ = ["RelDistillAdapter"]
