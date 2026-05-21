"""CBR-Flash student adapter for contract-budgeted residual distillation.

This is the canonical implementation of the lightweight student used after
the RAER/LREE teacher stage.  The student distills the teacher's final residual
policy under the same bounded-residual contract; it does not expose auxiliary
per-relation contribution or gate matching heads.

Architecture
------------

The student shares a single MLP trunk fed the *concatenation* of base
embedding and all R per-relation evidence vectors (NOT a mean across
relations — Codex round-3 minor 1)::

    input    = [base_z ; e_{i,1} ; ... ; e_{i,R}]      (dim = d_z + R*9)
    h        = MLP(input)
    Δ_φ      = δ_max * tanh(head_delta(h))            (scalar per node, zero-init)
    s_S      = base_logit + Δ_φ                        (final scalar logit)

The four hard contracts of ``AGENTS.md §1 Problem formulation``:

* **C1 base-freeze**: ``base_logit`` is ``.detach()``-ed; no gradient flows
  to base parameters.
* **C2 score-blind input**: ``base_logit`` is NEVER read by the trunk nor by
  the residual head; it is added outside as ``s_S = b + Δ_φ``.  The trunk input
  is ``[base_z ; rel_features]`` only.
* **C3 train-only prototype**: this adapter never accesses prototype tensors;
  the teacher's train-only construction is preserved upstream.
* **C4 δ-bounded residual**: ``Δ_φ ∈ [-δ_max, δ_max]`` by architecture
  (``δ_max * tanh(...)``), not by post-hoc clipping.

The only distilled student signal is the final residual policy:
``s_S = base_logit + Δ_φ`` is matched to the teacher final logit/probability.
The residual head is zero-initialised so the epoch-0 student posterior equals the
frozen base posterior (Proposition P3 §5; safe deployment from epoch 0).

Parameter budget
----------------

For YelpChi/Amazon (d_z = 64, R = 3, hidden_dim = 32):

* trunk: (64 + 27)·32 + 32·32 = 2912 + 1024 = 3936
* head_delta: 32·1 + 1 = 33

Total: 3969 before biases in the trunk accounting above, and 4033 with all
linear biases in PyTorch; both remain within the 5000-parameter design target.

Compatibility
-------------

``FlashAdapter`` remains as a short alias for ``CBRFlashAdapter`` because the
tests and some local notebooks use that shorter class name. New training code
should import ``CBRFlashAdapter``.
"""

import torch
import torch.nn as nn
from torch import Tensor


class CBRFlashAdapter(nn.Module):
    """CBR-Flash student adapter with a single bounded residual head.

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

        # Score-blind: trunk input is [base_z ; rel_features], i.e. the
        # concatenation of base embedding and per-relation evidence vectors
        # (NOT a mean across relations — Codex round-3 minor 1).
        in_dim = self.base_z_dim + self.num_relations * self.rel_stat_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Δ_φ head — scalar bounded residual (zero-init ⇒ Δ_φ ≡ 0 at epoch 0).
        self.head_delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

    def forward(
        self,
        base_z: Tensor,
        base_logit: Tensor,
        relation_features: Tensor,
        return_heads: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            base_z: ``(N, base_z_dim)`` base embeddings.
            base_logit: ``(N,)`` or ``(N, 1)`` base logits.
            relation_features: ``(N, R * rel_stat_dim)`` relation stats.
            return_heads: if True, additionally expose final-policy aliases
                ``logit``, ``p``, and ``delta_pre_tanh`` for distillation logs.

        Returns:
            Dict with::

                delta_phi    (N,)            bounded residual Δ_φ
                final_logit  (N,)            = base_logit + Δ_φ

            With ``return_heads=True`` additionally::

                logit          (N,)          alias of final_logit
                p              (N,)          σ(logit)
                delta_pre_tanh (N,)          = atanh(Δ_φ / δ_max); mirrors teacher
        """
        device = base_z.device
        dtype = base_z.dtype

        b_i = base_logit.detach().view(-1).to(device=device, dtype=dtype)

        # Score-blind trunk input — base_logit is NOT in the MLP.
        inp = torch.cat([
            base_z.detach(),
            relation_features.to(device=device, dtype=dtype),
        ], dim=-1)

        h = self.trunk(inp)

        # Δ_φ = δ_max · tanh(head_delta(h)) — C4 bounded residual.
        delta_phi = self.delta_max * torch.tanh(self.head_delta(h).view(-1))

        out: dict[str, Tensor] = {
            "delta_phi": delta_phi,
            "final_logit": b_i + delta_phi,
        }

        if return_heads:
            logit = out["final_logit"]
            # delta_pre_tanh: inverse of δ_max·tanh, expressed via the
            # head_delta output directly to avoid atanh numerical issues.
            delta_pre_tanh = self.head_delta(h).view(-1)
            out.update({
                "logit": logit,
                "p": torch.sigmoid(logit),
                "delta_pre_tanh": delta_pre_tanh,
            })

        return out


# Short alias retained for tests and notebooks.
FlashAdapter = CBRFlashAdapter


__all__ = ["CBRFlashAdapter", "FlashAdapter"]
