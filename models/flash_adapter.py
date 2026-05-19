"""FlashAdapter — G-OPD-Flash student adapter with three matching heads.

Replaces (and back-compatibly subsumes) ``models/rel_distill_adapter.py``'s
``RelDistillAdapter`` for the C3 G-OPD-Flash contribution
(``docs/OPD_FLASH_DESIGN_v3.md`` §3.3 Phase C).

Architecture
------------

The student shares a single MLP trunk fed the *concatenation* of base
embedding and all R per-relation evidence vectors (NOT a mean across
relations — Codex round-3 minor 1)::

    input    = [base_z ; e_{i,1} ; ... ; e_{i,R}]      (dim = d_z + R*9)
    h        = MLP(input)
    Δ_φ      = δ_max * tanh(head_delta(h))            (scalar per node, zero-init)
    s_S_r    = c_per_r_head(h)                         (R-dim pre-gate scalar, zero-init)
    g_S_r    = softmax(head_gate(h))                   (R-dim gate, zero-init)
    c_S_r    = g_S_r * s_S_r                           (R-dim gate-weighted contribution)
    s_S      = base_logit + Δ_φ                        (final scalar logit)

The four hard contracts of ``AGENTS.md §1 Problem formulation``:

* **C1 base-freeze**: ``base_logit`` is ``.detach()``-ed; no gradient flows
  to base parameters.
* **C2 score-blind input**: ``base_logit`` is NEVER read by the trunk nor by
  any head; it is added outside as ``s_S = b + Δ_φ``.  The trunk input is
  ``[base_z ; rel_features]`` only.  This is statically asserted by
  ``tests/test_opd_flash_contracts.py::test_score_blind_static_pass``.
* **C3 train-only prototype**: this adapter never accesses prototype tensors;
  the teacher's train-only construction is preserved upstream.
* **C4 δ-bounded residual**: ``Δ_φ ∈ [-δ_max, δ_max]`` by architecture
  (``δ_max * tanh(...)``), not by post-hoc clipping.

Three matching heads (consumed by G-OPD-Flash entropy-aware mixed KL loss):

* ``logit``    — final scalar logit, matched via Bernoulli KL to teacher ``p``
* ``c_per_r``  — gate-weighted per-relation contribution, matched to
                 teacher's ``c_per_r = g_r * Head_r(h_r)`` via MSE
* ``gate``     — softmax over R relations, matched to teacher's ``gate`` via
                 categorical KL

All heads are zero-initialised so the epoch-0 student posterior equals the
frozen base posterior (Proposition P3 §5; safe deployment from epoch 0).

Parameter budget
----------------

For YelpChi/Amazon (d_z = 64, R = 3, hidden_dim = 32):

* trunk: (64 + 27)·32 + 32·32 = 2912 + 1024 = 3936
* head_delta: 32·1 + 1 = 33
* c_per_r_head: 32·3 + 3 = 99
* head_gate: 32·3 + 3 = 99

Total: 4167 ≤ 5000 (design target).

Back-compat
-----------

``RelDistillAdapter`` is re-exported as an alias of ``FlashAdapter`` so
existing call sites in ``scripts/train_distill_adapter.py`` continue to load
without modification.  Such call sites will receive a strict superset of
the original return dict (``delta_phi``, ``gate_logits``, ``gate_probs``,
``final_logit``) since FlashAdapter preserves every key.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FlashAdapter(nn.Module):
    """G-OPD-Flash student adapter with logit / c_per_r / gate heads.

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

        # Per-relation pre-gate scalar head s_S_r (zero-init).
        # Together with gate (softmax of head_gate), these form the
        # G-OPD-Flash multi-head matching signals; ``c_S_r = g_S_r * s_S_r``.
        self.c_per_r_head = nn.Linear(hidden_dim, self.num_relations)
        nn.init.zeros_(self.c_per_r_head.weight)
        nn.init.zeros_(self.c_per_r_head.bias)

        # Gate head — R-dim logits, zero-init ⇒ uniform softmax at epoch 0.
        self.head_gate = nn.Linear(hidden_dim, self.num_relations)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)

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
            return_heads: if True, additionally expose ``logit, c_per_r, gate, p,
                s_per_r, delta_pre_tanh`` for matching against the teacher
                (mirror of the teacher's ``return_heads=True`` interface).

        Returns:
            Dict with (back-compat with RelDistillAdapter)::

                delta_phi    (N,)            bounded residual Δ_φ
                gate_logits  (N, R)          raw gate logits π_φ
                gate_probs   (N, R)          softmax(π_φ)
                final_logit  (N,)            = base_logit + Δ_φ

            With ``return_heads=True`` additionally::

                logit          (N,)          alias of final_logit
                s_per_r        (N, R)        per-relation pre-gate scalar
                c_per_r        (N, R)        g_S_r * s_S_r (gate-weighted)
                gate           (N, R)        alias of gate_probs
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

        # Gate (softmax over R relations).
        gate_logits = self.head_gate(h)
        gate_probs = F.softmax(gate_logits, dim=-1)

        out: dict[str, Tensor] = {
            "delta_phi": delta_phi,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "final_logit": b_i + delta_phi,
        }

        if return_heads:
            # Per-relation pre-gate scalar (zero at epoch 0).
            s_per_r = self.c_per_r_head(h)            # (N, R)
            # Gate-weighted per-relation contribution; matches teacher c_per_r.
            c_per_r = gate_probs * s_per_r            # (N, R)
            logit = out["final_logit"]
            # delta_pre_tanh: inverse of δ_max·tanh, expressed via the
            # head_delta output directly to avoid atanh numerical issues.
            delta_pre_tanh = self.head_delta(h).view(-1)
            out.update({
                "logit": logit,
                "s_per_r": s_per_r,
                "c_per_r": c_per_r,
                "gate": gate_probs,
                "p": torch.sigmoid(logit),
                "delta_pre_tanh": delta_pre_tanh,
            })

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat alias for existing call sites in scripts/train_distill_adapter.py
# ─────────────────────────────────────────────────────────────────────────────

#: Alias kept so the v3 G-OPD-Flash student is a strict superset of the v1
#: RelDistillAdapter: existing distill scripts can import ``RelDistillAdapter``
#: from this module unchanged and receive a FlashAdapter instance.  When
#: invoked with ``return_heads=False`` (default), the returned dict matches
#: the v1 RelDistillAdapter signature key-for-key.
RelDistillAdapter = FlashAdapter


__all__ = ["FlashAdapter", "RelDistillAdapter"]
