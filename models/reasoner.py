"""Evidence-conditioned reasoner for CoVER-FD.

Supports multiple gate modes to control how the residual correction is applied
to the base detector logit.  The default ``safe_residual`` mode uses an
independent gate head (sigmoid, bounded [0, 1]) and a tanh-bounded residual
head to avoid the saturation problem observed with the original signed-difference
gate.

Gate modes
----------
signed_diff_legacy
    Original design: ``gate = sigmoid(pos_logits).mean - sigmoid(neg_logits).mean``.
    Can saturate to -1 when counter-evidence dominates, pushing logits very negative.

safe_residual
    Independent gate head (sigmoid) + tanh-bounded residual head.
    ``final_logit = base_logit + rho * gate * delta``.
    Gate and delta are both bounded; residual shift is limited to
    ``rho * delta_scale`` in absolute value.

direct_tanh
    No gate.  ``delta = delta_scale * tanh(residual_head(h))``.
    ``final_logit = base_logit + rho * delta``.

aux_only
    ``final_logit = base_logit``.  Evidence heads are trained for
    supervision only; no residual correction is applied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from evidence.vocab import get_num_values, get_reason_types, get_evidence_slots

VALID_GATE_MODES = ("signed_diff_legacy", "safe_residual", "direct_tanh", "aux_only")


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
    """Evidence-conditioned reasoner with configurable gate modes.

    Args:
        z_dim: Dimension of base detector embeddings.
        num_slots: Number of evidence slots (auto-detected if None).
        num_values: Number of vocabulary values (auto-detected if None).
        num_types: Number of risk types (auto-detected if None).
        hidden_dim: Hidden dimension for the shared MLP.
        evidence_emb_dim: Embedding dimension for evidence tokens.
        rho: Residual correction strength.  0 disables correction entirely.
        gate_mode: One of ``signed_diff_legacy``, ``safe_residual``,
            ``direct_tanh``, ``aux_only``.
        delta_scale: Maximum absolute value of the residual delta (used by
            ``safe_residual`` and ``direct_tanh``).
        gate_bias_init: Initial bias for the gate head (``safe_residual``
            only).  -2.0 gives an initial gate ~0.12, encouraging conservatism.
        residual_init_zero: If True, initialise residual head weights and bias
            to zero so that initial delta ≈ 0.
    """

    def __init__(
        self,
        z_dim: int,
        num_slots: int | None = None,
        num_values: int | None = None,
        num_types: int | None = None,
        hidden_dim: int = 128,
        evidence_emb_dim: int = 16,
        rho: float = 0.3,
        gate_mode: str = "safe_residual",
        delta_scale: float = 2.0,
        gate_bias_init: float = -2.0,
        residual_init_zero: bool = True,
    ):
        super().__init__()
        if gate_mode not in VALID_GATE_MODES:
            raise ValueError(f"Unknown gate_mode={gate_mode!r}. Must be one of {VALID_GATE_MODES}")

        if num_slots is None:
            num_slots = len(get_evidence_slots())
        if num_values is None:
            num_values = get_num_values()
        if num_types is None:
            num_types = len(get_reason_types())

        self.rho = rho
        self.num_slots = num_slots
        self.num_types = num_types
        self.gate_mode = gate_mode
        self.delta_scale = delta_scale

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

        if gate_mode == "safe_residual":
            self.gate_head = nn.Linear(hidden_dim, 1)
            self.residual_head = nn.Linear(hidden_dim, 1)
            # sigmoid(-2) ≈ 0.12 — conservative initial gate
            nn.init.constant_(self.gate_head.bias, gate_bias_init)
            nn.init.xavier_uniform_(self.gate_head.weight)
        elif gate_mode in ("direct_tanh", "signed_diff_legacy"):
            self.residual_head = nn.Linear(hidden_dim, 1)

        if residual_init_zero and gate_mode in ("safe_residual", "direct_tanh", "signed_diff_legacy"):
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        z: Tensor,
        base_logit: Tensor,
        evidence_token_ids: Tensor,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        """Run the reasoner.

        Args:
            z: Node embeddings from the base detector (N, z_dim).
            base_logit: Base detector logits (N,) or (N, 1).
            evidence_token_ids: Encoded evidence tokens per node (N, num_slots).
            return_debug: If True, include gate/residual diagnostics.

        Returns:
            dict with at least ``final_logit``, ``type_logits``, ``pos_logits``,
            ``neg_logits``.  Additional keys when ``return_debug=True``.
        """
        g = self.evidence_encoder(evidence_token_ids)
        h = self.shared(torch.cat([z, g], dim=-1))

        type_logits = self.type_head(h)
        pos_logits = self.pos_head(h)
        neg_logits = self.neg_head(h)

        base = base_logit.view(-1, 1)

        if self.gate_mode == "signed_diff_legacy":
            gate = (
                torch.sigmoid(pos_logits).mean(dim=-1, keepdim=True)
                - torch.sigmoid(neg_logits).mean(dim=-1, keepdim=True)
            )
            delta_raw = self.residual_head(h)
            residual = gate * delta_raw
            final_logit = base + self.rho * residual

        elif self.gate_mode == "safe_residual":
            gate = torch.sigmoid(self.gate_head(h))
            delta_raw = self.residual_head(h)
            delta = self.delta_scale * torch.tanh(delta_raw)
            final_logit = base + self.rho * gate * delta

        elif self.gate_mode == "direct_tanh":
            delta_raw = self.residual_head(h)
            delta = self.delta_scale * torch.tanh(delta_raw)
            final_logit = base + self.rho * delta

        elif self.gate_mode == "aux_only":
            final_logit = base

        else:
            raise ValueError(f"Unknown gate_mode={self.gate_mode!r}")

        final_logit = final_logit.view(-1)

        outputs: dict[str, Tensor] = {
            "final_logit": final_logit,
            "type_logits": type_logits,
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
        }

        if return_debug:
            residual_shift = final_logit - base_logit.view(-1)
            outputs["residual_shift"] = residual_shift
            outputs["gate_mode"] = final_logit.new_tensor(
                list(VALID_GATE_MODES).index(self.gate_mode)
            )
            outputs["rho"] = final_logit.new_tensor(self.rho)
            outputs["delta_scale"] = final_logit.new_tensor(self.delta_scale)

            if self.gate_mode == "signed_diff_legacy":
                outputs["gate"] = gate
                outputs["delta_raw"] = delta_raw
                outputs["residual_raw"] = residual
            elif self.gate_mode == "safe_residual":
                outputs["gate"] = gate
                outputs["delta_raw"] = delta_raw
                outputs["delta"] = delta
            elif self.gate_mode == "direct_tanh":
                outputs["delta_raw"] = delta_raw
                outputs["delta"] = delta

        return outputs
