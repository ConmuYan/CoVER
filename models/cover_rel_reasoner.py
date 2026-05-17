"""Phase 2 unified CoVER-REL Reasoner — cls-only canonical.

Forward (rel-only)::

    z_i = b_i + Δ_rel_i,   |Δ_rel_i| ≤ delta_rel_max

This is the *only* CoVER-REL forward path. There is no LLM judge, no
additive ``α·Δ_llm`` residual, no auxiliary signal injection. All
regularisation is structural: bounded ``tanh`` saturation, zero-initialised
per-relation residual heads, ``detach``-isolated base inputs, and
independent (non-shared) per-relation expert MLPs.

Where:

* ``b_i = base_logit.detach()`` is the frozen base detector logit.
* ``h_{i,r} = Expert_r(E_{i,r})`` is a per-relation hidden vector produced
  by an independent MLP expert from the 9-dim relation statistics
  ``E_{i,r}`` (see ``evidence/relation_features.py``).
* ``g_i = softmax(a_i / tau_gate, dim=-1)`` is a softmax-normalised schema
  gate over the ``R`` relations (``Σ_r g_{i,r} = 1``).
* ``u_i = Σ_r g_{i,r} · Head_r(h_{i,r})`` mixes the per-relation scalar
  residuals by the gate; each ``Head_r`` is zero-initialised so the
  starting point is ``Δ_rel ≡ 0`` (do-nothing baseline).
* ``Δ_rel_i = delta_rel_max · tanh(u_i)`` is the bounded relation residual.

See ``AGENTS.md`` §§3–7 for the full TPAMI-style derivation, including
the 5-seed paired-t evidence that retired every deleted route
(LLM judge, ``L_intervention``, ``L_sparse``, ``L_align``, CV-SCD,
CoVER-DIR, CoVER-LIFT, LEQA, B3 PRTAE).

Backward compatibility
----------------------

The ``__init__`` and ``forward`` signatures preserve the old judge-related
keyword arguments (``use_judge``, ``judge_feature_dim``, ``alpha_max``,
``delta_llm_max``, ``judge_features``, ``judge_mask``, ...) so legacy
configs and call sites that still pass the *off* defaults continue to load.
Any *judge-on* configuration raises ``NotImplementedError`` immediately —
the route was falsified at 5 seeds (paired t = +0.03, p = 0.976; see
``artifacts/tables/paper_negative_routes.md``).

The output dict still contains ``alpha_llm``, ``delta_llm``, and
``judge_used_mask`` as zero / False tensors so any downstream diagnostic
code that reads them keeps working without ``KeyError``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_DEPRECATION_MSG = (
    "Judge fusion path is permanently removed from the canonical CoVER-REL. "
    "The α·Δ_llm additive residual was 5-seed paired-t falsified "
    "(t = +0.03, p = 0.976 with 2000 vLLM judge packets); LEQA and B3 PRTAE "
    "follow-ups were equally null. See AGENTS.md §9 and "
    "artifacts/tables/paper_negative_routes.md for the negative-route ledger."
)


def _make_relation_expert(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Per-relation expert MLP.

    Layout (for ``num_layers=2``)::

        Linear(in_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, hidden_dim) -> ReLU
        LayerNorm(hidden_dim)
        Dropout(dropout)   # only when dropout > 0
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(max(int(num_layers), 1)):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        prev = hidden_dim
    layers.append(nn.LayerNorm(hidden_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _make_relation_head(hidden_dim: int) -> nn.Linear:
    """Per-relation scalar residual head.

    Zero-initialised so ``Head_r(h) = 0`` at the start of training; this gives
    ``Δ_rel ≈ 0`` initially and lets the frozen base logit speak for itself.
    """
    head = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class CoVERRelReasoner(nn.Module):
    """Phase 2 unified reasoner with softmax relation gate.

    Args:
        base_z_dim: Dimension of base detector embeddings ``base_z`` passed at forward.
        relation_names: Ordered list of relation names (uppercased internally).
            The order is used to split ``relation_features`` into per-relation
            chunks of size ``rel_stat_dim``.
        anchor_relation: Primary anchor relation (must be present in
            ``relation_names``).  Diagnostic only — the softmax gate is
            symmetric across all relations.
        rel_stat_dim: Per-relation statistic dim (default 9 to match
            ``RELATION_STAT_NAMES``).
        rel_hidden_dim: Per-relation expert output dim.
        rel_num_layers: Number of hidden layers in each expert MLP.
        rel_dropout: Dropout applied after the expert ``LayerNorm``.
        tau_gate: Softmax temperature for the schema gate.
        delta_rel_max: Tanh saturation bound for the relation residual.

        use_judge, judge_feature_dim, judge_hidden_dim, judge_dropout,
        delta_llm_max, alpha_max, alpha_bias_init: **Deprecated noop kwargs.**
            Kept for backward compatibility with rel-only configs that
            still pass them with default (off) values.  Any judge-on
            configuration raises ``NotImplementedError`` per
            ``PHASE2_DEPRECATION_PLAN_CORRECTION.md``.
    """

    def __init__(
        self,
        base_z_dim: int,
        relation_names: list[str],
        anchor_relation: str,
        rel_stat_dim: int = 9,
        rel_hidden_dim: int = 64,
        rel_num_layers: int = 2,
        rel_dropout: float = 0.30,
        tau_gate: float = 0.7,
        delta_rel_max: float = 2.0,
        # ----- Idea-1 ablation toggles (config-driven switches) -----
        gate_mode: str = "softmax",
        evidence_groups: list[str] | None = None,
        expert_shared: bool = False,
        residual_activation: str = "tanh",
        # ----- Deprecated noop kwargs (Commit 1 v2) -----
        use_judge: bool = False,
        judge_feature_dim: int = 0,
        judge_hidden_dim: int = 32,
        judge_dropout: float = 0.30,
        delta_llm_max: float = 0.75,
        alpha_max: float = 0.0,
        alpha_bias_init: float = -3.0,
    ):
        super().__init__()

        # ----- Reject any judge-on configuration -----
        if bool(use_judge) or float(alpha_max) > 0.0 or int(judge_feature_dim) > 0:
            raise NotImplementedError(_DEPRECATION_MSG)

        if not relation_names:
            raise ValueError("relation_names must be non-empty")
        rel_names = [str(name).upper() for name in relation_names]
        if anchor_relation is None:
            raise ValueError("anchor_relation is required")
        anchor = str(anchor_relation).upper()
        if anchor not in rel_names:
            raise ValueError(
                f"anchor_relation={anchor!r} not in relation_names={rel_names}"
            )

        # ----- Validate ablation toggles -----
        if gate_mode not in ("softmax", "uniform"):
            raise ValueError(
                f"gate_mode must be 'softmax' or 'uniform', got {gate_mode!r}"
            )
        ev_groups = ["A", "B", "C"] if evidence_groups is None else [
            str(g).upper() for g in evidence_groups
        ]
        for g in ev_groups:
            if g not in {"A", "B", "C"}:
                raise ValueError(
                    f"evidence_groups must be a subset of ['A','B','C'], got {g!r}"
                )
        if not ev_groups:
            raise ValueError("evidence_groups must contain at least one of A, B, C")
        if residual_activation not in ("tanh", "identity"):
            raise ValueError(
                f"residual_activation must be 'tanh' or 'identity', got {residual_activation!r}"
            )

        self.base_z_dim = int(base_z_dim)
        self.relation_names = rel_names
        self.anchor_relation = anchor
        self.num_relations = len(rel_names)
        self.rel_stat_dim = int(rel_stat_dim)
        self.rel_hidden_dim = int(rel_hidden_dim)
        self.rel_num_layers = int(rel_num_layers)
        self.rel_dropout = float(rel_dropout)
        self.tau_gate = max(float(tau_gate), 1e-6)
        self.delta_rel_max = float(delta_rel_max)
        self.gate_mode = gate_mode
        self.evidence_groups = ev_groups
        self.expert_shared = bool(expert_shared)
        self.residual_activation = residual_activation

        # ----- Evidence-group mask buffer (zeros the dropped-group dims) -----
        # 9-dim per-relation layout:
        #   A (structural)         : indices [0, 1, 2]
        #   B (feature-neighbor)   : indices [3, 4, 5]
        #   C (prototype-relative) : indices [6, 7, 8]
        # If rel_stat_dim != 9, the mask becomes all-ones (no group semantics).
        if self.rel_stat_dim == 9:
            mask = torch.zeros(9, dtype=torch.float32)
            if "A" in ev_groups:
                mask[0:3] = 1.0
            if "B" in ev_groups:
                mask[3:6] = 1.0
            if "C" in ev_groups:
                mask[6:9] = 1.0
        else:
            mask = torch.ones(self.rel_stat_dim, dtype=torch.float32)
        self.register_buffer("evidence_mask", mask, persistent=False)

        # ----- Backward-compat attributes (always read as judge-off) -----
        self.use_judge = False
        self.judge_feature_dim = 0
        self.judge_hidden_dim = int(judge_hidden_dim)
        self.judge_dropout = float(judge_dropout)
        self.delta_llm_max = float(delta_llm_max)
        self.alpha_max = 0.0
        self.alpha_bias_init = float(alpha_bias_init)
        self.judge_encoder = None
        self.head_alpha = None
        self.head_llm = None

        # ----- Per-relation experts and scalar residual heads -----
        # Independent (default) OR shared MLP with one-hot relation id input.
        if self.expert_shared:
            # Single MLP taking (E_{i,r} ; one-hot(r)) → h_{i,r}.
            self.relation_experts = _make_relation_expert(
                in_dim=self.rel_stat_dim + self.num_relations,
                hidden_dim=self.rel_hidden_dim,
                num_layers=self.rel_num_layers,
                dropout=self.rel_dropout,
            )
            # Shared scalar head (zero-init preserves do-nothing start).
            self.relation_heads = _make_relation_head(self.rel_hidden_dim)
        else:
            self.relation_experts = nn.ModuleDict({
                name: _make_relation_expert(
                    in_dim=self.rel_stat_dim,
                    hidden_dim=self.rel_hidden_dim,
                    num_layers=self.rel_num_layers,
                    dropout=self.rel_dropout,
                )
                for name in self.relation_names
            })
            self.relation_heads = nn.ModuleDict({
                name: _make_relation_head(self.rel_hidden_dim)
                for name in self.relation_names
            })

        # ----- Softmax gate over R relations -----
        gate_in_dim = self.base_z_dim + self.num_relations * self.rel_hidden_dim
        self.gate_logit_head = nn.Linear(gate_in_dim, self.num_relations)
        nn.init.zeros_(self.gate_logit_head.weight)
        nn.init.zeros_(self.gate_logit_head.bias)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        base_z: Tensor,
        base_logit: Tensor,
        relation_features: Tensor,
        judge_features: Tensor | None = None,
        judge_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the rel-only forward pass.

        Args:
            base_z: Base detector embeddings ``(N, base_z_dim)``.  Detached
                internally so no gradient flows back to the base detector.
            base_logit: Base detector logits ``(N,)`` or ``(N, 1)``.  Detached.
            relation_features: Concatenated per-relation 9-dim stats arranged
                in ``relation_names`` order, shape ``(N, R * rel_stat_dim)``.
            judge_features, judge_mask: **Deprecated.**  Must be ``None``.
                Passing non-None raises ``NotImplementedError``.

        Returns:
            A dict with::

                final_logit       (N,)            # = b + Δ_rel
                rel_only_logit    (N,)            # = b + Δ_rel
                relation_gate     (N, R)          # softmax-normalised
                delta_rel         (N,)
                delta_llm         (N,)            # always zeros (backward-compat)
                alpha_llm         (N,)            # always zeros (backward-compat)
                judge_used_mask   (N,) bool       # always False (backward-compat)
                fused_rel_h       (N, rel_hidden_dim)
                relation_strength (N, R)          # |Head_r(h_i,r)| post-Head
        """
        if judge_features is not None or (judge_mask is not None and bool(judge_mask.any())):
            raise NotImplementedError(_DEPRECATION_MSG)

        if base_z.dim() != 2:
            raise ValueError(
                f"base_z must be 2D (N, base_z_dim); got {tuple(base_z.shape)}"
            )
        if base_z.shape[1] != self.base_z_dim:
            raise ValueError(
                f"base_z dim mismatch: {base_z.shape[1]} != base_z_dim={self.base_z_dim}"
            )

        n = base_z.shape[0]
        device = base_z.device
        dtype = base_z.dtype

        z_detached = base_z.detach()
        b_i = base_logit.detach().view(-1).to(device=device, dtype=dtype)

        expected_rel_dim = self.num_relations * self.rel_stat_dim
        if relation_features.shape[1] != expected_rel_dim:
            raise ValueError(
                f"relation_features dim mismatch: "
                f"{relation_features.shape[1]} != R*rel_stat_dim={expected_rel_dim}"
            )
        rel_feats = relation_features.to(device=device, dtype=dtype)
        # ---- Evidence-group mask: zero-out dropped per-relation dims --------
        mask_one = self.evidence_mask.to(device=device, dtype=dtype)
        if mask_one.numel() == self.rel_stat_dim:
            mask_full = mask_one.repeat(self.num_relations).view(1, -1)
            rel_feats = rel_feats * mask_full
        chunks = torch.split(rel_feats, self.rel_stat_dim, dim=1)

        # h_i,r (per-relation hidden) and Head_r(h_i,r) (per-relation scalar)
        h_per_rel: list[Tensor] = []
        head_out_per_rel: list[Tensor] = []
        if self.expert_shared:
            # Shared MLP with one-hot relation id appended to each chunk.
            eye = torch.eye(self.num_relations, device=device, dtype=dtype)
            for r_idx, (name, chunk) in enumerate(zip(self.relation_names, chunks)):
                onehot = eye[r_idx].unsqueeze(0).expand(chunk.shape[0], -1)
                augmented = torch.cat([chunk, onehot], dim=-1)
                h = self.relation_experts(augmented)
                h_per_rel.append(h)
                head_out_per_rel.append(self.relation_heads(h).view(-1))
        else:
            for name, chunk in zip(self.relation_names, chunks):
                h = self.relation_experts[name](chunk)            # (N, rel_hidden_dim)
                h_per_rel.append(h)
                head_out_per_rel.append(self.relation_heads[name](h).view(-1))  # (N,)

        # ----- Schema gate g_i = softmax(a_i / tau_gate) (or uniform) -----
        all_h = torch.cat(h_per_rel, dim=-1)
        if self.gate_mode == "uniform":
            # Bypass gate net entirely: g_{i,r} = 1/R for all i, r.
            relation_gate = torch.full(
                (n, self.num_relations),
                1.0 / float(self.num_relations),
                device=device,
                dtype=dtype,
            )
        else:
            gate_input = torch.cat([z_detached, all_h], dim=-1)
            gate_logits = self.gate_logit_head(gate_input)
            relation_gate = F.softmax(gate_logits / self.tau_gate, dim=-1)

        # ----- Relation residual u_i = Σ_r g_i,r * Head_r(h_i,r) -----
        head_stack = torch.stack(head_out_per_rel, dim=1)
        u_i = (relation_gate * head_stack).sum(dim=-1)
        if self.residual_activation == "identity":
            # Unbounded ablation: Δ_rel = δ_max · u (still scaled, but linear).
            delta_rel = self.delta_rel_max * u_i
        else:
            delta_rel = self.delta_rel_max * torch.tanh(u_i)

        # Gate-weighted fused expert hidden — used to condition downstream code
        # that previously consumed it (LEQA in Commit 2 will reuse this).
        h_stack = torch.stack(h_per_rel, dim=1)
        fused_rel_h = (relation_gate.unsqueeze(-1) * h_stack).sum(dim=1)

        # Dominance signal for L_sparse.
        relation_strength = head_stack.abs()

        rel_only_logit = b_i + delta_rel
        final_logit = rel_only_logit

        # ----- Backward-compat zero tensors for deprecated judge outputs -----
        zero_n = torch.zeros(n, device=device, dtype=dtype)
        zero_n_bool = torch.zeros(n, device=device, dtype=torch.bool)

        return {
            "final_logit": final_logit,
            "rel_only_logit": rel_only_logit,
            "relation_gate": relation_gate,
            "delta_rel": delta_rel,
            "delta_llm": zero_n,
            "alpha_llm": zero_n,
            "judge_used_mask": zero_n_bool,
            "fused_rel_h": fused_rel_h,
            "relation_strength": relation_strength,
        }


__all__ = ["CoVERRelReasoner"]
