"""Phase2 unified CoVER-REL Reasoner.

Single forward path
-------------------

For each node ``i`` the final logit decomposes additively as::

    z_i = b_i + Δ_rel_i + α_i · Δ_llm_i

where:

* ``b_i = base_logit.detach()`` is the frozen base detector logit.
* ``h_i,r = Expert_r(E_i,r)`` is the per-relation hidden produced by an MLP
  expert from the 9-dim relation statistics ``E_i,r``.
* ``g_i = softmax(a_i / tau_gate, dim=-1)`` is a **softmax-normalised** schema
  gate over the ``R`` relations (so ``Σ_r g_i,r = 1``).  Unlike the legacy
  ``EvidenceReasoner`` (anchor + per-optional-relation sigmoid gates), this
  module shares one softmax over all relations.
* ``u_i = Σ_r g_i,r · Head_r(h_i,r)`` mixes the per-relation scalar residual
  contributions by the gate.
* ``Δ_rel_i = delta_rel_max · tanh(u_i)`` is the bounded relation residual.
* When ``use_judge`` is enabled and a judge is available for node ``i``::

      z_judge_i = JudgeEncoder(J_i)
      cond_i    = cat[z_judge_i, fused_h_i, g_i]
      α_i       = mask_judge_i · alpha_max · sigmoid(Head_alpha(cond_i))
      Δ_llm_i   = delta_llm_max · tanh(Head_llm(cond_i))

  ``mask_judge_i`` is 1 only when the judge packet was accepted; otherwise
  ``α_i`` is forcibly zero so the LLM path contributes nothing.

The module is intentionally additive and side-effect free relative to
``models/reasoner.py``; both can coexist without import-time clashes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    ``Δ_rel ≈ 0`` initially and lets the BWGNN base logit speak for itself.
    """
    head = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def _make_alpha_head(in_dim: int, hidden_dim: int, alpha_bias_init: float) -> nn.Sequential:
    """Judge gate head.  Last-layer bias init -3.0 ⇒ sigmoid(-3) ≈ 0.0474."""
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    final = head[-1]
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(alpha_bias_init))
    return head


def _make_llm_head(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """Judge residual head.  Zero-initialised so ``Δ_llm ≈ 0`` at start."""
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    final = head[-1]
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    return head


class CoVERRelReasoner(nn.Module):
    """Phase2 unified reasoner with softmax relation gate and optional LLM judge.

    Args:
        base_z_dim: Dimension of base detector embeddings ``base_z`` passed at forward.
        relation_names: Ordered list of relation names (uppercased internally).
            The order is used to split ``relation_features`` into per-relation
            chunks of size ``rel_stat_dim``.
        anchor_relation: Primary anchor relation (must be present in
            ``relation_names``).  Diagnostic only — the softmax gate is
            symmetric across all relations.  Downstream code may use this
            to pick a "default" relation for logs.
        rel_stat_dim: Per-relation statistic dim (default 9 to match
            ``RELATION_STAT_NAMES``).
        rel_hidden_dim: Per-relation expert output dim.
        rel_num_layers: Number of hidden layers in each expert MLP.
        rel_dropout: Dropout applied after the expert ``LayerNorm``.
        tau_gate: Softmax temperature for the schema gate.
        delta_rel_max: Tanh saturation bound for the relation residual.
        use_judge: If False, the judge encoder/heads are not constructed and
            ``α``/``Δ_llm`` are returned as zero tensors at forward.
        judge_feature_dim: Required when ``use_judge=True``; matches the
            second dim of the judge feature tensor.
        judge_hidden_dim: Hidden dim of the judge encoder and the
            ``α``/``Δ_llm`` head bottleneck.
        judge_dropout: Dropout applied after the judge encoder ``LayerNorm``.
        delta_llm_max: Tanh saturation bound for the LLM residual.
        alpha_max: Upper bound for the judge gate (after sigmoid).
        alpha_bias_init: Last-layer bias for the alpha head.  ``-3.0`` ⇒
            ``sigmoid(-3) ≈ 0.0474`` so the LLM path starts ~off.
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
        use_judge: bool = False,
        judge_feature_dim: int = 0,
        judge_hidden_dim: int = 32,
        judge_dropout: float = 0.30,
        delta_llm_max: float = 0.75,
        alpha_max: float = 0.10,
        alpha_bias_init: float = -3.0,
    ):
        super().__init__()
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
        self.use_judge = bool(use_judge)
        self.judge_feature_dim = int(judge_feature_dim)
        self.judge_hidden_dim = int(judge_hidden_dim)
        self.judge_dropout = float(judge_dropout)
        self.delta_llm_max = float(delta_llm_max)
        self.alpha_max = float(alpha_max)
        self.alpha_bias_init = float(alpha_bias_init)

        # ----- Per-relation experts and scalar residual heads -----
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
        # Zero init ⇒ initial logits = 0 ⇒ uniform 1/R gate, lets training
        # discover dominance from data instead of a baked-in prior.
        nn.init.zeros_(self.gate_logit_head.weight)
        nn.init.zeros_(self.gate_logit_head.bias)

        # ----- Optional judge path -----
        if self.use_judge:
            if self.judge_feature_dim <= 0:
                raise ValueError(
                    "judge_feature_dim must be > 0 when use_judge=True"
                )
            judge_layers: list[nn.Module] = [
                nn.Linear(self.judge_feature_dim, self.judge_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.judge_hidden_dim),
            ]
            if self.judge_dropout > 0:
                judge_layers.append(nn.Dropout(self.judge_dropout))
            self.judge_encoder = nn.Sequential(*judge_layers)

            cond_dim = (
                self.judge_hidden_dim + self.rel_hidden_dim + self.num_relations
            )
            self.head_alpha = _make_alpha_head(
                cond_dim, self.judge_hidden_dim, self.alpha_bias_init
            )
            self.head_llm = _make_llm_head(cond_dim, self.judge_hidden_dim)
        else:
            self.judge_encoder = None
            self.head_alpha = None
            self.head_llm = None

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        base_z: Tensor,
        base_logit: Tensor,
        relation_features: Tensor,
        judge_features: Tensor | None = None,
        judge_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the unified forward pass.

        Args:
            base_z: Base detector embeddings ``(N, base_z_dim)``.  Detached
                internally so no gradient flows back to the base detector.
            base_logit: Base detector logits ``(N,)`` or ``(N, 1)``.  Detached
                internally.
            relation_features: Concatenated per-relation 9-dim stats arranged
                in ``relation_names`` order, shape ``(N, R * rel_stat_dim)``.
            judge_features: Encoded judge features ``(N, judge_feature_dim)``
                or ``None`` when ``use_judge=False``.
            judge_mask: Boolean acceptance mask ``(N,)`` or ``None``.  When a
                node's mask is False, the alpha gate is forced to zero.

        Returns:
            A dict with at least::

                final_logit       (N,)
                rel_only_logit    (N,)            # b + Δ_rel
                relation_gate     (N, R)          # softmax-normalised
                delta_rel         (N,)
                delta_llm         (N,)            # zeros when use_judge=False
                alpha_llm         (N,)            # zeros for rejected/missing judge
                judge_used_mask   (N,) bool       # always False when use_judge=False
                fused_rel_h       (N, rel_hidden_dim)
                relation_strength (N, R)          # ||Head_r(h_i,r)||₂ (post-Head)
        """
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
        chunks = torch.split(rel_feats, self.rel_stat_dim, dim=1)

        # h_i,r (per-relation hidden) and Head_r(h_i,r) (per-relation scalar)
        h_per_rel: list[Tensor] = []
        head_out_per_rel: list[Tensor] = []
        for name, chunk in zip(self.relation_names, chunks):
            h = self.relation_experts[name](chunk)            # (N, rel_hidden_dim)
            h_per_rel.append(h)
            head_out_per_rel.append(self.relation_heads[name](h).view(-1))  # (N,)

        # ----- Schema gate g_i = softmax(a_i / tau_gate) -----
        all_h = torch.cat(h_per_rel, dim=-1)                   # (N, R * rel_hidden_dim)
        gate_input = torch.cat([z_detached, all_h], dim=-1)    # (N, base_z_dim + R*rel_hidden_dim)
        gate_logits = self.gate_logit_head(gate_input)         # (N, R)
        relation_gate = F.softmax(gate_logits / self.tau_gate, dim=-1)  # (N, R)

        # ----- Relation residual u_i = Σ_r g_i,r * Head_r(h_i,r) -----
        head_stack = torch.stack(head_out_per_rel, dim=1)      # (N, R)
        u_i = (relation_gate * head_stack).sum(dim=-1)         # (N,)
        delta_rel = self.delta_rel_max * torch.tanh(u_i)       # (N,)

        # Gate-weighted fused expert hidden — used to condition α / Δ_llm.
        h_stack = torch.stack(h_per_rel, dim=1)                # (N, R, rel_hidden_dim)
        fused_rel_h = (relation_gate.unsqueeze(-1) * h_stack).sum(dim=1)  # (N, rel_hidden_dim)

        # Dominance signal for L_sparse.  Head_r outputs a scalar so the L2 norm
        # collapses to the absolute value; this still measures the magnitude of
        # the per-relation contribution to u_i (before gate weighting).
        relation_strength = head_stack.abs()                   # (N, R)

        rel_only_logit = b_i + delta_rel                       # (N,)

        # ----- Optional judge path -----
        alpha_llm = torch.zeros(n, device=device, dtype=dtype)
        delta_llm = torch.zeros(n, device=device, dtype=dtype)
        judge_used_mask = torch.zeros(n, device=device, dtype=torch.bool)

        if self.use_judge:
            if judge_features is None:
                judge_features = torch.zeros(
                    n, self.judge_feature_dim, device=device, dtype=dtype
                )
            if judge_features.shape[1] != self.judge_feature_dim:
                raise ValueError(
                    f"judge_features dim mismatch: "
                    f"{judge_features.shape[1]} != judge_feature_dim={self.judge_feature_dim}"
                )
            j_feats = judge_features.to(device=device, dtype=dtype)

            if judge_mask is None:
                mask_bool = torch.zeros(n, device=device, dtype=torch.bool)
            else:
                mask_bool = judge_mask.to(device=device).view(-1).to(torch.bool)
            judge_used_mask = mask_bool
            mask_f = mask_bool.to(dtype=dtype).view(-1, 1)     # (N, 1)

            z_judge = self.judge_encoder(j_feats)              # (N, judge_hidden_dim)
            cond = torch.cat(
                [z_judge, fused_rel_h, relation_gate], dim=-1
            )                                                  # (N, judge_h + rel_h + R)

            alpha_raw = self.head_alpha(cond)                  # (N, 1)
            # mask_f forces α=0 for rejected/missing judge.
            alpha_llm = (
                self.alpha_max * torch.sigmoid(alpha_raw) * mask_f
            ).view(-1)                                         # (N,)

            delta_raw = self.head_llm(cond)                    # (N, 1)
            delta_llm = (
                self.delta_llm_max * torch.tanh(delta_raw)
            ).view(-1)                                         # (N,)

        final_logit = rel_only_logit + alpha_llm * delta_llm   # (N,)

        return {
            "final_logit": final_logit,
            "rel_only_logit": rel_only_logit,
            "relation_gate": relation_gate,
            "delta_rel": delta_rel,
            "delta_llm": delta_llm,
            "alpha_llm": alpha_llm,
            "judge_used_mask": judge_used_mask,
            "fused_rel_h": fused_rel_h,
            "relation_strength": relation_strength,
        }


__all__ = ["CoVERRelReasoner"]
