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

from evidence.vocab import (
    get_num_values,
    get_reason_types,
    get_evidence_slots,
    get_direction_num_classes,
)

VALID_GATE_MODES = ("signed_diff_legacy", "safe_residual", "direct_tanh", "aux_only")
VALID_RELATION_FUSION_MODES = ("single", "concat", "anchor_gate", "base_additive_gate")
VALID_LLM_FUSION_MODES = ("none", "gated_llm_residual")


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
        latent_dim: int = 256,
        relation_dim: int = 0,
        relation_hidden_dim: int = 32,
        relation_fusion_mode: str = "concat",
        relation_names: list[str] | None = None,
        relation_stat_dim: int | None = None,
        anchor_relation: str | None = None,
        optional_relations: list[str] | None = None,
        relation_dropout: float = 0.0,
        gate_hidden_dim: int = 64,
        gate_temperature: float = 1.0,
        use_llm_judge: bool = False,
        judge_feature_dim: int = 0,
        fusion_mode: str = "none",
        llm_delta_scale: float = 1.0,
        alpha_max: float = 1.0,
        strength_aware_alpha: bool = False,
        judge_hidden_dim: int = 32,
    ):
        super().__init__()
        if gate_mode not in VALID_GATE_MODES:
            raise ValueError(f"Unknown gate_mode={gate_mode!r}. Must be one of {VALID_GATE_MODES}")
        if relation_fusion_mode not in VALID_RELATION_FUSION_MODES:
            raise ValueError(
                f"Unknown relation_fusion_mode={relation_fusion_mode!r}. "
                f"Must be one of {VALID_RELATION_FUSION_MODES}"
            )
        if fusion_mode not in VALID_LLM_FUSION_MODES:
            raise ValueError(f"Unknown fusion_mode={fusion_mode!r}. Must be one of {VALID_LLM_FUSION_MODES}")

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
        self.latent_dim = latent_dim
        self.relation_dim = int(relation_dim)
        self.relation_fusion_mode = relation_fusion_mode
        self.relation_names = [str(name).upper() for name in (relation_names or [])]
        self.relation_stat_dim = int(relation_stat_dim or 0)
        self.anchor_relation = str(anchor_relation).upper() if anchor_relation else None
        self.optional_relations = [str(name).upper() for name in (optional_relations or [])]
        self.relation_dropout = float(relation_dropout)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.use_llm_judge = bool(use_llm_judge)
        self.judge_feature_dim = int(judge_feature_dim)
        self.fusion_mode = fusion_mode
        self.llm_delta_scale = float(llm_delta_scale)
        self.alpha_max = float(alpha_max)
        self.strength_aware_alpha = bool(strength_aware_alpha)

        self.evidence_encoder = EvidenceEncoder(num_values, num_slots, evidence_emb_dim, 64)
        relation_out_dim = 0
        if self.relation_dim > 0:
            relation_out_dim = int(relation_hidden_dim)
            if relation_fusion_mode in ("anchor_gate", "base_additive_gate"):
                self._init_relation_gate_modules(
                    z_dim=z_dim,
                    relation_hidden_dim=relation_out_dim,
                    gate_hidden_dim=gate_hidden_dim,
                )
            else:
                self.relation_encoder = nn.Sequential(
                    nn.Linear(self.relation_dim, relation_out_dim),
                    nn.ReLU(),
                    nn.LayerNorm(relation_out_dim),
                )
        in_dim = z_dim + 64 + relation_out_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.type_head = nn.Linear(hidden_dim, num_types)
        self.pos_head = nn.Linear(hidden_dim, num_slots)
        self.neg_head = nn.Linear(hidden_dim, num_slots)
        self.direction_head = nn.Linear(hidden_dim, get_direction_num_classes())
        self.strength_head = nn.Linear(hidden_dim, 3)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

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

        if self.use_llm_judge:
            if self.judge_feature_dim <= 0:
                raise ValueError("judge_feature_dim must be positive when use_llm_judge=True")
            self.judge_encoder = nn.Sequential(
                nn.Linear(self.judge_feature_dim, judge_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(judge_hidden_dim),
            )
            self.llm_delta_head = nn.Linear(judge_hidden_dim, 1)
            self.llm_alpha_head = nn.Sequential(
                nn.Linear(hidden_dim + judge_hidden_dim + 11, judge_hidden_dim),
                nn.ReLU(),
                nn.Linear(judge_hidden_dim, 1),
            )
            self.judge_only_head = nn.Linear(judge_hidden_dim, 1)
            nn.init.zeros_(self.llm_delta_head.weight)
            nn.init.zeros_(self.llm_delta_head.bias)
            final = self.llm_alpha_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, -2.0)

    def _init_relation_gate_modules(
        self,
        z_dim: int,
        relation_hidden_dim: int,
        gate_hidden_dim: int,
    ) -> None:
        if not self.relation_names:
            raise ValueError("relation_names are required for relation gate fusion modes")
        if self.relation_dim % len(self.relation_names) != 0:
            raise ValueError(
                f"relation_dim={self.relation_dim} is not divisible by "
                f"{len(self.relation_names)} relation_names"
            )
        inferred_stat_dim = self.relation_dim // len(self.relation_names)
        if self.relation_stat_dim <= 0:
            self.relation_stat_dim = inferred_stat_dim
        if self.relation_stat_dim * len(self.relation_names) != self.relation_dim:
            raise ValueError(
                "relation_stat_dim * num_relations must equal relation_dim: "
                f"{self.relation_stat_dim} * {len(self.relation_names)} != {self.relation_dim}"
            )

        self.relation_experts = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.relation_stat_dim, relation_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(relation_hidden_dim),
            )
            for name in self.relation_names
        })

        if self.relation_fusion_mode == "anchor_gate":
            if self.anchor_relation is None:
                raise ValueError("anchor_relation is required for anchor_gate fusion")
            if self.anchor_relation not in self.relation_names:
                raise ValueError(
                    f"anchor_relation={self.anchor_relation!r} not in relation_names={self.relation_names}"
                )
            gate_names = self.optional_relations or [
                name for name in self.relation_names if name != self.anchor_relation
            ]
            unknown = [name for name in gate_names if name not in self.relation_names]
            if unknown:
                raise ValueError(f"Unknown optional relation(s): {unknown}")
            self.optional_relations = [name for name in gate_names if name != self.anchor_relation]
            gate_input_dim = z_dim + 2 * relation_hidden_dim + self.relation_stat_dim
            self.relation_gate_names = self.optional_relations
        else:
            gate_input_dim = z_dim + relation_hidden_dim + self.relation_stat_dim
            self.relation_gate_names = self.relation_names

        self.relation_gate_heads = nn.ModuleDict({
            name: self._make_relation_gate_head(gate_input_dim, gate_hidden_dim)
            for name in self.relation_gate_names
        })

    @staticmethod
    def _make_relation_gate_head(in_dim: int, hidden_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        final = head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -2.0)
        return head

    def forward(
        self,
        z: Tensor,
        base_logit: Tensor,
        evidence_token_ids: Tensor,
        relation_features: Tensor | None = None,
        judge_features: Tensor | None = None,
        judge_mask: Tensor | None = None,
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
        parts = [z, g]
        relation_gate_values = None
        relation_contribution_norms = None
        relation_gate_sparse_loss = z.new_tensor(0.0)
        relation_fused = None
        if self.relation_dim > 0:
            if relation_features is None:
                relation_features = z.new_zeros((z.shape[0], self.relation_dim))
            if relation_features.shape[1] != self.relation_dim:
                raise ValueError(
                    f"relation_features dim mismatch: {relation_features.shape[1]} != {self.relation_dim}"
                )
            rel_features = relation_features.to(device=z.device, dtype=z.dtype)
            if self.relation_fusion_mode in ("anchor_gate", "base_additive_gate"):
                (
                    relation_fused,
                    relation_gate_values,
                    relation_contribution_norms,
                    relation_gate_sparse_loss,
                ) = self._fuse_relation_experts(z, rel_features)
                parts.append(relation_fused)
            else:
                relation_fused = self.relation_encoder(rel_features)
                parts.append(relation_fused)
        h = self.shared(torch.cat(parts, dim=-1))

        type_logits = self.type_head(h)
        pos_logits = self.pos_head(h)
        neg_logits = self.neg_head(h)
        direction_logits = self.direction_head(h)
        strength_logits = self.strength_head(h)
        z_student = self.latent_head(h)

        base = base_logit.detach().view(-1, 1)

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

        rel_only_logit = final_logit.view(-1)
        final_logit = rel_only_logit
        alpha_llm = None
        delta_llm = None
        llm_residual = None
        judge_only_logit = None

        if self.use_llm_judge and self.fusion_mode == "gated_llm_residual":
            if judge_features is None:
                judge_features = z.new_zeros((z.shape[0], self.judge_feature_dim))
            if judge_features.shape[1] != self.judge_feature_dim:
                raise ValueError(
                    f"judge_features dim mismatch: {judge_features.shape[1]} != {self.judge_feature_dim}"
                )
            if judge_mask is None:
                judge_mask = z.new_zeros(z.shape[0], dtype=torch.bool)
            judge_features = judge_features.to(device=z.device, dtype=z.dtype)
            judge_mask_f = judge_mask.to(device=z.device).view(-1, 1).to(dtype=z.dtype)
            z_judge = self.judge_encoder(judge_features)
            agreement = self._judge_agreement_features(rel_only_logit, judge_features)
            delta_llm = self.llm_delta_scale * torch.tanh(self.llm_delta_head(z_judge))
            alpha_llm = torch.sigmoid(self.llm_alpha_head(torch.cat([h, z_judge, agreement], dim=-1)))
            alpha_llm = alpha_llm * self.alpha_max
            if self.strength_aware_alpha:
                alpha_llm = alpha_llm * self._judge_strength_factor(judge_features)
            alpha_llm = alpha_llm * judge_mask_f
            llm_residual = alpha_llm * delta_llm
            judge_only_logit = self.judge_only_head(z_judge).view(-1)
            final_logit = rel_only_logit + llm_residual.view(-1)

        final_logit = final_logit.view(-1)
        residual_shift = final_logit - base.view(-1)

        outputs: dict[str, Tensor] = {
            "final_logit": final_logit,
            "type_logits": type_logits,
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "direction_logits": direction_logits,
            "strength_logits": strength_logits,
            "z_student": z_student,
            "residual": residual_shift,
            "rel_only_logit": rel_only_logit,
            # CoVER-LIFT canonical aliases.  Keep legacy names above for
            # existing CV-SCD scripts and checkpoints.
            "risk_type_logits": type_logits,
            "support_mask_logits": pos_logits,
            "counter_mask_logits": neg_logits,
        }
        if relation_gate_values is not None:
            outputs["relation_gate_values"] = relation_gate_values
            outputs["relation_contribution_norms"] = relation_contribution_norms
            outputs["relation_gate_sparse_loss"] = relation_gate_sparse_loss
            outputs["relation_fused"] = relation_fused
        if alpha_llm is not None and delta_llm is not None and llm_residual is not None:
            outputs["alpha_llm"] = alpha_llm.view(-1)
            outputs["delta_llm"] = delta_llm.view(-1)
            outputs["llm_residual"] = llm_residual.view(-1)
            outputs["judge_only_logit"] = judge_only_logit if judge_only_logit is not None else final_logit

        if return_debug:
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

    def _judge_agreement_features(self, rel_only_logit: Tensor, judge_features: Tensor) -> Tensor:
        verdict = judge_features[:, :3] if judge_features.shape[1] >= 3 else judge_features.new_zeros((judge_features.shape[0], 3))
        strength = judge_features[:, 3:6] if judge_features.shape[1] >= 6 else judge_features.new_zeros((judge_features.shape[0], 3))
        key_primary = judge_features[:, 6:7] if judge_features.shape[1] >= 7 else judge_features.new_zeros((judge_features.shape[0], 1))
        support_count = judge_features[:, 7:8] if judge_features.shape[1] >= 8 else judge_features.new_zeros((judge_features.shape[0], 1))
        counter_count = judge_features[:, 8:9] if judge_features.shape[1] >= 9 else judge_features.new_zeros((judge_features.shape[0], 1))
        rel_fake = (rel_only_logit.view(-1, 1) >= 0).to(dtype=judge_features.dtype)
        agree = verdict[:, 0:1] * rel_fake + verdict[:, 1:2] * (1.0 - rel_fake)
        return torch.cat([verdict, rel_fake, agree, strength, key_primary, support_count, counter_count], dim=-1)

    def _judge_strength_factor(self, judge_features: Tensor) -> Tensor:
        verdict = judge_features[:, :3] if judge_features.shape[1] >= 3 else judge_features.new_zeros((judge_features.shape[0], 3))
        strength = judge_features[:, 3:6] if judge_features.shape[1] >= 6 else judge_features.new_zeros((judge_features.shape[0], 3))
        weak = strength[:, 0:1]
        moderate = strength[:, 1:2]
        strong = strength[:, 2:3]
        factor = 0.3 * weak + 0.6 * moderate + strong
        factor = torch.where(verdict[:, 2:3] > 0.5, factor.new_full(factor.shape, 0.2), factor)
        no_strength = strength.sum(dim=1, keepdim=True) <= 0
        return torch.where(no_strength, factor.new_ones(factor.shape), factor)

    def _fuse_relation_experts(
        self,
        z: Tensor,
        relation_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chunks = torch.split(relation_features, self.relation_stat_dim, dim=1)
        expert_by_name = {
            name: self.relation_experts[name](chunk)
            for name, chunk in zip(self.relation_names, chunks)
        }
        feature_by_name = {
            name: chunk
            for name, chunk in zip(self.relation_names, chunks)
        }
        gate_values = z.new_zeros((z.shape[0], len(self.relation_names)))
        contribution_norms = z.new_zeros((z.shape[0], len(self.relation_names)))
        sparse_gates: list[Tensor] = []

        if self.relation_fusion_mode == "anchor_gate":
            assert self.anchor_relation is not None
            fused = expert_by_name[self.anchor_relation]
            anchor_idx = self.relation_names.index(self.anchor_relation)
            gate_values[:, anchor_idx] = 1.0
            contribution_norms[:, anchor_idx] = fused.norm(dim=1)
            anchor_h = expert_by_name[self.anchor_relation]

            for name in self.optional_relations:
                rel_h = expert_by_name[name]
                raw_features = feature_by_name[name]
                gate_input = torch.cat([z, anchor_h, rel_h, raw_features], dim=-1)
                gate = torch.sigmoid(self.relation_gate_heads[name](gate_input) / self.gate_temperature)
                gate = self._apply_relation_dropout(gate, allow_drop=True)
                contribution = gate * rel_h
                rel_idx = self.relation_names.index(name)
                fused = fused + contribution
                gate_values[:, rel_idx] = gate.view(-1)
                contribution_norms[:, rel_idx] = contribution.norm(dim=1)
                sparse_gates.append(gate)

        elif self.relation_fusion_mode == "base_additive_gate":
            fused = z.new_zeros((z.shape[0], next(iter(expert_by_name.values())).shape[1]))
            for name in self.relation_names:
                rel_h = expert_by_name[name]
                raw_features = feature_by_name[name]
                gate_input = torch.cat([z, rel_h, raw_features], dim=-1)
                gate = torch.sigmoid(self.relation_gate_heads[name](gate_input) / self.gate_temperature)
                gate = self._apply_relation_dropout(gate, allow_drop=True)
                contribution = gate * rel_h
                rel_idx = self.relation_names.index(name)
                fused = fused + contribution
                gate_values[:, rel_idx] = gate.view(-1)
                contribution_norms[:, rel_idx] = contribution.norm(dim=1)
                sparse_gates.append(gate)
        else:
            raise ValueError(f"Unsupported relation_fusion_mode={self.relation_fusion_mode}")

        if sparse_gates:
            sparse_loss = torch.cat(sparse_gates, dim=1).sum(dim=1).mean()
        else:
            sparse_loss = z.new_tensor(0.0)
        return fused, gate_values, contribution_norms, sparse_loss

    def _apply_relation_dropout(self, gate: Tensor, allow_drop: bool) -> Tensor:
        if not allow_drop or not self.training or self.relation_dropout <= 0:
            return gate
        keep = torch.rand_like(gate) >= self.relation_dropout
        return gate * keep.to(dtype=gate.dtype)
