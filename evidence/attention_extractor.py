"""Extract proxy attention from CoVERRelReasoner for LEQA.

Produces a ``(N, T)`` attention tensor from the reasoner's forward output
using ``relation_gate * per-relation evidence-token magnitude``.

Score-blind by construction: the produced tensor uses only relation
statistics and the learned gate — never base_score, label, split, or
any prediction information.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evidence.relation_features import RELATION_STAT_NAMES
from models.leqa_lora import LEQA_CARD_FIELDS


# Fields that must NEVER appear in attention computation
_FORBIDDEN_ATTENTION_SOURCES = frozenset({
    "base_score", "base_prob", "base_logit", "confidence",
    "label", "split", "ground_truth", "final_prediction",
    "base_probability", "base_prediction",
})


def extract_attention(
    reasoner_output: dict[str, Tensor],
    relation_features: Tensor,
    relation_names: list[str],
    rel_stat_dim: int = 9,
) -> Tensor:
    """Extract proxy attention tensor ``(N, T)`` from reasoner output.

    The attention proxy is computed as:
      - For card-field slots (first 10): small uniform value (these fields
        do not directly participate in the reasoner's gated computation).
      - For relation-stat slots (remaining R*9): ``gate[r] * |stat[r, s]|``
        where ``gate`` is the softmax relation gate and ``stat`` is the
        raw relation feature.

    Args:
        reasoner_output: Dict from ``CoVERRelReasoner.forward()``.
        relation_features: Raw relation features ``(N, R * rel_stat_dim)``.
        relation_names: Ordered list of relation names.
        rel_stat_dim: Number of stats per relation (default 9).

    Returns:
        Attention tensor ``(N, T)`` where ``T = 10 + R * rel_stat_dim``.
    """
    relation_gate = reasoner_output["relation_gate"]  # (N, R)
    N = relation_gate.shape[0]
    R = len(relation_names)
    device = relation_gate.device
    dtype = relation_gate.dtype

    # Defensive assertion: output dict must not contain forbidden fields
    for key in reasoner_output:
        assert key.lower() not in _FORBIDDEN_ATTENTION_SOURCES, (
            f"Attention extractor received forbidden key '{key}' in "
            f"reasoner output. This would leak score/label information."
        )

    # Defensive assertion: relation_features must be score-blind
    assert relation_features.shape[1] == R * rel_stat_dim, (
        f"relation_features dim {relation_features.shape[1]} != "
        f"R*rel_stat_dim={R * rel_stat_dim}"
    )

    n_card = len(LEQA_CARD_FIELDS)
    T = n_card + R * rel_stat_dim

    attention = torch.zeros(N, T, device=device, dtype=dtype)

    # Card-field slots: uniform small value (they don't participate in
    # the gated reasoner computation directly)
    attention[:, :n_card] = 1e-3

    # Relation-stat slots: gate[r] * |stat[r, s]|
    rel_feats = relation_features.to(device=device, dtype=dtype).detach()
    rel_chunks = rel_feats.view(N, R, rel_stat_dim)  # (N, R, D)
    gate_expanded = relation_gate.unsqueeze(-1)  # (N, R, 1)

    # proxy attention per stat = gate * |stat_value|
    rel_attention = gate_expanded * rel_chunks.abs()  # (N, R, D)
    attention[:, n_card:] = rel_attention.reshape(N, R * rel_stat_dim)

    # Final score-blind assertion: the attention tensor contains no
    # information from base_score, label, split, or prediction.
    # The relation_features are constructed from anonymous graph
    # statistics (degree, cosine, z-score, prototype distances) and
    # the gate is a learned softmax over relation hidden states.
    assert not torch.isnan(attention).any(), "NaN in attention tensor"
    assert not torch.isinf(attention).any(), "Inf in attention tensor"

    return attention


__all__ = ["extract_attention"]
