"""Error-aware trace sampler for Stage2 evidence generation.

Sampling pools (with default ratios):
1. train_false_negative  (0.30): train node, y=1, base_prob < 0.5
2. train_false_positive  (0.20): train node, y=0, base_prob >= 0.5
3. train_high_loss       (0.20): train node, high BCE loss
4. val_boundary          (0.15): val node, base_prob near 0.5 (logits only, no labels)
5. high_conf_fraud       (0.075): train/val, high structural fraud evidence (no test)
6. high_conf_benign      (0.075): train/val, high benign consistency (no test)

HARD CONSTRAINTS:
- Trace selection CAN use train labels.
- val_boundary can only use val logits/uncertainty, NOT val labels as correction target.
- Test labels CANNOT be used at all.
- base_score/prob/logit only for calibration channel and trace selection,
  NOT in teacher_payload.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


DEFAULT_RATIOS = {
    "train_false_negative": 0.30,
    "train_false_positive": 0.20,
    "train_high_loss": 0.20,
    "val_boundary": 0.15,
    "high_conf_fraud_candidate": 0.075,
    "high_conf_benign_candidate": 0.075,
}

COUNTER_FOCUSED_RATIOS = {
    "train_base_fp": 0.35,
    "train_benign_high_loss": 0.20,
    "train_benign_dominant_payload": 0.20,
    "high_base_prob_benign_train": 0.10,
    "val_boundary_benign_like": 0.15,
}


def _bce_loss_per_node(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-node BCE loss."""
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    return -(labels * probs.log() + (1 - labels) * (1 - probs).log())


def _neighbor_label_consistency(
    edge_index: torch.Tensor, y: torch.Tensor, num_nodes: int,
) -> torch.Tensor:
    """Fraction of neighbors sharing the same label (uses labels, not scores)."""
    row, col = edge_index
    same = (y[row] == y[col]).float()
    consistency = torch.zeros(num_nodes)
    count = torch.zeros(num_nodes)
    consistency.scatter_add_(0, row, same)
    count.scatter_add_(0, row, torch.ones(row.shape[0]))
    return consistency / count.clamp(min=1)


def _high_freq_anomaly_score(extras: dict[str, torch.Tensor] | None, num_nodes: int) -> torch.Tensor:
    """Extract high-frequency response as fraud signal. Higher = more anomalous."""
    if extras is None:
        return torch.zeros(num_nodes)
    if "high_freq_response" in extras:
        return extras["high_freq_response"]
    return torch.zeros(num_nodes)


def sample_traces(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    edge_index: torch.Tensor,
    trace_size: int,
    seed: int,
    ratios: dict[str, float] | None = None,
    extras: dict[str, torch.Tensor] | None = None,
    output_dir: Path | None = None,
) -> tuple[list[int], dict]:
    """Select trace nodes using error-aware sampling.

    Returns (trace_node_ids, stats_dict).
    """
    ratios = ratios or DEFAULT_RATIOS
    rng = np.random.RandomState(seed)
    num_nodes = y.shape[0]
    base_probs = torch.sigmoid(base_logits)

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]

    y_train = y[train_idx]
    probs_train = base_probs[train_idx]

    # Pool 1: train false negatives (y=1, prob < 0.5)
    fn_mask = (y_train == 1) & (probs_train < 0.5)
    fn_pool = train_idx[fn_mask].tolist()

    # Pool 2: train false positives (y=0, prob >= 0.5)
    fp_mask = (y_train == 0) & (probs_train >= 0.5)
    fp_pool = train_idx[fp_mask].tolist()

    # Pool 3: train high BCE loss
    train_losses = _bce_loss_per_node(base_logits[train_idx], y[train_idx].float())
    loss_thresh = torch.quantile(train_losses, 0.75).item()
    high_loss_mask = train_losses >= loss_thresh
    hl_pool = train_idx[high_loss_mask].tolist()

    # Pool 4: val boundary (use val logits only, NOT val labels)
    val_probs = base_probs[val_idx]
    boundary_width = 0.15
    boundary_mask = (val_probs >= 0.5 - boundary_width) & (val_probs <= 0.5 + boundary_width)
    vb_pool = val_idx[boundary_mask].tolist()

    # Pool 5: high-confidence fraud candidates (structural evidence, no test)
    hfr_score = _high_freq_anomaly_score(extras, num_nodes)
    neighbor_cons = _neighbor_label_consistency(edge_index, y, num_nodes)
    candidate_mask = train_mask | val_mask
    candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
    fraud_signal = hfr_score[candidate_idx] * (1 - neighbor_cons[candidate_idx])
    if fraud_signal.numel() > 0:
        fraud_thresh = torch.quantile(fraud_signal, 0.90).item()
        fraud_mask = fraud_signal >= fraud_thresh
        hcf_pool = candidate_idx[fraud_mask].tolist()
    else:
        hcf_pool = []

    # Pool 6: high-confidence benign candidates (no test)
    benign_signal = neighbor_cons[candidate_idx] * (1 - hfr_score[candidate_idx])
    if benign_signal.numel() > 0:
        benign_thresh = torch.quantile(benign_signal, 0.90).item()
        benign_mask = benign_signal >= benign_thresh
        hcb_pool = candidate_idx[benign_mask].tolist()
    else:
        hcb_pool = []

    pools = {
        "train_false_negative": fn_pool,
        "train_false_positive": fp_pool,
        "train_high_loss": hl_pool,
        "val_boundary": vb_pool,
        "high_conf_fraud_candidate": hcf_pool,
        "high_conf_benign_candidate": hcb_pool,
    }

    # Allocate slots per pool
    trace_set: set[int] = set()
    allocation = {}
    actual_drawn = {}

    for pool_name, pool in pools.items():
        n_alloc = max(0, round(trace_size * ratios[pool_name]))
        allocation[pool_name] = n_alloc

        available = [n for n in pool if n not in trace_set]
        rng.shuffle(available)
        drawn = available[:n_alloc]
        trace_set.update(drawn)
        actual_drawn[pool_name] = drawn

    # If under-allocated, fill remainder from train pool
    remainder = trace_size - len(trace_set)
    if remainder > 0:
        train_list = train_idx.tolist()
        available = [n for n in train_list if n not in trace_set]
        rng.shuffle(available)
        extra = available[:remainder]
        trace_set.update(extra)
        actual_drawn.setdefault("_remainder", extra)

    trace_nodes = sorted(trace_set)

    # Build stats
    stats = {
        "trace_size_requested": trace_size,
        "trace_size_actual": len(trace_nodes),
        "seed": seed,
        "ratios_used": ratios,
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "pool_allocation": allocation,
        "pool_drawn": {k: len(v) for k, v in actual_drawn.items()},
        "val_boundary_used_val_labels": False,
        "test_labels_used": False,
        "train_labels_used_for": ["train_false_negative", "train_false_positive", "train_high_loss"],
        "structural_signals_used_for": ["high_conf_fraud_candidate", "high_conf_benign_candidate"],
        "base_prob_used_for": "trace_selection_only_not_teacher_payload",
    }

    if output_dir is not None:
        from utils.paths import ensure_dir
        ensure_dir(output_dir)
        with open(output_dir / "trace_sampling_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    return trace_nodes, stats


def _draw_from_pools(
    pools: dict[str, list[int]],
    ratios: dict[str, float],
    trace_size: int,
    rng: np.random.RandomState,
) -> tuple[list[int], dict[str, int], dict[str, int], dict[str, list[int]]]:
    trace_set: set[int] = set()
    allocation: dict[str, int] = {}
    actual_drawn: dict[str, list[int]] = {}

    for pool_name, pool in pools.items():
        n_alloc = max(0, round(trace_size * ratios.get(pool_name, 0.0)))
        allocation[pool_name] = n_alloc
        available = [n for n in pool if n not in trace_set]
        rng.shuffle(available)
        drawn = available[:n_alloc]
        trace_set.update(drawn)
        actual_drawn[pool_name] = drawn

    remainder = trace_size - len(trace_set)
    if remainder > 0:
        fallback: list[int] = []
        for pool in pools.values():
            fallback.extend(pool)
        available = [n for n in fallback if n not in trace_set]
        rng.shuffle(available)
        extra = available[:remainder]
        trace_set.update(extra)
        actual_drawn["_remainder"] = extra

    return sorted(trace_set), allocation, {k: len(v) for k, v in actual_drawn.items()}, actual_drawn


def build_counter_focused_candidates(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    trace_size: int,
    seed: int,
    oversample_factor: int = 5,
) -> tuple[list[int], dict]:
    """Build a score/label-based candidate set for later polarity filtering.

    Uses train labels and base probabilities only for trace selection. It does
    not inspect validation labels and never touches test nodes.
    """
    rng = np.random.RandomState(seed)
    base_probs = torch.sigmoid(base_logits)
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]
    y_train = y[train_idx]
    probs_train = base_probs[train_idx]

    benign_train = train_idx[y_train == 0]
    benign_probs = base_probs[benign_train]
    train_losses = _bce_loss_per_node(base_logits[benign_train], torch.zeros_like(benign_probs))

    fp_pool = train_idx[(y_train == 0) & (probs_train >= 0.5)].tolist()

    if train_losses.numel() > 0:
        loss_thresh = torch.quantile(train_losses, 0.75).item()
        benign_high_loss_pool = benign_train[train_losses >= loss_thresh].tolist()
    else:
        benign_high_loss_pool = []

    if benign_probs.numel() > 0:
        high_prob_thresh = torch.quantile(benign_probs, 0.80).item()
        high_prob_benign_pool = benign_train[benign_probs >= high_prob_thresh].tolist()
    else:
        high_prob_benign_pool = []

    val_probs = base_probs[val_idx]
    boundary_width = 0.20
    val_boundary_pool = val_idx[(val_probs >= 0.5 - boundary_width) & (val_probs <= 0.5 + boundary_width)].tolist()

    pools = {
        "train_base_fp": fp_pool,
        "train_benign_high_loss": benign_high_loss_pool,
        "high_base_prob_benign_train": high_prob_benign_pool,
        "val_boundary": val_boundary_pool,
    }
    candidate_size = max(trace_size, trace_size * oversample_factor)
    candidate_ratios = {
        "train_base_fp": 0.35,
        "train_benign_high_loss": 0.25,
        "high_base_prob_benign_train": 0.20,
        "val_boundary": 0.20,
    }
    candidates, allocation, drawn_counts, drawn_nodes = _draw_from_pools(
        pools, candidate_ratios, candidate_size, rng,
    )
    stats = {
        "candidate_size_requested": candidate_size,
        "candidate_size_actual": len(candidates),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "candidate_pool_allocation": allocation,
        "candidate_pool_drawn": drawn_counts,
        "candidate_pool_drawn_nodes": drawn_nodes,
        "test_labels_used": False,
        "val_labels_used": False,
        "train_labels_used_for": ["train_base_fp", "train_benign_high_loss", "high_base_prob_benign_train"],
        "base_prob_used_for": [
            "train_base_fp",
            "train_benign_high_loss",
            "high_base_prob_benign_train",
            "val_boundary_boundary_only",
        ],
    }
    return candidates, stats


def sample_counter_focused_traces(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    trace_size: int,
    seed: int,
    polarity_by_node: dict[int, str],
    candidate_nodes: list[int],
    ratios: dict[str, float] | None = None,
    output_dir: Path | None = None,
    candidate_stats: dict | None = None,
) -> tuple[list[int], dict]:
    """Draw a counter-focused trace after card polarity is available."""
    ratios = ratios or COUNTER_FOCUSED_RATIOS
    rng = np.random.RandomState(seed)
    candidate_set = set(candidate_nodes)
    base_probs = torch.sigmoid(base_logits)
    train_idx = [int(i) for i in train_mask.nonzero(as_tuple=True)[0].tolist() if int(i) in candidate_set]
    val_idx = [int(i) for i in val_mask.nonzero(as_tuple=True)[0].tolist() if int(i) in candidate_set]

    def is_benign_train(n: int) -> bool:
        return bool(train_mask[n]) and int(y[n].item()) == 0

    train_base_fp = [n for n in train_idx if is_benign_train(n) and base_probs[n].item() >= 0.5]

    benign_train = [n for n in train_idx if is_benign_train(n)]
    if benign_train:
        losses = _bce_loss_per_node(base_logits[benign_train], torch.zeros(len(benign_train)))
        thresh = torch.quantile(losses, 0.75).item()
        train_benign_high_loss = [n for n, loss in zip(benign_train, losses.tolist()) if loss >= thresh]
        probs = base_probs[benign_train]
        prob_thresh = torch.quantile(probs, 0.80).item()
        high_base_prob_benign_train = [n for n in benign_train if base_probs[n].item() >= prob_thresh]
    else:
        train_benign_high_loss = []
        high_base_prob_benign_train = []

    train_benign_dominant_payload = [
        n for n in benign_train
        if polarity_by_node.get(n) == "benign_dominant"
    ]

    val_boundary_benign_like = [
        n for n in val_idx
        if 0.30 <= base_probs[n].item() <= 0.70
        and polarity_by_node.get(n) in ("benign_dominant", "mixed")
    ]

    pools = {
        "train_base_fp": train_base_fp,
        "train_benign_high_loss": train_benign_high_loss,
        "train_benign_dominant_payload": train_benign_dominant_payload,
        "high_base_prob_benign_train": high_base_prob_benign_train,
        "val_boundary_benign_like": val_boundary_benign_like,
    }
    trace_nodes, allocation, drawn_counts, drawn_nodes = _draw_from_pools(pools, ratios, trace_size, rng)
    stats = {
        "trace_sampler_mode": "counter_focused",
        "trace_size_requested": trace_size,
        "trace_size_actual": len(trace_nodes),
        "seed": seed,
        "ratios_used": ratios,
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "pool_allocation": allocation,
        "pool_drawn": drawn_counts,
        "pool_drawn_nodes": drawn_nodes,
        "candidate_stats": candidate_stats or {},
        "payload_polarity_counts": dict(Counter(polarity_by_node.get(n, "unknown") for n in candidate_nodes)),
        "test_labels_used": False,
        "val_labels_used": False,
        "train_labels_used_for": ["train_base_fp", "train_benign_high_loss", "train_benign_dominant_payload", "high_base_prob_benign_train"],
        "base_prob_used_for": ["trace_selection_only_not_teacher_payload"],
        "payload_polarity_used_for": ["train_benign_dominant_payload", "val_boundary_benign_like"],
    }
    if output_dir is not None:
        from utils.paths import ensure_dir
        ensure_dir(output_dir)
        with open(output_dir / "trace_sampling_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    return trace_nodes, stats
