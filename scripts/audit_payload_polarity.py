"""Payload Polarity Audit: data-only diagnostic for evidence polarity balance.

Samples 30 nodes from the dataset using stratified error-aware strategy
and audits the polarity distribution of generated evidence tokens WITHOUT
calling any LLM.

Sampling pools:
  - 10 train false negatives (label=1, pred<0.5)
  - 10 train false positives (label=0, pred>=0.5)
  - 5  train high-loss (top 25% BCE loss)
  - 5  val boundary (pred in [0.35, 0.65])

Reports polarity metrics and exits 0 if passing conditions are met.

Usage:
    python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml
    python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --seed 42
    python scripts/audit_payload_polarity.py --config configs/yelpchi_bwgnn.yaml --debug
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.adapter import EvidenceAdapter
from evidence.prototypes import PrototypeBuilder
from evidence.trace_sampler import _bce_loss_per_node
from evidence.vocab import (
    TOKEN_POLARITY_BENIGN,
    TOKEN_POLARITY_FRAUD,
    TOKEN_POLARITY_NEUTRAL,
)
from models.gnn import build_detector
from utils.paths import get_base_checkpoint_path


# ---------------------------------------------------------------------------
# Sampling (mirrors run_stage2_microbenchmark.sample_microbenchmark_nodes)
# ---------------------------------------------------------------------------

def sample_audit_nodes(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    seed: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Sample 30 nodes: 10 FN, 10 FP, 5 high-loss, 5 val-boundary."""
    rng = np.random.RandomState(seed)
    base_probs = torch.sigmoid(base_logits)

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]

    y_train = y[train_idx]
    probs_train = base_probs[train_idx]

    # Pool 1: train FN (y=1, prob<0.5)
    fn_mask = (y_train == 1) & (probs_train < 0.5)
    fn_pool = train_idx[fn_mask].tolist()
    rng.shuffle(fn_pool)

    # Pool 2: train FP (y=0, prob>=0.5)
    fp_mask = (y_train == 0) & (probs_train >= 0.5)
    fp_pool = train_idx[fp_mask].tolist()
    rng.shuffle(fp_pool)

    # Pool 3: train high-loss (top 25% BCE)
    train_losses = _bce_loss_per_node(base_logits[train_idx], y[train_idx].float())
    loss_thresh = torch.quantile(train_losses, 0.75).item()
    hl_mask = train_losses >= loss_thresh
    hl_pool = train_idx[hl_mask].tolist()
    rng.shuffle(hl_pool)

    # Pool 4: val boundary (pred in [0.35, 0.65])
    val_probs = base_probs[val_idx]
    boundary_mask = (val_probs >= 0.35) & (val_probs <= 0.65)
    vb_pool = val_idx[boundary_mask].tolist()
    rng.shuffle(vb_pool)

    drawn: dict[str, list[int]] = {
        "train_fn": fn_pool[:10],
        "train_fp": fp_pool[:10],
        "train_high_loss": hl_pool[:5],
        "val_boundary": vb_pool[:5],
    }

    all_nodes: list[int] = []
    for nodes in drawn.values():
        all_nodes.extend(nodes)

    return all_nodes, drawn


# ---------------------------------------------------------------------------
# Node category lookup
# ---------------------------------------------------------------------------

def _node_category(node_id: int, drawn: dict[str, list[int]]) -> str:
    for cat, nodes in drawn.items():
        if node_id in nodes:
            return cat
    return "unknown"


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def audit_payloads(
    cards: list,
    drawn: dict[str, list[int]],
) -> dict:
    """Audit evidence polarity across all sampled payloads."""
    n = len(cards)
    if n == 0:
        return {}

    # Counters
    benign_dominant_count = 0
    fraud_dominant_count = 0
    mixed_count = 0
    weak_count = 0
    payloads_with_benign_token = 0
    payloads_with_fraud_token = 0

    # Token polarity tallies
    total_fraud_tokens = 0
    total_benign_tokens = 0
    total_neutral_tokens = 0

    # Per-category coverage
    category_benign_coverage: dict[str, int] = {}
    category_fraud_coverage: dict[str, int] = {}
    category_totals: dict[str, int] = {}

    # Per-node token lists for debug
    node_token_details: list[dict] = []

    for card in cards:
        node_id = card.node_id
        reasoning = card.reasoning
        polarity = reasoning.evidence_polarity

        # Use the token counts already computed in the card
        fraud_ct = reasoning.fraud_token_count
        benign_ct = reasoning.benign_token_count
        neutral_ct = reasoning.neutral_token_count

        # Aggregate token counts
        total_fraud_tokens += fraud_ct
        total_benign_tokens += benign_ct
        total_neutral_tokens += neutral_ct

        # Signal availability
        if benign_ct > 0:
            payloads_with_benign_token += 1
        if fraud_ct > 0:
            payloads_with_fraud_token += 1

        # Polarity bucket
        if polarity == "benign_dominant":
            benign_dominant_count += 1
        elif polarity == "fraud_dominant":
            fraud_dominant_count += 1
        elif polarity == "mixed":
            mixed_count += 1
        else:
            weak_count += 1

        # Per-category coverage
        cat = _node_category(node_id, drawn)
        category_totals[cat] = category_totals.get(cat, 0) + 1
        if benign_ct > 0:
            category_benign_coverage[cat] = category_benign_coverage.get(cat, 0) + 1
        if fraud_ct > 0:
            category_fraud_coverage[cat] = category_fraud_coverage.get(cat, 0) + 1

        node_token_details.append({
            "node_id": node_id,
            "category": cat,
            "polarity": polarity,
            "fraud_tokens": fraud_ct,
            "benign_tokens": benign_ct,
            "neutral_tokens": neutral_ct,
        })

    return {
        "benign_signal_available_rate": payloads_with_benign_token / n if n else 0.0,
        "fraud_signal_available_rate": payloads_with_fraud_token / n if n else 0.0,
        "benign_dominant_payload_count": benign_dominant_count,
        "fraud_dominant_payload_count": fraud_dominant_count,
        "mixed_payload_count": mixed_count,
        "weak_payload_count": weak_count,
        "total_fraud_tokens": total_fraud_tokens,
        "total_benign_tokens": total_benign_tokens,
        "total_neutral_tokens": total_neutral_tokens,
        "category_benign_coverage": category_benign_coverage,
        "category_fraud_coverage": category_fraud_coverage,
        "category_totals": category_totals,
        "node_token_details": node_token_details,
        "n_total": n,
    }


# ---------------------------------------------------------------------------
# Pass conditions
# ---------------------------------------------------------------------------

def check_pass_conditions(audit: dict) -> dict[str, bool]:
    """Check the 5 polarity pass conditions."""
    return {
        "benign_dominant_gte_5": audit.get("benign_dominant_payload_count", 0) >= 5,
        "fraud_dominant_gte_5": audit.get("fraud_dominant_payload_count", 0) >= 5,
        "mixed_gte_3": audit.get("mixed_payload_count", 0) >= 3,
        "benign_signal_rate_gte_30pct": audit.get("benign_signal_available_rate", 0) >= 0.30,
        "fraud_signal_rate_gte_50pct": audit.get("fraud_signal_available_rate", 0) >= 0.50,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(audit: dict, pass_conditions: dict[str, bool], dataset_name: str, model_name: str, seed: int) -> None:
    """Print formatted audit report."""
    print(f"\n{'='*64}")
    print(f"Payload Polarity Audit Report")
    print(f"{'='*64}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model:   {model_name}")
    print(f"  Seed:    {seed}")
    print(f"  Nodes:   {audit.get('n_total', 0)}")
    print()

    # -- Signal availability --
    print(f"  Signal Availability:")
    print(f"    benign_signal_available_rate:  {audit.get('benign_signal_available_rate', 0):.2%}")
    print(f"    fraud_signal_available_rate:   {audit.get('fraud_signal_available_rate', 0):.2%}")
    print()

    # -- Polarity distribution --
    print(f"  Polarity Distribution:")
    print(f"    benign_dominant_payload_count: {audit.get('benign_dominant_payload_count', 0)}")
    print(f"    fraud_dominant_payload_count:  {audit.get('fraud_dominant_payload_count', 0)}")
    print(f"    mixed_payload_count:          {audit.get('mixed_payload_count', 0)}")
    print(f"    weak_payload_count:           {audit.get('weak_payload_count', 0)}")
    print()

    # -- Token polarity distribution --
    print(f"  Token Polarity Distribution:")
    print(f"    Total fraud tokens:   {audit.get('total_fraud_tokens', 0)}")
    print(f"    Total benign tokens:  {audit.get('total_benign_tokens', 0)}")
    print(f"    Total neutral tokens: {audit.get('total_neutral_tokens', 0)}")
    print()

    # -- Per-category coverage --
    cat_totals = audit.get("category_totals", {})
    cat_benign = audit.get("category_benign_coverage", {})
    cat_fraud = audit.get("category_fraud_coverage", {})

    print(f"  Benign Token Coverage by Category:")
    for cat in sorted(cat_totals.keys()):
        total = cat_totals.get(cat, 0)
        covered = cat_benign.get(cat, 0)
        rate = covered / total if total > 0 else 0.0
        print(f"    {cat:20s}: {covered:2d}/{total:2d} ({rate:.0%})")
    print()

    print(f"  Fraud Token Coverage by Category:")
    for cat in sorted(cat_totals.keys()):
        total = cat_totals.get(cat, 0)
        covered = cat_fraud.get(cat, 0)
        rate = covered / total if total > 0 else 0.0
        print(f"    {cat:20s}: {covered:2d}/{total:2d} ({rate:.0%})")
    print()

    # -- Pass conditions --
    all_pass = all(pass_conditions.values())
    print(f"  Pass Conditions:")
    for name, ok in pass_conditions.items():
        print(f"    {name:35s}: {'PASS' if ok else 'FAIL'}")
    print()
    print(f"  Overall: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Payload polarity audit (data-only, no LLM)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Use tiny synthetic graph for quick test")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    model_name = config["model"]["name"]
    seed = args.seed

    torch.manual_seed(seed)

    # ---- Load data ----
    if args.debug:
        print("[DEBUG] Using tiny synthetic graph")
        data = load_fraud_dataset("tiny", seed=seed)
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    checkpoint_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        try:
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except RuntimeError:
            print(f"Warning: Checkpoint shape mismatch, using random weights")
    else:
        print(f"Warning: No checkpoint at {checkpoint_path}, using random weights")

    model.eval()

    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = output.logits.cpu()
        embeddings = output.embeddings.cpu()
        extras = {}
        if output.extras:
            for k, v in output.extras.items():
                extras[k] = v.cpu() if isinstance(v, torch.Tensor) else v

    # ---- Sample nodes ----
    val_mask = getattr(data, "val_mask", None)
    if val_mask is None:
        val_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)

    nodes, drawn = sample_audit_nodes(
        y=data.y,
        train_mask=data.train_mask,
        val_mask=val_mask,
        base_logits=base_logits,
        seed=seed,
    )
    print(f"Sampled {len(nodes)} nodes: " + ", ".join(f"{k}={len(v)}" for k, v in drawn.items()))

    # ---- Build adapter ----
    adapter = EvidenceAdapter(
        detector_name=model_name,
        x=data.x,
        edge_index=data.edge_index,
    )

    # No prototype building needed — polarity audit only needs graph tokens
    old_prototypes = None

    # ---- Extract EvidenceCards for sampled nodes ----
    cards = adapter.extract_batch(
        node_ids=nodes,
        base_logits=base_logits,
        embeddings=embeddings,
        extras=extras if extras else None,
        prototypes=old_prototypes,
    )

    # ---- Audit ----
    audit = audit_payloads(cards, drawn)
    pass_conditions = check_pass_conditions(audit)

    print_report(audit, pass_conditions, dataset_name, model_name, seed)

    # Exit code
    if all(pass_conditions.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
