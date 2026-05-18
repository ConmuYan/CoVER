"""Active Learning loop for Idea 3 — REL-Curriculum.

Trains a CoVER-REL reasoner inside an AL loop, using various acquisition
functions to select the most informative train nodes to label.

CLI::

    python scripts/al_loop.py \\
        --dataset yelpchi --base_model bwgnn --seed 42 \\
        --budget_pct 5 --af random --device cuda:0

Outputs::

    artifacts/results/al/{dataset}/{base_model}/{af}/seed_{seed}/budget_{pct}/
        learning_curve.json   (per-round AUPRC + metadata)

Safety contracts preserved: score-blind, frozen-base, train-only-prototype,
bounded-intervention.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import RELATION_SCHEMAS
from models.cover_rel_reasoner import CoVERRelReasoner
from sklearn.metrics import average_precision_score
from training.metrics import compute_metrics
from training.phase2_losses import compute_phase2_loss
from utils.paths import ensure_dir


# ════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, ValueError):
            pass


# ════════════════════════════════════════════════════════════════════
# Acquisition functions
# ════════════════════════════════════════════════════════════════════

def af_random(
    base_logits: torch.Tensor,
    rel_output: dict | None,
    candidate_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random acquisition — uniform random selection from candidates."""
    chosen = rng.choice(len(candidate_idx), size=min(k, len(candidate_idx)), replace=False)
    return candidate_idx[chosen]


def af_uncertainty(
    base_logits: torch.Tensor,
    rel_output: dict | None,
    candidate_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uncertainty acquisition — select nodes with highest base-logit entropy.

    For binary classification with logit ``z``:
        p = σ(z),  H = -p·log(p) - (1-p)·log(1-p)
    Higher entropy → more uncertain → more informative to label.
    """
    logits = base_logits[candidate_idx]
    p = torch.sigmoid(logits)
    eps = 1e-8
    entropy = -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))
    scores = entropy.detach().cpu().numpy()
    top_k_idx = np.argsort(scores)[-k:][::-1]
    return candidate_idx[top_k_idx]


def af_mitigate(
    base_logits: torch.Tensor,
    rel_output: dict | None,
    candidate_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """MITIGATE-style acquisition (ported from external/MITIGATE/sampling_methods.py).

    Uses entropy of base predictions + anomaly score difference.  Since we
    don't have a separate normal-class classifier here, we use base-logit
    entropy as the primary signal (matches MITIGATE's ``query_nent_diff``
    with weight=0.5).

    For the k-medoids variant we need adjacency embeddings which are not
    directly available in this loop, so we use the simplified entropy+diff
    version that operates purely on scores (no clustering overhead).
    """
    logits = base_logits[candidate_idx]
    p = torch.sigmoid(logits)
    eps = 1e-8

    # Entropy score (proxy for MITIGATE's normal-class entropy)
    entropy = -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))
    n_entropy = (entropy - entropy.mean()) / (entropy.std() + eps)

    # Anomaly score (base probability of fraud)
    a_scores = (p - p.mean()) / (p.std() + eps)

    # Disagreement between entropy and anomaly score (MITIGATE's nent_diff)
    scores_diff = torch.abs(n_entropy - a_scores)

    # Combined score: weight * entropy + (1-weight) * |entropy - anomaly|
    weight = 0.5
    scores = (weight * n_entropy + (1 - weight) * scores_diff).detach().cpu().numpy()

    top_k_idx = np.argsort(scores)[-k:][::-1]
    return candidate_idx[top_k_idx]


def af_rel(
    base_logits: torch.Tensor,
    rel_output: dict | None,
    candidate_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """REL-acquisition function — uses REL diagnostic signals.

    Combines three signals:
    1. |Δ_rel| — residual magnitude (REL says base is wrong here)
    2. H(π_rel) — gate entropy (high = relations disagree)
    3. Disagree — base vs REL prediction flip
    """
    if rel_output is None:
        # Fallback to uncertainty if REL output unavailable
        return af_uncertainty(base_logits, rel_output, candidate_idx, k, rng)

    # rel_output is already indexed over candidates (0..len(candidates))
    delta_rel = rel_output["delta_rel"]
    gate = rel_output["relation_gate"]

    # Signal 1: residual magnitude
    residual_mag = delta_rel.abs()

    # Signal 2: gate entropy
    eps = 1e-8
    gate_entropy = -(gate * torch.log(gate.clamp_min(eps))).sum(dim=1)

    # Signal 3: prediction flip (base vs REL)
    base_cand_logits = base_logits[candidate_idx]
    base_pred = (base_cand_logits > 0).float()
    rel_pred = ((base_cand_logits + delta_rel) > 0).float()
    disagree = (base_pred != rel_pred).float()

    # Normalize each signal to [0, 1] range for fair combination
    def _norm(x: torch.Tensor) -> torch.Tensor:
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    score = _norm(residual_mag) + _norm(gate_entropy) + _norm(disagree)
    scores = score.detach().cpu().numpy()

    top_k_idx = np.argsort(scores)[-k:][::-1]
    return candidate_idx[top_k_idx]


AF_REGISTRY: dict[str, callable] = {
    "random": af_random,
    "uncertainty": af_uncertainty,
    "mitigate_af": af_mitigate,
    "rel_af": af_rel,
}


# ════════════════════════════════════════════════════════════════════
# Core AL helpers
# ════════════════════════════════════════════════════════════════════

def stratified_seed_selection(
    train_idx: np.ndarray,
    labels: np.ndarray,
    n_seed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select initial seed set preserving class balance (at least 1 fraud)."""
    fraud_idx = train_idx[labels[train_idx] == 1]
    normal_idx = train_idx[labels[train_idx] == 0]

    n_fraud = max(1, n_seed // 4)  # at least 25% fraud in seed
    n_fraud = min(n_fraud, len(fraud_idx))
    n_normal = n_seed - n_fraud
    n_normal = min(n_normal, len(normal_idx))

    seed_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
    seed_normal = rng.choice(normal_idx, size=n_normal, replace=False)

    seed_set = np.concatenate([seed_fraud, seed_normal])
    rng.shuffle(seed_set)
    return seed_set


def load_frozen_base_cached(
    config: dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    data,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load frozen base logits + embeddings (reuses Phase 2 cache)."""
    from scripts.train_phase2_reasoner import load_frozen_base
    # Use fixed_v1_100ep checkpoint (canonical base for all Phase 2+ work)
    ckpt_path = Path("artifacts") / "checkpoints" / dataset_name / model_name / "fixed_v1_100ep" / f"seed_{seed}" / "base.pt"
    base_logits, base_z, _ = load_frozen_base(
        config, dataset_name, model_name, seed, data, device,
        ckpt_override=str(ckpt_path) if ckpt_path.exists() else None,
    )
    return base_logits, base_z


def load_rel_features_cached(
    dataset_name: str,
    model_name: str,
    seed: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Load relation features (reuses Phase 2 cache)."""
    from scripts.train_phase2_reasoner import load_relation_features_for_phase2
    rel_features, _ = load_relation_features_for_phase2(
        dataset_name, model_name, seed, num_nodes,
    )
    return rel_features.to(device)


def train_reasoner_subset(
    reasoner: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    val_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor | None,
    epochs: int,
    device: torch.device,
) -> dict:
    """Train reasoner on labeled subset for a fixed number of epochs.

    Returns best val metrics (AUPRC) from the training run.
    """
    best_auprc = -1.0
    best_state = None
    patience_counter = 0
    patience = max(10, epochs // 3)  # adaptive patience

    for epoch in range(1, epochs + 1):
        # Train
        reasoner.train()
        z_l = base_z[labeled_mask]
        bl_l = base_logits[labeled_mask]
        rf_l = rel_features[labeled_mask]
        y_l = y[labeled_mask]
        all_train = torch.ones(z_l.shape[0], dtype=torch.bool, device=device)

        out = reasoner(z_l, bl_l, rf_l)
        loss, _ = compute_phase2_loss(out, y_l, all_train, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_norm=1.0)
        optimizer.step()

        # Eval on val (every epoch for fast convergence detection)
        if epoch % 1 == 0:
            reasoner.eval()
            with torch.no_grad():
                out_val = reasoner(base_z[val_mask], base_logits[val_mask], rel_features[val_mask])
                prob_val = torch.sigmoid(out_val["final_logit"]).cpu().numpy()
                y_val = y[val_mask].cpu().numpy()
                try:
                    auprc = float(average_precision_score(y_val, prob_val))
                except Exception:
                    auprc = 0.0

            if auprc > best_auprc:
                best_auprc = auprc
                best_state = copy.deepcopy(reasoner.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        reasoner.load_state_dict(best_state)

    return {"val_auprc": best_auprc}


def evaluate_on_val(
    reasoner: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    y: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """Evaluate reasoner on validation set."""
    reasoner.eval()
    with torch.no_grad():
        out = reasoner(base_z[val_mask], base_logits[val_mask], rel_features[val_mask])
        prob = torch.sigmoid(out["final_logit"]).cpu().numpy()
        y_np = y[val_mask].cpu().numpy()

    metrics = compute_metrics(y_np, prob)
    return metrics


def get_rel_output_for_candidates(
    reasoner: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    candidate_idx: np.ndarray,
    device: torch.device,
) -> dict:
    """Get REL output for candidate nodes (for rel_af acquisition)."""
    reasoner.eval()
    with torch.no_grad():
        idx_t = torch.tensor(candidate_idx, device=device, dtype=torch.long)
        out = reasoner(base_z[idx_t], base_logits[idx_t], rel_features[idx_t])
    return out


# ════════════════════════════════════════════════════════════════════
# Main AL loop
# ════════════════════════════════════════════════════════════════════

def run_al_loop(
    *,
    dataset_name: str,
    base_model: str,
    seed: int,
    budget_pct: float,
    af_name: str,
    device: torch.device,
    al_epochs: int = 30,
    initial_seed_pct: float = 1.0,
) -> dict:
    """Run one complete AL experiment.

    Args:
        dataset_name: 'yelpchi' or 'amazon'
        base_model: 'bwgnn', 'gat', 'gcn', 'sage'
        seed: random seed
        budget_pct: total budget as % of train set (e.g. 5 = 5%)
        af_name: acquisition function name
        device: torch device
        al_epochs: REL training epochs per AL round
        initial_seed_pct: initial labeled set as % of train (default 1%)

    Returns:
        dict with learning curve and metadata
    """
    t0 = time.time()
    set_seed(seed)

    # ---- Dataset config (matches Phase 2 canonical) ----
    dataset_paths = {
        "yelpchi": "datasets/YelpChi.mat",
        "amazon": "datasets/Amazon.mat",
    }
    path = dataset_paths.get(dataset_name)
    if path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = {
        "dataset": {
            "name": dataset_name,
            "path": path,
            "format": "mat",
            "split_mode": "supervised",
            "train_ratio": 0.4,
            "val_test_ratio": [1, 2],
            "scarcity_ratio": 1.0,
            "stratified": True,
        },
        "model": {
            "name": base_model,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
        },
    }

    # Add model-specific config
    if base_model == "bwgnn":
        config["model"]["num_bands"] = 3
        config["model"]["agg"] = "concat"
    elif base_model == "gat":
        config["model"]["attention_heads"] = 4

    # Phase 2 reasoner config (canonical)
    p2_cfg = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "relation_names": ["RUR", "RSR", "RTR"],
        "anchor_relation": "RUR",
        "rel_hidden_dim": 64,
        "rel_num_layers": 2,
        "rel_dropout": 0.3,
        "tau_gate": 0.7,
        "delta_rel_max": 2.0,
        "gate_mode": "softmax",
        "evidence_groups": ["A", "B", "C"],
    }

    # ---- Load data ----
    data = load_fraud_dataset(
        name=dataset_name,
        path=path,
        format="mat",
        seed=seed,
        train_ratio=0.4,
        val_test_ratio=[1, 2],
        stratified=True,
    )

    # ---- Load frozen base ----
    base_logits, base_z = load_frozen_base_cached(
        config, dataset_name, base_model, seed, data, device,
    )

    # ---- Load relation features ----
    num_nodes = int(data.x.shape[0])
    rel_features = load_rel_features_cached(
        dataset_name, base_model, seed, num_nodes, device,
    )

    # ---- Get relation names from schema ----
    relation_names = list(RELATION_SCHEMAS[dataset_name].keys())

    # ---- Extract train/val/test indices ----
    y = data.y.to(device)
    train_mask = data.train_mask.cpu().numpy().astype(bool)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    train_idx = np.where(train_mask)[0]
    val_idx_np = data.val_mask.cpu().numpy().astype(bool)
    val_idx = np.where(val_idx_np)[0]

    n_train = len(train_idx)
    n_total_budget = max(1, int(n_train * budget_pct / 100.0))
    n_initial = max(2, int(n_train * initial_seed_pct / 100.0))

    # Ensure initial seed < total budget
    n_initial = min(n_initial, n_total_budget // 2) if n_total_budget > 4 else max(2, n_total_budget // 2)

    # Number of AL rounds: each round adds (budget - current) / remaining_rounds
    # We do ~5 rounds by default, spacing out the budget additions
    n_rounds = min(5, max(1, n_total_budget - n_initial))
    if n_rounds == 0:
        n_rounds = 1
    per_round = max(1, (n_total_budget - n_initial) // n_rounds)

    # ---- Prepare RNG ----
    rng = np.random.default_rng(seed)

    # ---- Initial labeled set (stratified) ----
    labeled_set = set(stratified_seed_selection(train_idx, y.cpu().numpy(), n_initial, rng).tolist())
    unlabeled_set = set(train_idx.tolist()) - labeled_set

    # ---- Acquisition function ----
    af_fn = AF_REGISTRY.get(af_name)
    if af_fn is None:
        raise ValueError(f"Unknown acquisition function: {af_name}. Choose from {list(AF_REGISTRY.keys())}")

    # ---- Build reasoner ----
    z_dim = int(base_z.shape[1])
    reasoner = CoVERRelReasoner(
        base_z_dim=z_dim,
        relation_names=relation_names,
        anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=9,
        rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=p2_cfg.get("rel_num_layers", 2),
        rel_dropout=p2_cfg.get("rel_dropout", 0.3),
        tau_gate=p2_cfg.get("tau_gate", 0.7),
        delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
        gate_mode=p2_cfg.get("gate_mode", "softmax"),
        evidence_groups=p2_cfg.get("evidence_groups", None),
    ).to(device)

    # ---- Compute pos_weight for imbalanced BCE ----
    y_train_np = y.cpu().numpy()[train_mask]
    n_pos = max(1, int(y_train_np.sum()))
    n_neg = max(1, len(y_train_np) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    # ---- Learning curve storage ----
    learning_curve: list[dict] = []

    def _record_round(round_num: int, n_labeled: int, metrics: dict, elapsed: float) -> None:
        entry = {
            "round": round_num,
            "n_labeled": n_labeled,
            "budget_pct": round(100.0 * n_labeled / n_train, 2),
            "val_auprc": round(float(metrics.get("auprc", 0.0)), 6),
            "val_roc_auc": round(float(metrics.get("roc_auc", 0.0)), 6),
            "val_macro_f1": round(float(metrics.get("macro_f1", 0.0)), 6),
            "elapsed_sec": round(elapsed, 1),
        }
        learning_curve.append(entry)
        print(
            f"  [Round {round_num}] labeled={n_labeled} "
            f"({entry['budget_pct']:.1f}%) "
            f"AUPRC={entry['val_auprc']:.4f} "
            f"ROC={entry['val_roc_auc']:.4f} "
            f"elapsed={entry['elapsed_sec']:.1f}s"
        )

    # ---- Evaluate at round 0 (initial seed only) ----
    print(f"[AL] Starting: dataset={dataset_name}, base={base_model}, seed={seed}, "
          f"af={af_name}, budget={budget_pct}%, initial_seed={n_initial}")

    labeled_mask_t = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    labeled_mask_t[list(labeled_set)] = True

    reasoner_init = copy.deepcopy(reasoner)
    optimizer_init = torch.optim.AdamW(
        reasoner_init.parameters(),
        lr=float(p2_cfg.get("lr", 1e-3)),
        weight_decay=float(p2_cfg.get("weight_decay", 1e-4)),
    )
    t_round = time.time()
    train_reasoner_subset(
        reasoner_init, base_z, base_logits, rel_features, y,
        labeled_mask_t, val_mask, optimizer_init, pos_weight,
        epochs=al_epochs, device=device,
    )
    metrics_0 = evaluate_on_val(reasoner_init, base_z, base_logits, rel_features, y, val_mask, device)
    _record_round(0, len(labeled_set), metrics_0, time.time() - t_round)
    del reasoner_init

    # ---- AL loop ----
    for rnd in range(1, n_rounds + 1):
        t_round = time.time()
        current_budget = min(n_initial + rnd * per_round, n_total_budget)
        k = current_budget - len(labeled_set)
        if k <= 0 or len(unlabeled_set) == 0:
            break

        candidate_idx = np.array(sorted(unlabeled_set))

        # Build labeled mask
        labeled_mask_t = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        labeled_mask_t[list(labeled_set)] = True

        # Re-init reasoner for this round (fresh start on current labeled set)
        reasoner_r = CoVERRelReasoner(
            base_z_dim=z_dim,
            relation_names=relation_names,
            anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
            rel_stat_dim=9,
            rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
            rel_num_layers=p2_cfg.get("rel_num_layers", 2),
            rel_dropout=p2_cfg.get("rel_dropout", 0.3),
            tau_gate=p2_cfg.get("tau_gate", 0.7),
            delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
            gate_mode=p2_cfg.get("gate_mode", "softmax"),
            evidence_groups=p2_cfg.get("evidence_groups", None),
        ).to(device)
        optimizer_r = torch.optim.AdamW(
            reasoner_r.parameters(),
            lr=float(p2_cfg.get("lr", 1e-3)),
            weight_decay=float(p2_cfg.get("weight_decay", 1e-4)),
        )

        # Train on current labeled set
        train_reasoner_subset(
            reasoner_r, base_z, base_logits, rel_features, y,
            labeled_mask_t, val_mask, optimizer_r, pos_weight,
            epochs=al_epochs, device=device,
        )

        # Get REL output for candidates (needed by rel_af)
        rel_output = None
        if af_name == "rel_af":
            rel_output = get_rel_output_for_candidates(
                reasoner_r, base_z, base_logits, rel_features,
                candidate_idx, device,
            )

        # Acquire new labels
        selected = af_fn(base_logits, rel_output, candidate_idx, k, rng)
        labeled_set.update(selected.tolist())
        unlabeled_set -= set(selected.tolist())

        # Evaluate on val
        metrics_r = evaluate_on_val(reasoner_r, base_z, base_logits, rel_features, y, val_mask, device)
        _record_round(rnd, len(labeled_set), metrics_r, time.time() - t_round)
        del reasoner_r

    # ---- Final evaluation at full budget ----
    if len(labeled_set) < n_total_budget and len(unlabeled_set) > 0:
        t_round = time.time()
        remaining = n_total_budget - len(labeled_set)
        candidate_idx = np.array(sorted(unlabeled_set))

        labeled_mask_t = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        labeled_mask_t[list(labeled_set)] = True

        reasoner_f = CoVERRelReasoner(
            base_z_dim=z_dim,
            relation_names=relation_names,
            anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
            rel_stat_dim=9,
            rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
            rel_num_layers=p2_cfg.get("rel_num_layers", 2),
            rel_dropout=p2_cfg.get("rel_dropout", 0.3),
            tau_gate=p2_cfg.get("tau_gate", 0.7),
            delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
            gate_mode=p2_cfg.get("gate_mode", "softmax"),
            evidence_groups=p2_cfg.get("evidence_groups", None),
        ).to(device)
        optimizer_f = torch.optim.AdamW(
            reasoner_f.parameters(),
            lr=float(p2_cfg.get("lr", 1e-3)),
            weight_decay=float(p2_cfg.get("weight_decay", 1e-4)),
        )

        rel_output = None
        if af_name == "rel_af":
            rel_output = get_rel_output_for_candidates(
                reasoner_f, base_z, base_logits, rel_features,
                candidate_idx, device,
            )

        selected = af_fn(base_logits, rel_output, candidate_idx, remaining, rng)
        labeled_set.update(selected.tolist())

        labeled_mask_t = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        labeled_mask_t[list(labeled_set)] = True

        train_reasoner_subset(
            reasoner_f, base_z, base_logits, rel_features, y,
            labeled_mask_t, val_mask, optimizer_f, pos_weight,
            epochs=al_epochs, device=device,
        )
        metrics_f = evaluate_on_val(reasoner_f, base_z, base_logits, rel_features, y, val_mask, device)
        _record_round(len(learning_curve), len(labeled_set), metrics_f, time.time() - t_round)
        del reasoner_f

    total_time = time.time() - t0
    print(f"[AL] Done in {total_time:.1f}s. Final AUPRC={learning_curve[-1]['val_auprc']:.4f}")

    return {
        "dataset": dataset_name,
        "base_model": base_model,
        "seed": seed,
        "budget_pct": budget_pct,
        "af": af_name,
        "al_epochs": al_epochs,
        "learning_curve": learning_curve,
        "total_time_sec": round(total_time, 1),
    }


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active Learning loop for Idea 3 (REL-Curriculum)")
    p.add_argument("--dataset", type=str, default="yelpchi", choices=["yelpchi", "amazon"])
    p.add_argument("--base_model", type=str, default="bwgnn", choices=["bwgnn", "gat", "gcn", "sage"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget_pct", type=float, default=5.0,
                   help="Total AL budget as %% of train set (e.g. 5 = 5%%)")
    p.add_argument("--af", type=str, default="random",
                   choices=list(AF_REGISTRY.keys()),
                   help="Acquisition function")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--al_epochs", type=int, default=30,
                   help="REL training epochs per AL round")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    result = run_al_loop(
        dataset_name=args.dataset,
        base_model=args.base_model,
        seed=args.seed,
        budget_pct=args.budget_pct,
        af_name=args.af,
        device=device,
        al_epochs=args.al_epochs,
    )

    # ---- Save results ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path("artifacts") / "results" / "al"
            / args.dataset / args.base_model / args.af
            / f"seed_{args.seed}" / f"budget_{args.budget_pct:.0f}"
        )
    ensure_dir(out_dir)

    out_path = out_dir / "learning_curve.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[AL] Results saved to {out_path}")


if __name__ == "__main__":
    main()
