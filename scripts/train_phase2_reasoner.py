"""Phase2 CoVER-REL Reasoner training script.

Trains a CoVERRelReasoner on top of a frozen base BWGNN, using relation-gated
evidence fusion with optional LLM judge residual.

Usage:
    python scripts/train_phase2_reasoner.py \
        --config configs/phase2_yelpchi_E2_judge_residual.yaml \
        --seed 42 --device cuda:0

    # debug smoke test (3 epochs, tiny graph)
    python scripts/train_phase2_reasoner.py \
        --config configs/phase2_yelpchi_E0_relgate.yaml \
        --seed 42 --debug --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.judge_encoder import load_jsonl
from evidence.relation_features import load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.gnn import build_detector
from training.metrics import compute_metrics, g_means as compute_g_means, precision_recall_at_k
from training.phase2_losses import build_judge_relation_targets, compute_phase2_loss
from utils.paths import (
    ensure_dir,
    get_base_checkpoint_path,
    get_checkpoint_dir,
    get_logs_dir,
    get_results_dir,
)
from utils.threshold import evaluate_with_threshold, find_best_threshold


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int, model_name: str = "") -> None:
    # Phase2 sweeps are usually launched as multiple independent processes.
    # Keep each process from spawning a large BLAS/OpenMP thread pool.
    num_threads = int(os.environ.get("COVER_NUM_THREADS", "1"))
    torch.set_num_threads(max(num_threads, 1))
    try:
        torch.set_num_interop_threads(max(int(os.environ.get("COVER_INTEROP_THREADS", "1")), 1))
    except RuntimeError:
        # PyTorch only allows setting interop threads before parallel work starts.
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # GAT's deterministic scatter_add_ uses 3x more GPU memory (OOM on dense graphs).
    # Skip deterministic mode for GAT; rely on manual seeding for reproducibility.
    if model_name != "gat":
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def runtime_device_info(device: torch.device) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    info: dict[str, object] = {
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": visible,
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return info
    visible_index = device.index if device.index is not None else torch.cuda.current_device()
    physical_ids = [item.strip() for item in visible.split(",")] if visible else []
    physical_id = physical_ids[visible_index] if visible_index < len(physical_ids) else str(visible_index)
    info.update({
        "visible_device_index": visible_index,
        "physical_device_id": physical_id,
        "device_name": torch.cuda.get_device_name(visible_index),
    })
    return info


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

def load_frozen_base(config: dict, dataset_name: str, model_name: str, seed: int, data, device: torch.device):
    """Load frozen base BWGNN and cache its logits + embeddings."""
    extra_kwargs = {}
    if "attention_heads" in config["model"]:
        extra_kwargs["attention_heads"] = config["model"]["attention_heads"]
    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
        **extra_kwargs,
    ).to(device)

    ckpt_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, weights_only=True)
    base_model.load_state_dict(state)
    print(f"Loaded base checkpoint from {ckpt_path}")

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        output = base_model(x, edge_index, return_output=True)
        base_logits = output.logits.detach()
        base_z = output.embeddings.detach()

    return base_logits, base_z


def load_relation_features_for_phase2(
    dataset_name: str, model_name: str, seed: int, num_nodes: int,
) -> tuple[torch.Tensor, dict]:
    """Load pre-computed relation statistics."""
    rel_path = (
        Path("artifacts") / "relation_features" / dataset_name / model_name
        / f"seed_{seed}" / "all" / "rel_stats.pt"
    )
    if not rel_path.exists():
        raise FileNotFoundError(f"Relation features not found: {rel_path}")
    rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=num_nodes)
    print(f"Loaded relation features {tuple(rel_stats.shape)} from {rel_path}")
    return rel_stats, rel_meta


def load_judge_data(
    dataset_name: str, model_name: str, seed: int, num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict], dict]:
    """Load judge features, mask, accepted records, and validate audit."""
    judge_dir = (
        Path("artifacts") / "judge_packets" / dataset_name / model_name
        / "cover_rel_judge" / f"seed_{seed}"
    )

    # Load features + mask
    feat_path = judge_dir / "judge_features.pt"
    if not feat_path.exists():
        raise FileNotFoundError(f"Judge features not found: {feat_path}")
    payload = torch.load(feat_path, map_location="cpu", weights_only=False)
    judge_features = payload["features"].float()
    judge_mask = payload["mask"].bool()
    if judge_features.shape[0] != num_nodes:
        raise ValueError(
            f"Judge features shape mismatch: {judge_features.shape[0]} != {num_nodes}"
        )

    # Load accepted judge records
    accepted_path = judge_dir / "accepted_judge.jsonl"
    accepted_records = load_jsonl(accepted_path)
    print(
        f"Loaded judge features {tuple(judge_features.shape)} + "
        f"{len(accepted_records)} accepted records from {judge_dir}"
    )

    # Validate audit
    audit_path = judge_dir / "judge_forbidden_field_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Judge audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed", False):
        raise RuntimeError(
            f"Judge forbidden field audit FAILED (passed={audit.get('passed')}): {audit_path}"
        )
    if not audit.get("accepted_outputs_passed", False):
        raise RuntimeError(
            f"Judge accepted outputs audit FAILED "
            f"(accepted_outputs_passed={audit.get('accepted_outputs_passed')}): {audit_path}"
        )

    return judge_features, judge_mask, accepted_records, audit


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_phase2(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    judge_features: torch.Tensor | None,
    judge_mask: torch.Tensor | None,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate the reasoner on a split (val or test)."""
    reasoner.eval()
    outputs = reasoner(
        base_z[mask],
        base_logits[mask],
        rel_features[mask].to(device),
        judge_features=judge_features[mask].to(device) if judge_features is not None else None,
        judge_mask=judge_mask[mask].to(device) if judge_mask is not None else None,
    )
    final_logit = outputs["final_logit"]
    prob = torch.sigmoid(final_logit).cpu().numpy()
    y_np = y[mask].cpu().numpy()

    if threshold is not None:
        metrics = evaluate_with_threshold(y_np, prob, threshold)
        # Add g_means and precision@K/recall@K
        y_pred_binary = (prob >= threshold).astype(int)
        metrics["g_means"] = compute_g_means(y_np, y_pred_binary)
        for k in (k_values or [50, 100, 200]):
            pk, rk = precision_recall_at_k(y_np, prob, k)
            metrics[f"precision@{k}"] = pk
            metrics[f"recall@{k}"] = rk
        return metrics
    else:
        return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200])


@torch.no_grad()
def compute_epoch_diagnostics(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    judge_features: torch.Tensor | None,
    judge_mask: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    """Compute per-epoch diagnostic stats (delta_rel, alpha_llm, gates, etc.)."""
    reasoner.eval()
    outputs = reasoner(
        base_z,
        base_logits,
        rel_features.to(device),
        judge_features=judge_features.to(device) if judge_features is not None else None,
        judge_mask=judge_mask.to(device) if judge_mask is not None else None,
    )
    diag: dict[str, float] = {}

    # mean |delta_rel|
    delta_rel = outputs.get("delta_rel")
    if delta_rel is not None:
        diag["mean_abs_delta_rel"] = float(delta_rel.abs().mean().item())

    # alpha_llm stats
    alpha_llm = outputs.get("alpha_llm")
    if alpha_llm is not None:
        diag["mean_alpha_llm"] = float(alpha_llm.mean().item())
        if judge_mask is not None:
            jm = judge_mask.to(device).bool()
            if jm.any():
                diag["mean_alpha_llm_accepted"] = float(alpha_llm[jm].mean().item())
            rejected = ~jm
            if rejected.any():
                diag["mean_alpha_llm_rejected"] = float(alpha_llm[rejected].mean().item())
                max_alpha_rejected = float(alpha_llm[rejected].abs().max().item())
                diag["max_abs_alpha_llm_rejected"] = max_alpha_rejected
                # Assert alpha_llm for non-judge nodes is ~0
                assert max_alpha_rejected < 1e-6, (
                    f"alpha_llm leak: max |alpha_llm[~mask]| = {max_alpha_rejected:.8f} >= 1e-6"
                )

    # gate entropy
    gate_values = outputs.get("relation_gate")
    if gate_values is not None:
        eps = 1e-8
        g = gate_values.clamp(eps, 1.0)
        entropy = -(g * g.log()).sum(dim=-1).mean()
        diag["mean_gate_entropy"] = float(entropy.item())

        # per-relation gate weight
        for r_idx in range(gate_values.shape[1]):
            diag[f"gate_weight_rel_{r_idx}"] = float(gate_values[:, r_idx].mean().item())

    # dominance rho from relation_strength
    rel_strength = outputs.get("relation_strength")
    if rel_strength is not None and rel_strength.shape[1] >= 2:
        sorted_s, _ = torch.sort(rel_strength, dim=-1, descending=True)
        rho = sorted_s[:, 0] - sorted_s[:, 1]
        diag["mean_dominance_rho"] = float(rho.mean().item())

    return diag


# ════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════

def write_phase2_epoch_log(log_dir: Path, rows: list[dict]) -> None:
    """Write per-epoch training log as JSONL and CSV."""
    jsonl_path = log_dir / "phase2_train_log.jsonl"
    csv_path = log_dir / "phase2_train_log.csv"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        fields = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


# ════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    judge_features: torch.Tensor | None,
    judge_mask: torch.Tensor | None,
    judge_align: dict[str, torch.Tensor] | None,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    p2_cfg: dict,
    epoch: int,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """One training epoch with phase2 4-term loss."""
    reasoner.train()
    device = base_z.device

    # Slice train nodes
    z_t = base_z[train_mask]
    bl_t = base_logits[train_mask]
    rf_t = rel_features[train_mask].to(device)
    jf_t = judge_features[train_mask].to(device) if judge_features is not None else None
    jm_t = judge_mask[train_mask].to(device) if judge_mask is not None else None
    y_t = y[train_mask]

    # Slice judge_align dict values to train nodes
    ja_t: dict[str, torch.Tensor] | None = None
    if judge_align is not None:
        tm_cpu = train_mask.cpu()
        ja_t = {
            "q_target": judge_align["q_target"][tm_cpu],
            "w_weight": judge_align["w_weight"][tm_cpu],
            "has_target_mask": judge_align["has_target_mask"][tm_cpu],
        }

    outputs = reasoner(z_t, bl_t, rf_t, judge_features=jf_t, judge_mask=jm_t)

    # All-true mask since we already sliced to train nodes
    all_train = torch.ones(z_t.shape[0], dtype=torch.bool, device=device)

    pw = pos_weight if pos_weight is not None else torch.tensor(1.0, device=device)

    loss, loss_dict = compute_phase2_loss(
        outputs=outputs,
        y=y_t,
        train_mask=all_train,
        judge_align=ja_t,
        pos_weight=pw,
        lambda_trust=p2_cfg.get("lambda_trust", 3e-3),
        lambda_sparse=p2_cfg.get("lambda_sparse", 1e-3),
        lambda_align=p2_cfg.get("lambda_align", 1e-3),
        eta_llm=p2_cfg.get("eta_llm", 2.0),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_dict


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase2 CoVER-REL Reasoner Training")
    parser.add_argument("--config", type=str, required=True, help="Path to phase2 config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 3 epochs, tiny graph")
    # CLI overrides for phase2_reasoner knobs
    parser.add_argument("--use_judge", type=int, default=None, choices=[0, 1], help="Override use_judge (0/1)")
    parser.add_argument("--alpha_max", type=float, default=None)
    parser.add_argument("--lambda_trust", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    parser.add_argument("--lambda_align", type=float, default=None)
    parser.add_argument("--eta_llm", type=float, default=None)
    parser.add_argument("--delta_rel_max", type=float, default=None)
    parser.add_argument("--delta_llm_max", type=float, default=None)
    parser.add_argument("--tau_gate", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Evaluate validation/diagnostics every N epochs; default preserves config/1.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name")
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    p2_cfg = config.get("phase2_reasoner", {})

    # CLI overrides
    if args.use_judge is not None:
        p2_cfg["use_judge"] = bool(args.use_judge)
    if args.alpha_max is not None:
        p2_cfg["alpha_max"] = args.alpha_max
    if args.lambda_trust is not None:
        p2_cfg["lambda_trust"] = args.lambda_trust
    if args.lambda_sparse is not None:
        p2_cfg["lambda_sparse"] = args.lambda_sparse
    if args.lambda_align is not None:
        p2_cfg["lambda_align"] = args.lambda_align
    if args.eta_llm is not None:
        p2_cfg["eta_llm"] = args.eta_llm
    if args.delta_rel_max is not None:
        p2_cfg["delta_rel_max"] = args.delta_rel_max
    if args.delta_llm_max is not None:
        p2_cfg["delta_llm_max"] = args.delta_llm_max
    if args.tau_gate is not None:
        p2_cfg["tau_gate"] = args.tau_gate
    if args.lr is not None:
        p2_cfg["lr"] = args.lr
    if args.patience is not None:
        p2_cfg["patience"] = args.patience
    if args.early_stop_metric is not None:
        p2_cfg["early_stop_metric"] = args.early_stop_metric
    if args.eval_interval is not None:
        p2_cfg["eval_interval"] = args.eval_interval
    if args.run_name is not None:
        p2_cfg["run_name"] = args.run_name

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    model_name = config["model"]["name"]
    seed = args.seed if args.seed is not None else config["train"]["seed"]
    run_name = p2_cfg.get("run_name", "phase2")
    relation_names = p2_cfg.get("relation_names", [])

    # ── Seed + device ──
    set_seed(seed, model_name=model_name)

    requested_device = str(
        args.device or config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    device_info = runtime_device_info(device)
    print(f"[Phase2] seed={seed} device={device} run_name={run_name}")

    # ── Load dataset ──
    # Phase2 always loads the real dataset since it needs a matching base
    # checkpoint.  Debug mode simply reduces epochs.
    data = load_fraud_dataset(
        dataset_name,
        path=dataset_path,
        seed=seed,
        split_mode=config["dataset"].get("split_mode", "supervised"),
        train_ratio=config["dataset"].get("train_ratio", 0.4),
        val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        stratified=config["dataset"].get("stratified", True),
    )
    epochs = p2_cfg.get("epochs", 300)
    if args.debug:
        epochs = config["train"].get("debug_epochs", 3)
        print(f"[DEBUG] Using real dataset with {epochs} epochs")

    num_nodes = data.x.shape[0]
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # ── Load frozen base BWGNN ──
    base_logits, base_z = load_frozen_base(config, dataset_name, model_name, seed, data, device)
    z_dim = base_z.shape[1]

    # ── Load relation features ──
    rel_stats, rel_meta = load_relation_features_for_phase2(
        dataset_name, model_name, seed, num_nodes,
    )
    rel_features = rel_stats.to(device)  # (N, rel_dim)

    # ── Load judge data (if enabled) ──
    use_judge = p2_cfg.get("use_judge", False)
    judge_features = None
    judge_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    judge_align = None
    accepted_records: list[dict] = []

    if use_judge:
        judge_features, judge_mask_tensor, accepted_records, _audit = load_judge_data(
            dataset_name, model_name, seed, num_nodes,
        )
        judge_features = judge_features.to(device)
        judge_mask_tensor = judge_mask_tensor.to(device)
        judge_dir = (
            Path("artifacts") / "judge_packets" / dataset_name / model_name
            / "cover_rel_judge" / f"seed_{seed}"
        )
        judge_align = build_judge_relation_targets(
            jsonl_path=judge_dir / "accepted_judge.jsonl",
            num_nodes=num_nodes,
            relation_names=relation_names,
        )
    else:
        print("[Phase2] Judge disabled (E0 mode)")

    # ── Instantiate CoVERRelReasoner ──
    reasoner = CoVERRelReasoner(
        base_z_dim=z_dim,
        relation_names=relation_names,
        anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=p2_cfg.get("rel_num_layers", 2),
        rel_dropout=p2_cfg.get("rel_dropout", 0.3),
        tau_gate=p2_cfg.get("tau_gate", 0.7),
        delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
        use_judge=use_judge,
        judge_feature_dim=judge_features.shape[1] if judge_features is not None else 0,
        judge_hidden_dim=p2_cfg.get("judge_hidden_dim", 32),
        judge_dropout=p2_cfg.get("judge_dropout", 0.3),
        delta_llm_max=p2_cfg.get("delta_llm_max", 0.75),
        alpha_max=p2_cfg.get("alpha_max", 0.0),
        alpha_bias_init=p2_cfg.get("alpha_bias_init", -3.0),
    ).to(device)

    print(f"[Phase2] Reasoner params: {sum(p.numel() for p in reasoner.parameters()):,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=p2_cfg.get("lr", 1e-3),
        weight_decay=p2_cfg.get("weight_decay", 1e-4),
    )

    # ── Training ──
    patience = p2_cfg.get("patience", 50)
    early_stop_metric = p2_cfg.get("early_stop_metric", "val_auprc")
    eval_interval = max(int(p2_cfg.get("eval_interval", 1)), 1)
    k_values = config.get("eval", {}).get("k_values", [50, 100, 200])
    debug_epochs = config["train"].get("debug_epochs", 3)
    if args.debug:
        epochs = debug_epochs

    best_val_score = float("-inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_rows: list[dict] = []
    start_time = time.time()

    print(
        f"[Phase2] Training for up to {epochs} epochs, patience={patience}, "
        f"metric={early_stop_metric}, eval_interval={eval_interval}"
    )

    # Compute pos_weight from train labels
    y_train = y[train_mask]
    n_pos = y_train.sum().item()
    n_neg = (y_train.numel() - n_pos)
    pw = torch.tensor(max(n_neg / max(n_pos, 1), 1.0), device=device)
    print(f"[Phase2] pos_weight={pw.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    for epoch in range(1, epochs + 1):
        # Train
        loss, loss_dict = train_one_epoch(
            reasoner, base_z, base_logits, rel_features, judge_features,
            judge_mask_tensor, judge_align, y, train_mask, optimizer, p2_cfg, epoch,
            pos_weight=pw,
        )

        should_eval = epoch == 1 or epoch == epochs or (epoch % eval_interval == 0)
        val_metrics: dict[str, float] = {}
        diag: dict[str, float] = {}
        if should_eval:
            # Validation metrics and full-graph diagnostics are the CPU-heavy
            # part of Phase2 sweeps.  In sensitivity runs we can evaluate less
            # frequently while keeping epochs, patience, and train loss fixed.
            val_metrics = evaluate_phase2(
                reasoner, base_z, base_logits, rel_features, judge_features,
                judge_mask_tensor, val_mask, y, device, k_values=k_values,
            )

            diag = compute_epoch_diagnostics(
                reasoner, base_z, base_logits, rel_features, judge_features,
                judge_mask_tensor, device,
            )

        # Build epoch log row
        row: dict[str, float] = {"epoch": float(epoch), "loss": loss}
        row.update({f"loss/{k}": float(v) for k, v in loss_dict.items()})
        row.update({f"val/{k}": float(v) for k, v in val_metrics.items()})
        row.update(diag)
        # Named gate weights with relation names
        for r_idx, rname in enumerate(relation_names):
            key = f"gate_weight_rel_{r_idx}"
            if key in diag:
                row[f"gate/{rname}"] = diag[key]
        epoch_rows.append(row)

        # Print progress
        if should_eval and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"  Epoch {epoch:3d} | Loss {loss:.4f} | "
                f"Val AUPRC {val_metrics.get('auprc', 0):.4f} | "
                f"Val AUC {val_metrics.get('roc_auc', 0):.4f} | "
                f"Val Macro-F1 {val_metrics.get('macro_f1', 0):.4f}"
            )

        # Early stopping
        if should_eval:
            val_score = val_metrics.get(
                early_stop_metric.replace("val_", ""), val_metrics.get("auprc", 0),
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in reasoner.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += eval_interval
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
                    break

    elapsed = time.time() - start_time

    # ── Restore best + final test ──
    if best_state is not None:
        reasoner.load_state_dict(best_state)

    # Calibrate threshold on val, then evaluate test
    reasoner.eval()
    with torch.no_grad():
        val_out = reasoner(
            base_z[val_mask], base_logits[val_mask],
            rel_features[val_mask].to(device),
            judge_features=judge_features[val_mask].to(device) if judge_features is not None else None,
            judge_mask=judge_mask_tensor[val_mask].to(device) if judge_mask_tensor is not None else None,
        )
        val_prob = torch.sigmoid(val_out["final_logit"]).cpu().numpy()
    val_y = y[val_mask].cpu().numpy()
    best_threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")

    test_metrics = evaluate_phase2(
        reasoner, base_z, base_logits, rel_features, judge_features,
        judge_mask_tensor, test_mask, y, device,
        threshold=best_threshold, k_values=k_values,
    )

    print(f"\n=== Phase2 Test Results (threshold={best_threshold:.3f}) ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # ── Save artifacts ──
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))

    # Checkpoint
    torch.save(
        {"model_state_dict": best_state, "config": p2_cfg, "seed": seed, "epoch": best_epoch},
        ckpt_dir / "reasoner.pt",
    )

    # Epoch log
    write_phase2_epoch_log(log_dir, epoch_rows)

    # Test metrics (stage3_metrics.json for backward compat)
    with open(results_dir / "stage3_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Diagnostics summary
    final_diag = compute_epoch_diagnostics(
        reasoner, base_z, base_logits, rel_features, judge_features,
        judge_mask_tensor, device,
    )
    diagnostics = {
        "config": config,
        "phase2_reasoner": p2_cfg,
        "seed": seed,
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "device": device_info,
        "git_hash": get_git_hash(),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "early_stop_metric": early_stop_metric,
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "final_diagnostics": final_diag,
        "relation_feature_meta": rel_meta,
        "num_accepted_judge": int(judge_mask_tensor.sum().item()),
    }
    with open(log_dir / "phase2_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    print(f"\nCheckpoint: {ckpt_dir / 'reasoner.pt'}")
    print(f"Metrics:    {results_dir / 'stage3_metrics.json'}")
    print(f"Log:        {log_dir / 'phase2_train_log.jsonl'}")
    print(f"Diag:       {log_dir / 'phase2_diagnostics.json'}")


if __name__ == "__main__":
    main()
