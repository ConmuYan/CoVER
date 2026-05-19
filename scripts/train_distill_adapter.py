"""Phase 2 distillation adapter trainer (Idea 2C).

Trains a lightweight ``RelDistillAdapter`` to mimic a frozen CoVER-REL teacher.
The adapter is ~5K params vs teacher's ~14K; at inference it replaces the full
teacher pipeline (R expert MLPs + gate net + per-relation heads) with a single
small MLP.

CLI::

    python scripts/train_distill_adapter.py \\
        --config configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml \\
        --seed 42 \\
        --device cuda:2 \\
        --run_name idea2c_distill_adapter \\
        --teacher_ckpt artifacts/checkpoints/yelpchi/bwgnn/idea1_canonical_clsonly/seed_42/reasoner.pt \\
        --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/fixed_v1_100ep/seed_42/base.pt

Outputs::

    artifacts/checkpoints/{ds}/{base}/idea2c_distill_adapter/seed_{s}/adapter.pt
    artifacts/results/{ds}/{base}/idea2c_distill_adapter/seed_{s}/stage3_metrics.json
    artifacts/results/{ds}/{base}/idea2c_distill_adapter/seed_{s}/phase2_summary.json
    artifacts/logs/{ds}/{base}/idea2c_distill_adapter/seed_{s}/phase2_diagnostics.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.learned_extractor import build_learned_extractor
from evidence.relation_features import RELATION_SCHEMAS, load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.gnn import build_detector
from models.rel_distill_adapter import RelDistillAdapter
from training.metrics import compute_metrics, g_means, precision_recall_at_k
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir
from utils.threshold import evaluate_with_threshold, find_best_threshold


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


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ════════════════════════════════════════════════════════════════════
# Data loading (reuses existing infrastructure)
# ════════════════════════════════════════════════════════════════════

def get_base_output_cache_path(dataset: str, model: str, seed: int) -> Path:
    return Path("artifacts") / "base_outputs" / dataset / model / f"seed_{seed}" / "base_outputs.pt"


def load_frozen_base(config, dataset_name, model_name, seed, data, device, ckpt_override=None):
    """Load frozen base and return (base_logits, base_z, ckpt_path)."""
    if ckpt_override is not None:
        ckpt_path = Path(ckpt_override)
    else:
        from utils.paths import get_base_checkpoint_path
        ckpt_path = get_base_checkpoint_path(dataset_name, model_name, seed)

    default_cache = get_base_output_cache_path(dataset_name, model_name, seed)
    if ckpt_override is not None:
        override_cache = default_cache.parent / f"_override_{ckpt_path.parent.name}_{ckpt_path.stem}.pt"
    else:
        override_cache = None

    # Try all possible cache paths
    cache_candidates = [default_cache]
    if override_cache is not None:
        cache_candidates.append(override_cache)
    # Also try any _override_*.pt in the directory
    if default_cache.parent.exists():
        for f in sorted(default_cache.parent.glob("_override_*.pt")):
            if f not in cache_candidates:
                cache_candidates.append(f)

    for cache_path in cache_candidates:
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception:
                continue
            meta = payload.get("meta", {})
            if int(meta.get("num_nodes", -1)) == int(data.x.shape[0]):
                base_logits = payload["base_logits"].to(device).detach()
                base_z = payload["base_z"].to(device).detach()
                print(f"[Distill] Loaded cached base outputs from {cache_path}")
                return base_logits, base_z, ckpt_path

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

    base_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"[Distill] Loaded base checkpoint from {ckpt_path}")

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    with torch.no_grad():
        out = base_model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = out.logits.detach()
        base_z = out.embeddings.detach()

    return base_logits, base_z, ckpt_path


def load_relation_features(dataset_name, model_name, seed, num_nodes):
    rel_path = (
        Path("artifacts") / "relation_features" / dataset_name / model_name
        / f"seed_{seed}" / "all" / "rel_stats.pt"
    )
    if not rel_path.exists():
        raise FileNotFoundError(f"Relation features not found: {rel_path}")
    rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=num_nodes)
    print(f"[Distill] Loaded relation features {tuple(rel_stats.shape)} from {rel_path}")
    return rel_stats, rel_meta


def load_relation_adjs_for_extractor(
    dataset_name: str,
    dataset_path: str | Path,
    relation_names: list[str],
    num_nodes: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load per-relation sparse COO adjacencies for a learned extractor."""
    from scipy.io import loadmat
    from scipy.sparse import coo_matrix, issparse

    schema = RELATION_SCHEMAS[dataset_name]
    mat = loadmat(str(dataset_path))
    adjs: list[torch.Tensor] = []
    for rel_name in relation_names:
        key = rel_name.upper()
        if key not in schema:
            raise KeyError(f"relation {key!r} not in schema for {dataset_name}")
        mat_key = schema[key]["mat_key"]
        if mat_key not in mat:
            raise KeyError(f"matrix key {mat_key!r} not found in {dataset_path}")
        m = mat[mat_key]
        sp = m.tocoo() if issparse(m) else coo_matrix(m)
        if sp.shape[0] != num_nodes or sp.shape[1] != num_nodes:
            raise ValueError(
                f"adj shape {sp.shape} != ({num_nodes}, {num_nodes}) for relation {key}"
            )
        indices = torch.tensor(
            np.vstack([sp.row, sp.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(sp.data, dtype=torch.float32, device=device)
        adjs.append(torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce())
    return adjs


@torch.no_grad()
def build_learned_teacher_features(
    config: dict,
    data,
    relation_names: list[str],
    extractor_ckpt_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """Recreate frozen Idea-2B learned evidence used by the teacher."""
    p2_cfg = config["phase2_reasoner"]
    evidence_cfg = p2_cfg.get("evidence", {}) or {}
    ext_cfg = evidence_cfg.get("extractor", {}) or {}
    extractor = build_learned_extractor(
        x_dim=int(data.x.shape[1]),
        num_relations=len(relation_names),
        cfg={**ext_cfg, "out_dim_per_rel": int(ext_cfg.get("out_dim_per_rel", 9))},
    ).to(device)
    extractor.load_state_dict(torch.load(extractor_ckpt_path, weights_only=True, map_location=device))

    dataset_path = config["dataset"].get("path")
    if dataset_path is None:
        raise ValueError("dataset.path required for learned teacher evidence")
    x_dev = data.x.to(device)
    relation_adjs = load_relation_adjs_for_extractor(
        dataset_name=config["dataset"]["name"],
        dataset_path=dataset_path,
        relation_names=relation_names,
        num_nodes=int(data.x.shape[0]),
        device=device,
    )
    extractor.prepare(x_dev, relation_adjs)
    extractor.eval()
    rel_features = extractor(x_dev, data.train_mask.to(device), data.y.to(device).long())
    print(
        f"[Distill] Built learned teacher features {tuple(rel_features.shape)} "
        f"from {extractor_ckpt_path}"
    )
    return rel_features.detach()


# ════════════════════════════════════════════════════════════════════
# Teacher cache generation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_teacher_cache(
    teacher: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run frozen teacher on all nodes; return Δ_rel and gate probs."""
    teacher.eval()
    out = teacher(base_z, base_logits, rel_features.to(device))
    return {
        "delta_rel": out["delta_rel"].detach().cpu(),
        "gate_probs": out["relation_gate"].detach().cpu(),
    }


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_adapter(
    adapter: RelDistillAdapter,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    adapter.eval()
    out = adapter(base_z[mask], base_logits[mask], rel_features[mask].to(device))
    prob = torch.sigmoid(out["final_logit"]).cpu().numpy()
    # Sanitize NaN (can occur if base_logits contain extreme values)
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_np = y[mask].cpu().numpy()

    if threshold is not None:
        metrics = evaluate_with_threshold(y_np, prob, threshold)
        pred = (prob >= threshold).astype(int)
        metrics["g_means"] = g_means(y_np, pred)
        for k in (k_values or [50, 100, 200]):
            pk, rk = precision_recall_at_k(y_np, prob, k)
            metrics[f"precision@{k}"] = pk
            metrics[f"recall@{k}"] = rk
        return metrics, prob, y_np
    return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200]), prob, y_np


@torch.no_grad()
def evaluate_base_only(
    base_logits: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate base-only (no REL, no adapter)."""
    prob = torch.sigmoid(base_logits[mask]).cpu().numpy()
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_np = y[mask].cpu().numpy()
    if threshold is not None:
        metrics = evaluate_with_threshold(y_np, prob, threshold)
        pred = (prob >= threshold).astype(int)
        metrics["g_means"] = g_means(y_np, pred)
        for k in (k_values or [50, 100, 200]):
            pk, rk = precision_recall_at_k(y_np, prob, k)
            metrics[f"precision@{k}"] = pk
            metrics[f"recall@{k}"] = rk
        return metrics
    return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200])


@torch.no_grad()
def evaluate_teacher(
    teacher: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate full CoVER-REL teacher."""
    teacher.eval()
    out = teacher(base_z[mask], base_logits[mask], rel_features[mask].to(device))
    prob = torch.sigmoid(out["final_logit"]).cpu().numpy()
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_np = y[mask].cpu().numpy()
    if threshold is not None:
        metrics = evaluate_with_threshold(y_np, prob, threshold)
        pred = (prob >= threshold).astype(int)
        metrics["g_means"] = g_means(y_np, pred)
        for k in (k_values or [50, 100, 200]):
            pk, rk = precision_recall_at_k(y_np, prob, k)
            metrics[f"precision@{k}"] = pk
            metrics[f"recall@{k}"] = rk
        return metrics
    return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200])


# ════════════════════════════════════════════════════════════════════
# Distillation loss
# ════════════════════════════════════════════════════════════════════

def compute_distill_loss(
    adapter_out: dict[str, torch.Tensor],
    y: torch.Tensor,
    train_mask: torch.Tensor,
    teacher_delta: torch.Tensor,
    teacher_gate: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    lambda_distill: float = 1.0,
    gamma_kl: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Three-way distillation loss.

    L = L_bce + λ_distill · MSE(Δ_φ, sg(Δ_rel)) + γ_kl · KL(π_φ ‖ sg(π_rel))

    All teacher tensors are stop-gradient (detached before loss).
    """
    device = adapter_out["final_logit"].device
    train_bool = train_mask.to(device).view(-1).to(torch.bool)
    y_f = y.to(device=device, dtype=torch.float32).view(-1)

    n_train = int(train_bool.sum().item())
    if n_train == 0:
        zero = adapter_out["final_logit"].sum() * 0.0
        return zero, {"l_bce": 0.0, "l_distill": 0.0, "l_kl": 0.0, "total": 0.0}

    # L_bce
    l_bce = F.binary_cross_entropy_with_logits(
        adapter_out["final_logit"][train_bool], y_f[train_bool], pos_weight=pos_weight,
    )

    # L_distill: MSE(Δ_φ, sg(Δ_rel))
    teacher_delta_dev = teacher_delta.to(device=device, dtype=torch.float32).view(-1)
    delta_adapter = adapter_out["delta_phi"][train_bool]
    delta_teacher = teacher_delta_dev[train_bool].detach()
    l_distill = F.mse_loss(delta_adapter, delta_teacher)

    # L_kl: KL(adapter ‖ teacher) = sum(adapter * (log(adapter) - log(teacher)))
    # Spec: KL(softmax(π_φ) ‖ sg(softmax(π_rel_teacher)))
    # F.kl_div(log_input, target) = sum(target * (log(target) - log_input))
    # → log_input = log(teacher), target = adapter  ⟹  KL(adapter ‖ teacher)
    teacher_gate_dev = teacher_gate.to(device=device, dtype=torch.float32)
    gate_adapter = adapter_out["gate_probs"][train_bool]
    gate_teacher = teacher_gate_dev[train_bool].detach()
    eps = 1e-8
    l_kl = F.kl_div(
        (gate_teacher + eps).log(),   # log(teacher) — input
        gate_adapter,                  # adapter — target
        reduction="batchmean",
    )

    total = l_bce + lambda_distill * l_distill + gamma_kl * l_kl

    stats = {
        "l_bce": float(l_bce.detach().item()),
        "l_distill": float(l_distill.detach().item()),
        "l_kl": float(l_kl.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, stats


# ════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════

def train_distill_adapter(
    *,
    adapter: RelDistillAdapter,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    teacher_cache: dict[str, torch.Tensor],
    data,
    distill_cfg: dict,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> tuple[dict, list[dict]]:
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(distill_cfg.get("lr", 1e-3)),
        weight_decay=float(distill_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(distill_cfg.get("epochs", 300))
    patience = int(distill_cfg.get("patience", 50))
    lambda_distill = float(distill_cfg.get("lambda_distill", 1.0))
    gamma_kl = float(distill_cfg.get("gamma_kl", 0.5))

    teacher_delta = teacher_cache["delta_rel"]
    teacher_gate = teacher_cache["gate_probs"]

    best_val = -float("inf")
    best_state: dict | None = None
    best_epoch = -1
    no_improve = 0
    rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        adapter.train()
        out = adapter(base_z, base_logits, rel_features)
        loss, loss_stats = compute_distill_loss(
            out, y, train_mask, teacher_delta, teacher_gate,
            pos_weight=pos_weight,
            lambda_distill=lambda_distill,
            gamma_kl=gamma_kl,
        )
        optimizer.zero_grad()
        loss.backward()
        grad_clip = distill_cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics, _, _ = evaluate_adapter(
                adapter, base_z, base_logits, rel_features, val_mask, y, device,
            )

            row = {
                "epoch": epoch,
                "loss": loss_stats["total"],
                "l_bce": loss_stats["l_bce"],
                "l_distill": loss_stats["l_distill"],
                "l_kl": loss_stats["l_kl"],
                "lr": optimizer.param_groups[0]["lr"],
                "val/auprc": float(val_metrics.get("auprc", 0.0)),
                "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
            }
            rows.append(row)

            metric_value = float(val_metrics.get("auprc", 0.0))
            if metric_value > best_val:
                best_val = metric_value
                best_epoch = epoch
                best_state = copy.deepcopy(adapter.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[Distill] Early stop at epoch {epoch} (best val_auprc={best_val:.4f} @ {best_epoch})")
                    break

    if best_state is not None:
        adapter.load_state_dict(best_state)

    # Final eval with best weights
    _, val_prob, val_y = evaluate_adapter(
        adapter, base_z, base_logits, rel_features, val_mask, y, device,
    )
    threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")
    val_metrics, _, _ = evaluate_adapter(
        adapter, base_z, base_logits, rel_features, val_mask, y, device,
        threshold=threshold,
    )
    test_metrics, _, _ = evaluate_adapter(
        adapter, base_z, base_logits, rel_features, test_mask, y, device,
        threshold=threshold,
    )

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, rows


# ════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════

def write_epoch_log(log_dir: Path, rows: list[dict]) -> None:
    jsonl_path = log_dir / "distill_train_log.jsonl"
    csv_path = log_dir / "distill_train_log.csv"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 distillation adapter trainer (Idea 2C)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_name", type=str, default="idea2c_distill_adapter")
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="Path to frozen CoVER-REL teacher reasoner.pt")
    p.add_argument("--teacher_extractor_ckpt", type=str, default=None,
                   help="Optional Idea-2B learned evidence_extractor.pt for the teacher")
    p.add_argument("--base_ckpt_path", type=str, default=None,
                   help="Override path to base.pt")
    p.add_argument("--no-tensorboard", action="store_true")
    # Distillation hyperparams
    p.add_argument("--lambda_distill", type=float, default=1.0)
    p.add_argument("--gamma_kl", type=float, default=0.5)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    p2_cfg = config["phase2_reasoner"]
    seed = int(args.seed)
    run_name = args.run_name

    set_seed(seed)
    device = torch.device(args.device)

    # ----- Output dirs -----
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    result_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))

    # Repro
    repro_cmd = " ".join(sys.argv)
    (log_dir / "repro_command.txt").write_text(repro_cmd + "\n")
    (log_dir / "repro_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # ----- Load dataset -----
    ds_cfg = config["dataset"]
    data = load_fraud_dataset(
        name=dataset_name,
        path=ds_cfg.get("path"),
        format=ds_cfg.get("format"),
        seed=seed,
        scarcity_ratio=float(ds_cfg.get("scarcity_ratio", 1.0)),
        split_mode=ds_cfg.get("split_mode", "supervised"),
        train_ratio=float(ds_cfg.get("train_ratio", 0.4)),
        val_test_ratio=list(ds_cfg.get("val_test_ratio", [1, 2])),
        stratified=bool(ds_cfg.get("stratified", True)),
    )

    # ----- Load frozen base -----
    base_logits, base_z, base_ckpt_path = load_frozen_base(
        config, dataset_name, model_name, seed, data, device,
        ckpt_override=args.base_ckpt_path,
    )

    # ----- Load relation features -----
    rel_features, rel_meta = load_relation_features(
        dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
    )
    rel_features = rel_features.to(device)

    relation_names: list[str] = [
        str(name).upper() for name in
        p2_cfg.get("relation_names", rel_meta.get("relations", []))
    ]
    if not relation_names:
        relation_names = list(RELATION_SCHEMAS[dataset_name].keys())
    num_relations = len(relation_names)

    # ----- Optional: replace hand-crafted evidence with frozen 2B learned evidence -----
    if args.teacher_extractor_ckpt is not None:
        extractor_ckpt_path = Path(args.teacher_extractor_ckpt)
        if not extractor_ckpt_path.exists():
            raise FileNotFoundError(f"Teacher extractor checkpoint not found: {extractor_ckpt_path}")
        rel_features = build_learned_teacher_features(
            config=config,
            data=data,
            relation_names=relation_names,
            extractor_ckpt_path=extractor_ckpt_path,
            device=device,
        )
        rel_meta = {
            **rel_meta,
            "evidence_source": "learned",
            "teacher_extractor_ckpt": str(extractor_ckpt_path),
        }

    # ----- Load frozen teacher -----
    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")

    teacher = CoVERRelReasoner(
        base_z_dim=int(base_z.shape[1]),
        relation_names=relation_names,
        anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=p2_cfg.get("rel_num_layers", 2),
        rel_dropout=p2_cfg.get("rel_dropout", 0.3),
        tau_gate=p2_cfg.get("tau_gate", 0.7),
        delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
        gate_mode=p2_cfg.get("gate_mode", "softmax"),
        evidence_groups=p2_cfg.get("evidence_groups", None),
        expert_shared=p2_cfg.get("expert_shared", False),
        residual_activation=p2_cfg.get("residual_activation", "tanh"),
    ).to(device)
    teacher_raw = torch.load(teacher_ckpt_path, weights_only=False, map_location=device)
    teacher_state = teacher_raw["model_state_dict"] if isinstance(teacher_raw, dict) and "model_state_dict" in teacher_raw else teacher_raw
    teacher.load_state_dict(teacher_state)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"[Distill] Loaded frozen teacher from {teacher_ckpt_path}")
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"[Distill] Teacher params: {n_teacher}")

    # ----- Generate teacher cache -----
    t0_cache = time.time()
    teacher_cache = generate_teacher_cache(teacher, base_z, base_logits, rel_features, device)
    print(f"[Distill] Teacher cache generated in {time.time() - t0_cache:.1f}s")

    # ----- Build adapter -----
    distill_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "lambda_distill": args.lambda_distill,
        "gamma_kl": args.gamma_kl,
    }
    adapter = RelDistillAdapter(
        base_z_dim=int(base_z.shape[1]),
        num_relations=num_relations,
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        delta_max=p2_cfg.get("delta_rel_max", 2.0),
    ).to(device)
    n_adapter = sum(p.numel() for p in adapter.parameters())
    print(f"[Distill] Adapter params: {n_adapter} ({n_adapter / n_teacher:.2f}x teacher)")

    # ----- Pos-weight -----
    y_train_int = data.y[data.train_mask].long()
    n_pos = int((y_train_int == 1).sum().item())
    n_neg = int((y_train_int == 0).sum().item())
    pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)), device=device)

    # ----- Train -----
    t0 = time.time()
    train_summary, rows = train_distill_adapter(
        adapter=adapter,
        base_z=base_z, base_logits=base_logits, rel_features=rel_features,
        teacher_cache=teacher_cache, data=data, distill_cfg=distill_cfg,
        device=device, pos_weight=pos_weight,
    )
    elapsed = time.time() - t0

    # ----- Save adapter -----
    write_epoch_log(log_dir, rows)
    torch.save(adapter.state_dict(), ckpt_dir / "adapter.pt")

    # ----- Evaluate baselines for comparison -----
    k_values = [50, 100, 200]

    # Base-only
    base_threshold = find_best_threshold(
        data.y[data.val_mask].numpy(),
        torch.sigmoid(base_logits[data.val_mask]).cpu().numpy(),
        metric="macro_f1",
    )[0]
    base_only_metrics = evaluate_base_only(
        base_logits, data.test_mask.to(device), data.y.to(device),
        threshold=base_threshold, k_values=k_values,
    )

    # Full teacher — calibrate threshold on val, then evaluate on test
    teacher.eval()
    with torch.no_grad():
        teacher_val_out = teacher(base_z[data.val_mask.to(device)], base_logits[data.val_mask.to(device)], rel_features[data.val_mask.to(device)].to(device))
        teacher_val_prob = torch.sigmoid(teacher_val_out["final_logit"]).cpu().numpy()
        teacher_val_y = data.y[data.val_mask].cpu().numpy()
    teacher_threshold = find_best_threshold(teacher_val_y, teacher_val_prob, metric="macro_f1")[0]
    teacher_metrics = evaluate_teacher(
        teacher, base_z, base_logits, rel_features,
        data.test_mask.to(device), data.y.to(device), device,
        threshold=teacher_threshold, k_values=k_values,
    )

    # ----- Write diagnostics -----
    diagnostics = {
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "git_hash": get_git_hash(),
        "elapsed_seconds": elapsed,
        "adapter_params": n_adapter,
        "teacher_params": n_teacher,
        "param_ratio": round(n_adapter / n_teacher, 4),
        "teacher_ckpt": str(teacher_ckpt_path),
        "relation_names": relation_names,
        "distill_cfg": distill_cfg,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics_adapter": train_summary["test_metrics"],
        "test_metrics_base_only": base_only_metrics,
        "test_metrics_teacher": teacher_metrics,
    }
    (log_dir / "phase2_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    # ----- Write results (same format as train_phase2_reasoner.py) -----
    summary = {
        "run_name": run_name,
        "seed": seed,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics": train_summary["test_metrics"],
        "elapsed_seconds": elapsed,
    }
    (result_dir / "phase2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (result_dir / "stage3_metrics.json").write_text(
        json.dumps(train_summary["test_metrics"], indent=2, sort_keys=True) + "\n"
    )

    # ----- Console summary -----
    ta = train_summary["test_metrics"]
    tb = base_only_metrics
    tt = teacher_metrics
    print(f"\n[Distill] Done in {elapsed:.1f}s | best_epoch={train_summary['best_epoch']}")
    print(f"  Adapter AUPRC  = {ta.get('auprc', 0.0):.4f}  AUROC = {ta.get('roc_auc', 0.0):.4f}")
    print(f"  Base-only AUPRC= {tb.get('auprc', 0.0):.4f}  AUROC = {tb.get('roc_auc', 0.0):.4f}")
    print(f"  Teacher AUPRC  = {tt.get('auprc', 0.0):.4f}  AUROC = {tt.get('roc_auc', 0.0):.4f}")
    print(f"  Params: adapter={n_adapter} teacher={n_teacher} ratio={n_adapter / n_teacher:.2f}x")
    print(f"\nCheckpoint: {ckpt_dir / 'adapter.pt'}")
    print(f"Metrics:    {result_dir / 'stage3_metrics.json'}")


if __name__ == "__main__":
    main()
