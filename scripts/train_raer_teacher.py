"""RAER teacher trainer — relation-aware evidence residual learning.

This is the canonical teacher trainer for standard frozen base detectors
(BWGNN / SAGE / GCN / GAT). It trains the RAER teacher with a single
``L = L_cls`` objective.

CLI::

    python scripts/train_raer_teacher.py \\
        --config configs/raer_fd/teacher/raer_lree/yelpchi_bwgnn.yaml \\
        --seed 42 \\
        --device cuda:0 \\
        --run_name raer_lree \\
        --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt

Outputs::

    artifacts/checkpoints/{ds}/{base}/{run}/seed_{s}/raer_teacher.pt
    artifacts/results/{ds}/{base}/{run}/seed_{s}/test_metrics.json
    artifacts/results/{ds}/{base}/{run}/seed_{s}/raer_summary.json
    artifacts/logs/{ds}/{base}/{run}/seed_{s}/raer_diagnostics.json
    artifacts/logs/{ds}/{base}/{run}/seed_{s}/raer_train_log.{jsonl,csv}
    artifacts/logs/{ds}/{base}/{run}/seed_{s}/repro_command.txt
    artifacts/logs/{ds}/{base}/{run}/seed_{s}/repro_config.yaml
    artifacts/tensorboard/raer/{ds}/{base}/{run}/seed_{s}/  (optional)
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Set CUBLAS workspace config BEFORE importing torch (matches BWGNN protocol).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.lree import (
    LearnedRelationEvidenceExtractor,
    build_lree_extractor,
)
from evidence.relation_features import (
    RELATION_SCHEMAS,
    load_relation_stats,
)
from evidence.scalable_lree import ScalableLREE, build_scalable_lree_extractor
from models.raer_teacher import RAERTeacher
from models.gnn import build_detector
from training.metrics import compute_metrics, g_means, precision_recall_at_k
from training.raer_losses import compute_raer_loss
from utils.paths import (
    ensure_dir,
    get_base_checkpoint_path,
    get_checkpoint_dir,
    get_logs_dir,
    get_results_dir,
)
from utils.tensorboard import HAS_TENSORBOARD, TensorBoardLogger
from utils.threshold import evaluate_with_threshold, find_best_threshold


# ════════════════════════════════════════════════════════════════════
# Reproducibility helpers
# ════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Set deterministic seeds across libraries (BWGNN-faithful protocol)."""
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


def runtime_device_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        info["cuda_device_index"] = idx
        info["cuda_device_name"] = torch.cuda.get_device_name(idx)
    return info


# ════════════════════════════════════════════════════════════════════
# Base-freeze SHA-256 contract
# ════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    flat = t.detach().cpu().contiguous().view(-1)
    chunk = 1_000_000
    for start in range(0, flat.numel(), chunk):
        h.update(flat[start:start + chunk].numpy().tobytes())
    return h.hexdigest()


def snapshot_base_freeze(
    ckpt_path: Path,
    base_logits: torch.Tensor,
    base_z: torch.Tensor,
) -> dict[str, str | None]:
    return {
        "base_ckpt_sha256": _sha256_file(ckpt_path),
        "base_logits_sha256": _sha256_tensor(base_logits),
        "base_z_sha256": _sha256_tensor(base_z),
    }


def verify_base_frozen(
    before: dict[str, str | None],
    after: dict[str, str | None],
) -> dict[str, object]:
    diffs = [
        k for k in ("base_ckpt_sha256", "base_logits_sha256", "base_z_sha256")
        if before.get(k) != after.get(k)
    ]
    return {
        "before": before,
        "after": after,
        "differences": diffs,
        "verdict": "frozen" if not diffs else "MUTATED",
    }


# ════════════════════════════════════════════════════════════════════
# Base + relation feature loading
# ════════════════════════════════════════════════════════════════════

def get_base_output_cache_path(dataset: str, model: str, seed: int) -> Path:
    return Path("artifacts") / "base_outputs" / dataset / model / f"seed_{seed}" / "base_outputs.pt"


def get_base_output_override_cache_paths(default_cache: Path, ckpt_path: Path) -> list[Path]:
    """Cache candidates for an explicit base checkpoint override.

    The new name includes both the run directory and seed directory. The legacy
    fallback preserves compatibility with older caches named only by
    ``seed_x/base.pt``.
    """
    seed_dir = ckpt_path.parent.name
    run_dir = ckpt_path.parent.parent.name if ckpt_path.parent.parent != ckpt_path.parent else "override"
    new_path = default_cache.parent / f"_override_{run_dir}_{seed_dir}_{ckpt_path.stem}.pt"
    legacy_path = default_cache.parent / f"_override_{seed_dir}_{ckpt_path.stem}.pt"
    if new_path == legacy_path:
        return [new_path]
    return [new_path, legacy_path]


def load_frozen_base(
    config: dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    data,
    device: torch.device,
    ckpt_override: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Load frozen base detector and return cached ``(base_logits, base_z, ckpt_path)``.

    Cache key includes the ckpt path so different bases (e.g. 100ep vs 400ep)
    cannot collide.
    """
    if ckpt_override is not None:
        ckpt_path = Path(ckpt_override)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"base ckpt_override not found: {ckpt_path}")
        default_cache = get_base_output_cache_path(dataset_name, model_name, seed)
        cache_candidates = get_base_output_override_cache_paths(default_cache, ckpt_path)
        cache_path = cache_candidates[0]
    else:
        ckpt_path = get_base_checkpoint_path(dataset_name, model_name, seed)
        cache_path = get_base_output_cache_path(dataset_name, model_name, seed)
        cache_candidates = [cache_path]

    for candidate in cache_candidates:
        if not candidate.exists():
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {})
        if (
            int(meta.get("num_nodes", -1)) == int(data.x.shape[0])
            and str(meta.get("checkpoint_path", "")) == str(ckpt_path)
        ):
            base_logits = payload["base_logits"].float().to(device).detach()
            base_z = payload["base_z"].float().to(device).detach()
            print(f"[RAER] Loaded cached base outputs from {candidate}")
            return base_logits, base_z, ckpt_path

    extra_kwargs: dict[str, object] = {}
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

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    base_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"[RAER] Loaded base checkpoint from {ckpt_path}")

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    with torch.no_grad():
        out = base_model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = out.logits.detach()
        base_z = out.embeddings.detach()

    ensure_dir(cache_path.parent)
    torch.save(
        {
            "base_logits": base_logits.detach().cpu(),
            "base_z": base_z.detach().cpu(),
            "meta": {
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(seed),
                "num_nodes": int(data.x.shape[0]),
                "checkpoint_path": str(ckpt_path),
                "git_hash": get_git_hash(),
            },
        },
        cache_path,
    )
    print(f"[RAER] Cached base outputs to {cache_path}")
    return base_logits, base_z, ckpt_path


def load_relation_features_for_raer(
    dataset_name: str, model_name: str, seed: int, num_nodes: int,
) -> tuple[torch.Tensor, dict]:
    rel_path = (
        Path("artifacts") / "relation_features" / dataset_name / model_name
        / f"seed_{seed}" / "all" / "rel_stats.pt"
    )
    if not rel_path.exists():
        raise FileNotFoundError(f"Relation features not found: {rel_path}")
    rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=num_nodes)
    print(f"[RAER] Loaded relation features {tuple(rel_stats.shape)} from {rel_path}")
    return rel_stats, rel_meta


def load_relation_adjs_for_extractor(
    dataset_name: str,
    dataset_path: str | Path,
    relation_names: list[str],
    num_nodes: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load per-relation sparse-COO adjacency matrices from the .mat file.

    Used by LREE to compute online evidence features.
    Returns adjacencies in the same order as ``relation_names``.
    """
    from scipy.io import loadmat
    from scipy.sparse import issparse, coo_matrix

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
        adj = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes)
        ).coalesce()
        adjs.append(adj)
        print(f"[RAER] Loaded adj for {key}: nnz={adj._nnz()}")
    return adjs


def build_sparse_adj_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge_index = edge_index.to(device=device, dtype=torch.long)
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        edge_index, values, (num_nodes, num_nodes),
    ).coalesce()


def load_relation_adjs_for_extractor_from_data(
    dataset_name: str,
    dataset_path: str | Path,
    relation_names: list[str],
    data,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load LREE adjacencies from either .mat relations or data.edge_index."""
    schema = RELATION_SCHEMAS[dataset_name]
    adjs: list[torch.Tensor] = []
    mat_relation_names: list[str] = []
    mat_positions: list[int] = []
    for pos, rel_name in enumerate(relation_names):
        key = str(rel_name).upper()
        if key not in schema:
            raise KeyError(f"relation {key!r} not in schema for {dataset_name}")
        mat_key = schema[key]["mat_key"]
        if mat_key == "edge_index":
            adjs.append(
                build_sparse_adj_from_edge_index(
                    data.edge_index,
                    num_nodes=int(data.x.shape[0]),
                    device=device,
                )
            )
            print(f"[RAER] Loaded adj for {key} from data.edge_index: nnz={adjs[-1]._nnz()}")
        else:
            adjs.append(None)  # type: ignore[arg-type]
            mat_relation_names.append(key)
            mat_positions.append(pos)

    if mat_relation_names:
        loaded = load_relation_adjs_for_extractor(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            relation_names=mat_relation_names,
            num_nodes=int(data.x.shape[0]),
            device=device,
        )
        for pos, adj in zip(mat_positions, loaded):
            adjs[pos] = adj
    return adjs


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_split(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    reasoner.eval()
    out = reasoner(base_z[mask], base_logits[mask], rel_features[mask].to(device))
    prob = torch.sigmoid(out["final_logit"]).cpu().numpy()
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
def evaluate_split_chunked(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
    chunk_size: int = 131_072,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    reasoner.eval()
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, idx.numel(), chunk):
        batch_idx = idx[start:start + chunk]
        out = reasoner(
            base_z[batch_idx],
            base_logits[batch_idx],
            rel_features[batch_idx].to(device),
        )
        probs.append(torch.sigmoid(out["final_logit"]).cpu())
        labels.append(y[batch_idx].cpu())
    prob = torch.cat(probs).numpy() if probs else np.asarray([], dtype=np.float32)
    y_np = torch.cat(labels).numpy() if labels else np.asarray([], dtype=np.int64)
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
def compute_full_diagnostics(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Per-epoch observability: mean |delta_rel|, gate entropy, dominance rho."""
    reasoner.eval()
    out = reasoner(base_z, base_logits, rel_features.to(device))
    diag: dict[str, float] = {}
    delta_rel = out["delta_rel"]
    diag["mean_abs_delta_rel"] = float(delta_rel.abs().mean().item())

    gate = out["relation_gate"]
    eps = 1e-8
    log_g = torch.log(gate.clamp_min(eps))
    ent = -(gate * log_g).sum(dim=-1)
    diag["mean_gate_entropy"] = float(ent.mean().item())
    R = gate.shape[1]
    if R >= 2:
        rho = (1.0 - ent / float(np.log(R))).clamp(min=0.0, max=1.0)
        diag["mean_dominance_rho"] = float(rho.mean().item())
        for r in range(R):
            diag[f"gate_weight_rel_{r}"] = float(gate[:, r].mean().item())
    return diag


@torch.no_grad()
def compute_diagnostics_sampled(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    device: torch.device,
    max_nodes: int = 200_000,
) -> dict[str, float]:
    n = int(base_z.shape[0])
    if n <= max_nodes:
        return compute_full_diagnostics(reasoner, base_z, base_logits, rel_features, device)
    idx = torch.linspace(0, n - 1, steps=int(max_nodes), dtype=torch.long, device=base_z.device)
    return compute_full_diagnostics(
        reasoner,
        base_z[idx],
        base_logits[idx],
        rel_features[idx],
        device,
    )


# ════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════

def write_epoch_log(log_dir: Path, rows: list[dict]) -> None:
    jsonl_path = log_dir / "raer_train_log.jsonl"
    csv_path = log_dir / "raer_train_log.csv"
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


def default_tensorboard_dir(dataset: str, model: str, run_name: str, seed: int) -> Path:
    return Path("artifacts") / "tensorboard" / "raer" / dataset / model / run_name / f"seed_{seed}"


def log_to_tensorboard(
    logger: TensorBoardLogger | None,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
    diag: dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> None:
    if logger is None:
        return

    def emit(tag: str, value: object) -> None:
        try:
            v = float(value)
            if np.isfinite(v):
                logger.log_scalar(tag, v, epoch)
        except (TypeError, ValueError):
            pass

    emit("train/loss", train_loss)
    emit("train/lr", optimizer.param_groups[0].get("lr", 0.0))
    for k in ("auprc", "roc_auc", "macro_f1", "g_means"):
        if k in val_metrics:
            emit(f"val/{k}", val_metrics[k])
    for k, v in diag.items():
        emit(f"diag/{k}", v)


# ════════════════════════════════════════════════════════════════════
# Reasoner construction + training
# ════════════════════════════════════════════════════════════════════

def build_raer_teacher(
    raer_cfg: dict,
    z_dim: int,
    relation_names: list[str],
    device: torch.device,
) -> RAERTeacher:
    """Construct the cls-only RAERTeacher from a RAER config block."""
    return RAERTeacher(
        base_z_dim=z_dim,
        relation_names=relation_names,
        anchor_relation=raer_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=raer_cfg.get("rel_stat_dim", 9),
        rel_hidden_dim=raer_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=raer_cfg.get("rel_num_layers", 2),
        rel_dropout=raer_cfg.get("rel_dropout", 0.3),
        tau_gate=raer_cfg.get("tau_gate", 0.7),
        delta_rel_max=raer_cfg.get("delta_rel_max", 2.0),
        gate_mode=raer_cfg.get("gate_mode", "softmax"),
        evidence_groups=raer_cfg.get("evidence_groups", None),
        expert_shared=raer_cfg.get("expert_shared", False),
        residual_activation=raer_cfg.get("residual_activation", "tanh"),
    ).to(device)


def train_one_epoch(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    raer_cfg: dict,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """One BCE-only training epoch (cls-only canonical)."""
    reasoner.train()
    device = base_z.device

    z_t = base_z[train_mask]
    bl_t = base_logits[train_mask]
    rf_t = rel_features[train_mask].to(device)
    y_t = y[train_mask]
    all_train = torch.ones(z_t.shape[0], dtype=torch.bool, device=device)

    out = reasoner(z_t, bl_t, rf_t)
    loss, stats = compute_raer_loss(out, y_t, all_train, pos_weight=pos_weight)

    optimizer.zero_grad()
    loss.backward()
    grad_clip = raer_cfg.get("grad_clip")
    if grad_clip is not None and float(grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_norm=float(grad_clip))
    optimizer.step()

    return float(loss.item()), stats


def train_one_epoch_indexed(
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    raer_cfg: dict,
    pos_weight: torch.Tensor | None = None,
    batch_size: int = 65_536,
    generator: torch.Generator | None = None,
) -> tuple[float, dict[str, float]]:
    reasoner.train()
    device = base_z.device
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(train_idx.numel(), generator=generator, device="cpu").to(train_idx.device)
    order = train_idx[perm]
    losses: list[float] = []
    stats_acc: dict[str, float] = {}
    total_seen = 0
    batch = max(int(batch_size), 1)
    for start in range(0, order.numel(), batch):
        idx = order[start:start + batch]
        z_t = base_z[idx]
        bl_t = base_logits[idx]
        rf_t = rel_features[idx].to(device)
        y_t = y[idx]
        all_train = torch.ones(idx.shape[0], dtype=torch.bool, device=device)
        out = reasoner(z_t, bl_t, rf_t)
        loss, stats = compute_raer_loss(out, y_t, all_train, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        grad_clip = raer_cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        seen = int(idx.numel())
        losses.append(float(loss.detach()) * seen)
        total_seen += seen
        for k, v in stats.items():
            stats_acc[k] = stats_acc.get(k, 0.0) + float(v) * seen

    denom = max(total_seen, 1)
    stats_mean = {k: v / denom for k, v in stats_acc.items()}
    return float(sum(losses) / denom), stats_mean


def train_one_epoch_scalable_lree(
    reasoner: RAERTeacher,
    extractor: ScalableLREE,
    relation_basis: torch.Tensor,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    raer_cfg: dict,
    pos_weight: torch.Tensor | None = None,
    batch_size: int = 65_536,
    generator: torch.Generator | None = None,
) -> tuple[float, dict[str, float]]:
    reasoner.train()
    extractor.train()
    device = base_z.device
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(train_idx.numel(), generator=generator, device="cpu").to(train_idx.device)
    order = train_idx[perm]
    losses: list[float] = []
    stats_acc: dict[str, float] = {}
    total_seen = 0
    batch = max(int(batch_size), 1)
    for start in range(0, order.numel(), batch):
        idx = order[start:start + batch]
        rel_features = extractor(relation_basis[idx].to(device))
        all_train = torch.ones(idx.shape[0], dtype=torch.bool, device=device)
        out = reasoner(base_z[idx], base_logits[idx], rel_features)
        loss, stats = compute_raer_loss(out, y[idx], all_train, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        grad_clip = raer_cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(reasoner.parameters()) + list(extractor.parameters()),
                max_norm=float(grad_clip),
            )
        optimizer.step()

        seen = int(idx.numel())
        losses.append(float(loss.detach()) * seen)
        total_seen += seen
        for k, v in stats.items():
            stats_acc[k] = stats_acc.get(k, 0.0) + float(v) * seen

    denom = max(total_seen, 1)
    stats_mean = {k: v / denom for k, v in stats_acc.items()}
    return float(sum(losses) / denom), stats_mean


@torch.no_grad()
def materialize_scalable_lree_features(
    extractor: ScalableLREE,
    relation_basis: torch.Tensor,
    device: torch.device,
    batch_size: int = 262_144,
    output_device: torch.device | str = "cpu",
) -> torch.Tensor:
    extractor.eval()
    parts: list[torch.Tensor] = []
    chunk = max(int(batch_size), 1)
    for start in range(0, relation_basis.shape[0], chunk):
        basis = relation_basis[start:start + chunk].to(device)
        parts.append(extractor(basis).detach().to(output_device))
    return torch.cat(parts, dim=0)


def run_training(
    *,
    reasoner: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    data,
    raer_cfg: dict,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    tb_logger: TensorBoardLogger | None,
    extractor: LearnedRelationEvidenceExtractor | None = None,
    evidence_inputs: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Full training loop with val-AUPRC early stopping and threshold calibration.

    If ``extractor`` is provided (LREE), ``rel_features`` is recomputed
    online each epoch via ``extractor(x, train_mask, train_labels)`` where
    ``evidence_inputs`` supplies the (device-resident) ``x``, ``train_mask``,
    ``train_labels`` tensors. The extractor's parameters are added to the
    optimizer alongside the reasoner's so gradients flow end-to-end.
    """
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)

    use_extractor = extractor is not None
    if use_extractor and evidence_inputs is None:
        raise ValueError("evidence_inputs required when extractor is provided")
    ev_x: torch.Tensor | None = None
    ev_train_mask: torch.Tensor | None = None
    ev_train_labels: torch.Tensor | None = None
    if use_extractor:
        ev_x = evidence_inputs["x"]
        ev_train_mask = evidence_inputs["train_mask"]
        ev_train_labels = evidence_inputs["train_labels"]

    params: list = list(reasoner.parameters())
    if use_extractor:
        params += list(extractor.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(raer_cfg.get("lr", 1e-3)),
        weight_decay=float(raer_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(raer_cfg.get("epochs", 300))
    patience = int(raer_cfg.get("patience", 50))
    eval_interval = int(raer_cfg.get("eval_interval", 1))
    select_metric = str(raer_cfg.get("early_stop_metric", "val_auprc")).replace("val_", "")

    best_val = -float("inf")
    best_state: dict | None = None
    best_extractor_state: dict | None = None
    best_epoch = -1
    no_improve = 0
    rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        # ----- Recompute rel_features online if learned extractor active -----
        if use_extractor:
            extractor.train()
            rel_features = extractor(ev_x, ev_train_mask, ev_train_labels)

        loss, loss_stats = train_one_epoch(
            reasoner, base_z, base_logits, rel_features, y, train_mask,
            optimizer, raer_cfg, pos_weight=pos_weight,
        )

        if epoch % eval_interval == 0 or epoch == epochs:
            # Eval-mode rel_features (no grad, no dropout) for val/test passes
            if use_extractor:
                extractor.eval()
                with torch.no_grad():
                    rel_features_eval = extractor(ev_x, ev_train_mask, ev_train_labels)
            else:
                rel_features_eval = rel_features

            val_metrics, _, _ = evaluate_split(
                reasoner, base_z, base_logits, rel_features_eval, val_mask, y, device,
            )
            diag = compute_full_diagnostics(reasoner, base_z, base_logits, rel_features_eval, device)

            row = {
                "epoch": epoch,
                "loss": loss,
                "lr": optimizer.param_groups[0]["lr"],
                "val/auprc": float(val_metrics.get("auprc", 0.0)),
                "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                "val/g_means": float(val_metrics.get("g_means", 0.0)),
                **{f"loss/{k}": float(v) for k, v in loss_stats.items()},
                **{f"diag/{k}": float(v) for k, v in diag.items()},
            }
            rows.append(row)
            log_to_tensorboard(tb_logger, epoch, loss, val_metrics, diag, optimizer)

            metric_value = float(val_metrics.get(select_metric, 0.0))
            if metric_value > best_val:
                best_val = metric_value
                best_epoch = epoch
                best_state = copy.deepcopy(reasoner.state_dict())
                if use_extractor:
                    best_extractor_state = copy.deepcopy(extractor.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[RAER] Early stop at epoch {epoch} (best {select_metric}={best_val:.4f} @ {best_epoch})")
                    break

    if best_state is not None:
        reasoner.load_state_dict(best_state)
    if use_extractor and best_extractor_state is not None:
        extractor.load_state_dict(best_extractor_state)

    # ----- Final eval with best weights; recompute rel_features once more -----
    if use_extractor:
        extractor.eval()
        with torch.no_grad():
            rel_features_final = extractor(ev_x, ev_train_mask, ev_train_labels)
    else:
        rel_features_final = rel_features

    # Calibrate threshold on val, evaluate on test
    _, val_prob, val_y = evaluate_split(
        reasoner, base_z, base_logits, rel_features_final, val_mask, y, device,
    )
    threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")
    val_metrics, _, _ = evaluate_split(
        reasoner, base_z, base_logits, rel_features_final, val_mask, y, device,
        threshold=threshold,
    )
    test_metrics, _, _ = evaluate_split(
        reasoner, base_z, base_logits, rel_features_final, test_mask, y, device,
        threshold=threshold,
    )
    diag_test = compute_full_diagnostics(reasoner, base_z, base_logits, rel_features_final, device)

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "diagnostics": diag_test,
    }, rows


def run_training_scalable_lree(
    *,
    reasoner: RAERTeacher,
    extractor: ScalableLREE,
    relation_basis: torch.Tensor,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    data,
    raer_cfg: dict,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    tb_logger: TensorBoardLogger | None,
) -> tuple[dict, list[dict]]:
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)
    train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)

    params: list = list(reasoner.parameters()) + list(extractor.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(raer_cfg.get("lr", 1e-3)),
        weight_decay=float(raer_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(raer_cfg.get("epochs", 300))
    patience = int(raer_cfg.get("patience", 50))
    eval_interval = int(raer_cfg.get("eval_interval", 1))
    select_metric = str(raer_cfg.get("early_stop_metric", "val_auprc")).replace("val_", "")
    batch_size = int(raer_cfg.get("batch_size", 65_536))
    eval_batch_size = int(raer_cfg.get("eval_batch_size", 262_144))
    feature_batch_size = int(raer_cfg.get("feature_batch_size", 262_144))

    best_val = -float("inf")
    best_state: dict | None = None
    best_extractor_state: dict | None = None
    best_epoch = -1
    no_improve = 0
    rows: list[dict] = []
    generator = torch.Generator(device="cpu").manual_seed(int(raer_cfg.get("shuffle_seed", 0)))

    for epoch in range(1, epochs + 1):
        loss, loss_stats = train_one_epoch_scalable_lree(
            reasoner=reasoner,
            extractor=extractor,
            relation_basis=relation_basis,
            base_z=base_z,
            base_logits=base_logits,
            y=y,
            train_idx=train_idx,
            optimizer=optimizer,
            raer_cfg=raer_cfg,
            pos_weight=pos_weight,
            batch_size=batch_size,
            generator=generator,
        )

        if epoch % eval_interval == 0 or epoch == epochs:
            rel_features_eval = materialize_scalable_lree_features(
                extractor,
                relation_basis,
                device,
                batch_size=feature_batch_size,
                output_device=device,
            )
            val_metrics, _, _ = evaluate_split_chunked(
                reasoner,
                base_z,
                base_logits,
                rel_features_eval,
                val_mask,
                y,
                device,
                chunk_size=eval_batch_size,
            )
            diag = compute_diagnostics_sampled(
                reasoner,
                base_z,
                base_logits,
                rel_features_eval,
                device,
                max_nodes=int(raer_cfg.get("diagnostic_max_nodes", 200_000)),
            )

            row = {
                "epoch": epoch,
                "loss": loss,
                "lr": optimizer.param_groups[0]["lr"],
                "val/auprc": float(val_metrics.get("auprc", 0.0)),
                "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                "val/g_means": float(val_metrics.get("g_means", 0.0)),
                **{f"loss/{k}": float(v) for k, v in loss_stats.items()},
                **{f"diag/{k}": float(v) for k, v in diag.items()},
            }
            rows.append(row)
            log_to_tensorboard(tb_logger, epoch, loss, val_metrics, diag, optimizer)

            metric_value = float(val_metrics.get(select_metric, 0.0))
            if metric_value > best_val:
                best_val = metric_value
                best_epoch = epoch
                best_state = copy.deepcopy(reasoner.state_dict())
                best_extractor_state = copy.deepcopy(extractor.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[RAER] Early stop at epoch {epoch} (best {select_metric}={best_val:.4f} @ {best_epoch})")
                    break

    if best_state is not None:
        reasoner.load_state_dict(best_state)
    if best_extractor_state is not None:
        extractor.load_state_dict(best_extractor_state)

    rel_features_final = materialize_scalable_lree_features(
        extractor,
        relation_basis,
        device,
        batch_size=feature_batch_size,
        output_device=device,
    )
    _, val_prob, val_y = evaluate_split_chunked(
        reasoner,
        base_z,
        base_logits,
        rel_features_final,
        val_mask,
        y,
        device,
        chunk_size=eval_batch_size,
    )
    threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")
    val_metrics, _, _ = evaluate_split_chunked(
        reasoner,
        base_z,
        base_logits,
        rel_features_final,
        val_mask,
        y,
        device,
        threshold=threshold,
        chunk_size=eval_batch_size,
    )
    test_metrics, _, _ = evaluate_split_chunked(
        reasoner,
        base_z,
        base_logits,
        rel_features_final,
        test_mask,
        y,
        device,
        threshold=threshold,
        chunk_size=eval_batch_size,
    )
    diag_test = compute_diagnostics_sampled(
        reasoner,
        base_z,
        base_logits,
        rel_features_final,
        device,
        max_nodes=int(raer_cfg.get("diagnostic_max_nodes", 200_000)),
    )

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "diagnostics": diag_test,
    }, rows


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAER teacher trainer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--base_ckpt_path", type=str, default=None,
                   help="Override path to base.pt (default: derived from config/seed).")
    p.add_argument("--single-stage", action="store_true", help="No-op; RAER teacher is cls-only.")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard writer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    raer_cfg = config["raer_teacher"]
    run_name = args.run_name
    seed = int(args.seed)

    set_seed(seed)
    device = torch.device(args.device)

    # ----- Setup output dirs -----
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    result_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))

    # Repro fingerprint
    repro_cmd = " ".join(sys.argv)
    (log_dir / "repro_command.txt").write_text(repro_cmd + "\n")
    (log_dir / "repro_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # ----- Load dataset (split is deterministic per seed) -----
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
        hsd_invert=ds_cfg.get("hsd_invert"),
        hsd_chunk_size=int(ds_cfg.get("hsd_chunk_size", 250_000)),
        append_hsd=bool(ds_cfg.get("append_hsd", True)),
    )

    # ----- Load frozen base -----
    base_logits, base_z, base_ckpt_path = load_frozen_base(
        config, dataset_name, model_name, seed, data, device,
        ckpt_override=args.base_ckpt_path,
    )
    base_freeze_before = snapshot_base_freeze(base_ckpt_path, base_logits, base_z)

    # ----- Resolve evidence source and relation names -----
    evidence_cfg = raer_cfg.get("evidence", {}) or {}
    evidence_source = str(evidence_cfg.get("source", "hand_crafted")).lower()
    if evidence_source not in {"hand_crafted", "learned", "scalable_lree"}:
        raise ValueError(
            "raer.evidence.source must be 'hand_crafted', 'learned', or "
            f"'scalable_lree', got {evidence_source!r}"
        )
    relation_names: list[str] = [
        str(name).upper() for name in raer_cfg.get("relation_names", [])
    ]
    if not relation_names:
        relation_names = list(RELATION_SCHEMAS[dataset_name].keys())

    relation_basis: torch.Tensor | None = None

    # ----- Load relation features only for RAER-HC -----
    if evidence_source == "hand_crafted":
        rel_features, rel_meta = load_relation_features_for_raer(
            dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
        )
        rel_features = rel_features.to(device)
        if not raer_cfg.get("relation_names"):
            relation_names = [str(name).upper() for name in rel_meta.get("relations", relation_names)]
    elif evidence_source == "learned":
        rel_meta = {
            "relations": relation_names,
            "source": "learned",
            "score_blind": True,
            "prototype_labels": "train_only",
        }
        ext_cfg_for_shape = evidence_cfg.get("extractor", {}) or {}
        out_dim_per_rel = int(ext_cfg_for_shape.get("out_dim_per_rel", raer_cfg.get("rel_stat_dim", 9)))
        rel_features = torch.empty(
            0,
            len(relation_names) * out_dim_per_rel,
            dtype=torch.float32,
        )
    else:
        relation_basis, rel_meta = load_relation_features_for_raer(
            dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
        )
        if not raer_cfg.get("relation_names"):
            relation_names = [str(name).upper() for name in rel_meta.get("relations", relation_names)]
        ext_cfg_for_shape = evidence_cfg.get("extractor", {}) or {}
        out_dim_per_rel = int(ext_cfg_for_shape.get("out_dim_per_rel", raer_cfg.get("rel_stat_dim", 9)))
        rel_features = torch.empty(
            0,
            len(relation_names) * out_dim_per_rel,
            dtype=torch.float32,
        )
        rel_meta = {
            **rel_meta,
            "source": "scalable_lree",
            "score_blind": True,
            "prototype_labels": rel_meta.get("prototype_labels", "train_only"),
            "basis": "compact_relation_features",
        }

    # ----- Optional: build LREE -----
    extractor: LearnedRelationEvidenceExtractor | None = None
    scalable_extractor: ScalableLREE | None = None
    evidence_inputs: dict | None = None
    if evidence_source == "learned":
        ext_cfg = evidence_cfg.get("extractor", {}) or {}
        rel_stat_dim_override = int(ext_cfg.get("out_dim_per_rel", 9))
        extractor = build_lree_extractor(
            x_dim=int(data.x.shape[1]),
            num_relations=len(relation_names),
            cfg={**ext_cfg, "out_dim_per_rel": rel_stat_dim_override},
        ).to(device)
        # Load per-relation adjacency matrices once
        dataset_path = ds_cfg.get("path")
        if dataset_path is None:
            raise ValueError("dataset.path required for learned extractor (.mat lookup)")
        x_dev = data.x.to(device)
        relation_adjs = load_relation_adjs_for_extractor_from_data(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            relation_names=relation_names,
            data=data,
            device=device,
        )
        extractor.prepare(x_dev, relation_adjs)
        evidence_inputs = {
            "x": x_dev,
            "train_mask": data.train_mask.to(device),
            "train_labels": data.y.to(device).long(),
        }
        n_params = sum(p.numel() for p in extractor.parameters())
        print(
            f"[RAER] LearnedRelationEvidenceExtractor active: "
            f"R={len(relation_names)} x_dim={data.x.shape[1]} "
            f"out_dim_per_rel={rel_stat_dim_override} params={n_params}"
        )
        # First forward to populate rel_features for any downstream code that
        # might read rel_features before run_training (e.g. logging).
        with torch.no_grad():
            rel_features = extractor(
                evidence_inputs["x"], evidence_inputs["train_mask"],
                evidence_inputs["train_labels"],
            )
    elif evidence_source == "scalable_lree":
        ext_cfg = evidence_cfg.get("extractor", {}) or {}
        if relation_basis is None:
            raise RuntimeError("relation_basis is required for scalable_lree")
        if relation_basis.shape[1] % len(relation_names) != 0:
            raise ValueError(
                "relation basis width is not divisible by number of relations: "
                f"{relation_basis.shape[1]} vs R={len(relation_names)}"
            )
        relation_basis = relation_basis.to(device)
        in_dim_per_rel = int(relation_basis.shape[1] // len(relation_names))
        rel_stat_dim_override = int(ext_cfg.get("out_dim_per_rel", raer_cfg.get("rel_stat_dim", 9)))
        scalable_extractor = build_scalable_lree_extractor(
            num_relations=len(relation_names),
            in_dim_per_rel=in_dim_per_rel,
            cfg={**ext_cfg, "out_dim_per_rel": rel_stat_dim_override},
        ).to(device)
        n_params = sum(p.numel() for p in scalable_extractor.parameters())
        print(
            f"[RAER] ScalableLREE active: R={len(relation_names)} "
            f"in_dim_per_rel={in_dim_per_rel} out_dim_per_rel={rel_stat_dim_override} "
            f"params={n_params}"
        )

    # ----- Construct cls-only reasoner -----
    reasoner = build_raer_teacher(raer_cfg, z_dim=int(base_z.shape[1]),
                              relation_names=relation_names, device=device)
    print(f"[RAER] RAERTeacher relations={relation_names} "
          f"gate_mode={reasoner.gate_mode} evidence_groups={reasoner.evidence_groups} "
          f"expert_shared={getattr(reasoner, 'expert_shared', False)} "
          f"residual_activation={getattr(reasoner, 'residual_activation', 'tanh')}")

    # ----- Pos-weight for imbalanced BCE -----
    y_train_int = data.y[data.train_mask].long()
    n_pos = int((y_train_int == 1).sum().item())
    n_neg = int((y_train_int == 0).sum().item())
    pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)), device=device)

    # ----- TensorBoard logger -----
    tb_enabled = HAS_TENSORBOARD and not args.no_tensorboard
    tb_logger = (
        TensorBoardLogger(
            default_tensorboard_dir(dataset_name, model_name, run_name, seed),
            enabled=True,
        ) if tb_enabled else None
    )

    # ----- Train -----
    t0 = time.time()
    if scalable_extractor is not None:
        if relation_basis is None:
            raise RuntimeError("relation_basis is required for scalable_lree")
        train_summary, rows = run_training_scalable_lree(
            reasoner=reasoner,
            extractor=scalable_extractor,
            relation_basis=relation_basis,
            base_z=base_z,
            base_logits=base_logits,
            data=data,
            raer_cfg=raer_cfg,
            device=device,
            pos_weight=pos_weight,
            tb_logger=tb_logger,
        )
    else:
        train_summary, rows = run_training(
            reasoner=reasoner,
            base_z=base_z, base_logits=base_logits, rel_features=rel_features,
            data=data, raer_cfg=raer_cfg, device=device, pos_weight=pos_weight,
            tb_logger=tb_logger,
            extractor=extractor,
            evidence_inputs=evidence_inputs,
        )
    elapsed = time.time() - t0

    if tb_logger is not None:
        tb_logger.close()

    # ----- Save reasoner + epoch log -----
    write_epoch_log(log_dir, rows)
    torch.save(reasoner.state_dict(), ckpt_dir / "raer_teacher.pt")
    if extractor is not None:
        torch.save(extractor.state_dict(), ckpt_dir / "lree.pt")
        print(f"[RAER] Saved learned extractor to {ckpt_dir / 'lree.pt'}")
    if scalable_extractor is not None:
        torch.save(
            {
                "state_dict": scalable_extractor.state_dict(),
                "num_relations": len(relation_names),
                "in_dim_per_rel": int(scalable_extractor.in_dim_per_rel),
                "out_dim_per_rel": int(scalable_extractor.out_dim_per_rel),
                "relation_names": relation_names,
                "source": "scalable_lree",
            },
            ckpt_dir / "scalable_lree.pt",
        )
        print(f"[RAER] Saved scalable LREE extractor to {ckpt_dir / 'scalable_lree.pt'}")

    # ----- Verify base unchanged -----
    base_freeze_after = snapshot_base_freeze(base_ckpt_path, base_logits, base_z)
    base_freeze_check = verify_base_frozen(base_freeze_before, base_freeze_after)
    (ckpt_dir / "base_freeze_check.json").write_text(
        json.dumps(base_freeze_check, indent=2, sort_keys=True) + "\n"
    )

    # ----- Write diagnostics (consumed by aggregators) -----
    diagnostics = {
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "config_path": str(config_path),
        "git_hash": get_git_hash(),
        "elapsed_seconds": elapsed,
        "relation_names": relation_names,
        "relation_feature_meta": rel_meta,
        "base_ckpt_path": str(base_ckpt_path),
        "base_freeze_check": base_freeze_check,
        "best_val": train_summary["best_val"],
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics": train_summary["test_metrics"],
        "diagnostics_test": train_summary["diagnostics"],
        "runtime": runtime_device_info(device),
    }
    (log_dir / "raer_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "run_name": run_name,
        "seed": seed,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics": train_summary["test_metrics"],
        "elapsed_seconds": elapsed,
    }
    (result_dir / "raer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (result_dir / "test_metrics.json").write_text(
        json.dumps(train_summary["test_metrics"], indent=2, sort_keys=True) + "\n"
    )

    # ----- Console summary -----
    tm = train_summary["test_metrics"]
    print(f"\n[RAER] Done in {elapsed:.1f}s | best_epoch={train_summary['best_epoch']} "
          f"threshold={train_summary['best_threshold']:.3f}")
    print(f"  test AUPRC = {tm.get('auprc', 0.0):.4f}")
    print(f"  test AUROC = {tm.get('roc_auc', 0.0):.4f}")
    print(f"  test macro_f1 = {tm.get('macro_f1', 0.0):.4f}")
    print(f"  test g_means  = {tm.get('g_means', 0.0):.4f}")
    print(f"  base freeze: {base_freeze_check['verdict']}")
    print(f"\nCheckpoint: {ckpt_dir / 'raer_teacher.pt'}")
    print(f"Metrics:    {result_dir / 'test_metrics.json'}")
    print(f"Summary:    {result_dir / 'raer_summary.json'}")
    print(f"Log:        {log_dir / 'raer_train_log.jsonl'}")
    print(f"Diag:       {log_dir / 'raer_diagnostics.json'}")
    if tb_enabled:
        print(f"TensorBoard:{default_tensorboard_dir(dataset_name, model_name, run_name, seed)}")


if __name__ == "__main__":
    main()
