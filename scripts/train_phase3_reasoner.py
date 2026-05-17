"""Phase 3 CoVER-REL Reasoner training script with LEQA audit loss.

Trains a CoVERRelReasoner on top of a frozen base BWGNN, using
relation-gated evidence fusion with optional LEQA (LLM-as-Evidence-
Quality-Auditor) audit loss.  Replaces the deprecated Phase 2 trainer.

The judge residual path (alpha, delta_llm, L_align) is fully removed.
The only new loss term is L_audit from ``training.losses_audit``.

Usage:
    python scripts/train_phase3_reasoner.py \
        --config configs/phase3/yelpchi_bwgnn/leqa/E4.yaml \
        --seed 42 --device cuda:0

    # with pre-cached LoRA quality tensor
    python scripts/train_phase3_reasoner.py \
        --config configs/phase3/yelpchi_bwgnn/leqa/E4.yaml \
        --seed 42 --device cuda:0 \
        --lora_leqa_cache_path artifacts/cache/lora_leqa/yelpchi/seed_42/inference.parquet

    # debug smoke test (3 epochs)
    python scripts/train_phase3_reasoner.py \
        --config configs/phase3/yelpchi_bwgnn/leqa/E0.yaml \
        --seed 42 --debug --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
from evidence.relation_features import load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.gnn import build_detector
from models.leqa_lora import build_leqa_token_names
from training.losses_audit import compute_audit_loss
from training.metrics import compute_metrics, g_means as compute_g_means, precision_recall_at_k
from training.phase2_losses import (
    build_relation_evidence_distribution,
    compute_phase2_loss,
)
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
    num_threads = int(os.environ.get("COVER_NUM_THREADS", "1"))
    torch.set_num_threads(max(num_threads, 1))
    try:
        torch.set_num_interop_threads(max(int(os.environ.get("COVER_INTEROP_THREADS", "1")), 1))
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def get_base_output_cache_path(dataset_name: str, model_name: str, seed: int) -> Path:
    return (
        Path("artifacts")
        / "base_outputs"
        / dataset_name
        / model_name
        / f"seed_{seed}"
        / "base_outputs.pt"
    )


# ════════════════════════════════════════════════════════════════════
# Base-freeze verification
# ════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(str(tuple(t.shape)).encode("utf-8"))
    cpu = t.detach().to("cpu").contiguous()
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def snapshot_base_freeze(
    dataset_name: str,
    model_name: str,
    seed: int,
    base_logits: torch.Tensor,
    base_z: torch.Tensor,
    ckpt_override: str | Path | None = None,
) -> dict[str, str | None]:
    ckpt_path = (
        Path(ckpt_override) if ckpt_override is not None
        else get_base_checkpoint_path(dataset_name, model_name, seed)
    )
    return {
        "base_ckpt_path": str(ckpt_path),
        "base_ckpt_sha256": _sha256_file(ckpt_path),
        "base_logits_sha256": _sha256_tensor(base_logits),
        "base_z_sha256": _sha256_tensor(base_z),
    }


def verify_base_frozen(
    before: dict[str, str | None],
    after: dict[str, str | None],
) -> dict[str, object]:
    diffs: list[str] = []
    for key in ("base_ckpt_sha256", "base_logits_sha256", "base_z_sha256"):
        b = before.get(key)
        a = after.get(key)
        if b is None and a is None:
            continue
        if b != a:
            diffs.append(key)
    return {
        "before": before,
        "after": after,
        "differences": diffs,
        "verdict": "frozen" if not diffs else "MUTATED",
    }


# ════════════════════════════════════════════════════════════════════
# Frozen base loading
# ════════════════════════════════════════════════════════════════════

def load_frozen_base(
    config: dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    data,
    device: torch.device,
    ckpt_override: str | Path | None = None,
):
    """Load frozen base BWGNN and cache its logits + embeddings."""
    if ckpt_override is not None:
        ckpt_path = Path(ckpt_override)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"base ckpt_override not found: {ckpt_path}")
        default_cache = get_base_output_cache_path(dataset_name, model_name, seed)
        cache_path = default_cache.parent / f"_override_{ckpt_path.parent.name}_{ckpt_path.stem}.pt"
        print(f"[Phase3] base ckpt override: {ckpt_path} (cache -> {cache_path.name})")
    else:
        ckpt_path = get_base_checkpoint_path(dataset_name, model_name, seed)
        cache_path = get_base_output_cache_path(dataset_name, model_name, seed)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {})
        if (
            int(meta.get("num_nodes", -1)) == int(data.x.shape[0])
            and str(meta.get("checkpoint_path", "")) == str(ckpt_path)
        ):
            base_logits = payload["base_logits"].to(device).detach()
            base_z = payload["base_z"].to(device).detach()
            print(f"Loaded cached base outputs from {cache_path}")
            return base_logits, base_z
        print(f"[Phase3] Ignoring stale base output cache: {cache_path}")

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
    print(f"Cached base outputs to {cache_path}")
    return base_logits, base_z


def load_relation_features_for_phase3(
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


# ════════════════════════════════════════════════════════════════════
# LEQA quality tensor loading
# ════════════════════════════════════════════════════════════════════

def load_lora_q_tensor(
    cache_path: str | Path,
    num_nodes: int,
    relation_names: list[str],
    use_random: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load (N, T) quality tensor from LEQA parquet cache.

    Returns:
        q_tensor: ``(num_nodes, T)`` float in [0, 1], default 1.0 for
            nodes without cached LoRA output.
        meta: Dict with loading statistics.
    """
    token_names = build_leqa_token_names(relation_names)
    T = len(token_names)

    if use_random:
        q_tensor = torch.rand(num_nodes, T)
        meta = {
            "source": "random",
            "num_nodes": num_nodes,
            "lora_signal_dim": T,
            "covered_nodes": 0,
            "forbidden_field_audit_count": 0,
        }
        print(f"[Phase3:LEQA] Using RANDOM q tensor {tuple(q_tensor.shape)} (E4' falsification)")
        return q_tensor, meta

    import pandas as pd

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"LoRA LEQA cache not found: {cache_path}")

    df = pd.read_parquet(cache_path)
    q_tensor = torch.ones(num_nodes, T, dtype=torch.float32)
    token_to_idx = {name: i for i, name in enumerate(token_names)}

    # Vectorised assignment from long-format parquet
    valid_tokens = df["token"].isin(token_to_idx)
    valid_nodes = (df["node_id"] >= 0) & (df["node_id"] < num_nodes)
    valid = df[valid_tokens & valid_nodes]

    if len(valid) > 0:
        node_ids = valid["node_id"].values.astype(np.int64)
        token_indices = valid["token"].map(token_to_idx).values.astype(np.int64)
        q_vals = np.clip(valid["q"].values.astype(np.float32), 0.0, 1.0)
        q_tensor[node_ids, token_indices] = torch.from_numpy(q_vals)

    covered_nodes = int(df["node_id"].nunique()) if len(df) > 0 else 0

    # Forbidden-field audit on raw parquet (count rows where reason looks suspicious)
    forbidden_audit_count = 0
    if "raw_text" in df.columns:
        from models.leqa_lora import audit_leqa_output
        forbidden_audit_count = int(
            (~df["raw_text"].apply(audit_leqa_output)).sum()
        )

    meta = {
        "source": str(cache_path),
        "num_nodes": num_nodes,
        "lora_signal_dim": T,
        "total_rows": len(df),
        "valid_rows": len(valid),
        "covered_nodes": covered_nodes,
        "forbidden_field_audit_count": forbidden_audit_count,
    }
    print(
        f"[Phase3:LEQA] Loaded q tensor {tuple(q_tensor.shape)} from {cache_path} "
        f"({covered_nodes} nodes, {len(valid)} valid rows)"
    )
    return q_tensor, meta


def build_leqa_proxy_attention(
    relation_gate: torch.Tensor,
    lora_signal_dim: int,
    num_relations: int,
) -> torch.Tensor:
    """Build (N, T) proxy attention from relation gate for LEQA audit loss.

    Distributes gate weights across the LEQA vocabulary (T = 8*R + 4):
    - 7*R relation_evidence bucket tokens: gate[r] for each of 7 buckets
    - 3 graph_diagnostic tokens: uniform 1/T
    - 1 anchor_relation token: max(gate)
    - R optional_relation_gate_bucket tokens: gate values directly
    """
    N = relation_gate.shape[0]
    R = num_relations
    T = lora_signal_dim
    device = relation_gate.device
    dtype = relation_gate.dtype

    attn = torch.zeros(N, T, device=device, dtype=dtype)

    # 7 tokens per relation (relation_evidence buckets)
    for r in range(R):
        for j in range(7):
            idx = r * 7 + j
            if idx < T:
                attn[:, idx] = relation_gate[:, r]

    offset = 7 * R
    # 3 graph diagnostic tokens: uniform
    for j in range(3):
        if offset + j < T:
            attn[:, offset + j] = 1.0 / max(T, 1)
    # 1 anchor relation token: max gate
    if offset + 3 < T:
        attn[:, offset + 3] = relation_gate.max(dim=-1).values
    # R per-relation gate bucket tokens
    for r in range(R):
        if offset + 4 + r < T:
            attn[:, offset + 4 + r] = relation_gate[:, r]

    return attn


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_phase3(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
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
    )
    final_logit = outputs["final_logit"]
    prob = torch.sigmoid(final_logit).cpu().numpy()
    y_np = y[mask].cpu().numpy()

    if threshold is not None:
        metrics = evaluate_with_threshold(y_np, prob, threshold)
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
    device: torch.device,
    y: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    relation_evidence_pi: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute per-epoch diagnostic stats (delta_rel, gates, etc.)."""
    reasoner.eval()
    outputs = reasoner(
        base_z,
        base_logits,
        rel_features.to(device),
    )
    diag: dict[str, float] = {}

    delta_rel = outputs.get("delta_rel")
    if delta_rel is not None:
        diag["mean_abs_delta_rel"] = float(delta_rel.abs().mean().item())
        # Intervention is just delta_rel (no alpha_llm * delta_llm)
        diag["mean_intervention"] = float(delta_rel.abs().mean().item())

        if mask is not None and y is not None:
            m = mask.to(device).bool()
            y_bool = y.to(device).view(-1).float() >= 0.5
            base_pred = base_logits.to(device).view(-1) >= 0
            base_correct = m & (base_pred == y_bool)
            base_wrong = m & (~(base_pred == y_bool))
            if base_correct.any():
                diag["mean_intervention_base_correct"] = float(delta_rel[base_correct].abs().mean().item())
            if base_wrong.any():
                diag["mean_intervention_base_wrong"] = float(delta_rel[base_wrong].abs().mean().item())

    # gate entropy
    gate_values = outputs.get("relation_gate")
    if gate_values is not None:
        eps = 1e-8
        g = gate_values.clamp(eps, 1.0)
        entropy = -(g * g.log()).sum(dim=-1).mean()
        diag["mean_gate_entropy"] = float(entropy.item())
        for r_idx in range(gate_values.shape[1]):
            diag[f"gate_weight_rel_{r_idx}"] = float(gate_values[:, r_idx].mean().item())

    # Evidence distribution diagnostics
    rel_strength = outputs.get("relation_strength")
    if relation_evidence_pi is not None and gate_values is not None and gate_values.shape[1] >= 2:
        eps = 1e-8
        pi = relation_evidence_pi.to(device).detach().clamp_min(eps)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_r = float(np.log(max(pi.shape[1], 2)))
        ent_pi = -(pi * torch.log(pi + eps)).sum(dim=-1)
        rho = (1.0 - ent_pi / log_r).clamp(min=0.0, max=1.0)
        diag["mean_dominance_rho"] = float(rho.mean().item())
        diag["mean_evidence_entropy"] = float(ent_pi.mean().item())
        gate_kl = (pi * (torch.log(pi + eps) - torch.log(gate_values.clamp_min(eps)))).sum(dim=-1)
        diag["mean_evidence_gate_kl"] = float(gate_kl.mean().item())
        for r_idx in range(pi.shape[1]):
            diag[f"evidence_weight_rel_{r_idx}"] = float(pi[:, r_idx].mean().item())
    elif rel_strength is not None and rel_strength.shape[1] >= 2:
        eps = 1e-8
        log_r = float(np.log(max(rel_strength.shape[1], 2)))
        pi = torch.softmax(rel_strength, dim=-1)
        ent_pi = -(pi * torch.log(pi + eps)).sum(dim=-1)
        rho = (1.0 - ent_pi / log_r).clamp(min=0.0, max=1.0)
        diag["mean_dominance_rho"] = float(rho.mean().item())

    return diag


# ════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════

def write_epoch_log(log_dir: Path, rows: list[dict]) -> None:
    """Write per-epoch training log as JSONL and CSV."""
    jsonl_path = log_dir / "phase3_train_log.jsonl"
    csv_path = log_dir / "phase3_train_log.csv"
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


def default_tensorboard_dir(dataset_name: str, model_name: str, run_name: str, seed: int) -> Path:
    return (
        Path("artifacts")
        / "tensorboard"
        / "phase3"
        / dataset_name
        / model_name
        / run_name
        / f"seed_{seed}"
    )


def log_epoch_to_tensorboard(
    logger: TensorBoardLogger | None,
    epoch: int,
    loss: float,
    loss_dict: dict[str, float],
    val_metrics: dict[str, float],
    diagnostics: dict[str, float],
    relation_names: list[str],
    optimizer: torch.optim.Optimizer,
    p3_cfg: dict | None = None,
) -> None:
    """Write train loss, validation metrics, and diagnostics for one epoch."""
    if logger is None:
        return

    def log_number(tag: str, value: object) -> None:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value_f):
            logger.log_scalar(tag, value_f, epoch)

    log_number("train/loss", loss)
    log_number("train/lr", optimizer.param_groups[0].get("lr", 0.0))
    log_number("loss/total", loss_dict.get("total", loss))
    for key in ("l_cls", "l_intervention", "l_sparse", "l_audit"):
        if key in loss_dict:
            log_number(f"loss/{key}", loss_dict[key])
    for key in ("auprc", "roc_auc", "macro_f1", "g_means"):
        if key in val_metrics:
            log_number(f"val/{key}", val_metrics[key])

    # Budget tags (simplified: no judge path)
    if p3_cfg is not None:
        l_cls = float(loss_dict.get("l_cls", 0.0))
        l_int = float(loss_dict.get("l_intervention", 0.0))
        l_sp = float(loss_dict.get("l_sparse", 0.0))

        delta_rel_max = float(p3_cfg.get("delta_rel_max", 2.0))
        num_relations = max(len(relation_names), 2)

        m_int = delta_rel_max ** 2  # no alpha*delta_llm
        m_sparse = math.log(num_relations) + 0.5

        intervention_headroom = max(m_int - l_int, 0.0)
        evidence_alignment = max(m_sparse - l_sp, 0.0)

        lam_int = float(p3_cfg.get("lambda_int", 3e-3))
        lam_sparse = float(p3_cfg.get("lambda_sparse", 1e-3))

        display_total = (
            l_cls
            + lam_int * intervention_headroom
            + lam_sparse * evidence_alignment
        )

        log_number("budget/cls_residual", l_cls)
        log_number("budget/intervention_headroom", intervention_headroom)
        log_number("budget/evidence_alignment", evidence_alignment)
        log_number("budget/display_total", display_total)


def get_lambda_int(p3_cfg: dict) -> float:
    """Read lambda_int with one-release lambda_trust compatibility."""
    if "lambda_int" in p3_cfg:
        return float(p3_cfg["lambda_int"])
    return float(p3_cfg.get("lambda_trust", 3e-3))


# ════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    relation_evidence_pi: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    p3_cfg: dict,
    epoch: int,
    pos_weight: torch.Tensor | None = None,
    q_tensor: torch.Tensor | None = None,
    lora_signal_dim: int = 28,
    num_relations: int = 3,
    lambda_audit: float = 0.0,
    delta_rel_clip: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """One training epoch with Phase 3 loss (3-term base + optional audit)."""
    reasoner.train()
    device = base_z.device

    # Slice train nodes
    z_t = base_z[train_mask]
    bl_t = base_logits[train_mask]
    rf_t = rel_features[train_mask].to(device)
    epi_t = relation_evidence_pi[train_mask].to(device)
    y_t = y[train_mask]

    outputs = reasoner(z_t, bl_t, rf_t)
    outputs["relation_evidence_pi"] = epi_t

    # All-true mask since we already sliced to train nodes
    all_train = torch.ones(z_t.shape[0], dtype=torch.bool, device=device)
    pw = pos_weight if pos_weight is not None else torch.tensor(1.0, device=device)

    loss, loss_dict = compute_phase2_loss(
        outputs=outputs,
        y=y_t,
        train_mask=all_train,
        pos_weight=pw,
        lambda_int=get_lambda_int(p3_cfg),
        lambda_sparse=p3_cfg.get("lambda_sparse", 1e-3),
    )

    # LEQA audit loss
    if q_tensor is not None and lambda_audit > 0:
        q_t = q_tensor[train_mask].to(device)
        attention = build_leqa_proxy_attention(
            outputs["relation_gate"], lora_signal_dim, num_relations,
        )
        loss_audit = compute_audit_loss(
            attention, q_t, outputs["delta_rel"], clip=delta_rel_clip,
        )
        loss = loss + lambda_audit * loss_audit
        loss_dict["l_audit"] = float(loss_audit.detach().item())
        loss_dict["total"] = float(loss.detach().item())

    optimizer.zero_grad()
    loss.backward()
    grad_clip = p3_cfg.get("grad_clip")
    if grad_clip is not None and float(grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_norm=float(grad_clip))
    optimizer.step()

    return loss.item(), loss_dict


def build_phase3_reasoner(
    *,
    p3_cfg: dict,
    z_dim: int,
    relation_names: list[str],
    device: torch.device,
) -> CoVERRelReasoner:
    """Construct the reasoner without any judge-path kwargs."""
    reasoner = CoVERRelReasoner(
        base_z_dim=z_dim,
        relation_names=relation_names,
        anchor_relation=p3_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=p3_cfg.get("rel_stat_dim", 9),
        rel_hidden_dim=p3_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=p3_cfg.get("rel_num_layers", 2),
        rel_dropout=p3_cfg.get("rel_dropout", 0.3),
        tau_gate=p3_cfg.get("tau_gate", 0.7),
        delta_rel_max=p3_cfg.get("delta_rel_max", 2.0),
        use_judge=False,
        judge_feature_dim=0,
        judge_hidden_dim=32,
        judge_dropout=0.3,
        delta_llm_max=0.0,
        alpha_max=0.0,
        alpha_bias_init=-3.0,
    ).to(device)
    return reasoner


def run_training(
    *,
    reasoner,
    p3_cfg: dict,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    relation_evidence_pi: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    k_values: list[int],
    pos_weight: torch.Tensor,
    relation_names: list[str],
    tb_logger: TensorBoardLogger | None,
    q_tensor: torch.Tensor | None = None,
    lora_signal_dim: int = 28,
    lambda_audit: float = 0.0,
    delta_rel_clip: float = 2.0,
) -> dict[str, object]:
    """Train and return best state plus per-epoch rows."""
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=p3_cfg.get("lr", 1e-3),
        weight_decay=p3_cfg.get("weight_decay", 1e-4),
    )
    epochs = int(p3_cfg.get("epochs", 300))
    patience = int(p3_cfg.get("patience", 50))
    early_stop_metric = str(p3_cfg.get("early_stop_metric", "val_auprc"))
    eval_interval = max(int(p3_cfg.get("eval_interval", 1)), 1)

    # LR schedule
    lr_schedule = str(p3_cfg.get("lr_schedule", "constant")).lower()
    warmup_epochs = int(p3_cfg.get("warmup_epochs", 0))
    lr_min = float(p3_cfg.get("lr_min", 0.0))
    lr_base = float(p3_cfg.get("lr", 1e-3))
    min_ratio = lr_min / lr_base if lr_base > 0 else 0.0

    if lr_schedule == "cosine":
        def _lr_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return min_ratio + (1.0 - min_ratio) * (epoch_idx + 1) / warmup_epochs
            remaining = max(epochs - warmup_epochs, 1)
            progress = min(max(epoch_idx - warmup_epochs, 0), remaining) / remaining
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_factor
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        print(
            f"[Phase3] LR schedule: cosine, base={lr_base}, "
            f"min={lr_min} (ratio {min_ratio:.4g}), warmup={warmup_epochs}"
        )
    elif lr_schedule == "constant":
        scheduler = None
        print(f"[Phase3] LR schedule: constant {lr_base}")
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r} (use 'constant' or 'cosine')")

    best_val_score = float("-inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    rows: list[dict] = []
    start = time.time()
    R = len(relation_names)

    print(
        f"[Phase3] Training for up to {epochs} epochs, "
        f"patience={patience}, metric={early_stop_metric}, "
        f"eval_interval={eval_interval}, lambda_audit={lambda_audit}"
    )

    for epoch in range(1, epochs + 1):
        loss, loss_dict = train_one_epoch(
            reasoner,
            base_z,
            base_logits,
            rel_features,
            relation_evidence_pi,
            y,
            train_mask,
            optimizer,
            p3_cfg,
            epoch,
            pos_weight=pos_weight,
            q_tensor=q_tensor,
            lora_signal_dim=lora_signal_dim,
            num_relations=R,
            lambda_audit=lambda_audit,
            delta_rel_clip=delta_rel_clip,
        )

        if scheduler is not None:
            scheduler.step()

        should_eval = epoch == 1 or epoch == epochs or (epoch % eval_interval == 0)
        val_metrics: dict[str, float] = {}
        diag: dict[str, float] = {}
        if should_eval:
            val_metrics = evaluate_phase3(
                reasoner, base_z, base_logits, rel_features,
                val_mask, y, device, k_values=k_values,
            )
            diag = compute_epoch_diagnostics(
                reasoner, base_z, base_logits, rel_features,
                device, y=y, mask=val_mask,
                relation_evidence_pi=relation_evidence_pi,
            )

        row: dict[str, object] = {
            "epoch": float(epoch),
            "loss": loss,
        }
        row.update({f"loss/{k}": float(v) for k, v in loss_dict.items()})
        row.update({f"val/{k}": float(v) for k, v in val_metrics.items()})
        row.update(diag)
        for r_idx, rname in enumerate(relation_names):
            key = f"gate_weight_rel_{r_idx}"
            if key in diag:
                row[f"gate/{rname}"] = diag[key]
        rows.append(row)
        log_epoch_to_tensorboard(
            tb_logger, epoch, loss, loss_dict, val_metrics, diag, relation_names, optimizer,
            p3_cfg=p3_cfg,
        )

        if should_eval and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            audit_str = f" | L_audit {loss_dict.get('l_audit', 0):.4f}" if "l_audit" in loss_dict else ""
            print(
                f"  Epoch {epoch:3d} | Loss {loss:.4f} | "
                f"Val AUPRC {val_metrics.get('auprc', 0):.4f} | "
                f"Val AUC {val_metrics.get('roc_auc', 0):.4f} | "
                f"Val Macro-F1 {val_metrics.get('macro_f1', 0):.4f}"
                f"{audit_str}"
            )

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

    if best_state is not None:
        reasoner.load_state_dict(best_state)

    return {
        "reasoner": reasoner,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "epochs_trained": epoch,
        "elapsed_seconds": time.time() - start,
        "rows": rows,
    }


# ════════════════════════════════════════════════════════════════════
# LEQA diagnostics
# ════════════════════════════════════════════════════════════════════

def compute_lora_q_diagnostics(
    q_tensor: torch.Tensor,
    q_meta: dict[str, object],
) -> dict[str, object]:
    """Per-seed histogram of q values + audit count."""
    q_flat = q_tensor.flatten()
    valid_mask = q_flat < 1.0  # nodes with LoRA coverage have q < 1.0
    q_valid = q_flat[valid_mask] if valid_mask.any() else q_flat

    return {
        "lora_q_distribution": {
            "mean": float(q_valid.mean().item()),
            "std": float(q_valid.std().item()) if q_valid.numel() > 1 else 0.0,
            "p10": float(q_valid.quantile(0.1).item()) if q_valid.numel() > 0 else 0.0,
            "p50": float(q_valid.quantile(0.5).item()) if q_valid.numel() > 0 else 0.0,
            "p90": float(q_valid.quantile(0.9).item()) if q_valid.numel() > 0 else 0.0,
            "num_covered_nodes": int(q_meta.get("covered_nodes", 0)),
        },
        "forbidden_field_audit_count": int(q_meta.get("forbidden_field_audit_count", 0)),
    }


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 CoVER-REL Reasoner with LEQA")
    parser.add_argument("--config", type=str, required=True, help="Path to phase3 config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 3 epochs")

    # Phase 3 reasoner CLI overrides
    parser.add_argument("--lambda_int", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    parser.add_argument("--delta_rel_max", type=float, default=None)
    parser.add_argument("--tau_gate", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config.")
    parser.add_argument("--lr_schedule", type=str, default=None,
                        choices=["constant", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name")

    # LEQA-specific CLI args
    parser.add_argument(
        "--lora_leqa_cache_path", type=str, default=None,
        help="Path to inference.parquet from cache_lora_leqa_outputs.py",
    )
    parser.add_argument(
        "--lambda_audit", type=float, default=None,
        help="LEQA audit loss weight (default 3e-2 per FINAL_PROPOSAL section 3.4)",
    )
    parser.add_argument(
        "--delta_rel_clip", type=float, default=None,
        help="Clip bound for |delta_rel| in audit loss (default 2.0)",
    )
    parser.add_argument(
        "--use_random_lora_signal", action="store_true",
        help="Replace LoRA q with random Uniform[0,1] (E4' falsification cell)",
    )
    parser.add_argument(
        "--lora_frozen_after_epoch_1", action="store_true",
        help="Freeze LoRA signal after epoch 1 (E5 cell)",
    )
    parser.add_argument(
        "--lora_signal_dim", type=int, default=None,
        help="LEQA vocabulary size (default 28 = 8*R+4 for R=3)",
    )

    # Common overrides
    parser.add_argument(
        "--tensorboard_dir", type=str, default=None,
        help="Override TensorBoard log directory",
    )
    parser.add_argument(
        "--no_tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--base_ckpt_path", type=str, default=None,
        help="Override frozen-base checkpoint path.",
    )
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    p3_cfg = config.get("phase3_leqa", {})

    # CLI overrides
    cli_overrides = {
        "lambda_int": args.lambda_int,
        "lambda_sparse": args.lambda_sparse,
        "delta_rel_max": args.delta_rel_max,
        "tau_gate": args.tau_gate,
        "lr": args.lr,
        "epochs": args.epochs,
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "lr_min": args.lr_min,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "early_stop_metric": args.early_stop_metric,
        "eval_interval": args.eval_interval,
        "run_name": args.run_name,
        "lambda_audit": args.lambda_audit,
        "delta_rel_clip": args.delta_rel_clip,
        "lora_signal_dim": args.lora_signal_dim,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            p3_cfg[key] = val
    if args.lora_leqa_cache_path is not None:
        p3_cfg["lora_leqa_cache_path"] = args.lora_leqa_cache_path
    if args.use_random_lora_signal:
        p3_cfg["use_random_lora_signal"] = True
    if args.lora_frozen_after_epoch_1:
        p3_cfg["lora_frozen_after_epoch_1"] = True

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    model_name = config["model"]["name"]
    seed = args.seed if args.seed is not None else config["train"]["seed"]
    run_name = p3_cfg.get("run_name", "phase3")
    relation_names = p3_cfg.get("relation_names", [])

    # LEQA config
    lora_cache_path = p3_cfg.get("lora_leqa_cache_path")
    lambda_audit = float(p3_cfg.get("lambda_audit", 0.0))
    delta_rel_clip = float(p3_cfg.get("delta_rel_clip", 2.0))
    use_random_lora = bool(p3_cfg.get("use_random_lora_signal", False))
    R = len(relation_names) if relation_names else 3
    lora_signal_dim = int(p3_cfg.get("lora_signal_dim", 8 * R + 4))

    # ── Seed + device ──
    set_seed(seed, model_name=model_name)

    requested_device = str(
        args.device or config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    device_info = runtime_device_info(device)
    print(f"[Phase3] seed={seed} device={device} run_name={run_name}")

    # ── Load dataset ──
    data = load_fraud_dataset(
        dataset_name,
        path=dataset_path,
        seed=seed,
        split_mode=config["dataset"].get("split_mode", "supervised"),
        train_ratio=config["dataset"].get("train_ratio", 0.4),
        val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        stratified=config["dataset"].get("stratified", True),
    )
    epochs = p3_cfg.get("epochs", 300)
    if args.debug:
        epochs = config["train"].get("debug_epochs", 3)
        p3_cfg["epochs"] = epochs
        print(f"[DEBUG] Using real dataset with {epochs} epochs")

    num_nodes = data.x.shape[0]
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # ── Load frozen base ──
    base_logits, base_z = load_frozen_base(
        config, dataset_name, model_name, seed, data, device,
        ckpt_override=args.base_ckpt_path,
    )
    base_freeze_before = snapshot_base_freeze(
        dataset_name, model_name, seed, base_logits, base_z,
        ckpt_override=args.base_ckpt_path,
    )
    z_dim = base_z.shape[1]

    # ── Load relation features ──
    rel_stats, rel_meta = load_relation_features_for_phase3(
        dataset_name, model_name, seed, num_nodes,
    )
    rel_features = rel_stats.to(device)
    relation_evidence_pi = build_relation_evidence_distribution(
        rel_features,
        num_relations=len(relation_names),
        rel_stat_dim=p3_cfg.get("rel_stat_dim", 9),
        tau=p3_cfg.get("evidence_tau", 1.5),
    ).to(device)

    # ── Load LEQA quality tensor (if enabled) ──
    q_tensor: torch.Tensor | None = None
    q_meta: dict[str, object] = {}
    use_leqa = lora_cache_path is not None or use_random_lora
    if use_leqa and lambda_audit > 0:
        q_tensor, q_meta = load_lora_q_tensor(
            cache_path=lora_cache_path or "",
            num_nodes=num_nodes,
            relation_names=relation_names,
            use_random=use_random_lora,
        )
        # Validate dimension
        if q_tensor.shape[1] != lora_signal_dim:
            print(
                f"[Phase3:LEQA] WARNING: q_tensor dim {q_tensor.shape[1]} != "
                f"lora_signal_dim {lora_signal_dim}; adjusting lora_signal_dim"
            )
            lora_signal_dim = q_tensor.shape[1]
        print(f"[Phase3:LEQA] lambda_audit={lambda_audit} delta_rel_clip={delta_rel_clip}")
    elif use_leqa and lambda_audit == 0:
        print("[Phase3:LEQA] LEQA enabled but lambda_audit=0 (E2 mode: signal without loss)")
    else:
        print("[Phase3] LEQA disabled (E0 mode)")

    # ── TensorBoard ──
    tb_dir: Path | None = None
    tb_logger: TensorBoardLogger | None = None
    if not args.no_tensorboard:
        tb_dir = (
            Path(args.tensorboard_dir)
            if args.tensorboard_dir is not None
            else default_tensorboard_dir(dataset_name, model_name, run_name, seed)
        )
        tb_logger = TensorBoardLogger(tb_dir, enabled=True)
        if tb_logger.enabled:
            print(f"[Phase3] TensorBoard: {tb_dir}")
        else:
            print("[Phase3] TensorBoard unavailable; install tensorboard to enable")
            tb_logger = None

    # ── Training ──
    k_values = config.get("eval", {}).get("k_values", [50, 100, 200])

    # Compute pos_weight from train labels
    y_train = y[train_mask]
    n_pos = y_train.sum().item()
    n_neg = (y_train.numel() - n_pos)
    pw = torch.tensor(max(n_neg / max(n_pos, 1), 1.0), device=device)
    print(f"[Phase3] pos_weight={pw.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    reasoner = build_phase3_reasoner(
        p3_cfg=p3_cfg,
        z_dim=z_dim,
        relation_names=relation_names,
        device=device,
    )
    print(f"[Phase3] Reasoner params: {sum(p.numel() for p in reasoner.parameters()):,}")

    start_time = time.time()
    result = run_training(
        reasoner=reasoner,
        p3_cfg=p3_cfg,
        base_z=base_z,
        base_logits=base_logits,
        rel_features=rel_features,
        relation_evidence_pi=relation_evidence_pi,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        device=device,
        k_values=k_values,
        pos_weight=pw,
        relation_names=relation_names,
        tb_logger=tb_logger,
        q_tensor=q_tensor,
        lora_signal_dim=lora_signal_dim,
        lambda_audit=lambda_audit,
        delta_rel_clip=delta_rel_clip,
    )
    reasoner = result["reasoner"]
    best_state = result["best_state"]
    best_epoch = int(result["best_epoch"])
    best_val_score = float(result["best_val_score"])
    epoch_rows = result["rows"]
    elapsed = time.time() - start_time

    # ── Threshold calibration + test eval ──
    reasoner.eval()
    with torch.no_grad():
        val_out = reasoner(
            base_z[val_mask], base_logits[val_mask],
            rel_features[val_mask].to(device),
        )
        val_prob = torch.sigmoid(val_out["final_logit"]).cpu().numpy()
    val_y = y[val_mask].cpu().numpy()
    best_threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")

    test_metrics = evaluate_phase3(
        reasoner, base_z, base_logits, rel_features,
        test_mask, y, device,
        threshold=best_threshold, k_values=k_values,
    )

    print(f"\n=== Phase 3 Test Results (threshold={best_threshold:.3f}) ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # ── Save artifacts ──
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))

    # Checkpoint
    torch.save(
        {
            "model_state_dict": best_state,
            "config": p3_cfg,
            "seed": seed,
            "epoch": best_epoch,
            "train_protocol": "phase3_leqa",
        },
        ckpt_dir / "reasoner.pt",
    )

    # Epoch log
    write_epoch_log(log_dir, epoch_rows)

    # Test metrics
    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Reproducibility config
    repro_config = copy.deepcopy(config)
    repro_config["phase3_leqa"] = p3_cfg
    for out_dir in (log_dir, results_dir):
        with open(out_dir / "repro_config.yaml", "w") as f:
            yaml.safe_dump(repro_config, f, sort_keys=False)
        command = [
            sys.executable,
            "scripts/train_phase3_reasoner.py",
            "--config",
            str(out_dir / "repro_config.yaml"),
            "--seed",
            str(seed),
            "--device",
            str(device),
            "--run_name",
            str(run_name),
        ]
        (out_dir / "repro_command.txt").write_text(" ".join(command) + "\n")

    # ── Diagnostics ──
    final_diag = compute_epoch_diagnostics(
        reasoner, base_z, base_logits, rel_features,
        device, y=y, mask=test_mask,
        relation_evidence_pi=relation_evidence_pi,
    )

    # LEQA diagnostics
    leqa_diag: dict[str, object] = {}
    if q_tensor is not None:
        leqa_diag = compute_lora_q_diagnostics(q_tensor, q_meta)
    # audit_loss_per_seed: last epoch's l_audit
    if epoch_rows:
        last_audit = epoch_rows[-1].get("loss/l_audit")
        if last_audit is not None:
            leqa_diag["audit_loss_per_seed"] = float(last_audit)

    diagnostics = {
        "config": config,
        "phase3_leqa": p3_cfg,
        "seed": seed,
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "train_protocol": "phase3_leqa",
        "device": device_info,
        "tensorboard": {
            "enabled": bool(tb_logger is not None and HAS_TENSORBOARD),
            "path": str(tb_dir) if tb_dir is not None else None,
        },
        "git_hash": get_git_hash(),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "early_stop_metric": p3_cfg.get("early_stop_metric", "val_auprc"),
        "epochs_trained": result["epochs_trained"],
        "elapsed_seconds": elapsed,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "final_diagnostics": final_diag,
        "leqa_diagnostics": leqa_diag,
        "relation_feature_meta": rel_meta,
        "base_freeze_check": verify_base_frozen(
            before=base_freeze_before,
            after=snapshot_base_freeze(
                dataset_name, model_name, seed, base_logits, base_z,
                ckpt_override=args.base_ckpt_path,
            ),
        ),
    }
    with open(log_dir / "phase3_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    summary_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "run_name": run_name,
        "seed": seed,
        "cell": p3_cfg.get("cell", "unknown"),
        "git_hash": diagnostics["git_hash"],
        "device": device_info,
        "tensorboard": diagnostics["tensorboard"],
        "train_protocol": "phase3_leqa",
        "config_path": str(args.config),
        "repro_config": str(results_dir / "repro_config.yaml"),
        "repro_command": str(results_dir / "repro_command.txt"),
        "checkpoint": str(ckpt_dir / "reasoner.pt"),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "final_diagnostics": final_diag,
        "leqa_diagnostics": leqa_diag,
    }
    for out_dir in (log_dir, results_dir):
        with open(out_dir / "phase3_summary.json", "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)

    print(f"\nCheckpoint: {ckpt_dir / 'reasoner.pt'}")
    print(f"Metrics:    {results_dir / 'final_metrics.json'}")
    print(f"Summary:    {results_dir / 'phase3_summary.json'}")
    print(f"Log:        {log_dir / 'phase3_train_log.jsonl'}")
    print(f"Diag:       {log_dir / 'phase3_diagnostics.json'}")
    if tb_logger is not None:
        tb_logger.close()
        print(f"TensorBoard:{tb_dir}")


if __name__ == "__main__":
    main()
