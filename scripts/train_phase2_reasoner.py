"""Phase2 CoVER-REL Reasoner training script.

Trains a CoVERRelReasoner on top of a frozen base BWGNN, using relation-gated
evidence fusion with optional LLM judge residual.

Usage:
    python scripts/train_phase2_reasoner.py \
        --config configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E2_judge_residual.yaml \
        --seed 42 --device cuda:0

    # debug smoke test (3 epochs, tiny graph)
    python scripts/train_phase2_reasoner.py \
        --config configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E0_relgate.yaml \
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
from evidence.judge_encoder import load_jsonl
from evidence.relation_features import load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.gnn import build_detector
from training.metrics import compute_metrics, g_means as compute_g_means, precision_recall_at_k
from training.phase2_losses import (
    build_judge_relation_targets,
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


def get_base_output_cache_path(dataset_name: str, model_name: str, seed: int) -> Path:
    """Cache frozen Phase1 logits/embeddings for repeated Phase2 sweeps."""
    return (
        Path("artifacts")
        / "base_outputs"
        / dataset_name
        / model_name
        / f"seed_{seed}"
        / "base_outputs.pt"
    )


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# Base-freeze verification (D0.5: prove Phase 2 cannot mutate base)
# ════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of a file, or ``None`` if the file is missing."""
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    """Return hex SHA-256 of a tensor's raw bytes (dtype + shape + bytes)."""
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
) -> dict[str, str | None]:
    """Capture content hashes for the frozen-base artefacts at a point in time.

    The Phase 2 training loop reads ``base_logits`` and ``base_z`` from an
    on-disk cache and never holds the base detector as a trainable module,
    so by construction it cannot mutate base parameters.  This snapshot is
    the empirical witness for that claim — paired snapshots taken before
    and after Phase 2 training must agree bit-for-bit.

    Returns a dict with three SHA-256 hex digests:

    * ``base_ckpt_sha256``: hash of ``base.pt`` on disk (may be ``None``
      if the cached output was used and the checkpoint is absent).
    * ``base_logits_sha256``: hash of the in-memory ``base_logits`` tensor.
    * ``base_z_sha256``:      hash of the in-memory ``base_z`` tensor.
    """
    ckpt_path = get_base_checkpoint_path(dataset_name, model_name, seed)
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
    """Compare two base-freeze snapshots and return a verdict block.

    Verdict is ``"frozen"`` when every comparable hash matches and
    ``"MUTATED"`` if any disagree.  The structured payload is suitable
    for direct inclusion in the Phase 2 diagnostics JSON.
    """
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
# Frozen base
# ════════════════════════════════════════════════════════════════════

def load_frozen_base(config: dict, dataset_name: str, model_name: str, seed: int, data, device: torch.device):
    """Load frozen base BWGNN and cache its logits + embeddings."""
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
        print(f"[Phase2] Ignoring stale base output cache: {cache_path}")

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
    y: torch.Tensor | None = None,
    judge_align: dict[str, torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
    relation_evidence_pi: torch.Tensor | None = None,
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

    alpha_llm = outputs.get("alpha_llm")
    delta_llm = outputs.get("delta_llm")
    if delta_rel is not None and alpha_llm is not None and delta_llm is not None:
        intervention = delta_rel + alpha_llm * delta_llm
        diag["mean_intervention"] = float(intervention.abs().mean().item())
        diag["mean_abs_alpha_delta_llm"] = float((alpha_llm * delta_llm).abs().mean().item())
        if mask is not None and y is not None:
            m = mask.to(device).bool()
            y_bool = y.to(device).view(-1).float() >= 0.5
            base_pred = base_logits.to(device).view(-1) >= 0
            base_correct = m & (base_pred == y_bool)
            base_wrong = m & (~(base_pred == y_bool))
            if base_correct.any():
                diag["mean_intervention_base_correct"] = float(intervention[base_correct].abs().mean().item())
            if base_wrong.any():
                diag["mean_intervention_base_wrong"] = float(intervention[base_wrong].abs().mean().item())

    # alpha_llm stats
    if alpha_llm is not None:
        diag["mean_alpha_llm"] = float(alpha_llm.mean().item())
        if judge_mask is not None:
            jm = judge_mask.to(device).bool()
            if jm.any():
                diag["mean_alpha_llm_accepted"] = float(alpha_llm[jm].mean().item())
                if delta_llm is not None:
                    actual_j = (alpha_llm * delta_llm).abs()
                    diag["mean_abs_alpha_delta_llm_accepted"] = float(actual_j[jm].mean().item())
            rejected = ~jm
            if rejected.any():
                diag["mean_alpha_llm_rejected"] = float(alpha_llm[rejected].mean().item())
                if delta_llm is not None:
                    actual_j = (alpha_llm * delta_llm).abs()
                    diag["mean_abs_alpha_delta_llm_rejected"] = float(actual_j[rejected].mean().item())
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

    # Dominance from fixed relation evidence when available; otherwise fall
    # back to model relation_strength for backward diagnostics.
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

    if judge_align is not None and gate_values is not None and "key_idx" in judge_align:
        key_idx = judge_align["key_idx"].to(device).view(-1).long()
        has_target = judge_align["has_target_mask"].to(device).view(-1).bool()
        eligible = has_target & (key_idx >= 0) & (key_idx < gate_values.shape[1])
        if eligible.any():
            diag["gate_key_agreement"] = float(
                (gate_values[eligible].argmax(dim=-1) == key_idx[eligible]).float().mean().item()
            )

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


def default_tensorboard_dir(dataset_name: str, model_name: str, run_name: str, seed: int) -> Path:
    """Return the Phase2 TensorBoard directory for one run/seed."""
    return (
        Path("artifacts")
        / "tensorboard"
        / "phase2"
        / dataset_name
        / model_name
        / run_name
        / f"seed_{seed}"
    )


def log_phase2_epoch_to_tensorboard(
    logger: TensorBoardLogger | None,
    epoch: int,
    loss: float,
    loss_dict: dict[str, float],
    val_metrics: dict[str, float],
    diagnostics: dict[str, float],
    relation_names: list[str],
    optimizer: torch.optim.Optimizer,
    p2_cfg: dict | None = None,
) -> None:
    """Write train loss, validation metrics, and diagnostics for one epoch.

    Emits two parallel tag families:

    * ``loss/*`` — the raw optimisation objective values (``l_cls``,
      ``l_intervention``, ``l_sparse``, ``l_align``, ``total``).  These are
      the ground-truth quantities the optimiser actually minimises; the two
      regulariser values therefore *grow* during training because the model
      "spends" them in exchange for ``l_cls`` improvement.

    * ``budget/*`` — display-only complements designed so all four loss
      components and the derived ``display_total`` descend monotonically as
      training progresses, making dashboards visually consistent.  The
      reformulation is a pure additive-constant transformation (no gradient
      change, no effect on optimisation):

          budget/cls_residual            = l_cls
          budget/intervention_headroom   = M_int    - l_intervention
          budget/evidence_alignment      = M_sparse - l_sparse
          budget/judge_alignment_residual = l_align
          budget/display_total           = l_cls
              + λ_int    · (M_int    - l_intervention)
              + λ_sparse · (M_sparse - l_sparse)
              + λ_align  ·  l_align

      where ``M_int = (delta_rel_max + alpha_max * delta_llm_max) ** 2`` is
      the analytic upper bound on ``(z - b)²`` and
      ``M_sparse = log(R) + 0.5`` is a soft upper bound on
      ``KL(π_evidence || g)`` for ``R`` relations.
    """
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
    for key in ("l_cls", "l_intervention", "l_sparse", "l_align"):
        if key in loss_dict:
            log_number(f"loss/{key}", loss_dict[key])
    for key in ("auprc", "roc_auc", "macro_f1", "g_means"):
        if key in val_metrics:
            log_number(f"val/{key}", val_metrics[key])

    # ----- budget/* display tags (monotone-down by construction) -----
    if p2_cfg is not None:
        l_cls = float(loss_dict.get("l_cls", 0.0))
        l_int = float(loss_dict.get("l_intervention", 0.0))
        l_sp = float(loss_dict.get("l_sparse", 0.0))
        l_al = float(loss_dict.get("l_align", 0.0))

        delta_rel_max = float(p2_cfg.get("delta_rel_max", 2.0))
        alpha_max = float(p2_cfg.get("alpha_max", 0.10))
        delta_llm_max = float(p2_cfg.get("delta_llm_max", 0.75))
        num_relations = max(len(relation_names), 2)

        m_int = (delta_rel_max + alpha_max * delta_llm_max) ** 2
        m_sparse = math.log(num_relations) + 0.5  # soft upper bound for KL

        intervention_headroom = max(m_int - l_int, 0.0)
        evidence_alignment = max(m_sparse - l_sp, 0.0)

        lam_int = float(p2_cfg.get("lambda_int", p2_cfg.get("lambda_trust", 3e-3)))
        lam_sparse = float(p2_cfg.get("lambda_sparse", 1e-3))
        lam_align = float(p2_cfg.get("lambda_align", 1e-2))

        display_total = (
            l_cls
            + lam_int * intervention_headroom
            + lam_sparse * evidence_alignment
            + lam_align * l_al
        )

        log_number("budget/cls_residual", l_cls)
        log_number("budget/intervention_headroom", intervention_headroom)
        log_number("budget/evidence_alignment", evidence_alignment)
        log_number("budget/judge_alignment_residual", l_al)
        log_number("budget/display_total", display_total)


def get_lambda_int(p2_cfg: dict) -> float:
    """Read renamed lambda_int with one-release lambda_trust compatibility."""
    if "lambda_int" in p2_cfg:
        return float(p2_cfg["lambda_int"])
    return float(p2_cfg.get("lambda_trust", 3e-3))


def slice_judge_align(
    judge_align: dict[str, torch.Tensor] | None,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Slice judge alignment tensors by a boolean mask, preserving new/legacy keys."""
    if judge_align is None:
        return None
    mask_cpu = mask.detach().cpu()
    sliced: dict[str, torch.Tensor] = {}
    for key, value in judge_align.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == mask_cpu.shape[0]:
            sliced[key] = value[mask_cpu]
    return sliced


# ════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    relation_evidence_pi: torch.Tensor,
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
    epi_t = relation_evidence_pi[train_mask].to(device)
    jf_t = judge_features[train_mask].to(device) if judge_features is not None else None
    jm_t = judge_mask[train_mask].to(device) if judge_mask is not None else None
    y_t = y[train_mask]

    ja_t = slice_judge_align(judge_align, train_mask)

    outputs = reasoner(z_t, bl_t, rf_t, judge_features=jf_t, judge_mask=jm_t)
    outputs["relation_evidence_pi"] = epi_t

    # All-true mask since we already sliced to train nodes
    all_train = torch.ones(z_t.shape[0], dtype=torch.bool, device=device)

    pw = pos_weight if pos_weight is not None else torch.tensor(1.0, device=device)

    loss, loss_dict = compute_phase2_loss(
        outputs=outputs,
        y=y_t,
        train_mask=all_train,
        judge_align=ja_t,
        pos_weight=pw,
        lambda_int=get_lambda_int(p2_cfg),
        lambda_trust=p2_cfg.get("lambda_trust"),
        lambda_sparse=p2_cfg.get("lambda_sparse", 1e-3),
        lambda_align=p2_cfg.get("lambda_align", 1e-3),
        eta_llm=p2_cfg.get("eta_llm", 2.0),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_dict


def build_phase2_reasoner(
    *,
    p2_cfg: dict,
    z_dim: int,
    relation_names: list[str],
    judge_feature_dim: int,
    use_judge: bool,
    device: torch.device,
) -> CoVERRelReasoner:
    """Construct the reasoner from config with an explicit judge-path switch."""
    return CoVERRelReasoner(
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
        judge_feature_dim=judge_feature_dim if use_judge else 0,
        judge_hidden_dim=p2_cfg.get("judge_hidden_dim", 32),
        judge_dropout=p2_cfg.get("judge_dropout", 0.3),
        delta_llm_max=p2_cfg.get("delta_llm_max", 0.75),
        alpha_max=p2_cfg.get("alpha_max", 0.0),
        alpha_bias_init=p2_cfg.get("alpha_bias_init", -3.0),
    ).to(device)


def run_training_stage(
    *,
    stage_name: str,
    reasoner,
    p2_cfg: dict,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    relation_evidence_pi: torch.Tensor,
    judge_features: torch.Tensor | None,
    judge_mask_tensor: torch.Tensor | None,
    judge_align: dict[str, torch.Tensor] | None,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    k_values: list[int],
    pos_weight: torch.Tensor,
    relation_names: list[str],
    tb_logger: TensorBoardLogger | None,
    start_global_epoch: int,
) -> dict[str, object]:
    """Train one curriculum stage and return best state plus per-epoch rows."""
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=p2_cfg.get("lr", 1e-3),
        weight_decay=p2_cfg.get("weight_decay", 1e-4),
    )
    epochs = int(p2_cfg.get("epochs", 300))
    patience = int(p2_cfg.get("patience", 50))
    early_stop_metric = str(p2_cfg.get("early_stop_metric", "val_auprc"))
    eval_interval = max(int(p2_cfg.get("eval_interval", 1)), 1)

    best_val_score = float("-inf")
    best_state = None
    best_stage_epoch = 0
    patience_counter = 0
    rows: list[dict] = []
    stage_start = time.time()
    global_epoch = int(start_global_epoch)

    print(
        f"[Phase2:{stage_name}] Training for up to {epochs} epochs, "
        f"patience={patience}, metric={early_stop_metric}, eval_interval={eval_interval}"
    )

    for stage_epoch in range(1, epochs + 1):
        global_epoch += 1
        loss, loss_dict = train_one_epoch(
            reasoner,
            base_z,
            base_logits,
            rel_features,
            relation_evidence_pi,
            judge_features,
            judge_mask_tensor,
            judge_align,
            y,
            train_mask,
            optimizer,
            p2_cfg,
            stage_epoch,
            pos_weight=pos_weight,
        )

        should_eval = stage_epoch == 1 or stage_epoch == epochs or (stage_epoch % eval_interval == 0)
        val_metrics: dict[str, float] = {}
        diag: dict[str, float] = {}
        if should_eval:
            val_metrics = evaluate_phase2(
                reasoner, base_z, base_logits, rel_features, judge_features,
                judge_mask_tensor, val_mask, y, device, k_values=k_values,
            )

            diag = compute_epoch_diagnostics(
                reasoner, base_z, base_logits, rel_features, judge_features,
                judge_mask_tensor, device, y=y, judge_align=judge_align, mask=val_mask,
                relation_evidence_pi=relation_evidence_pi,
            )

        row: dict[str, object] = {
            "epoch": float(global_epoch),
            "global_epoch": float(global_epoch),
            "stage_epoch": float(stage_epoch),
            "stage": stage_name,
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
        log_phase2_epoch_to_tensorboard(
            tb_logger, global_epoch, loss, loss_dict, val_metrics, diag, relation_names, optimizer,
            p2_cfg=p2_cfg,
        )

        if should_eval and (stage_epoch % 10 == 0 or stage_epoch == 1 or stage_epoch == epochs):
            print(
                f"  [{stage_name}] Epoch {stage_epoch:3d} | Loss {loss:.4f} | "
                f"Val AUPRC {val_metrics.get('auprc', 0):.4f} | "
                f"Val AUC {val_metrics.get('roc_auc', 0):.4f} | "
                f"Val Macro-F1 {val_metrics.get('macro_f1', 0):.4f}"
            )

        if should_eval:
            val_score = val_metrics.get(
                early_stop_metric.replace("val_", ""), val_metrics.get("auprc", 0),
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in reasoner.state_dict().items()}
                best_stage_epoch = stage_epoch
                patience_counter = 0
            else:
                patience_counter += eval_interval
                if patience_counter >= patience:
                    print(f"  [{stage_name}] Early stopping at epoch {stage_epoch} (best={best_stage_epoch})")
                    break

    if best_state is not None:
        reasoner.load_state_dict(best_state)

    return {
        "reasoner": reasoner,
        "best_state": best_state,
        "best_epoch": best_stage_epoch,
        "best_val_score": best_val_score,
        "epochs_trained": stage_epoch,
        "elapsed_seconds": time.time() - stage_start,
        "rows": rows,
        "global_epoch": global_epoch,
    }


def _mean_float(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def compute_full_cover_diagnostics(
    *,
    reasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    judge_features: torch.Tensor | None,
    judge_mask: torch.Tensor | None,
    y: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    relation_names: list[str],
    accepted_records: list[dict],
    relation_evidence_pi: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute the four Full-CoVER diagnostic blocks required by augment.md."""
    reasoner.eval()
    outputs = reasoner(
        base_z,
        base_logits,
        rel_features.to(device),
        judge_features=judge_features.to(device) if judge_features is not None else None,
        judge_mask=judge_mask.to(device) if judge_mask is not None else None,
    )
    final_logit = outputs["final_logit"]
    gate = outputs["relation_gate"]
    delta_rel = outputs["delta_rel"]
    alpha = outputs["alpha_llm"]
    delta_llm = outputs["delta_llm"]
    actual_judge = alpha * delta_llm
    intervention = delta_rel + actual_judge

    y_bool = y.to(device).view(-1).float() >= 0.5
    test_bool = test_mask.to(device).view(-1).bool()
    base_pred = base_logits.to(device).view(-1) >= 0
    cover_pred = final_logit.view(-1) >= 0
    base_correct = base_pred == y_bool
    cover_correct = cover_pred == y_bool

    def count(mask: torch.Tensor) -> int:
        return int((test_bool & mask).sum().item())

    n_test = int(test_bool.sum().item())
    correction_counts = {
        "base_correct_cover_correct": count(base_correct & cover_correct),
        "base_correct_cover_wrong": count(base_correct & (~cover_correct)),
        "base_wrong_cover_correct": count((~base_correct) & cover_correct),
        "base_wrong_cover_wrong": count((~base_correct) & (~cover_correct)),
    }
    correction_table = {
        "n_test": n_test,
        "counts": correction_counts,
        "fractions": {
            key: (float(value) / max(n_test, 1))
            for key, value in correction_counts.items()
        },
    }

    judge_bool = (
        judge_mask.to(device).view(-1).bool()
        if judge_mask is not None
        else torch.zeros_like(test_bool)
    )

    strength_by_node: dict[int, str] = {}
    key_by_node: dict[int, str] = {}
    for rec in accepted_records:
        try:
            node_id = int(rec.get("node_id"))
        except (TypeError, ValueError):
            continue
        strength_by_node[node_id] = str(rec.get("evidence_strength", "missing")).lower()
        key_by_node[node_id] = str(rec.get("key_relation", "")).upper()

    strength_masks: dict[str, torch.Tensor] = {}
    for strength in ("strong", "moderate", "weak", "uncertain"):
        ids = [node_id for node_id, val in strength_by_node.items() if val == strength]
        m = torch.zeros_like(test_bool)
        if ids:
            valid = [idx for idx in ids if 0 <= idx < m.numel()]
            if valid:
                m[torch.tensor(valid, device=device, dtype=torch.long)] = True
        strength_masks[strength] = m

    def mean_abs_for(mask: torch.Tensor, tensor: torch.Tensor = intervention) -> float:
        m = mask.to(device).bool()
        return float(tensor[m].abs().mean().item()) if bool(m.any()) else 0.0

    intervention_magnitude: dict[str, object] = {
        "overall": mean_abs_for(test_bool),
        "base_correct": mean_abs_for(test_bool & base_correct),
        "base_wrong": mean_abs_for(test_bool & (~base_correct)),
        "judge_accepted": mean_abs_for(test_bool & judge_bool),
        "judge_rejected_or_missing": mean_abs_for(test_bool & (~judge_bool)),
        "by_evidence_strength": {
            strength: mean_abs_for(test_bool & mask)
            for strength, mask in strength_masks.items()
        },
    }

    eps = 1e-8
    gate_clamped = gate.clamp(eps, 1.0)
    gate_entropy = -(gate_clamped * gate_clamped.log()).sum(dim=-1)
    gate_argmax = gate.argmax(dim=-1)
    argmax_counts = {
        relation_names[idx]: int((test_bool & (gate_argmax == idx)).sum().item())
        for idx in range(len(relation_names))
    }
    rel_index = {name.upper(): idx for idx, name in enumerate(relation_names)}
    agreement_values: list[float] = []
    agreement_by_strength: dict[str, float] = {}
    for strength, strength_mask in strength_masks.items():
        vals: list[float] = []
        ids = [
            node_id for node_id, val in strength_by_node.items()
            if val == strength and key_by_node.get(node_id, "") in rel_index and 0 <= node_id < gate_argmax.numel()
        ]
        for node_id in ids:
            agree = float(gate_argmax[node_id].item() == rel_index[key_by_node[node_id]])
            vals.append(agree)
            agreement_values.append(agree)
        agreement_by_strength[strength] = _mean_float(vals)

    gate_explanation: dict[str, object] = {
        "mean_gate": {
            relation_names[idx]: float(gate[test_bool, idx].mean().item()) if n_test else 0.0
            for idx in range(len(relation_names))
        },
        "mean_entropy": float(gate_entropy[test_bool].mean().item()) if n_test else 0.0,
        "argmax_counts": argmax_counts,
        "argmax_fractions": {
            key: float(value) / max(n_test, 1)
            for key, value in argmax_counts.items()
        },
        "gate_key_agreement": _mean_float(agreement_values),
        "gate_key_agreement_by_strength": agreement_by_strength,
    }
    if relation_evidence_pi is not None:
        eps = 1e-8
        pi = relation_evidence_pi.to(device).detach().clamp_min(eps)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)
        gate_explanation["mean_evidence_pi"] = {
            relation_names[idx]: float(pi[test_bool, idx].mean().item()) if n_test else 0.0
            for idx in range(len(relation_names))
        }
        gate_explanation["mean_evidence_gate_kl"] = (
            float((pi[test_bool] * (torch.log(pi[test_bool] + eps) - torch.log(gate[test_bool].clamp_min(eps)))).sum(dim=-1).mean().item())
            if n_test else 0.0
        )

    def mean_for(mask: torch.Tensor, tensor: torch.Tensor) -> float:
        m = mask.to(device).bool()
        return float(tensor[m].mean().item()) if bool(m.any()) else 0.0

    judge_residual_behavior: dict[str, object] = {
        "mean_alpha": mean_for(test_bool, alpha),
        "mean_alpha_accepted": mean_for(test_bool & judge_bool, alpha),
        "mean_alpha_rejected_or_missing": mean_for(test_bool & (~judge_bool), alpha),
        "mean_abs_alpha_delta_llm": mean_abs_for(test_bool, actual_judge),
        "mean_abs_alpha_delta_llm_accepted": mean_abs_for(test_bool & judge_bool, actual_judge),
        "mean_abs_alpha_delta_llm_rejected_or_missing": mean_abs_for(test_bool & (~judge_bool), actual_judge),
        "by_evidence_strength": {
            strength: {
                "mean_alpha": mean_for(test_bool & mask, alpha),
                "mean_abs_alpha_delta_llm": mean_abs_for(test_bool & mask, actual_judge),
            }
            for strength, mask in strength_masks.items()
        },
        "by_base_correctness": {
            "base_correct": {
                "mean_alpha": mean_for(test_bool & base_correct, alpha),
                "mean_abs_alpha_delta_llm": mean_abs_for(test_bool & base_correct, actual_judge),
            },
            "base_wrong": {
                "mean_alpha": mean_for(test_bool & (~base_correct), alpha),
                "mean_abs_alpha_delta_llm": mean_abs_for(test_bool & (~base_correct), actual_judge),
            },
        },
    }

    return {
        "correction_table": correction_table,
        "intervention_magnitude": intervention_magnitude,
        "gate_explanation": gate_explanation,
        "judge_residual_behavior": judge_residual_behavior,
    }


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
    parser.add_argument("--lambda_int", type=float, default=None)
    parser.add_argument("--lambda_trust", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    parser.add_argument("--lambda_align", type=float, default=None)
    parser.add_argument("--eta_llm", type=float, default=None)
    parser.add_argument("--delta_rel_max", type=float, default=None)
    parser.add_argument("--delta_llm_max", type=float, default=None)
    parser.add_argument("--alpha_bias_init", type=float, default=None,
                        help="Override last-layer bias for alpha head. -3.0 (default) places "
                             "sigmoid near saturation and creates a gradient dead zone; set to "
                             "0.0 to keep alpha in the high-Jacobian regime so the judge "
                             "branch can actually be learned.")
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
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="Override TensorBoard log directory for this run/seed",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard logging for Phase2",
    )
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument(
        "--two-stage",
        dest="two_stage",
        action="store_true",
        default=None,
        help="Train relation-first Stage A, then judge-enabled Full CoVER Stage B (default).",
    )
    protocol.add_argument(
        "--single-stage",
        dest="two_stage",
        action="store_false",
        help="Train the configured Phase2 reasoner directly from scratch.",
    )
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
    if args.lambda_int is not None:
        p2_cfg["lambda_int"] = args.lambda_int
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
    if args.alpha_bias_init is not None:
        p2_cfg["alpha_bias_init"] = args.alpha_bias_init
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
    if "lambda_int" not in p2_cfg and "lambda_trust" in p2_cfg:
        p2_cfg["lambda_int"] = p2_cfg["lambda_trust"]

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    model_name = config["model"]["name"]
    seed = args.seed if args.seed is not None else config["train"]["seed"]
    run_name = p2_cfg.get("run_name", "phase2")
    relation_names = p2_cfg.get("relation_names", [])
    two_stage = bool(p2_cfg.get("two_stage", True) if args.two_stage is None else args.two_stage)
    p2_cfg["two_stage"] = two_stage

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
    base_freeze_before = snapshot_base_freeze(
        dataset_name, model_name, seed, base_logits, base_z
    )
    z_dim = base_z.shape[1]

    # ── Load relation features ──
    rel_stats, rel_meta = load_relation_features_for_phase2(
        dataset_name, model_name, seed, num_nodes,
    )
    rel_features = rel_stats.to(device)  # (N, rel_dim)
    relation_evidence_pi = build_relation_evidence_distribution(
        rel_features,
        num_relations=len(relation_names),
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        tau=p2_cfg.get("evidence_tau", 1.5),
    ).to(device)

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
            print(f"[Phase2] TensorBoard: {tb_dir}")
        else:
            print("[Phase2] TensorBoard unavailable; install tensorboard to enable event logs")
            tb_logger = None

    # ── Training ──
    k_values = config.get("eval", {}).get("k_values", [50, 100, 200])
    debug_epochs = config["train"].get("debug_epochs", 3)
    if args.debug:
        p2_cfg["epochs"] = debug_epochs

    epoch_rows: list[dict] = []
    start_time = time.time()
    stage_summaries: dict[str, dict[str, object]] = {}

    # Compute pos_weight from train labels
    y_train = y[train_mask]
    n_pos = y_train.sum().item()
    n_neg = (y_train.numel() - n_pos)
    pw = torch.tensor(max(n_neg / max(n_pos, 1), 1.0), device=device)
    print(f"[Phase2] pos_weight={pw.item():.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    judge_feature_dim = judge_features.shape[1] if judge_features is not None else 0
    global_epoch = 0
    stage_a_state = None
    if two_stage and use_judge:
        p2_stage_a = copy.deepcopy(p2_cfg)
        p2_stage_a["use_judge"] = False
        p2_stage_a["alpha_max"] = 0.0
        p2_stage_a["lambda_align"] = 0.0
        reasoner_a = build_phase2_reasoner(
            p2_cfg=p2_stage_a,
            z_dim=z_dim,
            relation_names=relation_names,
            judge_feature_dim=0,
            use_judge=False,
            device=device,
        )
        print(f"[Phase2:stage_A] Relation-first params: {sum(p.numel() for p in reasoner_a.parameters()):,}")
        stage_a = run_training_stage(
            stage_name="stage_A_relation_first",
            reasoner=reasoner_a,
            p2_cfg=p2_stage_a,
            base_z=base_z,
            base_logits=base_logits,
            rel_features=rel_features,
            relation_evidence_pi=relation_evidence_pi,
            judge_features=None,
            judge_mask_tensor=torch.zeros(num_nodes, dtype=torch.bool, device=device),
            judge_align=None,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            device=device,
            k_values=k_values,
            pos_weight=pw,
            relation_names=relation_names,
            tb_logger=tb_logger,
            start_global_epoch=global_epoch,
        )
        epoch_rows.extend(stage_a["rows"])  # type: ignore[arg-type]
        global_epoch = int(stage_a["global_epoch"])
        stage_a_state = stage_a["best_state"]
        stage_summaries["stage_A_relation_first"] = {
            "best_epoch": stage_a["best_epoch"],
            "best_val_score": stage_a["best_val_score"],
            "epochs_trained": stage_a["epochs_trained"],
            "elapsed_seconds": stage_a["elapsed_seconds"],
        }

    final_use_judge = bool(use_judge)
    reasoner = build_phase2_reasoner(
        p2_cfg=p2_cfg,
        z_dim=z_dim,
        relation_names=relation_names,
        judge_feature_dim=judge_feature_dim,
        use_judge=final_use_judge,
        device=device,
    )
    print(f"[Phase2] Final reasoner params: {sum(p.numel() for p in reasoner.parameters()):,}")
    if stage_a_state is not None:
        load_msg = reasoner.load_state_dict(stage_a_state, strict=False)
        print(
            "[Phase2:stage_B] Loaded Stage A relation weights; "
            f"missing={len(load_msg.missing_keys)} unexpected={len(load_msg.unexpected_keys)}"
        )

    stage_name = "stage_B_full_cover" if two_stage and use_judge else "single_stage"
    stage_b = run_training_stage(
        stage_name=stage_name,
        reasoner=reasoner,
        p2_cfg=p2_cfg,
        base_z=base_z,
        base_logits=base_logits,
        rel_features=rel_features,
        relation_evidence_pi=relation_evidence_pi,
        judge_features=judge_features,
        judge_mask_tensor=judge_mask_tensor,
        judge_align=judge_align,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        device=device,
        k_values=k_values,
        pos_weight=pw,
        relation_names=relation_names,
        tb_logger=tb_logger,
        start_global_epoch=global_epoch,
    )
    reasoner = stage_b["reasoner"]
    best_state = stage_b["best_state"]
    best_epoch = int(stage_b["best_epoch"])
    best_val_score = float(stage_b["best_val_score"])
    epoch_rows.extend(stage_b["rows"])  # type: ignore[arg-type]
    global_epoch = int(stage_b["global_epoch"])
    stage_summaries[stage_name] = {
        "best_epoch": stage_b["best_epoch"],
        "best_val_score": stage_b["best_val_score"],
        "epochs_trained": stage_b["epochs_trained"],
        "elapsed_seconds": stage_b["elapsed_seconds"],
    }
    elapsed = time.time() - start_time

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
        {
            "model_state_dict": best_state,
            "config": p2_cfg,
            "seed": seed,
            "epoch": best_epoch,
            "train_protocol": "two_stage" if two_stage and use_judge else "single_stage",
            "stage_summaries": stage_summaries,
        },
        ckpt_dir / "reasoner.pt",
    )
    if stage_a_state is not None:
        torch.save(
            {
                "model_state_dict": stage_a_state,
                "config": {**p2_cfg, "use_judge": False, "alpha_max": 0.0, "lambda_align": 0.0},
                "seed": seed,
                "stage": "stage_A_relation_first",
                "stage_summaries": stage_summaries.get("stage_A_relation_first", {}),
            },
            ckpt_dir / "reasoner_stage_A.pt",
        )

    # Epoch log
    write_phase2_epoch_log(log_dir, epoch_rows)

    # Test metrics (stage3_metrics.json for backward compat)
    with open(results_dir / "stage3_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Reproducibility config and command.
    repro_config = copy.deepcopy(config)
    repro_config["phase2_reasoner"] = p2_cfg
    for out_dir in (log_dir, results_dir):
        with open(out_dir / "repro_config.yaml", "w") as f:
            yaml.safe_dump(repro_config, f, sort_keys=False)
        command = [
            sys.executable,
            "scripts/train_phase2_reasoner.py",
            "--config",
            str(out_dir / "repro_config.yaml"),
            "--seed",
            str(seed),
            "--device",
            str(device),
            "--run_name",
            str(run_name),
            "--two-stage" if two_stage else "--single-stage",
        ]
        (out_dir / "repro_command.txt").write_text(" ".join(command) + "\n")

    # Diagnostics summary
    final_diag = compute_epoch_diagnostics(
        reasoner, base_z, base_logits, rel_features, judge_features,
        judge_mask_tensor, device, y=y, judge_align=judge_align, mask=test_mask,
        relation_evidence_pi=relation_evidence_pi,
    )
    full_cover_diagnostics = compute_full_cover_diagnostics(
        reasoner=reasoner,
        base_z=base_z,
        base_logits=base_logits,
        rel_features=rel_features,
        judge_features=judge_features,
        judge_mask=judge_mask_tensor,
        y=y,
        test_mask=test_mask,
        device=device,
        relation_names=relation_names,
        accepted_records=accepted_records,
        relation_evidence_pi=relation_evidence_pi,
    )
    diagnostics = {
        "config": config,
        "phase2_reasoner": p2_cfg,
        "seed": seed,
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "train_protocol": "two_stage" if two_stage and use_judge else "single_stage",
        "stage_summaries": stage_summaries,
        "device": device_info,
        "tensorboard": {
            "enabled": bool(tb_logger is not None and HAS_TENSORBOARD),
            "path": str(tb_dir) if tb_dir is not None else None,
        },
        "git_hash": get_git_hash(),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "early_stop_metric": p2_cfg.get("early_stop_metric", "val_auprc"),
        "epochs_trained": global_epoch,
        "elapsed_seconds": elapsed,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "final_diagnostics": final_diag,
        "full_cover_diagnostics": full_cover_diagnostics,
        "relation_feature_meta": rel_meta,
        "num_accepted_judge": int(judge_mask_tensor.sum().item()),
        "base_freeze_check": verify_base_frozen(
            before=base_freeze_before,
            after=snapshot_base_freeze(
                dataset_name, model_name, seed, base_logits, base_z
            ),
        ),
    }
    with open(log_dir / "phase2_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    summary_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "run_name": run_name,
        "seed": seed,
        "git_hash": diagnostics["git_hash"],
        "device": device_info,
        "tensorboard": diagnostics["tensorboard"],
        "train_protocol": diagnostics["train_protocol"],
        "stage_summaries": stage_summaries,
        "config_path": str(args.config),
        "repro_config": str(results_dir / "repro_config.yaml"),
        "repro_command": str(results_dir / "repro_command.txt"),
        "checkpoint": str(ckpt_dir / "reasoner.pt"),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "final_diagnostics": final_diag,
        "full_cover_diagnostics": full_cover_diagnostics,
    }
    for out_dir in (log_dir, results_dir):
        with open(out_dir / "phase2_summary.json", "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)

    print(f"\nCheckpoint: {ckpt_dir / 'reasoner.pt'}")
    print(f"Metrics:    {results_dir / 'stage3_metrics.json'}")
    print(f"Summary:    {results_dir / 'phase2_summary.json'}")
    print(f"Log:        {log_dir / 'phase2_train_log.jsonl'}")
    print(f"Diag:       {log_dir / 'phase2_diagnostics.json'}")
    if tb_logger is not None:
        tb_logger.close()
        print(f"TensorBoard:{tb_dir}")


if __name__ == "__main__":
    main()
