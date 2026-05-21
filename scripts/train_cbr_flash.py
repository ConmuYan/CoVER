"""CBR-Flash trainer for RAER-FD.

This is the canonical student trainer for Work 2. It distills a frozen RAER
teacher into a lightweight residual student under the contract

    final_logit = base_logit + delta_phi

where the base detector is frozen and the student trunk is score-blind. The
trainer intentionally exposes only the final CBR-Flash path. Earlier research
modes are archived under ``archive/legacy_raer_migration_20260520``.

Outputs::

    artifacts/checkpoints/{ds}/{base}/cbr_flash/seed_{s}/cbr_flash_student.pt
    artifacts/results/{ds}/{base}/cbr_flash/seed_{s}/test_metrics.json
    artifacts/results/{ds}/{base}/cbr_flash/seed_{s}/cbr_flash_summary.json
    artifacts/logs/{ds}/{base}/cbr_flash/seed_{s}/cbr_flash_diagnostics.json
    artifacts/logs/{ds}/{base}/cbr_flash/seed_{s}/cbr_flash_train_log.{jsonl,csv}
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.lree import build_lree_extractor
from evidence.relation_features import RELATION_SCHEMAS
from evidence.scalable_lree import build_scalable_lree_extractor
from models.cbr_flash_adapter import CBRFlashAdapter
from models.raer_teacher import RAERTeacher
from scripts.train_raer_teacher import (
    load_frozen_base,
    load_relation_adjs_for_extractor_from_data,
    load_relation_features_for_raer,
    materialize_scalable_lree_features,
)
from training.metrics import compute_metrics, g_means, precision_recall_at_k
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir
from utils.threshold import evaluate_with_threshold, find_best_threshold


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
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def bern_kl_rev(p_s: torch.Tensor, p_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reverse Bernoulli KL: KL(p_s || p_t)."""
    p_s = p_s.clamp(eps, 1.0 - eps)
    p_t = p_t.clamp(eps, 1.0 - eps)
    return p_s * (p_s.log() - p_t.log()) + (1.0 - p_s) * (
        (1.0 - p_s).log() - (1.0 - p_t).log()
    )


def bern_kl_fwd(p_s: torch.Tensor, p_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Forward Bernoulli KL: KL(p_t || p_s)."""
    return bern_kl_rev(p_t, p_s, eps=eps)


class NodeReliabilityHelper:
    """Node reliability used by CBR-Flash.

    The reliability term is intentionally separate from the teacher: it
    controls how strongly the student follows teacher residual signals versus
    the supervised BCE anchor.
    """

    def __init__(
        self,
        r_c: float = 1.0,
        cal_bins: list[float] | None = None,
        n_bins: int = 10,
        r_min: float = 0.1,
    ):
        self.r_c = float(max(min(r_c, 1.0), r_min))
        self.n_bins = int(n_bins)
        self.r_min = float(r_min)
        if cal_bins is None:
            self.cal_bins = torch.ones(self.n_bins, dtype=torch.float32)
        else:
            self.cal_bins = torch.tensor(cal_bins, dtype=torch.float32)
            if self.cal_bins.numel() != self.n_bins:
                raise ValueError(
                    f"cal_bins must have {self.n_bins} entries, got {self.cal_bins.numel()}"
                )

    def cal_bin_reliability(self, p: torch.Tensor) -> torch.Tensor:
        bin_idx = (p * float(self.n_bins)).floor().long().clamp_(0, self.n_bins - 1)
        cal = self.cal_bins.to(device=p.device, dtype=p.dtype)
        return (2.0 * cal[bin_idx] - 1.0).abs()

    def r_node(self, p_t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.r_c * self.cal_bin_reliability(p_t), min=self.r_min, max=1.0)


def load_curriculum_prior(
    cell_key: str,
    prior_path: Path | None,
    default_r_c: float = 1.0,
) -> float:
    if prior_path is None or not prior_path.exists():
        return default_r_c
    with open(prior_path) as f:
        priors = json.load(f)
    return float(priors.get(cell_key, default_r_c))


def load_calibration_bins(
    cell_key: str,
    bins_path: Path | None,
    n_bins: int = 10,
) -> list[float] | None:
    if bins_path is None or not bins_path.exists():
        return None
    with open(bins_path) as f:
        bins_data = json.load(f)
    bins = bins_data.get(cell_key)
    if bins is None:
        return None
    if len(bins) != n_bins:
        raise ValueError(f"{cell_key} has {len(bins)} calibration bins, expected {n_bins}")
    return bins


@torch.no_grad()
def build_lree_teacher_features(
    config: dict,
    data,
    relation_names: list[str],
    extractor_ckpt_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild the LREE evidence tensor used by a frozen RAER-LREE teacher."""
    raer_cfg = config["raer_teacher"]
    evidence_cfg = raer_cfg.get("evidence", {}) or {}
    ext_cfg = evidence_cfg.get("extractor", {}) or {}
    out_dim = int(ext_cfg.get("out_dim_per_rel", raer_cfg.get("rel_stat_dim", 9)))
    extractor = build_lree_extractor(
        x_dim=int(data.x.shape[1]),
        num_relations=len(relation_names),
        cfg={**ext_cfg, "out_dim_per_rel": out_dim},
    ).to(device)
    extractor.load_state_dict(
        torch.load(extractor_ckpt_path, weights_only=True, map_location=device)
    )

    dataset_path = config["dataset"].get("path")
    if dataset_path is None:
        raise ValueError("dataset.path is required to rebuild LREE evidence")
    relation_adjs = load_relation_adjs_for_extractor_from_data(
        dataset_name=config["dataset"]["name"],
        dataset_path=dataset_path,
        relation_names=relation_names,
        data=data,
        device=device,
    )
    x_dev = data.x.to(device)
    extractor.prepare(x_dev, relation_adjs)
    extractor.eval()
    rel_features = extractor(x_dev, data.train_mask.to(device), data.y.to(device).long())
    print(f"[CBR-Flash] Rebuilt LREE features {tuple(rel_features.shape)} from {extractor_ckpt_path}")
    return rel_features.detach()


@torch.no_grad()
def build_scalable_lree_teacher_features(
    config: dict,
    seed: int,
    num_nodes: int,
    relation_names: list[str],
    extractor_ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Rebuild cache-backed scalable LREE evidence for a frozen teacher."""
    raer_cfg = config["raer_teacher"]
    evidence_cfg = raer_cfg.get("evidence", {}) or {}
    ext_cfg = evidence_cfg.get("extractor", {}) or {}

    relation_basis, rel_meta = load_relation_features_for_raer(
        config["dataset"]["name"],
        config["model"]["name"],
        seed,
        num_nodes=num_nodes,
    )
    if not raer_cfg.get("relation_names"):
        relation_names = [
            str(name).upper() for name in rel_meta.get("relations", relation_names)
        ]
    if relation_basis.shape[1] % len(relation_names) != 0:
        raise ValueError(
            "relation basis width is not divisible by number of relations: "
            f"{relation_basis.shape[1]} vs R={len(relation_names)}"
        )

    payload = torch.load(extractor_ckpt_path, weights_only=True, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        ckpt_in_dim = int(payload.get("in_dim_per_rel", 0))
        ckpt_out_dim = int(payload.get("out_dim_per_rel", 0))
    else:
        state = payload
        ckpt_in_dim = 0
        ckpt_out_dim = 0

    in_dim = ckpt_in_dim or int(relation_basis.shape[1] // len(relation_names))
    out_dim = ckpt_out_dim or int(ext_cfg.get("out_dim_per_rel", raer_cfg.get("rel_stat_dim", 9)))
    extractor = build_scalable_lree_extractor(
        num_relations=len(relation_names),
        in_dim_per_rel=in_dim,
        cfg={**ext_cfg, "in_dim_per_rel": in_dim, "out_dim_per_rel": out_dim},
    ).to(device)
    extractor.load_state_dict(state)
    rel_features = materialize_scalable_lree_features(
        extractor,
        relation_basis,
        device,
        batch_size=int(raer_cfg.get("feature_batch_size", 262_144)),
        output_device=device,
    )
    rel_meta = {
        **rel_meta,
        "source": "scalable_lree",
        "score_blind": True,
        "prototype_labels": rel_meta.get("prototype_labels", "train_only"),
        "basis": "compact_relation_features",
        "extractor_ckpt": str(extractor_ckpt_path),
    }
    print(
        f"[CBR-Flash] Rebuilt scalable LREE features {tuple(rel_features.shape)} "
        f"from {extractor_ckpt_path}"
    )
    return rel_features.detach(), rel_meta


@torch.no_grad()
def generate_teacher_cache(
    teacher: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 262_144,
) -> dict[str, torch.Tensor]:
    teacher.eval()
    parts: dict[str, list[torch.Tensor]] = {
        "p": [],
        "logit": [],
        "delta_rel": [],
    }
    chunk = max(int(batch_size), 1)
    for start in range(0, int(base_z.shape[0]), chunk):
        end = min(start + chunk, int(base_z.shape[0]))
        out = teacher(
            base_z[start:end],
            base_logits[start:end],
            rel_features[start:end].to(device),
        )
        logit = out["final_logit"].detach().to(device)
        parts["logit"].append(logit)
        parts["p"].append(torch.sigmoid(logit))
        parts["delta_rel"].append(out["delta_rel"].detach().to(device))
    return {
        key: torch.cat(value, dim=0) if value else torch.empty(0, device=device)
        for key, value in parts.items()
    }


@torch.no_grad()
def evaluate_base_only(
    base_logits: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
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
    teacher: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
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


@torch.no_grad()
def evaluate_teacher_chunked(
    teacher: RAERTeacher,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
    chunk_size: int = 262_144,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    teacher.eval()
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, idx.numel(), chunk):
        batch_idx = idx[start:start + chunk]
        out = teacher(
            base_z[batch_idx],
            base_logits[batch_idx],
            rel_features[batch_idx].to(device),
        )
        probs.append(torch.sigmoid(out["final_logit"]).cpu())
        labels.append(y[batch_idx].cpu())
    prob = torch.cat(probs).numpy() if probs else np.asarray([], dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
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
def evaluate_student(
    student: CBRFlashAdapter,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    student.eval()
    out = student(base_z[mask], base_logits[mask], rel_features[mask].to(device))
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
        return metrics, prob, y_np
    return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200]), prob, y_np


@torch.no_grad()
def evaluate_student_chunked(
    student: CBRFlashAdapter,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    threshold: float | None = None,
    k_values: list[int] | None = None,
    chunk_size: int = 262_144,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    student.eval()
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, idx.numel(), chunk):
        batch_idx = idx[start:start + chunk]
        out = student(
            base_z[batch_idx],
            base_logits[batch_idx],
            rel_features[batch_idx].to(device),
        )
        probs.append(torch.sigmoid(out["final_logit"]).cpu())
        labels.append(y[batch_idx].cpu())
    prob = torch.cat(probs).numpy() if probs else np.asarray([], dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
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


def select_cbr_batch(
    student: CBRFlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    train_mask: torch.Tensor,
    k_eff: int,
    criterion: str,
    full_batch_size: int = 262_144,
) -> torch.Tensor:
    with torch.no_grad():
        full_score = torch.full(
            (int(base_z.shape[0]),),
            -float("inf"),
            dtype=base_z.dtype,
            device=base_z.device,
        )
        p_t = teacher_cache["p"].to(base_z.device)
        chunk = max(int(full_batch_size), 1)
        for start in range(0, int(base_z.shape[0]), chunk):
            end = min(start + chunk, int(base_z.shape[0]))
            s_out = student(
                base_z[start:end],
                base_logits[start:end],
                rel_features[start:end],
                return_heads=False,
            )
            p_s = torch.sigmoid(s_out["final_logit"])
            p_t_chunk = p_t[start:end]
            if criterion == "topk_student_entropy":
                score = bernoulli_entropy(p_s)
            elif criterion == "topk_teacher_entropy":
                score = bernoulli_entropy(p_t_chunk)
            elif criterion == "topk_teacher_student_disagreement":
                score = (p_s - p_t_chunk).abs()
            elif criterion == "random":
                score = torch.rand_like(p_s)
            else:
                raise ValueError(f"unknown mask_criterion={criterion!r}")
            score = score.clone()
            score[~train_mask[start:end]] = -float("inf")
            full_score[start:end] = score
        return torch.topk(full_score, k_eff).indices


def cbr_weight_from_sensitivity(sensitivity: torch.Tensor, form: str) -> torch.Tensor:
    if form == "linear":
        return 1.0 - sensitivity
    if form == "squared":
        return (1.0 - sensitivity) ** 2
    if form == "exp":
        return torch.exp(-sensitivity)
    if form == "binary_low":
        return (sensitivity < 0.3).to(sensitivity.dtype)
    raise ValueError(f"unknown cbr_weight_form={form!r}")


def compute_cbr_flash_loss(
    *,
    student: CBRFlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    idx: torch.Tensor,
    rel_helper: NodeReliabilityHelper,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    cfg: dict,
    pos_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    s_out = student(base_z[idx], base_logits[idx], rel_features[idx], return_heads=True)
    p_s = s_out["p"]
    s_logit = s_out["logit"]

    p_t = teacher_cache["p"][idx].to(p_s.device).detach()
    teacher_logit = teacher_cache["logit"][idx].to(p_s.device).detach()
    base_idx = base_logits[idx].to(p_s.device).detach()

    eta = bernoulli_entropy(p_t) / math.log(2.0)
    loss_logit = (1.0 - eta) * bern_kl_rev(p_s, p_t) + eta * bern_kl_fwd(p_s, p_t)

    r_node = rel_helper.r_node(p_t)
    lambda_min = float(cfg.get("lambda_min", 0.05))
    lambda_extra = float(cfg.get("lambda_extra", 0.50))
    lambda_bce = lambda_min + (1.0 - r_node) * lambda_extra
    y_idx = y[idx].to(dtype=p_s.dtype, device=p_s.device)
    mask_lab = train_mask[idx].to(dtype=p_s.dtype, device=p_s.device)

    distill_per = r_node * loss_logit
    loss_distill = distill_per.sum() / r_node.sum().clamp_min(1.0)

    bce_per = F.binary_cross_entropy_with_logits(
        s_logit,
        y_idx,
        reduction="none",
        pos_weight=pos_weight,
    )
    bce_weight = lambda_bce * mask_lab
    loss_bce = (bce_weight * bce_per).sum() / bce_weight.sum().clamp_min(1.0)

    delta_max = float(student.delta_max)
    teacher_sensitivity = ((teacher_logit - base_idx).abs() / delta_max).clamp(0.0, 1.0)
    student_delta = (s_logit - base_idx).abs() / delta_max
    budget_weight = cbr_weight_from_sensitivity(
        teacher_sensitivity,
        str(cfg.get("cbr_weight_form", "linear")),
    )
    loss_budget = (student_delta * budget_weight).mean()

    total = (
        loss_distill
        + float(cfg.get("bce_lambda", 1.0)) * loss_bce
        + float(cfg.get("cbr_lambda", 0.5)) * loss_budget
    )
    stats = {
        "total": float(total.detach()),
        "l_distill": float(loss_distill.detach()),
        "l_bce": float(loss_bce.detach()),
        "l_budget": float(loss_budget.detach()),
        "l_logit": float(
            (loss_logit.detach() * r_node.detach()).sum()
            / r_node.sum().clamp_min(1.0)
        ),
        "r_node_mean": float(r_node.detach().mean()),
        "lambda_bce_mean": float(lambda_bce.detach().mean()),
        "teacher_sensitivity_mean": float(teacher_sensitivity.detach().mean()),
        "student_delta_mean": float(student_delta.detach().mean()),
        "batch_size": int(idx.numel()),
    }
    return total, stats


def train_cbr_flash(
    *,
    student: CBRFlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    data,
    cfg: dict,
    rel_helper: NodeReliabilityHelper,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)
    train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
    if train_idx.numel() == 0:
        raise ValueError("empty training split")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 80))
    patience = int(cfg.get("patience", 10))
    eval_interval = int(cfg.get("eval_interval", 5))
    k_eff = min(int(cfg.get("K", 2048)), int(train_idx.numel()))
    mask_criterion = str(cfg.get("mask_criterion", "topk_student_entropy"))
    full_batch_size = int(cfg.get("full_batch_size", 262_144))
    eval_batch_size = int(cfg.get("eval_batch_size", full_batch_size))

    best_val = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        student.train()
        idx = select_cbr_batch(
            student=student,
            teacher_cache=teacher_cache,
            base_z=base_z,
            base_logits=base_logits,
            rel_features=rel_features,
            train_mask=train_mask,
            k_eff=k_eff,
            criterion=mask_criterion,
            full_batch_size=full_batch_size,
        )
        loss, stats = compute_cbr_flash_loss(
            student=student,
            teacher_cache=teacher_cache,
            base_z=base_z,
            base_logits=base_logits,
            rel_features=rel_features,
            idx=idx,
            rel_helper=rel_helper,
            y=y,
            train_mask=train_mask,
            cfg=cfg,
            pos_weight=pos_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_clip = cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        if epoch % eval_interval == 0 or epoch == 1 or epoch == epochs:
            val_metrics, _, _ = evaluate_student_chunked(
                student,
                base_z,
                base_logits,
                rel_features,
                val_mask,
                y,
                device,
                chunk_size=eval_batch_size,
            )
            row = {
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val/auprc": float(val_metrics.get("auprc", 0.0)),
                "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                **{f"loss/{k}": float(v) for k, v in stats.items()},
            }
            rows.append(row)

            val_score = float(val_metrics.get("auprc", 0.0))
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = copy.deepcopy(student.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"[CBR-Flash] Early stop at epoch {epoch} "
                        f"(best val_auprc={best_val:.4f} @ {best_epoch})"
                    )
                    break

    if best_state is not None:
        student.load_state_dict(best_state)

    _, val_prob, val_y = evaluate_student_chunked(
        student,
        base_z,
        base_logits,
        rel_features,
        val_mask,
        y,
        device,
        chunk_size=eval_batch_size,
    )
    threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")
    val_metrics, _, _ = evaluate_student_chunked(
        student,
        base_z,
        base_logits,
        rel_features,
        val_mask,
        y,
        device,
        threshold=threshold,
        chunk_size=eval_batch_size,
    )
    test_metrics, _, _ = evaluate_student_chunked(
        student,
        base_z,
        base_logits,
        rel_features,
        test_mask,
        y,
        device,
        threshold=threshold,
        chunk_size=eval_batch_size,
    )

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, rows


def write_epoch_log(log_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    jsonl_path = log_dir / "cbr_flash_train_log.jsonl"
    csv_path = log_dir / "cbr_flash_train_log.csv"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBR-Flash residual distillation trainer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_name", type=str, default="cbr_flash")
    p.add_argument("--mode", type=str, default="cbr_flash", choices=["cbr_flash"])
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--teacher_extractor_ckpt", type=str, default=None,
                   help="Optional LREE checkpoint matching the teacher.")
    p.add_argument("--base_ckpt_path", type=str, default=None)

    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--mask_criterion", type=str, default="topk_student_entropy",
                   choices=[
                       "topk_student_entropy",
                       "topk_teacher_entropy",
                       "topk_teacher_student_disagreement",
                       "random",
                   ])
    p.add_argument("--cbr_lambda", type=float, default=0.5)
    p.add_argument("--bce_lambda", type=float, default=1.0)
    p.add_argument("--cbr_weight_form", type=str, default="linear",
                   choices=["linear", "squared", "exp", "binary_low"])
    p.add_argument("--lambda_min", type=float, default=0.05)
    p.add_argument("--lambda_extra", type=float, default=0.50)

    p.add_argument("--curriculum_prior_path", type=str,
                   default="artifacts/teacher_curriculum_prior.json")
    p.add_argument("--calibration_bins_path", type=str,
                   default="artifacts/teacher_calibration_bins.json")
    p.add_argument("--r_min", type=float, default=0.1)
    p.add_argument("--n_cal_bins", type=int, default=10)

    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    raer_cfg = config["raer_teacher"]
    cbr_cfg = config.get("cbr_flash", {}) or {}
    seed = int(args.seed)
    run_name = str(args.run_name)

    set_seed(seed)
    device = torch.device(args.device)

    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    result_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
    (log_dir / "repro_command.txt").write_text(" ".join(sys.argv) + "\n")
    (log_dir / "repro_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

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

    base_logits, base_z, base_ckpt_path = load_frozen_base(
        config,
        dataset_name,
        model_name,
        seed,
        data,
        device,
        ckpt_override=args.base_ckpt_path,
    )
    base_sha256 = hashlib.sha256(Path(base_ckpt_path).read_bytes()).hexdigest()
    print(f"[CBR-Flash] Frozen base SHA-256 = {base_sha256[:16]}...")

    relation_names = [str(name).upper() for name in raer_cfg.get("relation_names", [])]
    if not relation_names:
        relation_names = list(RELATION_SCHEMAS[dataset_name].keys())

    if args.teacher_extractor_ckpt is not None:
        evidence_cfg = raer_cfg.get("evidence", {}) or {}
        evidence_source = str(evidence_cfg.get("source", "learned")).lower()
        extractor_path = Path(args.teacher_extractor_ckpt)
        if evidence_source == "scalable_lree" or extractor_path.name == "scalable_lree.pt":
            rel_features, rel_meta = build_scalable_lree_teacher_features(
                config=config,
                seed=seed,
                num_nodes=int(data.x.shape[0]),
                relation_names=relation_names,
                extractor_ckpt_path=extractor_path,
                device=device,
            )
        else:
            rel_meta = {
                "relations": relation_names,
                "source": "learned",
                "score_blind": True,
                "prototype_labels": "train_only",
            }
            rel_features = build_lree_teacher_features(
                config=config,
                data=data,
                relation_names=relation_names,
                extractor_ckpt_path=extractor_path,
                device=device,
            )
    else:
        rel_features, rel_meta = load_relation_features_for_raer(
            dataset_name,
            model_name,
            seed,
            num_nodes=int(data.x.shape[0]),
        )
        rel_features = rel_features.to(device)
        if not raer_cfg.get("relation_names"):
            relation_names = [
                str(name).upper()
                for name in rel_meta.get("relations", relation_names)
            ]

    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.exists():
        raise FileNotFoundError(teacher_ckpt_path)
    teacher = RAERTeacher(
        base_z_dim=int(base_z.shape[1]),
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
    raw = torch.load(teacher_ckpt_path, weights_only=False, map_location=device)
    if isinstance(raw, dict) and "raer_teacher" in raw:
        state = raw["raer_teacher"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw
    teacher.load_state_dict(state)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    print(f"[CBR-Flash] Loaded RAER teacher from {teacher_ckpt_path}")

    teacher_cache = generate_teacher_cache(
        teacher,
        base_z,
        base_logits,
        rel_features,
        device,
        batch_size=int(cbr_cfg.get("teacher_cache_batch_size", 262_144)),
    )

    cell_key = f"{dataset_name}/{model_name}/seed_{seed}"
    prior_path = Path(args.curriculum_prior_path) if args.curriculum_prior_path else None
    bins_path = Path(args.calibration_bins_path) if args.calibration_bins_path else None
    r_c = load_curriculum_prior(cell_key, prior_path, default_r_c=1.0)
    cal_bins = load_calibration_bins(cell_key, bins_path, n_bins=args.n_cal_bins)
    rel_helper = NodeReliabilityHelper(
        r_c=r_c,
        cal_bins=cal_bins,
        n_bins=args.n_cal_bins,
        r_min=args.r_min,
    )

    student = CBRFlashAdapter(
        base_z_dim=int(base_z.shape[1]),
        num_relations=len(relation_names),
        rel_stat_dim=raer_cfg.get("rel_stat_dim", 9),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        delta_max=raer_cfg.get("delta_rel_max", 2.0),
    ).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"[CBR-Flash] Student params={n_student:,}; teacher params={n_teacher:,}")

    y_train = data.y[data.train_mask].long()
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)), device=device)

    train_cfg = {
        "K": cbr_cfg.get("K", args.K),
        "mask_criterion": cbr_cfg.get("mask_criterion", args.mask_criterion),
        "cbr_lambda": cbr_cfg.get("cbr_lambda", args.cbr_lambda),
        "bce_lambda": cbr_cfg.get("bce_lambda", args.bce_lambda),
        "cbr_weight_form": cbr_cfg.get("cbr_weight_form", args.cbr_weight_form),
        "lambda_min": cbr_cfg.get("lambda_min", args.lambda_min),
        "lambda_extra": cbr_cfg.get("lambda_extra", args.lambda_extra),
        "lr": cbr_cfg.get("lr", args.lr),
        "weight_decay": cbr_cfg.get("weight_decay", args.weight_decay),
        "epochs": cbr_cfg.get("epochs", args.epochs),
        "patience": cbr_cfg.get("patience", args.patience),
        "eval_interval": cbr_cfg.get("eval_interval", args.eval_interval),
        "grad_clip": cbr_cfg.get("grad_clip", args.grad_clip),
        "full_batch_size": cbr_cfg.get("full_batch_size", 262_144),
        "eval_batch_size": cbr_cfg.get("eval_batch_size", cbr_cfg.get("full_batch_size", 262_144)),
    }

    start = time.time()
    train_summary, rows = train_cbr_flash(
        student=student,
        teacher_cache=teacher_cache,
        base_z=base_z,
        base_logits=base_logits,
        rel_features=rel_features,
        data=data,
        cfg=train_cfg,
        rel_helper=rel_helper,
        device=device,
        pos_weight=pos_weight,
    )
    elapsed = time.time() - start

    write_epoch_log(log_dir, rows)
    student_ckpt = ckpt_dir / "cbr_flash_student.pt"
    torch.save(student.state_dict(), student_ckpt)

    base_sha256_post = hashlib.sha256(Path(base_ckpt_path).read_bytes()).hexdigest()
    assert base_sha256 == base_sha256_post, (
        "base detector checkpoint changed during CBR-Flash training: "
        f"{base_ckpt_path}"
    )

    k_values = [50, 100, 200]
    base_thr = find_best_threshold(
        data.y[data.val_mask].numpy(),
        torch.sigmoid(base_logits[data.val_mask.to(device)]).cpu().numpy(),
        metric="macro_f1",
    )[0]
    base_metrics = evaluate_base_only(
        base_logits,
        data.test_mask.to(device),
        data.y.to(device),
        threshold=base_thr,
        k_values=k_values,
    )

    _, teacher_val_prob, teacher_val_y = evaluate_teacher_chunked(
        teacher,
        base_z,
        base_logits,
        rel_features,
        data.val_mask.to(device),
        data.y.to(device),
        device,
        chunk_size=int(train_cfg.get("eval_batch_size", 262_144)),
    )
    teacher_thr = find_best_threshold(teacher_val_y, teacher_val_prob, metric="macro_f1")[0]
    teacher_metrics, _, _ = evaluate_teacher_chunked(
        teacher,
        base_z,
        base_logits,
        rel_features,
        data.test_mask.to(device),
        data.y.to(device),
        device,
        threshold=teacher_thr,
        k_values=k_values,
        chunk_size=int(train_cfg.get("eval_batch_size", 262_144)),
    )

    diagnostics = {
        "run_name": run_name,
        "method": "CBR-Flash",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "git_hash": get_git_hash(),
        "elapsed_seconds": elapsed,
        "student_params": n_student,
        "teacher_params": n_teacher,
        "param_ratio_student_to_teacher": round(n_student / max(n_teacher, 1), 6),
        "teacher_ckpt": str(teacher_ckpt_path),
        "teacher_extractor_ckpt": args.teacher_extractor_ckpt,
        "base_ckpt": str(base_ckpt_path),
        "base_sha256": base_sha256,
        "relation_names": relation_names,
        "relation_feature_meta": rel_meta,
        "cell_key": cell_key,
        "r_c": r_c,
        "cal_bins_loaded": cal_bins is not None,
        "train_cfg": train_cfg,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics_student": train_summary["test_metrics"],
        "test_metrics_teacher": teacher_metrics,
        "test_metrics_base": base_metrics,
    }
    (log_dir / "cbr_flash_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "run_name": run_name,
        "method": "CBR-Flash",
        "seed": seed,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics": train_summary["test_metrics"],
        "elapsed_seconds": elapsed,
    }
    (result_dir / "cbr_flash_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (result_dir / "test_metrics.json").write_text(
        json.dumps(train_summary["test_metrics"], indent=2, sort_keys=True) + "\n"
    )

    student_metrics = train_summary["test_metrics"]
    print(
        f"\n[CBR-Flash] Done in {elapsed:.1f}s | "
        f"best_epoch={train_summary['best_epoch']}"
    )
    print(
        f"  Student AUPRC={student_metrics.get('auprc', 0.0):.4f} "
        f"AUROC={student_metrics.get('roc_auc', 0.0):.4f}"
    )
    print(
        f"  Teacher AUPRC={teacher_metrics.get('auprc', 0.0):.4f} "
        f"Base AUPRC={base_metrics.get('auprc', 0.0):.4f}"
    )
    print(f"Checkpoint: {student_ckpt}")
    print(f"Metrics:    {result_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
