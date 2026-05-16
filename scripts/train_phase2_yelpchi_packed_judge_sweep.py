"""GPU-packed YelpChi Phase2 judge sweep.

This script is intentionally specialized for the remaining YelpChi judge /
alpha sensitivity runs.  It trains multiple hyperparameter variants for the
same seed in one batched GPU model instead of launching one Python process per
variant.  That makes the work a larger GPU kernel and avoids repeated CPU-side
validation loops.
"""

from __future__ import annotations

import argparse
import csv
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.judge_encoder import load_jsonl
from scripts.train_phase2_reasoner import (
    load_frozen_base,
    load_judge_data,
    load_relation_features_for_phase2,
    runtime_device_info,
)
from training.phase2_losses import build_judge_relation_targets
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir


SEEDS = [42, 123, 456, 789, 2026]
RUN_SPECS = [
    {
        "run_names": ["phase2_yelp_s3_lalign_0"],
        "lambda_align": 0.0,
        "alpha_max": 0.0,
    },
    {
        "run_names": ["phase2_yelp_s3_lalign_1em3", "phase2_yelp_s3_alpha_0"],
        "lambda_align": 1.0e-3,
        "alpha_max": 0.0,
    },
    {
        "run_names": ["phase2_yelp_s3_lalign_3em3"],
        "lambda_align": 3.0e-3,
        "alpha_max": 0.0,
    },
    {
        "run_names": ["phase2_yelp_s3_lalign_1em2"],
        "lambda_align": 1.0e-2,
        "alpha_max": 0.0,
    },
    {
        "run_names": ["phase2_yelp_s3_alpha_01"],
        "lambda_align": 1.0e-3,
        "alpha_max": 0.1,
    },
    {
        "run_names": ["phase2_yelp_s3_alpha_03"],
        "lambda_align": 1.0e-3,
        "alpha_max": 0.3,
    },
]


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    torch.set_num_threads(max(int(os.environ.get("COVER_NUM_THREADS", "1")), 1))
    try:
        torch.set_num_interop_threads(max(int(os.environ.get("COVER_INTEROP_THREADS", "1")), 1))
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class PackedYelpJudgeReasoner(nn.Module):
    """Batched equivalent of CoVERRelReasoner for fixed YelpChi dimensions."""

    def __init__(
        self,
        num_models: int,
        base_z_dim: int,
        judge_feature_dim: int,
        rel_stat_dim: int = 9,
        rel_hidden_dim: int = 64,
        judge_hidden_dim: int = 32,
        num_relations: int = 3,
        tau_gate: float = 0.7,
        delta_rel_max: float = 2.0,
        delta_llm_max: float = 0.75,
        alpha_bias_init: float = -3.0,
        alpha_max: list[float] | Tensor | None = None,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.num_models = int(num_models)
        self.base_z_dim = int(base_z_dim)
        self.judge_feature_dim = int(judge_feature_dim)
        self.rel_stat_dim = int(rel_stat_dim)
        self.rel_hidden_dim = int(rel_hidden_dim)
        self.judge_hidden_dim = int(judge_hidden_dim)
        self.num_relations = int(num_relations)
        self.tau_gate = float(tau_gate)
        self.delta_rel_max = float(delta_rel_max)
        self.delta_llm_max = float(delta_llm_max)
        self.dropout = float(dropout)

        b = self.num_models
        r = self.num_relations
        h = self.rel_hidden_dim
        d = self.rel_stat_dim
        z = self.base_z_dim
        jdim = self.judge_feature_dim
        jh = self.judge_hidden_dim

        self.rel_w1 = nn.Parameter(torch.empty(b, r, h, d))
        self.rel_b1 = nn.Parameter(torch.empty(b, r, h))
        self.rel_w2 = nn.Parameter(torch.empty(b, r, h, h))
        self.rel_b2 = nn.Parameter(torch.empty(b, r, h))
        self.rel_ln_w = nn.Parameter(torch.ones(b, r, h))
        self.rel_ln_b = nn.Parameter(torch.zeros(b, r, h))
        self.rel_head_w = nn.Parameter(torch.zeros(b, r, h))
        self.rel_head_b = nn.Parameter(torch.zeros(b, r))

        gate_in = z + r * h
        self.gate_w = nn.Parameter(torch.zeros(b, r, gate_in))
        self.gate_b = nn.Parameter(torch.zeros(b, r))

        self.judge_w = nn.Parameter(torch.empty(b, jh, jdim))
        self.judge_b = nn.Parameter(torch.empty(b, jh))
        self.judge_ln_w = nn.Parameter(torch.ones(b, jh))
        self.judge_ln_b = nn.Parameter(torch.zeros(b, jh))

        cond_dim = jh + h + r
        self.alpha_w1 = nn.Parameter(torch.empty(b, jh, cond_dim))
        self.alpha_b1 = nn.Parameter(torch.empty(b, jh))
        self.alpha_w2 = nn.Parameter(torch.zeros(b, 1, jh))
        self.alpha_b2 = nn.Parameter(torch.full((b, 1), float(alpha_bias_init)))

        self.llm_w1 = nn.Parameter(torch.empty(b, jh, cond_dim))
        self.llm_b1 = nn.Parameter(torch.empty(b, jh))
        self.llm_w2 = nn.Parameter(torch.zeros(b, 1, jh))
        self.llm_b2 = nn.Parameter(torch.zeros(b, 1))

        alpha_tensor = torch.tensor(alpha_max if alpha_max is not None else [0.0] * b, dtype=torch.float32)
        self.register_buffer("alpha_max_vec", alpha_tensor.view(b))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(self.num_models):
            for rel in range(self.num_relations):
                nn.init.kaiming_uniform_(self.rel_w1[idx, rel], a=math.sqrt(5))
                fan_in = self.rel_w1.shape[-1]
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.rel_b1[idx, rel], -bound, bound)
                nn.init.kaiming_uniform_(self.rel_w2[idx, rel], a=math.sqrt(5))
                fan_in = self.rel_w2.shape[-1]
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.rel_b2[idx, rel], -bound, bound)
            nn.init.kaiming_uniform_(self.judge_w[idx], a=math.sqrt(5))
            bound = 1 / math.sqrt(self.judge_w.shape[-1])
            nn.init.uniform_(self.judge_b[idx], -bound, bound)
            nn.init.kaiming_uniform_(self.alpha_w1[idx], a=math.sqrt(5))
            bound = 1 / math.sqrt(self.alpha_w1.shape[-1])
            nn.init.uniform_(self.alpha_b1[idx], -bound, bound)
            nn.init.kaiming_uniform_(self.llm_w1[idx], a=math.sqrt(5))
            nn.init.uniform_(self.llm_b1[idx], -bound, bound)

    @staticmethod
    def _layer_norm(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-5) * weight + bias

    def forward(
        self,
        base_z: Tensor,
        base_logit: Tensor,
        relation_features: Tensor,
        judge_features: Tensor,
        judge_mask: Tensor,
    ) -> dict[str, Tensor]:
        n = base_z.shape[0]
        b = self.num_models
        rel = relation_features.view(n, self.num_relations, self.rel_stat_dim)

        h1 = torch.einsum("nrd,brhd->bnrh", rel, self.rel_w1) + self.rel_b1[:, None, :, :]
        h1 = F.relu(h1)
        h2 = torch.einsum("bnri,broi->bnro", h1, self.rel_w2) + self.rel_b2[:, None, :, :]
        h2 = F.relu(h2)
        h2 = self._layer_norm(h2, self.rel_ln_w[:, None, :, :], self.rel_ln_b[:, None, :, :])
        if self.training and self.dropout > 0:
            h2 = F.dropout(h2, p=self.dropout, training=True)

        head_stack = (h2 * self.rel_head_w[:, None, :, :]).sum(dim=-1) + self.rel_head_b[:, None, :]
        h_flat = h2.reshape(b, n, self.num_relations * self.rel_hidden_dim)
        z_expand = base_z.detach().unsqueeze(0).expand(b, -1, -1)
        gate_input = torch.cat([z_expand, h_flat], dim=-1)
        gate_logits = torch.einsum("bni,bri->bnr", gate_input, self.gate_w) + self.gate_b[:, None, :]
        gate = F.softmax(gate_logits / self.tau_gate, dim=-1)

        u = (gate * head_stack).sum(dim=-1)
        delta_rel = self.delta_rel_max * torch.tanh(u)
        rel_only_logit = base_logit.detach().view(1, n) + delta_rel

        fused_h = (gate.unsqueeze(-1) * h2).sum(dim=2)

        j = torch.einsum("nj,bhj->bnh", judge_features, self.judge_w) + self.judge_b[:, None, :]
        j = F.relu(j)
        j = self._layer_norm(j, self.judge_ln_w[:, None, :], self.judge_ln_b[:, None, :])
        if self.training and self.dropout > 0:
            j = F.dropout(j, p=self.dropout, training=True)
        cond = torch.cat([j, fused_h, gate], dim=-1)

        a1 = F.relu(torch.einsum("bni,bhi->bnh", cond, self.alpha_w1) + self.alpha_b1[:, None, :])
        alpha_raw = torch.einsum("bnh,boh->bno", a1, self.alpha_w2).squeeze(-1) + self.alpha_b2[:, 0:1]
        alpha = self.alpha_max_vec[:, None] * torch.sigmoid(alpha_raw) * judge_mask.view(1, n).to(cond.dtype)

        l1 = F.relu(torch.einsum("bni,bhi->bnh", cond, self.llm_w1) + self.llm_b1[:, None, :])
        delta_raw = torch.einsum("bnh,boh->bno", l1, self.llm_w2).squeeze(-1) + self.llm_b2[:, 0:1]
        delta_llm = self.delta_llm_max * torch.tanh(delta_raw)

        final_logit = rel_only_logit + alpha * delta_llm
        return {
            "final_logit": final_logit,
            "relation_gate": gate,
            "delta_rel": delta_rel,
            "delta_llm": delta_llm,
            "alpha_llm": alpha,
            "relation_strength": head_stack.abs(),
        }

    def single_state_dict(self, idx: int, relation_names: list[str]) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for rel_idx, name in enumerate(relation_names):
            prefix = f"relation_experts.{name}"
            state[f"{prefix}.0.weight"] = self.rel_w1[idx, rel_idx].detach().cpu().clone()
            state[f"{prefix}.0.bias"] = self.rel_b1[idx, rel_idx].detach().cpu().clone()
            state[f"{prefix}.2.weight"] = self.rel_w2[idx, rel_idx].detach().cpu().clone()
            state[f"{prefix}.2.bias"] = self.rel_b2[idx, rel_idx].detach().cpu().clone()
            state[f"{prefix}.4.weight"] = self.rel_ln_w[idx, rel_idx].detach().cpu().clone()
            state[f"{prefix}.4.bias"] = self.rel_ln_b[idx, rel_idx].detach().cpu().clone()
            state[f"relation_heads.{name}.weight"] = self.rel_head_w[idx, rel_idx].detach().cpu().view(1, -1).clone()
            state[f"relation_heads.{name}.bias"] = self.rel_head_b[idx, rel_idx].detach().cpu().view(1).clone()
        state["gate_logit_head.weight"] = self.gate_w[idx].detach().cpu().clone()
        state["gate_logit_head.bias"] = self.gate_b[idx].detach().cpu().clone()
        state["judge_encoder.0.weight"] = self.judge_w[idx].detach().cpu().clone()
        state["judge_encoder.0.bias"] = self.judge_b[idx].detach().cpu().clone()
        state["judge_encoder.2.weight"] = self.judge_ln_w[idx].detach().cpu().clone()
        state["judge_encoder.2.bias"] = self.judge_ln_b[idx].detach().cpu().clone()
        state["head_alpha.0.weight"] = self.alpha_w1[idx].detach().cpu().clone()
        state["head_alpha.0.bias"] = self.alpha_b1[idx].detach().cpu().clone()
        state["head_alpha.2.weight"] = self.alpha_w2[idx].detach().cpu().clone()
        state["head_alpha.2.bias"] = self.alpha_b2[idx].detach().cpu().clone()
        state["head_llm.0.weight"] = self.llm_w1[idx].detach().cpu().clone()
        state["head_llm.0.bias"] = self.llm_b1[idx].detach().cpu().clone()
        state["head_llm.2.weight"] = self.llm_w2[idx].detach().cpu().clone()
        state["head_llm.2.bias"] = self.llm_b2[idx].detach().cpu().clone()
        return state

    @torch.no_grad()
    def load_single_state_dict(self, idx: int, relation_names: list[str], state: dict[str, Tensor]) -> None:
        for rel_idx, name in enumerate(relation_names):
            prefix = f"relation_experts.{name}"
            self.rel_w1[idx, rel_idx].copy_(state[f"{prefix}.0.weight"].to(self.rel_w1.device))
            self.rel_b1[idx, rel_idx].copy_(state[f"{prefix}.0.bias"].to(self.rel_b1.device))
            self.rel_w2[idx, rel_idx].copy_(state[f"{prefix}.2.weight"].to(self.rel_w2.device))
            self.rel_b2[idx, rel_idx].copy_(state[f"{prefix}.2.bias"].to(self.rel_b2.device))
            self.rel_ln_w[idx, rel_idx].copy_(state[f"{prefix}.4.weight"].to(self.rel_ln_w.device))
            self.rel_ln_b[idx, rel_idx].copy_(state[f"{prefix}.4.bias"].to(self.rel_ln_b.device))
            self.rel_head_w[idx, rel_idx].copy_(state[f"relation_heads.{name}.weight"].view(-1).to(self.rel_head_w.device))
            self.rel_head_b[idx, rel_idx].copy_(state[f"relation_heads.{name}.bias"].view(()).to(self.rel_head_b.device))
        self.gate_w[idx].copy_(state["gate_logit_head.weight"].to(self.gate_w.device))
        self.gate_b[idx].copy_(state["gate_logit_head.bias"].to(self.gate_b.device))
        self.judge_w[idx].copy_(state["judge_encoder.0.weight"].to(self.judge_w.device))
        self.judge_b[idx].copy_(state["judge_encoder.0.bias"].to(self.judge_b.device))
        self.judge_ln_w[idx].copy_(state["judge_encoder.2.weight"].to(self.judge_ln_w.device))
        self.judge_ln_b[idx].copy_(state["judge_encoder.2.bias"].to(self.judge_ln_b.device))
        self.alpha_w1[idx].copy_(state["head_alpha.0.weight"].to(self.alpha_w1.device))
        self.alpha_b1[idx].copy_(state["head_alpha.0.bias"].to(self.alpha_b1.device))
        self.alpha_w2[idx].copy_(state["head_alpha.2.weight"].to(self.alpha_w2.device))
        self.alpha_b2[idx].copy_(state["head_alpha.2.bias"].to(self.alpha_b2.device))
        self.llm_w1[idx].copy_(state["head_llm.0.weight"].to(self.llm_w1.device))
        self.llm_b1[idx].copy_(state["head_llm.0.bias"].to(self.llm_b1.device))
        self.llm_w2[idx].copy_(state["head_llm.2.weight"].to(self.llm_w2.device))
        self.llm_b2[idx].copy_(state["head_llm.2.bias"].to(self.llm_b2.device))


def bce_per_model(logits: Tensor, labels: Tensor, pos_weight: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels.view(1, -1).expand_as(logits),
        pos_weight=pos_weight,
        reduction="none",
    )
    return loss.mean(dim=1)


def phase2_loss_vec(
    outputs: dict[str, Tensor],
    y: Tensor,
    pos_weight: Tensor,
    judge_align: dict[str, Tensor],
    lambda_trust: Tensor,
    lambda_sparse: Tensor,
    lambda_align: Tensor,
    eta_llm: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    logits = outputs["final_logit"]
    gate = outputs["relation_gate"]
    delta_rel = outputs["delta_rel"]
    delta_llm = outputs["delta_llm"]
    alpha = outputs["alpha_llm"]
    strength = outputs["relation_strength"]
    eps = 1e-8

    l_cls = bce_per_model(logits, y.float(), pos_weight)
    l_trust = (delta_rel.pow(2) + eta_llm * alpha * delta_llm.pow(2)).mean(dim=1)
    entropy = -(gate * torch.log(gate + eps)).sum(dim=-1)
    sorted_s, _ = torch.sort(strength, dim=-1, descending=True)
    rho = sorted_s[:, :, 0] - sorted_s[:, :, 1]
    rho_norm = rho / (rho.max(dim=1, keepdim=True).values.detach() + eps)
    l_sparse = (rho_norm * entropy).mean(dim=1)

    q = judge_align["q_target"].to(logits.device, dtype=logits.dtype)
    w = judge_align["w_weight"].to(logits.device, dtype=logits.dtype)
    has = judge_align["has_target_mask"].to(logits.device).bool()
    if bool(has.any()):
        g = gate[:, has, :]
        qh = q[has].unsqueeze(0)
        wh = w[has].unsqueeze(0)
        kl = (qh * (torch.log(qh + eps) - torch.log(g + eps))).sum(dim=-1)
        l_align = (wh * kl).mean(dim=1)
    else:
        l_align = torch.zeros_like(l_cls)

    total = l_cls + lambda_trust * l_trust + lambda_sparse * l_sparse + lambda_align * l_align
    return total, {
        "l_cls": l_cls,
        "l_trust": l_trust,
        "l_sparse": l_sparse,
        "l_align": l_align,
        "mean_abs_delta_rel": delta_rel.abs().mean(dim=1),
        "mean_alpha_llm": alpha.mean(dim=1),
        "mean_gate_entropy": entropy.mean(dim=1),
    }


@torch.no_grad()
def gpu_rank_metrics(logits: Tensor, y: Tensor) -> dict[str, Tensor]:
    """Exact AUROC/AP plus threshold-0.5 metrics for B x N logits."""
    scores = torch.sigmoid(logits)
    labels = y.float().view(1, -1).expand_as(scores)
    order = torch.argsort(scores, dim=1, descending=True)
    y_sorted = torch.gather(labels, 1, order)
    pos = labels.sum(dim=1).clamp_min(1.0)
    neg = (labels.shape[1] - labels.sum(dim=1)).clamp_min(1.0)

    tp = torch.cumsum(y_sorted, dim=1)
    fp = torch.cumsum(1.0 - y_sorted, dim=1)
    tpr = torch.cat([torch.zeros_like(tp[:, :1]), tp / pos[:, None]], dim=1)
    fpr = torch.cat([torch.zeros_like(fp[:, :1]), fp / neg[:, None]], dim=1)
    auroc = torch.trapz(tpr, fpr, dim=1)

    precision_at = tp / torch.arange(1, labels.shape[1] + 1, device=labels.device).view(1, -1)
    auprc = (precision_at * y_sorted).sum(dim=1) / pos

    pred = (scores >= 0.5).float()
    tp0 = ((pred == 1) & (labels == 1)).sum(dim=1).float()
    fp0 = ((pred == 1) & (labels == 0)).sum(dim=1).float()
    fn0 = ((pred == 0) & (labels == 1)).sum(dim=1).float()
    tn0 = ((pred == 0) & (labels == 0)).sum(dim=1).float()
    f1_pos = 2 * tp0 / (2 * tp0 + fp0 + fn0).clamp_min(1.0)
    f1_neg = 2 * tn0 / (2 * tn0 + fp0 + fn0).clamp_min(1.0)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    g_means = torch.sqrt((tp0 / (tp0 + fn0).clamp_min(1.0)) * (tn0 / (tn0 + fp0).clamp_min(1.0)))
    return {"roc_auc": auroc, "auprc": auprc, "macro_f1": macro_f1, "g_means": g_means}


@torch.no_grad()
def best_threshold_macro_f1(logits: Tensor, y: Tensor) -> Tensor:
    scores = torch.sigmoid(logits)
    labels = y.float().view(1, 1, -1)
    thresholds = torch.linspace(0.0, 1.0, 101, device=logits.device).view(1, -1, 1)
    pred = (scores.unsqueeze(1) >= thresholds).float()
    tp = ((pred == 1) & (labels == 1)).sum(dim=2).float()
    fp = ((pred == 1) & (labels == 0)).sum(dim=2).float()
    fn = ((pred == 0) & (labels == 1)).sum(dim=2).float()
    tn = ((pred == 0) & (labels == 0)).sum(dim=2).float()
    f1_pos = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
    f1_neg = 2 * tn / (2 * tn + fp + fn).clamp_min(1.0)
    idx = (0.5 * (f1_pos + f1_neg)).argmax(dim=1)
    return torch.linspace(0.0, 1.0, 101, device=logits.device)[idx]


@torch.no_grad()
def threshold_metrics(logits: Tensor, y: Tensor, thresholds: Tensor) -> dict[str, Tensor]:
    rank = gpu_rank_metrics(logits, y)
    scores = torch.sigmoid(logits)
    labels = y.float().view(1, -1).expand_as(scores)
    pred = (scores >= thresholds.view(-1, 1)).float()
    tp = ((pred == 1) & (labels == 1)).sum(dim=1).float()
    fp = ((pred == 1) & (labels == 0)).sum(dim=1).float()
    fn = ((pred == 0) & (labels == 1)).sum(dim=1).float()
    tn = ((pred == 0) & (labels == 0)).sum(dim=1).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
    f1_neg = 2 * tn / (2 * tn + fp + fn).clamp_min(1.0)
    rank["f1"] = f1
    rank["macro_f1"] = 0.5 * (f1 + f1_neg)
    rank["precision"] = tp / (tp + fp).clamp_min(1.0)
    rank["recall"] = tp / (tp + fn).clamp_min(1.0)
    rank["g_means"] = torch.sqrt((tp / (tp + fn).clamp_min(1.0)) * (tn / (tn + fp).clamp_min(1.0)))
    rank["threshold_used"] = thresholds
    rank["positive_prediction_rate"] = pred.mean(dim=1)
    return rank


def tensor_to_float_dict(metrics: dict[str, Tensor], idx: int) -> dict[str, float]:
    return {k: float(v[idx].detach().cpu().item()) for k, v in metrics.items()}


def needed_specs(seed: int) -> list[dict]:
    root = Path("artifacts/results/yelpchi/bwgnn")
    specs: list[dict] = []
    for spec in RUN_SPECS:
        if any(not (root / rn / f"seed_{seed}" / "stage3_metrics.json").exists() for rn in spec["run_names"]):
            specs.append(spec)
    return specs


def write_epoch_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path / "phase2_train_log.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        fields = list(rows[0])
        with (path / "phase2_train_log.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def train_seed(seed: int, args: argparse.Namespace) -> None:
    specs = needed_specs(seed)
    if not specs:
        print(f"[packed] seed={seed}: all requested artifacts already exist")
        return

    set_seed(seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    p2_cfg = config["phase2_reasoner"].copy()
    dataset_name = "yelpchi"
    model_name = "bwgnn"
    device = torch.device(args.device)

    data = load_fraud_dataset(
        dataset_name,
        path=config["dataset"].get("path"),
        seed=seed,
        split_mode=config["dataset"].get("split_mode", "supervised"),
        train_ratio=config["dataset"].get("train_ratio", 0.4),
        val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        stratified=config["dataset"].get("stratified", True),
    )
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    num_nodes = data.x.shape[0]
    base_logits, base_z = load_frozen_base(config, dataset_name, model_name, seed, data, device)
    rel_features, rel_meta = load_relation_features_for_phase2(dataset_name, model_name, seed, num_nodes)
    judge_features, judge_mask, accepted_records, _audit = load_judge_data(dataset_name, model_name, seed, num_nodes)
    rel_features = rel_features.to(device)
    judge_features = judge_features.to(device)
    judge_mask = judge_mask.to(device)
    judge_dir = Path("artifacts") / "judge_packets" / dataset_name / model_name / "cover_rel_judge" / f"seed_{seed}"
    judge_align = build_judge_relation_targets(
        judge_dir / "accepted_judge.jsonl",
        num_nodes=num_nodes,
        relation_names=p2_cfg["relation_names"],
    )
    judge_align_train = {
        "q_target": judge_align["q_target"][train_mask.cpu()],
        "w_weight": judge_align["w_weight"][train_mask.cpu()],
        "has_target_mask": judge_align["has_target_mask"][train_mask.cpu()],
    }

    alpha_max = [float(s["alpha_max"]) for s in specs]
    lambda_align = torch.tensor([float(s["lambda_align"]) for s in specs], device=device)
    lambda_trust = torch.full((len(specs),), 3.0e-3, device=device)
    lambda_sparse = torch.full((len(specs),), 1.0e-3, device=device)

    model = PackedYelpJudgeReasoner(
        num_models=len(specs),
        base_z_dim=base_z.shape[1],
        judge_feature_dim=judge_features.shape[1],
        alpha_max=alpha_max,
        tau_gate=p2_cfg.get("tau_gate", 0.7),
        delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
        delta_llm_max=p2_cfg.get("delta_llm_max", 0.75),
        dropout=p2_cfg.get("rel_dropout", 0.3),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p2_cfg.get("lr", 1e-3), weight_decay=p2_cfg.get("weight_decay", 1e-4))

    y_train = y[train_mask]
    n_pos = y_train.sum()
    n_neg = y_train.numel() - n_pos
    pos_weight = torch.tensor(float(max((n_neg / n_pos.clamp_min(1)).item(), 1.0)), device=device)

    best_scores = torch.full((len(specs),), float("-inf"), device=device)
    patience = torch.zeros(len(specs), dtype=torch.int64, device=device)
    active = torch.ones(len(specs), dtype=torch.bool, device=device)
    best_state: list[dict[str, Tensor] | None] = [None] * len(specs)
    best_epoch = [0] * len(specs)
    logs: list[list[dict]] = [[] for _ in specs]
    started = time.time()

    train_idx = train_mask
    val_idx = val_mask
    for epoch in range(1, args.epochs + 1):
        model.train()
        out = model(
            base_z[train_idx],
            base_logits[train_idx],
            rel_features[train_idx],
            judge_features[train_idx],
            judge_mask[train_idx],
        )
        loss_vec, loss_stats = phase2_loss_vec(
            out,
            y[train_idx],
            pos_weight,
            judge_align_train,
            lambda_trust=lambda_trust,
            lambda_sparse=lambda_sparse,
            lambda_align=lambda_align,
        )
        if bool(active.any()):
            loss = (loss_vec * active.float()).sum() / active.float().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        should_eval = epoch == 1 or epoch == args.epochs or epoch % args.eval_interval == 0
        if should_eval:
            model.eval()
            val_out = model(
                base_z[val_idx],
                base_logits[val_idx],
                rel_features[val_idx],
                judge_features[val_idx],
                judge_mask[val_idx],
            )
            val_metrics = gpu_rank_metrics(val_out["final_logit"], y[val_idx])
            scores = val_metrics["auprc"]
            improved = scores > best_scores
            for i in range(len(specs)):
                if bool(improved[i]):
                    best_scores[i] = scores[i]
                    best_state[i] = model.single_state_dict(i, p2_cfg["relation_names"])
                    best_epoch[i] = epoch
                    patience[i] = 0
                else:
                    patience[i] += args.eval_interval
                if patience[i] >= args.patience:
                    active[i] = False

                row = {
                    "epoch": float(epoch),
                    "loss": float(loss_vec[i].detach().cpu().item()),
                    "val/auprc": float(val_metrics["auprc"][i].detach().cpu().item()),
                    "val/roc_auc": float(val_metrics["roc_auc"][i].detach().cpu().item()),
                    "val/macro_f1": float(val_metrics["macro_f1"][i].detach().cpu().item()),
                    "loss/l_cls": float(loss_stats["l_cls"][i].detach().cpu().item()),
                    "loss/l_trust": float(loss_stats["l_trust"][i].detach().cpu().item()),
                    "loss/l_sparse": float(loss_stats["l_sparse"][i].detach().cpu().item()),
                    "loss/l_align": float(loss_stats["l_align"][i].detach().cpu().item()),
                }
                logs[i].append(row)
            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"[packed] seed={seed} epoch={epoch} active={int(active.sum().item())} "
                    f"best={best_scores.detach().cpu().numpy().round(4).tolist()}"
                )
            if not bool(active.any()):
                print(f"[packed] seed={seed} early stop at epoch={epoch}")
                break

    elapsed = time.time() - started
    if any(s is None for s in best_state):
        for i, state in enumerate(best_state):
            if state is None:
                best_state[i] = model.single_state_dict(i, p2_cfg["relation_names"])
                best_epoch[i] = epoch

    for i, spec in enumerate(specs):
        exported = best_state[i]
        assert exported is not None
        model.load_single_state_dict(i, p2_cfg["relation_names"], exported)

    model.eval()
    full_out = model(base_z, base_logits, rel_features, judge_features, judge_mask)
    val_out = model(base_z[val_mask], base_logits[val_mask], rel_features[val_mask], judge_features[val_mask], judge_mask[val_mask])
    thresholds = best_threshold_macro_f1(val_out["final_logit"], y[val_mask])
    test_out = model(base_z[test_mask], base_logits[test_mask], rel_features[test_mask], judge_features[test_mask], judge_mask[test_mask])
    test_metrics_all = threshold_metrics(test_out["final_logit"], y[test_mask], thresholds)
    final_metrics_all = threshold_metrics(full_out["final_logit"], y, torch.full_like(thresholds, 0.5))

    for i, spec in enumerate(specs):
        metrics = tensor_to_float_dict(test_metrics_all, i)
        diag = {
            "mean_abs_delta_rel": float(full_out["delta_rel"][i].abs().mean().detach().cpu().item()),
            "mean_alpha_llm": float(full_out["alpha_llm"][i].mean().detach().cpu().item()),
            "mean_alpha_llm_accepted": float(full_out["alpha_llm"][i][judge_mask].mean().detach().cpu().item()) if bool(judge_mask.any()) else 0.0,
            "mean_alpha_llm_rejected": float(full_out["alpha_llm"][i][~judge_mask].mean().detach().cpu().item()) if bool((~judge_mask).any()) else 0.0,
            "max_abs_alpha_llm_rejected": float(full_out["alpha_llm"][i][~judge_mask].abs().max().detach().cpu().item()) if bool((~judge_mask).any()) else 0.0,
            "mean_gate_entropy": float((-(full_out["relation_gate"][i] * torch.log(full_out["relation_gate"][i] + 1e-8)).sum(dim=-1)).mean().detach().cpu().item()),
            "gate_weight_rel_0": float(full_out["relation_gate"][i, :, 0].mean().detach().cpu().item()),
            "gate_weight_rel_1": float(full_out["relation_gate"][i, :, 1].mean().detach().cpu().item()),
            "gate_weight_rel_2": float(full_out["relation_gate"][i, :, 2].mean().detach().cpu().item()),
        }
        for run_name in spec["run_names"]:
            if (Path("artifacts/results/yelpchi/bwgnn") / run_name / f"seed_{seed}" / "stage3_metrics.json").exists():
                continue
            cfg = p2_cfg.copy()
            cfg.update({
                "run_name": run_name,
                "use_judge": True,
                "lambda_align": float(spec["lambda_align"]),
                "alpha_max": float(spec["alpha_max"]),
                "lambda_trust": 3.0e-3,
                "lambda_sparse": 1.0e-3,
                "packed_gpu_sweep": True,
            })
            ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
            log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
            results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
            torch.save(
                {"model_state_dict": best_state[i], "config": cfg, "seed": seed, "epoch": best_epoch[i]},
                ckpt_dir / "reasoner.pt",
            )
            write_epoch_log(log_dir, logs[i])
            with (results_dir / "stage3_metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)
            diagnostics = {
                "config": config,
                "phase2_reasoner": cfg,
                "seed": seed,
                "run_name": run_name,
                "dataset": dataset_name,
                "model": model_name,
                "device": runtime_device_info(device),
                "git_hash": get_git_hash(),
                "best_epoch": best_epoch[i],
                "best_val_score": float(best_scores[i].detach().cpu().item()),
                "early_stop_metric": "val_auprc",
                "epochs_trained": epoch,
                "elapsed_seconds": elapsed,
                "threshold": metrics.get("threshold_used", 0.5),
                "test_metrics": metrics,
                "final_diagnostics": diag,
                "relation_feature_meta": rel_meta,
                "num_accepted_judge": int(judge_mask.sum().detach().cpu().item()),
                "accepted_records": len(accepted_records),
                "packed_note": "Metrics were produced by the GPU-packed judge sweep trainer.",
            }
            with (log_dir / "phase2_diagnostics.json").open("w") as f:
                json.dump(diagnostics, f, indent=2, default=str)
            print(f"[packed] saved {run_name}/seed_{seed}: AUPRC={metrics['auprc']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E2_judge_residual.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=5)
    args = parser.parse_args()

    for seed in args.seeds:
        train_seed(seed, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
