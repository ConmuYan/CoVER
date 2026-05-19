"""G-OPD-Flash trainer — student-policy node-state distillation (Idea 2C → C3).

Implements ``docs/OPD_FLASH_DESIGN_v3.md`` (v3.1) Phase A–F training loop
with five ablation modes:

* ``off_policy``        — v1 vanilla full-graph KL distill (Idea 2C baseline).
* ``all_node_mh``       — multi-head distill, but on the full graph (no
                          student-policy sampling).  Tests "is OPD even needed?"
* ``det_mask``          — v1 deterministic entropy mask + reverse-KL final-only.
                          Tests "stochastic > deterministic mask?"
* ``g_opd_flash``       — full G-OPD-Flash (main method).
* ``opd_action_strict`` — Bernoulli action + REINFORCE single-step policy
                          gradient.  Tests "why not classic OPD-RL?"  (T6)

Per-epoch training loop (see ``docs/OPD_FLASH_DESIGN_v3.md`` §3.7)::

    # Phase A — student forward (no grad); build q_φ
    with torch.no_grad():
        s_S, δ_S, _ = student(z, feats)
        p_S = σ(s_S);  H_S = bern_entropy(p_S)
        q   = ε + H_S + α·p_S + β·|δ_S|
        q   = q / q.sum()                          # detached (GKD §3)

    # Phase B — sample 𝓑 ⊆ train; teacher sparse forward
    idx = multinomial(q, K, replacement=False)
    with torch.no_grad():
        t = teacher(z[idx], feats[idx], return_heads=True)
        p_T, c_T_r, g_T = t['p'], t['c_per_r'], t['gate']
        η = bern_entropy(p_T) / log 2

    # Phase C — student forward w/ grad on 𝓑, 3 heads
    s = student(z[idx], feats[idx], return_heads=True)

    # Phase D — entropy-aware mixed Bernoulli KL
    L_logit = (1-η)·KL_rev(p_S, p_T) + η·KL_fwd(p_S, p_T)
    L_rel   = MSE(c_S_r, c_T_r).mean(-1)
    L_gate  = KL(g_S || g_T)

    # Phase E — node-level reliability + adaptive BCE anchor (v3.2 MJ-4)
    r_node    = r_c · cal_bin(p_T)             # NO conf factor (v3.2)
    λ_bce_i   = λ_min + (1 - r_node)·λ_extra
    mask_lab  = is_labelled_train[idx]

    # Phase F — two-term independent-normaliser loss (v3.1 MF4)
    L_distill = Σ r_node · (α_f·L_logit + α_r·L_rel + α_g·L_gate) / Σ r_node
    L_bce     = Σ λ_bce · BCE(s_S, y) · mask_lab / Σ (λ_bce · mask_lab)
    L_total   = L_distill + L_bce

Outputs::

    artifacts/checkpoints/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/student.pt
    artifacts/results/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/stage3_metrics.json
    artifacts/results/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/phase2_summary.json
    artifacts/logs/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/{flash,diagnostics}.json
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from evidence.learned_extractor import build_learned_extractor
from evidence.relation_features import RELATION_SCHEMAS, load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.flash_adapter import FlashAdapter
from models.gnn import build_detector
from training.metrics import compute_metrics, g_means, precision_recall_at_k
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir
from utils.threshold import evaluate_with_threshold, find_best_threshold


SUPPORTED_MODES = (
    "off_policy",
    "all_node_mh",
    "det_mask",
    "g_opd_flash",
    "opd_action_strict",
    "opd_action_strict_mh",
)


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility (matches train_distill_adapter.py)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (delegated to train_distill_adapter helpers — shared infra)
# ═══════════════════════════════════════════════════════════════════════════

# Re-export the heavy data plumbing from the v1 trainer so the two scripts
# share a single source of truth for base/teacher feature construction.
from scripts.train_distill_adapter import (
    build_learned_teacher_features,
    evaluate_adapter,
    evaluate_base_only,
    evaluate_teacher,
    get_base_output_cache_path,
    load_frozen_base,
    load_relation_adjs_for_extractor,
    load_relation_features,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase A — student-policy sampling distribution q_φ
# ═══════════════════════════════════════════════════════════════════════════

def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scalar Bernoulli entropy H(p) = -p·log p - (1-p)·log(1-p)."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def build_student_policy(
    *,
    student: FlashAdapter,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    train_mask: torch.Tensor,
    alpha_q: float,
    beta_q: float,
    eps_q: float,
    tau: float,
) -> torch.Tensor:
    """Phase A: q_φ(i) ∝ ε + H(p^S_i) + α·p^S_i + β·|δ^S_i| on train nodes.

    Gradient is stopped — q_φ is a *detached* sampling distribution (GKD §3).
    Returns: q (N,) — zero outside ``train_mask``, normalised to sum to 1
    over ``train_mask``.
    """
    with torch.no_grad():
        out = student(base_z, base_logits, rel_features, return_heads=False)
        p_S = torch.sigmoid(out["final_logit"])
        delta_S = out["delta_phi"]
        H_S = bernoulli_entropy(p_S)
        q_raw = eps_q + H_S + alpha_q * p_S + beta_q * delta_S.abs()
        if tau != 1.0:
            q_raw = q_raw ** (1.0 / max(float(tau), 1e-6))
        # Restrict to train nodes (student-policy is over training distribution).
        q = torch.zeros_like(q_raw)
        q[train_mask] = q_raw[train_mask]
        q = q / q.sum().clamp_min(1e-12)
    return q.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Phase D — scalar Bernoulli mixed KL (entropy-aware OPD)
# ═══════════════════════════════════════════════════════════════════════════

def bern_kl_rev(p_S: torch.Tensor, p_T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Reverse KL: KL(p_S || p_T) — mode-seeking, what student concentrates on.

        KL_rev = p_S · log(p_S / p_T) + (1 - p_S) · log((1 - p_S) / (1 - p_T))
    """
    p_S = p_S.clamp(eps, 1.0 - eps)
    p_T = p_T.clamp(eps, 1.0 - eps)
    return p_S * (p_S.log() - p_T.log()) + (1.0 - p_S) * ((1.0 - p_S).log() - (1.0 - p_T).log())


def bern_kl_fwd(p_S: torch.Tensor, p_T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Forward KL: KL(p_T || p_S) — mass-covering, preserves recall on rare classes."""
    return bern_kl_rev(p_T, p_S, eps=eps)


def categorical_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) over the last dim (assumes both are valid distributions)."""
    p = p.clamp(eps, 1.0)
    q = q.clamp(eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Phase E — curriculum prior + ECE-style node reliability
# ═══════════════════════════════════════════════════════════════════════════

class NodeReliabilityHelper:
    """Computes r^node = r^c · cal_bin(p_T) for design v3.2 §3.5.

    **v3.2 fix (Critic round-4 MJ-4)**: removed the multiplicative
    ``conf(p_T) = |2p−1|`` factor.  The original v3.1 formula
    ``r^node = r^c · conf(p_T) · cal_bin(p_T)`` cancelled the
    entropy-aware mixed-KL contribution exactly where it was supposed
    to activate — when teacher is uncertain (η = H(p_T)/log 2 → 1),
    ``conf(p_T) = |2p−1| → 0`` zeroed the forward-KL term that
    "preserves recall on rare fraud calls".  By removing ``conf``,
    forward-KL on uncertain-teacher nodes is now actually applied
    (calibration still discounts miscalibrated bins).

    Inputs (loaded from T4 caches):
      * ``r_c``       cell-level curriculum prior (scalar)
      * ``cal_bins``  list of K (default 10) empirical positive rates,
                      one per equal-width confidence bin.  Optional —
                      falls back to uniform reliability 1.0 if unavailable.
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
                raise ValueError(f"cal_bins must have {self.n_bins} entries, got {self.cal_bins.numel()}")

    @staticmethod
    def confidence(p: torch.Tensor) -> torch.Tensor:
        """conf(p) = |2p - 1| ∈ [0, 1] — kept for diagnostic use ONLY.

        **NOT used in r_node anymore** (v3.2 MJ-4 fix); see class docstring.
        """
        return (2.0 * p - 1.0).abs()

    def cal_bin_reliability(self, p: torch.Tensor) -> torch.Tensor:
        """Per-bin reliability based on Bernoulli information content
        of the bin: returns ``|2·empirical_positive_rate − 1|``.

        A bin where outcomes are deterministic (positive rate near 0 or 1)
        returns ~1.0; a bin where outcomes are 50/50 returns ~0.0.

        **This is NOT a standard calibration measure** — a well-calibrated
        bin where ``positive_rate ≈ bin_midpoint`` will return a value far
        from 1.0 in the mid range (e.g., midpoint=0.5 → rate=0.5 → returns
        0.0).  The metric is intentionally chosen to discount distillation
        in *information-poor* regions of the teacher's prediction space —
        not in *miscalibrated* regions per se.

        **Implication for §4.2 (Opus round-5 #3)**: under a uniformly
        well-calibrated teacher, this discounts the mid-range bins where η
        is high and forward-KL would otherwise activate.  So the "forward-KL
        on uncertain-teacher nodes fires" claim is *conditional* on the
        teacher being non-trivially miscalibrated in those bins (e.g.,
        outcomes in [0.4, 0.6] are actually 90% positive due to label noise
        or domain skew).  T5 reports r_node histograms per cell to validate.
        """
        bin_idx = (p * float(self.n_bins)).floor().long().clamp_(0, self.n_bins - 1)
        cal = self.cal_bins.to(device=p.device, dtype=p.dtype)
        # cal[bin] is the empirical positive rate in that bin; convert to
        # a Bernoulli information-content scalar in [0, 1].
        cal_at = cal[bin_idx]
        return (2.0 * cal_at - 1.0).abs()

    def r_node(self, p_T: torch.Tensor) -> torch.Tensor:
        """r^node_i = r^c · cal_bin(p_T_i)  (v3.2 — conf factor dropped, MJ-4).

        Range: [r_min · cal_min, 1.0].  When teacher is on a miscalibrated
        bin (calibration ~ 0), r_node → 0 ⇒ λ_bce(i) → λ_min + λ_extra
        and BCE anchor takes over for that node.  When teacher is on a
        well-calibrated bin, r_node ≈ r_c and the entropy-aware mixed KL
        carries full weight, including forward-KL on uncertain teacher
        nodes (the §4.2 mass-covering contribution now actually fires).
        """
        cal_rel = self.cal_bin_reliability(p_T)
        return torch.clamp(self.r_c * cal_rel, min=self.r_min, max=1.0)


def load_curriculum_prior(
    cell_key: str,
    prior_path: Path | None,
    default_r_c: float = 1.0,
) -> float:
    if prior_path is None or not Path(prior_path).exists():
        return default_r_c
    with open(prior_path) as f:
        priors = json.load(f)
    return float(priors.get(cell_key, default_r_c))


def load_calibration_bins(
    cell_key: str,
    bins_path: Path | None,
    n_bins: int = 10,
) -> list[float] | None:
    if bins_path is None or not Path(bins_path).exists():
        return None
    with open(bins_path) as f:
        bins_data = json.load(f)
    return bins_data.get(cell_key, None)


# ═══════════════════════════════════════════════════════════════════════════
# Per-mode forward + loss assembly
# ═══════════════════════════════════════════════════════════════════════════

def _compute_distill_terms_on_indices(
    *,
    student: FlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    idx: torch.Tensor,
    rel_helper: NodeReliabilityHelper,
    alpha_f: float,
    alpha_r: float,
    alpha_g: float,
    lambda_min: float,
    lambda_extra: float,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Run Phase C+D+E+F on a set of node indices ``idx`` (the sampled batch
    in g_opd_flash mode, or all train nodes in all_node_mh mode).

    Returns (L_total, loss_stats_dict, diagnostic_tensors_dict).
    """
    s_out = student(base_z[idx], base_logits[idx], rel_features[idx], return_heads=True)
    p_S = s_out["p"]
    s_S_logit = s_out["logit"]
    c_S_r = s_out["c_per_r"]
    g_S_r = s_out["gate"]

    # Teacher cache supplies p^T, c_T_r, g_T on full graph; subset to idx.
    p_T = teacher_cache["p"][idx].to(p_S.device).detach()
    c_T_r = teacher_cache["c_per_r"][idx].to(p_S.device).detach()
    g_T_r = teacher_cache["gate"][idx].to(p_S.device).detach()

    # Phase D: entropy-aware mixed Bernoulli KL
    eta = bernoulli_entropy(p_T) / math.log(2.0)
    L_logit_per = (1.0 - eta) * bern_kl_rev(p_S, p_T) + eta * bern_kl_fwd(p_S, p_T)
    L_rel_per = ((c_S_r - c_T_r) ** 2).mean(dim=-1)
    L_gate_per = categorical_kl(g_S_r, g_T_r)

    # Phase E: r_node + adaptive λ_bce
    r_node = rel_helper.r_node(p_T)                            # (K,)
    lam_bce_i = lambda_min + (1.0 - r_node) * lambda_extra     # (K,)
    mask_lab = train_mask[idx].to(dtype=p_S.dtype, device=p_S.device)
    y_idx = y[idx].to(dtype=p_S.dtype, device=p_S.device)

    # Phase F: two-term independent-normaliser loss (v3.1 MF4)
    distill_per = r_node * (alpha_f * L_logit_per + alpha_r * L_rel_per + alpha_g * L_gate_per)
    denom_d = r_node.sum().clamp_min(1.0)
    L_distill = distill_per.sum() / denom_d

    # BCE term: per-node, pos_weight applies multiplicatively to y=1 only.
    bce_per = F.binary_cross_entropy_with_logits(
        s_S_logit, y_idx, reduction="none",
        pos_weight=pos_weight if pos_weight is not None else None,
    )
    bce_weight = lam_bce_i * mask_lab
    denom_b = bce_weight.sum().clamp_min(1.0)
    L_bce = (bce_weight * bce_per).sum() / denom_b

    L_total = L_distill + L_bce

    stats = {
        "l_logit": float((L_logit_per.detach() * r_node.detach()).sum() / denom_d.detach()),
        "l_rel": float((L_rel_per.detach() * r_node.detach()).sum() / denom_d.detach()),
        "l_gate": float((L_gate_per.detach() * r_node.detach()).sum() / denom_d.detach()),
        "l_distill": float(L_distill.detach()),
        "l_bce": float(L_bce.detach()),
        "total": float(L_total.detach()),
        "lam_bce_mean": float(lam_bce_i.detach().mean()),
        "r_node_mean": float(r_node.detach().mean()),
        "batch_size": int(idx.numel()),
    }
    diag = {
        "r_node": r_node.detach(),
        "lam_bce": lam_bce_i.detach(),
    }
    return L_total, stats, diag


def _train_step_off_policy(
    *,
    student: FlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    lambda_distill: float,
    gamma_kl: float,
    pos_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Vanilla off-policy KL distill (v1 Idea 2C baseline).

    L = L_bce(y, σ(s_S)) + λ_distill · MSE(Δ_φ, sg(Δ_rel)) + γ_kl · KL(π_φ || sg(π_T))
    """
    s_out = student(base_z, base_logits, rel_features, return_heads=False)
    device = s_out["final_logit"].device

    n_train = int(train_mask.sum())
    if n_train == 0:
        z = s_out["final_logit"].sum() * 0.0
        return z, {"l_bce": 0.0, "l_distill": 0.0, "l_kl": 0.0, "total": 0.0}

    train_bool = train_mask.view(-1).to(torch.bool)
    y_f = y.to(device=device, dtype=torch.float32).view(-1)

    l_bce = F.binary_cross_entropy_with_logits(
        s_out["final_logit"][train_bool], y_f[train_bool], pos_weight=pos_weight,
    )

    teacher_delta = teacher_cache["delta_rel"].to(device=device, dtype=torch.float32).view(-1).detach()
    l_distill = F.mse_loss(s_out["delta_phi"][train_bool], teacher_delta[train_bool])

    teacher_gate = teacher_cache["gate"].to(device=device, dtype=torch.float32).detach()
    eps = 1e-8
    l_kl = F.kl_div(
        (teacher_gate[train_bool] + eps).log(),
        s_out["gate_probs"][train_bool],
        reduction="batchmean",
    )

    total = l_bce + lambda_distill * l_distill + gamma_kl * l_kl
    stats = {
        "l_bce": float(l_bce.detach()),
        "l_distill": float(l_distill.detach()),
        "l_kl": float(l_kl.detach()),
        "total": float(total.detach()),
        "batch_size": int(n_train),
    }
    return total, stats


def _train_step_opd_action_strict(
    *,
    student: FlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    idx: torch.Tensor,
    rel_helper: NodeReliabilityHelper,
    alpha_f: float,
    lambda_min: float,
    lambda_extra: float,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    add_multihead: bool = False,
    alpha_r: float = 0.3,
    alpha_g: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """T6 strict-OPD: y ~ Bern(p^S); REINFORCE w/ batch-mean baseline.

        L = -α_f · Σ_i (r_i - bar{r}) · log π_S(y_i | i) + λ_bce·BCE-anchor

    where r_i = log p^T(y_i | i) is the teacher log-likelihood reward.

    **v3.2 MJ-6 fix**: with ``add_multihead=True``, a multi-head matching
    loss (``α_r · MSE(c_S, c_T) + α_g · KL(g_S || g_T)``) is added.  This
    is the new ``opd_action_strict_mh`` mode and is designed to be a fair
    3-way comparison against ``g_opd_flash``: REINFORCE + multi-head
    matching + BCE anchor — only the *sampling-gradient* mechanism
    (REINFORCE vs detached GKD) differs from the main method.
    """
    s_out = student(base_z[idx], base_logits[idx], rel_features[idx], return_heads=True)
    p_S = s_out["p"]
    s_S_logit = s_out["logit"]
    p_T = teacher_cache["p"][idx].to(p_S.device).detach()

    # Sample Bernoulli actions y_S ~ p_S (stop-gradient through sample).
    with torch.no_grad():
        y_action = torch.bernoulli(p_S)

    # log π_S(y_S | i)  — uses BCE-with-logits via the negative formula.
    log_pi_S = -F.binary_cross_entropy_with_logits(s_S_logit, y_action, reduction="none")

    # Reward r_i = log p^T(y_S | i).
    with torch.no_grad():
        log_pT_1 = torch.log(p_T.clamp_min(1e-8))
        log_pT_0 = torch.log((1.0 - p_T).clamp_min(1e-8))
        reward = y_action * log_pT_1 + (1.0 - y_action) * log_pT_0
        baseline = reward.mean()
        advantage = reward - baseline

    pol_loss = -(advantage * log_pi_S).mean()  # placeholder; overwritten below if add_multihead

    # ── v3.2 MJ-6 fix #2: r_node-weighted REINFORCE for fair comparison ──
    # When add_multihead=True (opd_action_strict_mh), the primary policy
    # loss MUST be r_node-weighted — matching how g_opd_flash treats its
    # primary logit-KL term.  Otherwise the comparison conflates two
    # axes (sampling-mechanism AND reliability-weighting), making it
    # impossible to attribute differences to the sampling mechanism alone.
    # When add_multihead=False (strict OPD baseline), keep the original
    # unweighted form — that arm exists explicitly as a strawman to show
    # what un-aided REINFORCE looks like.
    r_node = rel_helper.r_node(p_T)
    lam_bce_i = lambda_min + (1.0 - r_node) * lambda_extra
    mask_lab = train_mask[idx].to(dtype=p_S.dtype, device=p_S.device)
    y_idx = y[idx].to(dtype=p_S.dtype, device=p_S.device)

    if add_multihead:
        pol_per = -(advantage * log_pi_S)
        denom_p = r_node.sum().clamp_min(1.0)
        pol_loss = (r_node * pol_per).sum() / denom_p
    # else: keep the unweighted pol_loss assigned above
    bce_per = F.binary_cross_entropy_with_logits(
        s_S_logit, y_idx, reduction="none", pos_weight=pos_weight,
    )
    bce_weight = lam_bce_i * mask_lab
    denom_b = bce_weight.sum().clamp_min(1.0)
    L_bce = (bce_weight * bce_per).sum() / denom_b

    total = alpha_f * pol_loss + L_bce
    stats = {
        "l_policy": float(pol_loss.detach()),
        "reward_mean": float(reward.detach().mean()),
        "advantage_std": float(advantage.detach().std()),
        "l_bce": float(L_bce.detach()),
        "batch_size": int(idx.numel()),
    }

    # ── v3.2 MJ-6: optional multi-head matching for strict_mh variant ────
    if add_multihead:
        c_S_r = s_out["c_per_r"]
        g_S_r = s_out["gate"]
        c_T_r = teacher_cache["c_per_r"][idx].to(p_S.device).detach()
        g_T_r = teacher_cache["gate"][idx].to(p_S.device).detach()
        L_rel = ((c_S_r - c_T_r) ** 2).mean(dim=-1)
        L_gate = categorical_kl(g_S_r, g_T_r)
        # Reliability-weighted, same normaliser convention as g_opd_flash.
        mh_per = r_node * (alpha_r * L_rel + alpha_g * L_gate)
        denom_d = r_node.sum().clamp_min(1.0)
        L_mh = mh_per.sum() / denom_d
        total = total + L_mh
        stats["l_rel"] = float(L_rel.detach().mean())
        stats["l_gate"] = float(L_gate.detach().mean())
        stats["l_mh"] = float(L_mh.detach())

    stats["total"] = float(total.detach())
    return total, stats


# ═══════════════════════════════════════════════════════════════════════════
# Teacher cache with 3 heads (for G-OPD-Flash modes)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_teacher_cache_with_heads(
    teacher: CoVERRelReasoner,
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One-time full-graph teacher forward; keeps cache on ``device`` so
    sparse ``idx`` lookups during Phase B avoid GPU→CPU round-trips."""
    teacher.eval()
    out = teacher(base_z, base_logits, rel_features.to(device), return_heads=True)
    return {
        "p": out["p"].detach().to(device),
        "logit": out["logit"].detach().to(device),
        "c_per_r": out["c_per_r"].detach().to(device),
        "gate": out["gate"].detach().to(device),
        "s_per_r": out["s_per_r"].detach().to(device),
        # Back-compat for off_policy mode (uses delta_rel + gate_probs scheme).
        "delta_rel": out["delta_rel"].detach().to(device),
        "gate_probs": out["relation_gate"].detach().to(device),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation (same shape as train_distill_adapter)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_student(
    student: FlashAdapter,
    base_z, base_logits, rel_features,
    mask, y, device,
    threshold=None, k_values=None,
):
    student.eval()
    out = student(base_z[mask], base_logits[mask], rel_features[mask].to(device))
    prob = torch.sigmoid(out["final_logit"]).cpu().numpy()
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_np = y[mask].cpu().numpy()
    if threshold is not None:
        m = evaluate_with_threshold(y_np, prob, threshold)
        pred = (prob >= threshold).astype(int)
        m["g_means"] = g_means(y_np, pred)
        for k in (k_values or [50, 100, 200]):
            pk, rk = precision_recall_at_k(y_np, prob, k)
            m[f"precision@{k}"] = pk
            m[f"recall@{k}"] = rk
        return m, prob, y_np
    return compute_metrics(y_np, prob, k_values=k_values or [50, 100, 200]), prob, y_np


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_g_opd_flash(
    *,
    mode: str,
    student: FlashAdapter,
    teacher_cache: dict[str, torch.Tensor],
    base_z: torch.Tensor,
    base_logits: torch.Tensor,
    rel_features: torch.Tensor,
    data,
    distill_cfg: dict,
    rel_helper: NodeReliabilityHelper,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> tuple[dict, list[dict]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode {mode!r} not in {SUPPORTED_MODES}")

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)
    n_train_nodes = int(train_mask.sum().item())

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(distill_cfg.get("lr", 1e-3)),
        weight_decay=float(distill_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(distill_cfg.get("epochs", 80))
    patience = int(distill_cfg.get("patience", 10))
    K = int(distill_cfg.get("K", min(2048, n_train_nodes)))
    alpha_q = float(distill_cfg.get("alpha_q", 0.5))
    beta_q = float(distill_cfg.get("beta_q", 0.3))
    eps_q = float(distill_cfg.get("eps_q", 1e-3))
    tau = float(distill_cfg.get("tau", 1.0))
    alpha_f = float(distill_cfg.get("alpha_f", 1.0))
    alpha_r = float(distill_cfg.get("alpha_r", 0.3))
    alpha_g = float(distill_cfg.get("alpha_g", 0.2))
    lambda_min = float(distill_cfg.get("lambda_min", 0.05))
    lambda_extra = float(distill_cfg.get("lambda_extra", 0.50))
    lambda_distill_v1 = float(distill_cfg.get("lambda_distill_v1", 1.0))
    gamma_kl_v1 = float(distill_cfg.get("gamma_kl_v1", 0.5))

    train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
    K_eff = min(K, train_idx.numel())

    best_val = -float("inf")
    best_state: dict | None = None
    best_epoch = -1
    no_improve = 0
    rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        student.train()
        optimizer.zero_grad()

        if mode == "off_policy":
            loss, stats = _train_step_off_policy(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                y=y, train_mask=train_mask,
                lambda_distill=lambda_distill_v1, gamma_kl=gamma_kl_v1,
                pos_weight=pos_weight,
            )
        elif mode == "all_node_mh":
            loss, stats, _ = _compute_distill_terms_on_indices(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                idx=train_idx, rel_helper=rel_helper,
                alpha_f=alpha_f, alpha_r=alpha_r, alpha_g=alpha_g,
                lambda_min=lambda_min, lambda_extra=lambda_extra,
                y=y, train_mask=train_mask, pos_weight=pos_weight,
            )
        elif mode == "det_mask":
            # v1 deterministic entropy mask: top-K train nodes by H(p^S).
            with torch.no_grad():
                s_out = student(base_z, base_logits, rel_features, return_heads=False)
                p_S = torch.sigmoid(s_out["final_logit"])
                H_S = bernoulli_entropy(p_S)
                H_train = H_S.clone()
                H_train[~train_mask] = -float("inf")
                topk = torch.topk(H_train, K_eff).indices
            loss, stats, _ = _compute_distill_terms_on_indices(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                idx=topk, rel_helper=rel_helper,
                alpha_f=alpha_f, alpha_r=0.0, alpha_g=0.0,  # final-only per design
                lambda_min=lambda_min, lambda_extra=lambda_extra,
                y=y, train_mask=train_mask, pos_weight=pos_weight,
            )
        elif mode == "g_opd_flash":
            # Phase A: build q_φ
            q = build_student_policy(
                student=student, base_z=base_z, base_logits=base_logits,
                rel_features=rel_features, train_mask=train_mask,
                alpha_q=alpha_q, beta_q=beta_q, eps_q=eps_q, tau=tau,
            )
            # Phase B: sample 𝓑 ⊆ train
            idx = torch.multinomial(q, num_samples=K_eff, replacement=False)
            # Phases C+D+E+F
            loss, stats, diag = _compute_distill_terms_on_indices(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                idx=idx, rel_helper=rel_helper,
                alpha_f=alpha_f, alpha_r=alpha_r, alpha_g=alpha_g,
                lambda_min=lambda_min, lambda_extra=lambda_extra,
                y=y, train_mask=train_mask, pos_weight=pos_weight,
            )
            stats["q_max"] = float(q.max())
            stats["q_eff_frac"] = float((q > 0).sum() / max(n_train_nodes, 1))
        elif mode == "opd_action_strict":
            q = build_student_policy(
                student=student, base_z=base_z, base_logits=base_logits,
                rel_features=rel_features, train_mask=train_mask,
                alpha_q=alpha_q, beta_q=beta_q, eps_q=eps_q, tau=tau,
            )
            idx = torch.multinomial(q, num_samples=K_eff, replacement=False)
            loss, stats = _train_step_opd_action_strict(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                idx=idx, rel_helper=rel_helper,
                alpha_f=alpha_f, lambda_min=lambda_min, lambda_extra=lambda_extra,
                y=y, train_mask=train_mask, pos_weight=pos_weight,
                add_multihead=False,  # strict REINFORCE-only
            )
        elif mode == "opd_action_strict_mh":
            # v3.2 MJ-6: fair-comparison strict-OPD with multi-head matching.
            q = build_student_policy(
                student=student, base_z=base_z, base_logits=base_logits,
                rel_features=rel_features, train_mask=train_mask,
                alpha_q=alpha_q, beta_q=beta_q, eps_q=eps_q, tau=tau,
            )
            idx = torch.multinomial(q, num_samples=K_eff, replacement=False)
            loss, stats = _train_step_opd_action_strict(
                student=student, teacher_cache=teacher_cache,
                base_z=base_z, base_logits=base_logits, rel_features=rel_features,
                idx=idx, rel_helper=rel_helper,
                alpha_f=alpha_f, lambda_min=lambda_min, lambda_extra=lambda_extra,
                y=y, train_mask=train_mask, pos_weight=pos_weight,
                add_multihead=True, alpha_r=alpha_r, alpha_g=alpha_g,
            )
        else:
            raise AssertionError(f"unreachable: mode={mode!r}")

        loss.backward()
        grad_clip = distill_cfg.get("grad_clip")
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics, _, _ = _eval_student(
                student, base_z, base_logits, rel_features, val_mask, y, device,
            )
            row = {
                "epoch": epoch,
                "loss": stats.get("total", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
                "val/auprc": float(val_metrics.get("auprc", 0.0)),
                "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                **{f"loss/{k}": v for k, v in stats.items() if k != "total"},
            }
            rows.append(row)

            mv = float(val_metrics.get("auprc", 0.0))
            if mv > best_val:
                best_val = mv
                best_epoch = epoch
                best_state = copy.deepcopy(student.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[G-OPD-Flash:{mode}] Early stop at epoch {epoch} (best val_auprc={best_val:.4f} @ {best_epoch})")
                    break

    if best_state is not None:
        student.load_state_dict(best_state)

    # Final eval w/ best weights + threshold calibration on val
    _, val_prob, val_y = _eval_student(
        student, base_z, base_logits, rel_features, val_mask, y, device,
    )
    threshold, _, _ = find_best_threshold(val_y, val_prob, metric="macro_f1")
    val_metrics, _, _ = _eval_student(
        student, base_z, base_logits, rel_features, val_mask, y, device,
        threshold=threshold,
    )
    test_metrics, _, _ = _eval_student(
        student, base_z, base_logits, rel_features, test_mask, y, device,
        threshold=threshold,
    )

    return {
        "mode": mode,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, rows


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def write_epoch_log(log_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    jsonl_path = log_dir / "flash_train_log.jsonl"
    csv_path = log_dir / "flash_train_log.csv"
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_run_name(mode: str) -> str:
    return f"g_opd_flash_{mode}"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G-OPD-Flash trainer (Idea 2C → C3)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--mode", type=str, default="g_opd_flash", choices=list(SUPPORTED_MODES))
    p.add_argument("--run_name", type=str, default=None,
                   help="Override run name (default: g_opd_flash_<mode>)")
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--teacher_extractor_ckpt", type=str, default=None,
                   help="Optional Idea-2B learned evidence_extractor.pt for the teacher")
    p.add_argument("--base_ckpt_path", type=str, default=None)
    p.add_argument("--base_v2_ckpt_path", type=str, default=None,
                   help="T7 deployment-shift: if set, EVAL the trained student on base_v2 outputs while having TRAINED on base_v1.")

    # Distillation hyperparams (G-OPD-Flash design v3 defaults)
    p.add_argument("--K", type=int, default=2048,
                   help="Phase B sampled batch size (capped to N_train).")
    p.add_argument("--alpha_q", type=float, default=0.5)
    p.add_argument("--beta_q", type=float, default=0.3)
    p.add_argument("--eps_q", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--alpha_f", type=float, default=1.0)
    p.add_argument("--alpha_r", type=float, default=0.3)
    p.add_argument("--alpha_g", type=float, default=0.2)
    p.add_argument("--lambda_min", type=float, default=0.05)
    p.add_argument("--lambda_extra", type=float, default=0.50)
    p.add_argument("--lambda_distill_v1", type=float, default=1.0,
                   help="Off-policy v1: λ_distill on MSE(Δ_φ, Δ_T).")
    p.add_argument("--gamma_kl_v1", type=float, default=0.5,
                   help="Off-policy v1: γ_kl on gate KL.")

    # T4 curriculum prior & calibration bins (optional — fallback if absent)
    p.add_argument("--curriculum_prior_path", type=str,
                   default="artifacts/teacher_curriculum_prior.json")
    p.add_argument("--calibration_bins_path", type=str,
                   default="artifacts/teacher_calibration_bins.json")
    p.add_argument("--r_min", type=float, default=0.1)
    p.add_argument("--n_cal_bins", type=int, default=10)

    # Adapter arch
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=None)

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
    mode = args.mode
    run_name = args.run_name or make_run_name(mode)

    set_seed(seed)
    device = torch.device(args.device)

    # ----- Output dirs -----
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    result_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
    ckpt_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))

    (log_dir / "repro_command.txt").write_text(" ".join(sys.argv) + "\n")
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

    # ----- Load frozen base (v1) -----
    base_logits, base_z, base_ckpt_path = load_frozen_base(
        config, dataset_name, model_name, seed, data, device,
        ckpt_override=args.base_ckpt_path,
    )

    # ----- SHA-256 of base ckpt for C1 contract logging -----
    import hashlib
    base_sha256 = hashlib.sha256(Path(base_ckpt_path).read_bytes()).hexdigest()
    print(f"[G-OPD-Flash:{mode}] Base SHA-256 (epoch 0) = {base_sha256[:16]}…")

    # ----- Load relation features -----
    rel_features, rel_meta = load_relation_features(
        dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
    )
    rel_features = rel_features.to(device)

    relation_names: list[str] = [
        str(n).upper() for n in
        p2_cfg.get("relation_names", rel_meta.get("relations", []))
    ]
    if not relation_names:
        relation_names = list(RELATION_SCHEMAS[dataset_name].keys())
    num_relations = len(relation_names)

    # ----- Optional: learned-extractor teacher evidence -----
    if args.teacher_extractor_ckpt is not None:
        extractor_ckpt_path = Path(args.teacher_extractor_ckpt)
        if not extractor_ckpt_path.exists():
            raise FileNotFoundError(extractor_ckpt_path)
        rel_features = build_learned_teacher_features(
            config=config, data=data, relation_names=relation_names,
            extractor_ckpt_path=extractor_ckpt_path, device=device,
        )

    # ----- Load frozen teacher -----
    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.exists():
        raise FileNotFoundError(teacher_ckpt_path)
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
    raw = torch.load(teacher_ckpt_path, weights_only=False, map_location=device)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    teacher.load_state_dict(state)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"[G-OPD-Flash:{mode}] Loaded teacher from {teacher_ckpt_path}")
    n_teacher = sum(p.numel() for p in teacher.parameters())

    # ----- Teacher cache w/ heads (full-graph one-time) -----
    t0 = time.time()
    teacher_cache = generate_teacher_cache_with_heads(teacher, base_z, base_logits, rel_features, device)
    print(f"[G-OPD-Flash:{mode}] Teacher cache (3 heads) generated in {time.time() - t0:.1f}s")

    # ----- Curriculum prior + calibration bins (T4 caches; fallback to defaults) -----
    cell_key = f"{dataset_name}/{model_name}/seed_{seed}"
    prior_path = Path(args.curriculum_prior_path) if args.curriculum_prior_path else None
    bins_path = Path(args.calibration_bins_path) if args.calibration_bins_path else None
    r_c = load_curriculum_prior(cell_key, prior_path, default_r_c=1.0)
    cal_bins = load_calibration_bins(cell_key, bins_path, n_bins=args.n_cal_bins)
    rel_helper = NodeReliabilityHelper(r_c=r_c, cal_bins=cal_bins, n_bins=args.n_cal_bins, r_min=args.r_min)
    print(f"[G-OPD-Flash:{mode}] cell_key={cell_key} r_c={r_c:.3f} cal_bins={'loaded' if cal_bins else 'fallback=1.0'}")

    # ----- Build student adapter -----
    student = FlashAdapter(
        base_z_dim=int(base_z.shape[1]),
        num_relations=num_relations,
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        delta_max=p2_cfg.get("delta_rel_max", 2.0),
    ).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    print(f"[G-OPD-Flash:{mode}] Student params: {n_student} ({n_student / n_teacher:.2f}x teacher)")

    # ----- Pos-weight (matches v1) -----
    y_train_int = data.y[data.train_mask].long()
    n_pos = int((y_train_int == 1).sum())
    n_neg = int((y_train_int == 0).sum())
    pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)), device=device)

    # ----- Train -----
    distill_cfg = {
        "lr": args.lr, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "patience": args.patience,
        "K": args.K, "alpha_q": args.alpha_q, "beta_q": args.beta_q, "eps_q": args.eps_q, "tau": args.tau,
        "alpha_f": args.alpha_f, "alpha_r": args.alpha_r, "alpha_g": args.alpha_g,
        "lambda_min": args.lambda_min, "lambda_extra": args.lambda_extra,
        "lambda_distill_v1": args.lambda_distill_v1, "gamma_kl_v1": args.gamma_kl_v1,
        "grad_clip": args.grad_clip,
    }
    t0 = time.time()
    train_summary, rows = train_g_opd_flash(
        mode=mode, student=student, teacher_cache=teacher_cache,
        base_z=base_z, base_logits=base_logits, rel_features=rel_features,
        data=data, distill_cfg=distill_cfg, rel_helper=rel_helper,
        device=device, pos_weight=pos_weight,
    )
    elapsed = time.time() - t0

    # ----- Save student -----
    write_epoch_log(log_dir, rows)
    torch.save(student.state_dict(), ckpt_dir / "student.pt")

    # ===== C1 contract assertion (post-training SHA-256 re-check) =====
    # Design v3.1 §4.3: "SHA-256 hash logged at epoch 0 and 80; assertion
    # failure aborts run".  Verifies the base detector file was not mutated
    # during training — guard against accidental writes to the base ckpt
    # path or filesystem-level corruption.
    base_sha256_post = hashlib.sha256(Path(base_ckpt_path).read_bytes()).hexdigest()
    assert base_sha256 == base_sha256_post, (
        f"C1 VIOLATED: base SHA-256 changed during training\n"
        f"  pre  = {base_sha256}\n"
        f"  post = {base_sha256_post}\n"
        f"  base_ckpt = {base_ckpt_path}"
    )
    print(f"[G-OPD-Flash:{mode}] C1 contract OK: base SHA-256 unchanged (post-training)")

    # ----- T7: deployment-shift eval (optional) -----
    deploy_shift_metrics = None
    if args.base_v2_ckpt_path is not None:
        print(f"[G-OPD-Flash:{mode}] T7 deployment-shift eval: loading base_v2 from {args.base_v2_ckpt_path}")
        # Re-load with v2 base for eval only.
        base_logits_v2, base_z_v2, _ = load_frozen_base(
            config, dataset_name, model_name, seed, data, device,
            ckpt_override=args.base_v2_ckpt_path,
        )
        _, val_prob_v2, val_y_v2 = _eval_student(
            student, base_z_v2, base_logits_v2, rel_features, data.val_mask.to(device),
            data.y.to(device), device,
        )
        thr_v2, _, _ = find_best_threshold(val_y_v2, val_prob_v2, metric="macro_f1")
        deploy_shift_metrics, _, _ = _eval_student(
            student, base_z_v2, base_logits_v2, rel_features, data.test_mask.to(device),
            data.y.to(device), device, threshold=thr_v2,
        )

    # ----- Baselines -----
    k_values = [50, 100, 200]
    base_thr = find_best_threshold(
        data.y[data.val_mask].numpy(),
        torch.sigmoid(base_logits[data.val_mask]).cpu().numpy(),
        metric="macro_f1",
    )[0]
    base_only_metrics = evaluate_base_only(
        base_logits, data.test_mask.to(device), data.y.to(device),
        threshold=base_thr, k_values=k_values,
    )
    with torch.no_grad():
        teacher_val_out = teacher(
            base_z[data.val_mask.to(device)],
            base_logits[data.val_mask.to(device)],
            rel_features[data.val_mask.to(device)].to(device),
        )
        teacher_val_prob = torch.sigmoid(teacher_val_out["final_logit"]).cpu().numpy()
        teacher_val_y = data.y[data.val_mask].cpu().numpy()
    teacher_thr = find_best_threshold(teacher_val_y, teacher_val_prob, metric="macro_f1")[0]
    teacher_metrics = evaluate_teacher(
        teacher, base_z, base_logits, rel_features,
        data.test_mask.to(device), data.y.to(device), device,
        threshold=teacher_thr, k_values=k_values,
    )

    # ----- Diagnostics & summary -----
    diagnostics = {
        "run_name": run_name,
        "mode": mode,
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "git_hash": get_git_hash(),
        "elapsed_seconds": elapsed,
        "student_params": n_student,
        "teacher_params": n_teacher,
        "param_ratio": round(n_student / n_teacher, 4),
        "teacher_ckpt": str(teacher_ckpt_path),
        "base_ckpt": str(base_ckpt_path),
        "base_sha256": base_sha256,
        "relation_names": relation_names,
        "cell_key": cell_key,
        "r_c": r_c,
        "cal_bins_loaded": cal_bins is not None,
        "distill_cfg": distill_cfg,
        "best_epoch": train_summary["best_epoch"],
        "best_threshold": train_summary["best_threshold"],
        "val_metrics": train_summary["val_metrics"],
        "test_metrics_student": train_summary["test_metrics"],
        "test_metrics_base_only": base_only_metrics,
        "test_metrics_teacher": teacher_metrics,
        "test_metrics_deploy_shift": deploy_shift_metrics,
    }
    (log_dir / "phase2_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "run_name": run_name,
        "mode": mode,
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

    ta = train_summary["test_metrics"]
    tb = base_only_metrics
    tt = teacher_metrics
    print(f"\n[G-OPD-Flash:{mode}] Done in {elapsed:.1f}s | best_epoch={train_summary['best_epoch']}")
    print(f"  Student AUPRC  = {ta.get('auprc', 0.0):.4f}  AUROC = {ta.get('roc_auc', 0.0):.4f}")
    print(f"  Base-only AUPRC= {tb.get('auprc', 0.0):.4f}  AUROC = {tb.get('roc_auc', 0.0):.4f}")
    print(f"  Teacher AUPRC  = {tt.get('auprc', 0.0):.4f}  AUROC = {tt.get('roc_auc', 0.0):.4f}")
    capture = (ta.get("auprc", 0.0) / max(tt.get("auprc", 1e-9), 1e-9)) * 100
    print(f"  AUPRC capture  = {capture:.1f}% (target ≥ 95%)")
    if deploy_shift_metrics is not None:
        print(f"  Deploy-shift AUPRC = {deploy_shift_metrics.get('auprc', 0.0):.4f}  (T7)")
    print(f"  Params: student={n_student} teacher={n_teacher} ratio={n_student / n_teacher:.2f}x")
    print(f"\nCheckpoint: {ckpt_dir / 'student.pt'}")
    print(f"Metrics:    {result_dir / 'stage3_metrics.json'}")


if __name__ == "__main__":
    main()
