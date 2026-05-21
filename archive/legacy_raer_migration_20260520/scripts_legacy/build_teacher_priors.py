"""T4 — Build teacher curriculum prior + 10-bin ECE calibration cache.

For G-OPD-Flash node-level reliability ``r^node_i = r^c · cal_bin(p^T_i)``
(``docs/OPD_FLASH_DESIGN_v3.md`` §3.5, v3.2 MJ-4 fix — ``conf`` factor
dropped), we need two per-cell caches:

1. **Curriculum prior** ``r^c = clip((AUPRC^T_c − π_c) / (1 − π_c), r_min, 1)``
   — *lift over chance* for the teacher, normalised to be cross-dataset
   comparable (Codex round-2 I8 fix).

2. **10-bin ECE calibration table** — for each cell, ``cal_b`` is the
   empirical positive-rate inside confidence bin ``b`` (equal-width).
   Used to discount teacher reliability on miscalibrated probability ranges.

**v3.2 MJ-8 fix**: BOTH stats are now computed on a **stratified train
held-out subset (default 20 %)** instead of the validation set.  This
eliminates the v3.1 soft-leakage path where val labels shaped the
training loss via ``r_c`` and ``cal_bins``.  At deployment, the production
system can recompute these caches from labelled training data without
needing access to a held-out val set.

Outputs::

    artifacts/teacher_curriculum_prior.json   {cell_key: r_c}
    artifacts/teacher_calibration_bins.json   {cell_key: [bin_acc_0, ..., bin_acc_9]}

Cell key format: ``{dataset}/{base}/seed_{seed}``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import RELATION_SCHEMAS, load_relation_stats
from models.raer_teacher import CoVERRelReasoner
from models.gnn import build_detector
from training.metrics import compute_metrics

# Reuse train_distill_adapter loaders.
from scripts.train_distill_adapter import (
    build_learned_teacher_features,
    load_frozen_base,
    load_relation_features,
)


def _cell_key(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}/{model}/seed_{seed}"


def _config_for(dataset: str, model: str) -> Path:
    """Map (dataset, model) → canonical config path."""
    candidate = Path("configs/phase2_reasoner/ablation") / f"idea1_{dataset}_{model}_canonical_clsonly.yaml"
    if candidate.exists():
        return candidate
    # YelpChi-BWGNN fallback to the generic canonical config.
    fallback = Path("configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml")
    if fallback.exists() and dataset == "yelpchi" and model == "bwgnn":
        return fallback
    raise FileNotFoundError(f"No canonical config for cell ({dataset}, {model})")


@torch.no_grad()
def _eval_teacher_train_holdout(
    config: dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    teacher_ckpt: Path,
    extractor_ckpt: Path | None,
    device: torch.device,
    n_bins: int,
    r_min: float,
    holdout_frac: float = 0.2,
) -> tuple[float, list[float], float]:
    """Run frozen teacher on a stratified train held-out subset →
    return (r_c, cal_bins, holdout_auprc).

    **v3.2 MJ-8 fix**: previously used ``data.val_mask``, which caused
    a soft-leakage path (val labels shaped the training loss via
    ``r_c`` and ``cal_bins``).  Now uses a deterministic stratified 20 %
    held-out subset of ``data.train_mask`` so the priors are derived
    entirely from training-time-available labels.
    """
    p2_cfg = config["phase2_reasoner"]
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
    base_logits, base_z, _ = load_frozen_base(
        config, dataset_name, model_name, seed, data, device, ckpt_override=None,
    )
    rel_features, rel_meta = load_relation_features(
        dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
    )
    rel_features = rel_features.to(device)
    relation_names = [
        str(n).upper() for n in p2_cfg.get("relation_names", rel_meta.get("relations", []))
    ] or list(RELATION_SCHEMAS[dataset_name].keys())

    if extractor_ckpt is not None:
        rel_features = build_learned_teacher_features(
            config=config, data=data, relation_names=relation_names,
            extractor_ckpt_path=extractor_ckpt, device=device,
        )

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
    raw = torch.load(teacher_ckpt, weights_only=False, map_location=device)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    teacher.load_state_dict(state)
    teacher.eval()

    # ── v3.2 MJ-8: stratified train held-out construction ────────────────
    # Deterministic per seed: sample ``holdout_frac`` of positives and
    # negatives independently so the prior/calibration uses class-balanced
    # data without touching val/test.
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
    y_train = data.y[train_idx].long()
    rng = torch.Generator().manual_seed(seed + 7)
    pos_idx = train_idx[y_train == 1]
    neg_idx = train_idx[y_train == 0]
    n_pos_holdout = max(1, int(len(pos_idx) * holdout_frac))
    n_neg_holdout = max(1, int(len(neg_idx) * holdout_frac))
    pos_perm = torch.randperm(len(pos_idx), generator=rng)
    neg_perm = torch.randperm(len(neg_idx), generator=rng)
    holdout_pos = pos_idx[pos_perm[:n_pos_holdout]]
    holdout_neg = neg_idx[neg_perm[:n_neg_holdout]]
    holdout_idx = torch.cat([holdout_pos, holdout_neg]).to(device)

    out = teacher(base_z[holdout_idx], base_logits[holdout_idx], rel_features[holdout_idx])
    p_T = torch.sigmoid(out["final_logit"]).cpu().numpy()
    p_T = np.nan_to_num(p_T, nan=0.5, posinf=1.0, neginf=0.0)
    y_holdout = data.y[holdout_idx.cpu()].cpu().numpy()

    # Curriculum prior: r_c = (AUPRC_T - prevalence) / (1 - prevalence)
    metrics = compute_metrics(y_holdout, p_T)
    holdout_auprc = float(metrics.get("auprc", 0.0))
    prevalence = float(np.mean(y_holdout))
    r_c_raw = (holdout_auprc - prevalence) / max(1.0 - prevalence, 1e-9)
    r_c = float(np.clip(r_c_raw, r_min, 1.0))

    # 10-bin calibration: positive rate inside each equal-width bin.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cal_bins: list[float] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_T >= lo) & (p_T <= hi)
        else:
            mask = (p_T >= lo) & (p_T < hi)
        if int(mask.sum()) == 0:
            cal_bins.append(float((lo + hi) / 2))
        else:
            cal_bins.append(float(np.mean(y_holdout[mask])))
    return r_c, cal_bins, holdout_auprc


# Back-compat alias — kept so any external caller importing the old name
# continues to work; both names point at the v3.2 train-holdout implementation.
_eval_teacher_val = _eval_teacher_train_holdout


def _discover_teacher_ckpts(
    root: Path,
    pattern: str,
) -> list[tuple[str, str, int, Path]]:
    """Yield (dataset, model, seed, ckpt_path) for matching reasoner.pt files.

    Layout assumption (matches ``utils.paths.get_checkpoint_dir``)::

        artifacts/checkpoints/{dataset}/{model}/{run_name}/seed_{seed}/reasoner.pt
    """
    rgx = re.compile(pattern)
    out: list[tuple[str, str, int, Path]] = []
    for ckpt in root.rglob("reasoner.pt"):
        parts = ckpt.parts
        try:
            i = parts.index("checkpoints")
            dataset = parts[i + 1]
            model = parts[i + 2]
            run = parts[i + 3]
            seed_dir = parts[i + 4]
        except (ValueError, IndexError):
            continue
        if not rgx.search(run):
            continue
        m = re.match(r"seed_(-?\d+)", seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        out.append((dataset, model, seed, ckpt))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build G-OPD-Flash teacher priors (T4)")
    p.add_argument("--checkpoint_root", type=str, default="artifacts/checkpoints")
    p.add_argument("--pattern", type=str, default="canonical_clsonly|idea2b_learned",
                   help="regex matched against the run-name directory")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--r_min", type=float, default=0.1)
    p.add_argument("--prior_out", type=str, default="artifacts/teacher_curriculum_prior.json")
    p.add_argument("--bins_out", type=str, default="artifacts/teacher_calibration_bins.json")
    p.add_argument("--include_extractor", action="store_true",
                   help="If set, when an extractor ckpt sits next to the reasoner, use it.")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, only process the first N cells (debug).")
    p.add_argument("--dataset_filter", type=str, default="",
                   help="Only include this dataset (e.g. 'yelpchi'). Empty = all.")
    args = p.parse_args()

    device = torch.device(args.device)
    root = Path(args.checkpoint_root)
    cells = _discover_teacher_ckpts(root, args.pattern)
    if args.dataset_filter:
        cells = [c for c in cells if c[0] == args.dataset_filter]
    cells.sort()
    if args.limit > 0:
        cells = cells[: args.limit]

    print(f"[T4] Found {len(cells)} matching teacher cells (pattern={args.pattern!r}).")

    prior_path = Path(args.prior_out)
    bins_path = Path(args.bins_out)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    priors: dict[str, float] = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    bins_table: dict[str, list[float]] = json.loads(bins_path.read_text()) if bins_path.exists() else {}

    for ds, model, seed, ckpt in cells:
        key = _cell_key(ds, model, seed)
        if key in priors and key in bins_table:
            print(f"[T4] {key}: already cached, skipping")
            continue
        try:
            cfg_path = _config_for(ds, model)
        except FileNotFoundError as e:
            print(f"[T4] {key}: SKIP — {e}")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        extractor_ckpt = None
        if args.include_extractor:
            cand = ckpt.parent / "evidence_extractor.pt"
            if cand.exists():
                extractor_ckpt = cand

        try:
            r_c, cal_bins, holdout_auprc = _eval_teacher_train_holdout(
                config=cfg, dataset_name=ds, model_name=model, seed=seed,
                teacher_ckpt=ckpt, extractor_ckpt=extractor_ckpt,
                device=device, n_bins=args.n_bins, r_min=args.r_min,
            )
        except Exception as e:
            print(f"[T4] {key}: FAILED — {type(e).__name__}: {e}")
            continue

        priors[key] = r_c
        bins_table[key] = cal_bins
        print(f"[T4] {key}: r_c={r_c:.4f}  holdout_auprc={holdout_auprc:.4f}  cal_bins=[{cal_bins[0]:.2f}..{cal_bins[-1]:.2f}]")

        # Persist after each cell so partial runs are not lost.
        prior_path.write_text(json.dumps(priors, indent=2, sort_keys=True) + "\n")
        bins_path.write_text(json.dumps(bins_table, indent=2, sort_keys=True) + "\n")

    print(f"[T4] Wrote {len(priors)} entries to {prior_path}")
    print(f"[T4] Wrote {len(bins_table)} entries to {bins_path}")


if __name__ == "__main__":
    main()
