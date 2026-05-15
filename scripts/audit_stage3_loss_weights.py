from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from task89_common import DEFAULT_SEEDS, base_status_for_node, git_hash, load_base_outputs, load_config, mean_std
from utils.paths import ensure_dir


def audit_seed(config: dict, seed: int) -> dict:
    data, base_logits, _, _ = load_base_outputs(config, seed)
    train_mask = data.train_mask
    y_train = data.y[train_mask].float()
    base_train = base_logits[train_mask]
    base_prob = torch.sigmoid(base_train)
    per_node_bce = F.binary_cross_entropy_with_logits(base_train, y_train, reduction="none")

    correction_weight = float(config.get("reasoner", {}).get("correction_weight", 2.0))
    weights = torch.ones_like(y_train)
    base_fn = (y_train == 1) & (base_prob < 0.5)
    base_fp = (y_train == 0) & (base_prob >= 0.5)
    high_loss = base_fn | base_fp
    weights[base_fn] *= correction_weight
    weights[base_fp] *= correction_weight
    weights[high_loss] *= 1.5
    weighted = per_node_bce * weights

    pos = y_train == 1
    neg = y_train == 0
    statuses = [base_status_for_node(data.y, torch.sigmoid(base_logits), int(i)) for i in train_mask.nonzero(as_tuple=True)[0].tolist()]

    def sum_mask(mask: torch.Tensor) -> float:
        return float(weighted[mask].sum().item()) if mask.any() else 0.0

    def mean_weight(mask: torch.Tensor) -> float:
        return float(weights[mask].mean().item()) if mask.any() else 0.0

    row = {
        "seed": seed,
        "train_pos_count": int(pos.sum().item()),
        "train_neg_count": int(neg.sum().item()),
        "base_fn_count": int(base_fn.sum().item()),
        "base_fp_count": int(base_fp.sum().item()),
        "positive_bce_contribution": sum_mask(pos),
        "negative_bce_contribution": sum_mask(neg),
        "positive_contribution_fraction": sum_mask(pos) / float(weighted.sum().item()) if weighted.sum() > 0 else 0.0,
        "negative_contribution_fraction": sum_mask(neg) / float(weighted.sum().item()) if weighted.sum() > 0 else 0.0,
        "mean_weight_positive": mean_weight(pos),
        "mean_weight_negative": mean_weight(neg),
        "mean_weight_fn": mean_weight(base_fn),
        "mean_weight_fp": mean_weight(base_fp),
        "correction_weight": correction_weight,
    }
    for status in ("FN", "FP", "TP", "TN"):
        mask_list = [s == status for s in statuses]
        mask = torch.tensor(mask_list, dtype=torch.bool)
        row[f"{status}_weighted_bce_contribution"] = sum_mask(mask)
        row[f"{status}_mean_weight"] = mean_weight(mask)
    return row


def write_report(rows: list[dict]) -> None:
    report_dir = ensure_dir(Path("artifacts") / "reports")
    path = report_dir / "stage3_loss_weight_audit_5seed.md"

    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    lines = [
        "# Stage3 Loss Weight Audit",
        "",
        f"- Git hash: `{git_hash()}`",
        f"- Positive BCE contribution fraction: {mean_std(vals('positive_contribution_fraction'))}",
        f"- Negative BCE contribution fraction: {mean_std(vals('negative_contribution_fraction'))}",
        f"- Mean FN count: {mean_std(vals('base_fn_count'))}",
        f"- Mean FP count: {mean_std(vals('base_fp_count'))}",
        f"- Mean weight FN: {mean_std(vals('mean_weight_fn'))}",
        f"- Mean weight FP: {mean_std(vals('mean_weight_fp'))}",
        "",
        "| seed | pos_frac | neg_frac | FN | FP | mean_w_pos | mean_w_neg | mean_w_FN | mean_w_FP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {float(r['positive_contribution_fraction']):.4f} | "
            f"{float(r['negative_contribution_fraction']):.4f} | {r['base_fn_count']} | {r['base_fp_count']} | "
            f"{float(r['mean_weight_positive']):.4f} | {float(r['mean_weight_negative']):.4f} | "
            f"{float(r['mean_weight_fn']):.4f} | {float(r['mean_weight_fp']):.4f} |"
        )
    lines.append("")
    lines.append("This audit checks whether L_det weighting itself is asymmetrically pushing residuals toward one class.")
    path.write_text("\n".join(lines))
    print(f"Loss weight report: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = [audit_seed(config, seed) for seed in args.seeds]
    write_report(rows)


if __name__ == "__main__":
    main()
