"""Batch audit all splits across datasets and seeds.

Reads split_meta.json for each dataset/seed combination, collects
statistics, and generates a summary table in CSV and Markdown format.

Usage:
    python scripts/audit_all_splits.py \\
        --datasets yelpchi amazon \\
        --seeds 123 456 789 42 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_split_dir, ensure_dir, ARTIFACTS_ROOT


def audit_single_split(dataset: str, seed: int) -> dict | None:
    """Read split_meta.json for one dataset/seed and return a summary row.

    Returns None if the split does not exist.
    """
    split_dir = get_split_dir(dataset, seed)
    meta_path = split_dir / "split_meta.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    train_pr = meta.get("train_pos_rate", meta.get("pos_rate_train", 0.0))
    val_pr = meta.get("val_pos_rate", meta.get("pos_rate_val", 0.0))
    test_pr = meta.get("test_pos_rate", meta.get("pos_rate_test", 0.0))

    global_pr = meta.get("global_pos_rate")
    if global_pr is None:
        num_pos_total = (
            meta.get("num_pos_train", 0)
            + meta.get("num_pos_val", 0)
            + meta.get("num_pos_test", 0)
        )
        num_nodes = meta.get("num_nodes", 0)
        global_pr = num_pos_total / num_nodes if num_nodes > 0 else 0.0

    rates = [train_pr, val_pr, test_pr]
    max_gap = meta.get("max_pos_rate_gap")
    if max_gap is None:
        max_gap = max(rates) - min(rates)

    rel_gap = meta.get("relative_pos_rate_gap")
    if rel_gap is None:
        rel_gap = max_gap / global_pr if global_pr > 0 else 0.0

    overlap_info = meta.get("mask_overlap_counts", {})
    total_overlap = sum(
        overlap_info.get(k, 0)
        for k in ("train_val", "train_test", "val_test")
    )

    stratified_val = meta.get("stratified")
    if stratified_val is None:
        # Heuristic: very small gap suggests stratified split
        stratified_flag = "auto(low_gap)" if max_gap < 0.005 else "auto(high_gap)"
    elif stratified_val:
        stratified_flag = "true"
    else:
        stratified_flag = "false"

    # Warnings (reuse logic from check_split_sanity)
    warnings: list[str] = []

    if total_overlap > 0:
        warnings.append(f"mask_overlap={total_overlap}")
    if train_pr < 0.01:
        warnings.append(f"low_train_pos_rate={train_pr:.4f}")
    if val_pr < 0.01:
        warnings.append(f"low_val_pos_rate={val_pr:.4f}")
    if test_pr < 0.01:
        warnings.append(f"low_test_pos_rate={test_pr:.4f}")

    num_nodes = meta.get("num_nodes", 0)
    num_total_split = (
        meta.get("num_train", 0) + meta.get("num_val", 0) + meta.get("num_test", 0)
    )
    if num_nodes > 0 and num_total_split != num_nodes:
        warnings.append(
            f"coverage_gap={num_nodes - num_total_split}"
        )

    if dataset == "amazon" and test_pr > 0.5:
        warnings.append(f"high_test_pos_rate={test_pr:.4f}")

    return {
        "dataset": dataset,
        "seed": seed,
        "stratified": stratified_flag,
        "train_pos_rate": f"{train_pr:.6f}",
        "val_pos_rate": f"{val_pr:.6f}",
        "test_pos_rate": f"{test_pr:.6f}",
        "global_pos_rate": f"{global_pr:.6f}",
        "max_pos_rate_gap": f"{max_gap:.6f}",
        "relative_pos_rate_gap": f"{rel_gap:.6f}",
        "mask_overlap": total_overlap,
        "warnings": len(warnings),
        "warning_details": "; ".join(warnings) if warnings else "",
    }


COLUMNS = [
    "dataset",
    "seed",
    "stratified",
    "train_pos_rate",
    "val_pos_rate",
    "test_pos_rate",
    "global_pos_rate",
    "max_pos_rate_gap",
    "relative_pos_rate_gap",
    "mask_overlap",
    "warnings",
]


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No splits found.")
        return

    widths = {col: len(col) for col in COLUMNS}
    for row in rows:
        for col in COLUMNS:
            widths[col] = max(widths[col], len(str(row[col])))

    header = " | ".join(col.rjust(widths[col]) for col in COLUMNS)
    sep = "-+-".join("-" * widths[col] for col in COLUMNS)

    print(header)
    print(sep)
    for row in rows:
        line = " | ".join(str(row[col]).rjust(widths[col]) for col in COLUMNS)
        print(line)

    total = len(rows)
    missing_info = []
    any_overlap = sum(1 for r in rows if r["mask_overlap"] > 0)
    any_warnings = sum(1 for r in rows if r["warnings"] > 0)
    print()
    print(f"Total splits audited: {total}")
    if missing_info:
        print(f"Missing splits: {len(missing_info)}")
    if any_overlap:
        print(f"Splits with mask overlap: {any_overlap}")
    if any_warnings:
        print(f"Splits with warnings: {any_warnings}")
    print(f"All checks passed: {any_overlap == 0 and any_warnings == 0}")


def save_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {path}")


def save_markdown(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)

    lines = ["# Split Sanity Summary", ""]

    if not rows:
        lines.append("No splits found.")
        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return

    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")

    for row in rows:
        vals = " | ".join(str(row[col]) for col in COLUMNS)
        lines.append(f"| {vals} |")

    lines.append("")

    total = len(rows)
    any_overlap = sum(1 for r in rows if r["mask_overlap"] > 0)
    any_warnings = sum(1 for r in rows if r["warnings"] > 0)

    lines.append("")
    lines.append(f"**Total splits:** {total}")
    if any_overlap:
        lines.append(f"**Splits with mask overlap:** {any_overlap}")
    if any_warnings:
        lines.append(f"**Splits with warnings:** {any_warnings}")
    lines.append(
        f"**All checks passed:** "
        f"{'Yes' if any_overlap == 0 and any_warnings == 0 else 'No'}"
    )
    lines.append("")

    rows_with_warnings = [r for r in rows if r["warnings"] > 0]
    if rows_with_warnings:
        lines.append("## Warning Details")
        lines.append("")
        for r in rows_with_warnings:
            lines.append(
                f"- **{r['dataset']}/seed_{r['seed']}**: {r['warning_details']}"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch audit all splits across datasets and seeds."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["yelpchi", "amazon"],
        help="Dataset names to audit (default: yelpchi amazon)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 42, 123, 456, 789, 2026],
        help="Seeds to audit (default: 0 42 123 456 789 2026)",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    missing: list[str] = []

    for dataset in args.datasets:
        for seed in args.seeds:
            result = audit_single_split(dataset, seed)
            if result is not None:
                rows.append(result)
            else:
                missing.append(f"{dataset}/seed_{seed}")
                print(f"  [SKIP] {dataset}/seed_{seed}: split_meta.json not found")

    rows.sort(key=lambda r: (r["dataset"], int(r["seed"])))

    print()
    print_table(rows)

    if missing:
        print(f"\nMissing splits ({len(missing)}): {', '.join(missing)}")

    output_dir = ensure_dir(ARTIFACTS_ROOT / "tables")
    save_csv(rows, output_dir / "split_sanity_summary.csv")
    save_markdown(rows, output_dir / "split_sanity_summary.md")


if __name__ == "__main__":
    main()
