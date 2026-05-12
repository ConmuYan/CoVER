"""Check split sanity for graph fraud detection datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import (
    load_from_mat,
    load_from_npz,
    load_from_pkl,
    load_from_pt,
    load_synthetic_graph,
    load_tiny_graph,
)
from utils.paths import (
    ensure_dir,
    get_reports_dir_stratified,
    get_split_dir,
    get_split_meta_path,
    get_split_path,
    get_stratified_split_dir,
    get_stratified_split_meta_path,
    get_stratified_split_path,
)


DEFAULT_DATASET_PATHS = {
    "yelpchi": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/YelpChi.mat",
    "amazon": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/Amazon.mat",
}


def check_mask_overlap(train_mask, val_mask, test_mask):
    train_idx = set(train_mask.nonzero(as_tuple=True)[0].tolist())
    val_idx = set(val_mask.nonzero(as_tuple=True)[0].tolist())
    test_idx = set(test_mask.nonzero(as_tuple=True)[0].tolist())

    overlap_tv = train_idx & val_idx
    overlap_tt = train_idx & test_idx
    overlap_vt = val_idx & test_idx

    return {
        "train_val_overlap": len(overlap_tv),
        "train_test_overlap": len(overlap_tt),
        "val_test_overlap": len(overlap_vt),
        "total_overlap": len(overlap_tv) + len(overlap_tt) + len(overlap_vt),
    }


def compute_split_stats(y, train_mask, val_mask, test_mask):
    y = y.view(-1)

    num_nodes = int(y.shape[0])
    num_train = int(train_mask.sum().item())
    num_val = int(val_mask.sum().item())
    num_test = int(test_mask.sum().item())

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    num_pos_train = int(y_train.sum().item())
    num_pos_val = int(y_val.sum().item())
    num_pos_test = int(y_test.sum().item())
    num_pos_total = int(y.sum().item())

    pos_rate_train = num_pos_train / num_train if num_train > 0 else 0.0
    pos_rate_val = num_pos_val / num_val if num_val > 0 else 0.0
    pos_rate_test = num_pos_test / num_test if num_test > 0 else 0.0
    global_pos_rate = num_pos_total / num_nodes if num_nodes > 0 else 0.0

    split_pos_rates = [pos_rate_train, pos_rate_val, pos_rate_test]
    max_pos_rate_gap = max(split_pos_rates) - min(split_pos_rates)
    relative_pos_rate_gap = max_pos_rate_gap / global_pos_rate if global_pos_rate > 0 else 0.0

    return {
        "num_nodes": num_nodes,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "num_pos_total": num_pos_total,
        "num_pos_train": num_pos_train,
        "num_pos_val": num_pos_val,
        "num_pos_test": num_pos_test,
        "pos_rate_train": pos_rate_train,
        "pos_rate_val": pos_rate_val,
        "pos_rate_test": pos_rate_test,
        "global_pos_rate": global_pos_rate,
        "max_pos_rate_gap": max_pos_rate_gap,
        "relative_pos_rate_gap": relative_pos_rate_gap,
    }


def load_dataset_without_split(dataset_name: str, dataset_path: str | None, seed: int):
    name = dataset_name.lower()

    if name == "tiny":
        return load_tiny_graph(seed=seed)
    if name == "synthetic_small":
        return load_synthetic_graph(num_nodes=500, seed=seed)
    if name == "synthetic_medium":
        return load_synthetic_graph(num_nodes=2000, seed=seed)

    if name in ("yelpchi", "amazon"):
        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATHS.get(name)
        if dataset_path is None:
            raise ValueError(f"Path required for dataset '{dataset_name}'")
        return load_from_mat(dataset_path)

    if dataset_path is None:
        raise ValueError(f"Path required for dataset '{dataset_name}'")

    path = Path(dataset_path)
    ext = path.suffix.lower()
    if ext == ".pt":
        return load_from_pt(path)
    if ext == ".pkl":
        return load_from_pkl(path)
    if ext == ".npz":
        return load_from_npz(path)
    if ext == ".mat":
        return load_from_mat(path)

    raise ValueError(f"Unsupported dataset format: {ext}")


def resolve_split_artifacts(dataset: str, seed: int, requested_stratified: bool | None) -> tuple[Path, dict[str, Any] | None, bool]:
    candidate_order: list[bool]
    if requested_stratified is False:
        candidate_order = [False, True]
    else:
        candidate_order = [True, False]

    tried: list[str] = []
    for candidate_stratified in candidate_order:
        split_path = (
            get_stratified_split_path(dataset, candidate_stratified, seed)
            if candidate_stratified
            else get_split_path(dataset, seed)
        )
        tried.append(str(split_path))
        if not split_path.exists():
            continue

        meta_path = (
            get_stratified_split_meta_path(dataset, candidate_stratified, seed)
            if candidate_stratified
            else get_split_meta_path(dataset, seed)
        )

        meta: dict[str, Any] | None = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        if meta is not None and "stratified" in meta:
            stratified = bool(meta["stratified"])
        else:
            stratified = candidate_stratified

        return split_path, meta, stratified

    tried_paths = ", ".join(tried)
    raise FileNotFoundError(
        f"Could not find split file for dataset='{dataset}', seed={seed}. Tried: {tried_paths}"
    )


def build_messages(dataset_name: str, stats: dict[str, Any], overlap: dict[str, int]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if overlap["total_overlap"] > 0:
        errors.append(
            "ERROR: Mask overlap detected! "
            f"train_val={overlap['train_val_overlap']}, "
            f"train_test={overlap['train_test_overlap']}, "
            f"val_test={overlap['val_test_overlap']}"
        )

    if stats["num_train"] + stats["num_val"] + stats["num_test"] != stats["num_nodes"]:
        errors.append(
            "ERROR: Masks don't cover all nodes! "
            f"train+val+test={stats['num_train'] + stats['num_val'] + stats['num_test']}, "
            f"total={stats['num_nodes']}"
        )

    if stats["num_pos_train"] == 0 or stats["num_pos_val"] == 0 or stats["num_pos_test"] == 0:
        errors.append(
            "ERROR: Split with zero positive samples detected! "
            f"train={stats['num_pos_train']}, val={stats['num_pos_val']}, test={stats['num_pos_test']}"
        )

    if stats["max_pos_rate_gap"] > 0.02:
        warnings.append(f"WARNING: max_pos_rate_gap={stats['max_pos_rate_gap']:.6f} > 0.02")

    if stats["relative_pos_rate_gap"] > 0.3:
        warnings.append(f"WARNING: relative_pos_rate_gap={stats['relative_pos_rate_gap']:.6f} > 0.3")

    if stats["global_pos_rate"] < 0.01 and min(stats["num_pos_val"], stats["num_pos_test"]) < 10:
        warnings.append(
            "WARNING: global_pos_rate below 0.01 and val/test positive count below 10 "
            f"(global={stats['global_pos_rate']:.6f}, val_pos={stats['num_pos_val']}, test_pos={stats['num_pos_test']})"
        )

    if stats["pos_rate_train"] < 0.01:
        warnings.append(f"WARNING: Very low positive rate in train: {stats['pos_rate_train']:.4f}")
    if stats["pos_rate_test"] < 0.01:
        warnings.append(f"WARNING: Very low positive rate in test: {stats['pos_rate_test']:.4f}")

    if dataset_name == "amazon" and stats["pos_rate_test"] > 0.5:
        warnings.append(f"WARNING: Amazon test positive rate is unusually high: {stats['pos_rate_test']:.4f}")

    return errors, warnings


def render_summary_table(report: dict[str, Any], errors: list[str], warnings: list[str]) -> list[str]:
    columns = [
        "dataset",
        "seed",
        "stratified",
        "train_pos_rate",
        "val_pos_rate",
        "test_pos_rate",
        "max_pos_rate_gap",
        "warnings",
    ]

    row = {
        "dataset": report["dataset"],
        "seed": report["seed"],
        "stratified": str(report["stratified"]).lower(),
        "train_pos_rate": f"{report['pos_rate_train']:.4f}",
        "val_pos_rate": f"{report['pos_rate_val']:.4f}",
        "test_pos_rate": f"{report['pos_rate_test']:.4f}",
        "max_pos_rate_gap": f"{report['max_pos_rate_gap']:.6f}",
        "warnings": f"{len(errors)}E/{len(warnings)}W",
    }

    widths = {column: len(column) for column in columns}
    for column in columns:
        widths[column] = max(widths[column], len(str(row[column])))

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    values = " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
    return [header, separator, values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stratified", action="store_true", default=None)
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if config is not None:
        dataset_name = str(config["dataset"]["name"]).lower()
        dataset_path = config["dataset"].get("path")
        seed = args.seed if args.seed is not None else config["train"].get("seed", 0)
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
        requested_stratified = args.stratified if args.stratified is not None else config["dataset"].get("stratified")
    elif args.dataset:
        dataset_name = args.dataset.lower()
        dataset_path = DEFAULT_DATASET_PATHS.get(dataset_name)
        seed = args.seed if args.seed is not None else 0
        split_mode = "supervised"
        train_ratio = 0.4
        val_test_ratio = [1, 2]
        requested_stratified = args.stratified
    else:
        print("Error: Must provide --config or --dataset")
        sys.exit(1)

    print(f"Checking split sanity for {dataset_name}, seed={seed}")

    data = load_dataset_without_split(dataset_name, dataset_path, seed)
    split_path, meta, stratified = resolve_split_artifacts(dataset_name, seed, requested_stratified)

    split_state = torch.load(split_path, weights_only=True)
    data.train_mask = split_state["train_mask"]
    data.val_mask = split_state["val_mask"]
    data.test_mask = split_state["test_mask"]

    stats = compute_split_stats(data.y, data.train_mask, data.val_mask, data.test_mask)
    overlap = check_mask_overlap(data.train_mask, data.val_mask, data.test_mask)

    if meta is not None:
        split_mode = meta.get("split_mode", split_mode)
        train_ratio = meta.get("train_ratio", train_ratio)
        val_test_ratio = meta.get("val_test_ratio", val_test_ratio)

    errors, warnings = build_messages(dataset_name, stats, overlap)
    passed = len(errors) == 0

    report = {
        "dataset": dataset_name,
        "seed": seed,
        "split_mode": split_mode,
        "train_ratio": train_ratio,
        "val_test_ratio": val_test_ratio,
        "stratified": stratified,
        "split_path": str(split_path),
        **stats,
        **overlap,
        "errors": errors,
        "issues": errors,
        "warnings": warnings,
        "passed": passed,
    }

    # Empty model component keeps the requested stratified-aware report layout.
    report_dir = ensure_dir(get_reports_dir_stratified(dataset_name, "", stratified, seed))

    with open(report_dir / "split_sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Split Sanity Report: {dataset_name} seed={seed}",
        "",
        "## Split Configuration",
        f"- Mode: {split_mode}",
        f"- Train ratio: {train_ratio}",
        f"- Val:test: {val_test_ratio}",
        f"- Stratified: {stratified}",
        f"- Split path: {split_path}",
        "",
        "## Node Counts",
        f"- Total: {stats['num_nodes']}",
        f"- Train: {stats['num_train']} ({stats['num_train'] / stats['num_nodes']:.1%})",
        f"- Val: {stats['num_val']} ({stats['num_val'] / stats['num_nodes']:.1%})",
        f"- Test: {stats['num_test']} ({stats['num_test'] / stats['num_nodes']:.1%})",
        "",
        "## Positive Rates",
        f"- Train: {stats['pos_rate_train']:.4f} ({stats['num_pos_train']}/{stats['num_train']})",
        f"- Val: {stats['pos_rate_val']:.4f} ({stats['num_pos_val']}/{stats['num_val']})",
        f"- Test: {stats['pos_rate_test']:.4f} ({stats['num_pos_test']}/{stats['num_test']})",
        f"- Global: {stats['global_pos_rate']:.4f} ({stats['num_pos_total']}/{stats['num_nodes']})",
        f"- Max gap: {stats['max_pos_rate_gap']:.6f}",
        f"- Relative gap: {stats['relative_pos_rate_gap']:.6f}",
        "",
        "## Mask Overlap",
        f"- Train-Val: {overlap['train_val_overlap']}",
        f"- Train-Test: {overlap['train_test_overlap']}",
        f"- Val-Test: {overlap['val_test_overlap']}",
        f"- Total: {overlap['total_overlap']}",
    ]

    if errors:
        md_lines.extend(["", "## Errors"])
        md_lines.extend(f"- {error}" for error in errors)

    if warnings:
        md_lines.extend(["", "## Warnings"])
        md_lines.extend(f"- {warning}" for warning in warnings)

    md_lines.extend(["", f"## Result: {'PASS' if passed else 'FAIL'}"])

    with open(report_dir / "split_sanity_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'=' * 100}")
    for line in render_summary_table(report, errors, warnings):
        print(line)
    print(f"{'=' * 100}")
    print(f"Split Sanity Result: {'PASS' if passed else 'FAIL'}")
    print(
        f"Nodes: {stats['num_nodes']} (train={stats['num_train']}, val={stats['num_val']}, test={stats['num_test']})"
    )
    print(
        f"Positive rates: train={stats['pos_rate_train']:.4f}, val={stats['pos_rate_val']:.4f}, "
        f"test={stats['pos_rate_test']:.4f}, global={stats['global_pos_rate']:.4f}"
    )
    print(
        f"Gaps: max={stats['max_pos_rate_gap']:.6f}, relative={stats['relative_pos_rate_gap']:.6f}"
    )
    print(f"Mask overlap: {overlap['total_overlap']}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nReport saved to: {report_dir}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
