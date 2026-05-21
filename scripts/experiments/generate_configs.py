#!/usr/bin/env python3
"""Generate experiment configs for CoVER-FD.

Reads existing template configs from configs/raer_fd/ and produces new configs
with modified split parameters.  Usage:

    # Generate E1 care_712 configs
    python scripts/experiments/generate_configs.py --experiment E1 --split care_712

    # Generate E2 scarcity configs
    python scripts/experiments/generate_configs.py --experiment E2 --split care_712 --scarcity 0.05

    # Generate all E1-E4 configs at once
    python scripts/experiments/generate_configs.py --all
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_ROOT = ROOT / "configs" / "raer_fd"
EXP_ROOT = CONFIG_ROOT / "experiments"

SEEDS = [42, 123, 456, 789, 2026]

SPLIT_DEFS = {
    "care_712": {"train_ratio": 0.7, "val_test_ratio": [1, 2], "split_mode": "supervised"},
    "bwgnn_424": {"train_ratio": 0.4, "val_test_ratio": [1, 2], "split_mode": "supervised"},
    "bwgnn_semi": {"train_ratio": 0.01, "val_test_ratio": [1, 2], "split_mode": "semi-supervised"},
}

DATASET_INFO = {
    "yelpchi": {"format": "mat", "path": "datasets/YelpChi.mat"},
    "amazon": {"format": "mat", "path": "datasets/Amazon.mat"},
    "yelpnyc": {"format": "mat", "path": "datasets/YelpNYC.mat"},
    "yelpzip": {"format": "mat", "path": "datasets/YelpZip.mat"},
    "tfinance": {"format": "dgl", "path": "datasets/tfinance", "hsd_invert": True},
    "tsocial": {"format": "dgl", "path": "datasets/tsocial", "hsd_invert": True},
}

# --- Experiment definitions ------------------------------------------------

def e1_datasets_models():
    """E1: YelpChi & Amazon × 4 base models."""
    return {
        "datasets": ["yelpchi", "amazon"],
        "models": ["bwgnn", "sage", "gcn", "gat"],
        "splits": ["care_712", "bwgnn_semi"],
        "stages": ["base", "teacher", "student"],
    }


def e2_scarcity():
    """E2: YelpChi scarcity on care_712."""
    scarcity_ratios = [0.01, 0.05, 0.10, 0.20, 0.50]
    jobs = []
    for sr in scarcity_ratios:
        pct = int(sr * 100)
        jobs.append({
            "experiment": "E2_scarcity",
            "split": "care_712",
            "scarcity_ratio": sr,
            "scarcity_label": f"scarcity_{pct}pct",
            "datasets": ["yelpchi"],
            "models": ["bwgnn", "sage", "gcn", "gat"],
            "stages": ["base", "teacher", "student"],
        })
    return jobs


def e3_scaling():
    """E3: YelpNYC & YelpZip scaling with care_712."""
    return {
        "datasets": ["yelpnyc", "yelpzip"],
        "models": ["gcn", "gat", "sage"],
        "splits": ["care_712"],
        "stages": ["base", "teacher", "student"],
    }


def e4_generalization():
    """E4: TFinance & TSocial with bwgnn_424."""
    return {
        "datasets": ["tfinance", "tsocial"],
        "models": ["gcn", "gat", "sage"],
        "splits": ["bwgnn_424"],
        "stages": ["base", "teacher", "student"],
    }


# --- Config generation -----------------------------------------------------

def load_template(stage: str, dataset: str, model: str) -> dict | None:
    """Load an existing config file as template."""
    if stage == "base":
        path = CONFIG_ROOT / "base_detectors" / f"{dataset}_{model}.yaml"
    elif stage == "teacher":
        path = CONFIG_ROOT / "teacher" / "raer_lree" / f"{dataset}_{model}.yaml"
    elif stage == "student":
        path = CONFIG_ROOT / "student" / f"cbr_flash_{dataset}_{model}.yaml"
    else:
        return None
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def apply_split(cfg: dict, split_name: str, scarcity_ratio: float = 1.0) -> dict:
    """Apply split parameters to a config."""
    cfg = copy.deepcopy(cfg)
    sp = SPLIT_DEFS[split_name]
    cfg["dataset"]["train_ratio"] = sp["train_ratio"]
    cfg["dataset"]["val_test_ratio"] = sp["val_test_ratio"]
    cfg["dataset"]["split_mode"] = sp["split_mode"]
    cfg["dataset"]["scarcity_ratio"] = scarcity_ratio
    return cfg


def update_run_names(cfg: dict, stage: str, split_name: str,
                     scarcity_label: str = "") -> dict:
    """Update experiment/run names to reflect the split."""
    split_tag = split_name
    if scarcity_label:
        split_tag = f"{split_name}_{scarcity_label}"

    ds = cfg["dataset"]["name"]
    model = cfg["model"]["name"]

    if stage == "base":
        cfg["experiment"]["run_name"] = f"{ds}_{model}_base_{split_tag}"
    elif stage == "teacher":
        cfg["experiment"]["run_name"] = f"raer_lree_{split_tag}"
        cfg["raer_teacher"]["run_name"] = f"raer_lree_{split_tag}"
    elif stage == "student":
        cfg["experiment"]["run_name"] = f"cbr_flash_{split_tag}"
        cfg["cbr_flash"]["run_name"] = f"cbr_flash_{split_tag}"
    return cfg


def save_config(cfg: dict, exp_dir: str, split_name: str,
                stage: str, dataset: str, model: str,
                scarcity_label: str = "") -> Path:
    """Save config to experiments directory."""
    if stage == "base":
        subdir = "base_detectors"
    elif stage == "teacher":
        subdir = "teacher/raer_lree"
    elif stage == "student":
        subdir = "student"

    split_dir = split_name
    if scarcity_label:
        split_dir = f"{split_name}_{scarcity_label}"

    out_dir = EXP_ROOT / exp_dir / split_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{model}.yaml"

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def generate_group(exp_name: str, datasets: list[str], models: list[str],
                   splits: list[str], stages: list[str],
                   scarcity_ratio: float = 1.0,
                   scarcity_label: str = "") -> list[Path]:
    """Generate configs for a group of experiments."""
    generated = []
    for split_name in splits:
        for stage in stages:
            for ds in datasets:
                for model in models:
                    template = load_template(stage, ds, model)
                    if template is None:
                        print(f"  [skip] no template: {stage}/{ds}/{model}")
                        continue
                    cfg = apply_split(template, split_name, scarcity_ratio)
                    cfg = update_run_names(cfg, stage, split_name, scarcity_label)
                    path = save_config(cfg, exp_name, split_name, stage,
                                       ds, model, scarcity_label)
                    generated.append(path)
    return generated


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate CoVER-FD experiment configs")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        choices=["E1", "E2", "E3", "E4", "all"],
                        help="Which experiment group to generate")
    parser.add_argument("--split", type=str, default=None,
                        choices=list(SPLIT_DEFS.keys()),
                        help="Override split (for E1 single-split mode)")
    parser.add_argument("--scarcity", type=float, default=None,
                        help="Scarcity ratio (for E2)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all experiment groups (E1-E4)")
    args = parser.parse_args()

    if args.all:
        args.experiment = "all"

    total = []

    if args.experiment in ("E1", "all"):
        print("=== E1: Benchmark ===")
        e1 = e1_datasets_models()
        # If --split is given, only generate that split
        splits = [args.split] if args.split else e1["splits"]
        paths = generate_group("E1_benchmark", e1["datasets"], e1["models"],
                               splits, e1["stages"])
        total.extend(paths)
        print(f"  Generated {len(paths)} configs")

    if args.experiment in ("E2", "all"):
        print("=== E2: Scarcity ===")
        for job in e2_scarcity():
            if args.scarcity is not None and abs(job["scarcity_ratio"] - args.scarcity) > 1e-6:
                continue
            paths = generate_group(
                "E2_scarcity", job["datasets"], job["models"],
                [job["split"]], job["stages"],
                scarcity_ratio=job["scarcity_ratio"],
                scarcity_label=job["scarcity_label"],
            )
            total.extend(paths)
            print(f"  scarcity={job['scarcity_ratio']}: {len(paths)} configs")

    if args.experiment in ("E3", "all"):
        print("=== E3: Scaling ===")
        e3 = e3_scaling()
        paths = generate_group("E3_scaling", e3["datasets"], e3["models"],
                               e3["splits"], e3["stages"])
        total.extend(paths)
        print(f"  Generated {len(paths)} configs")

    if args.experiment in ("E4", "all"):
        print("=== E4: Generalization ===")
        e4 = e4_generalization()
        paths = generate_group("E4_generalization", e4["datasets"], e4["models"],
                               e4["splits"], e4["stages"])
        total.extend(paths)
        print(f"  Generated {len(paths)} configs")

    print(f"\nTotal: {len(total)} configs generated under {EXP_ROOT}")


if __name__ == "__main__":
    main()
