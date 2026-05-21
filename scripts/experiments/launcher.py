#!/usr/bin/env python3
"""Simple multi-GPU experiment launcher for CoVER-FD.

Usage:
    python scripts/experiments/launcher.py --gpus 1 2 --experiment E1 --split care_712
    python scripts/experiments/launcher.py --gpus 2 --experiment E2
    python scripts/experiments/launcher.py --gpus 0 1 2 3 --experiment E1 --split care_712 --stage base
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
SEEDS = [42, 123, 456, 789, 2026]
MODELS_ALL = ["bwgnn", "sage", "gcn", "gat"]
MODELS_LIGHT = ["gcn", "gat", "sage"]


def make_jobs(experiment: str, split: str | None = None,
              stage_filter: str | None = None) -> list[dict]:
    """Build the job list for a given experiment."""
    jobs = []

    if experiment == "E1":
        datasets = ["yelpchi", "amazon"]
        models = MODELS_ALL
        splits = [split] if split else ["care_712", "bwgnn_semi"]
        stages = ["base", "teacher", "student"]
        config_root = ROOT / "configs/raer_fd/experiments/E1_benchmark"

        for sp in splits:
            for st in stages:
                if stage_filter and st != stage_filter:
                    continue
                for ds in datasets:
                    for m in models:
                        cfg = _config_path(config_root, sp, st, ds, m)
                        if cfg.exists():
                            jobs.append({
                                "config": str(cfg),
                                "dataset": ds, "model": m,
                                "split": sp, "stage": st,
                                "seeds": SEEDS,
                            })

    elif experiment == "E2":
        datasets = ["yelpchi"]
        models = MODELS_ALL
        stages = ["base", "teacher", "student"]
        config_root = ROOT / "configs/raer_fd/experiments/E2_scarcity"

        for pct in [1, 5, 10, 20, 50]:
            sp = f"care_712_scarcity_{pct}pct"
            for st in stages:
                if stage_filter and st != stage_filter:
                    continue
                for m in models:
                    cfg = _config_path(config_root, sp, st, datasets[0], m)
                    if cfg.exists():
                        jobs.append({
                            "config": str(cfg),
                            "dataset": datasets[0], "model": m,
                            "split": sp, "stage": st,
                            "seeds": SEEDS,
                        })

    elif experiment == "E3":
        datasets = ["yelpnyc", "yelpzip"]
        models = MODELS_LIGHT
        stages = ["base", "teacher", "student"]
        config_root = ROOT / "configs/raer_fd/experiments/E3_scaling/care_712"

        for st in stages:
            if stage_filter and st != stage_filter:
                continue
            for ds in datasets:
                for m in models:
                    cfg = _config_path(config_root, "care_712", st, ds, m)
                    if cfg.exists():
                        jobs.append({
                            "config": str(cfg),
                            "dataset": ds, "model": m,
                            "split": "care_712", "stage": st,
                            "seeds": SEEDS,
                        })

    elif experiment == "E4":
        datasets = ["tfinance", "tsocial"]
        models = MODELS_LIGHT
        stages = ["base", "teacher", "student"]
        config_root = ROOT / "configs/raer_fd/experiments/E4_generalization/bwgnn_424"

        for st in stages:
            if stage_filter and st != stage_filter:
                continue
            for ds in datasets:
                for m in models:
                    cfg = _config_path(config_root, "bwgnn_424", st, ds, m)
                    if cfg.exists():
                        jobs.append({
                            "config": str(cfg),
                            "dataset": ds, "model": m,
                            "split": "bwgnn_424", "stage": st,
                            "seeds": SEEDS,
                        })

    elif experiment == "E6":
        dataset = "yelpchi"
        models = MODELS_LIGHT
        stages = ["base", "teacher"]
        config_root_e1 = ROOT / "configs/raer_fd/experiments/E1_benchmark/care_712"
        config_root_hc = ROOT / "configs/raer_fd/teacher/raer_hc"
        config_root_lree = ROOT / "configs/raer_fd/experiments/E1_benchmark/care_712/teacher/raer_lree"

        # Freeze-Base variant
        for m in models:
            cfg = config_root_e1 / "base_detectors" / f"{dataset}_{m}.yaml"
            if cfg.exists():
                jobs.append({
                    "config": str(cfg), "dataset": dataset, "model": m,
                    "split": "care_712", "stage": "base", "variant": "freeze",
                    "seeds": SEEDS,
                })
        # RAER-HC variant
        for m in models:
            cfg = config_root_hc / f"{dataset}_{m}.yaml"
            if cfg.exists():
                jobs.append({
                    "config": str(cfg), "dataset": dataset, "model": m,
                    "split": "care_712", "stage": "teacher_hc", "variant": "hc",
                    "seeds": SEEDS,
                })
        # RAER-LREE variant
        for m in models:
            cfg = config_root_lree / f"{dataset}_{m}.yaml"
            if cfg.exists():
                jobs.append({
                    "config": str(cfg), "dataset": dataset, "model": m,
                    "split": "care_712", "stage": "teacher_lree", "variant": "lree",
                    "seeds": SEEDS,
                })

    return jobs


def _config_path(root: Path, split: str, stage: str, ds: str, model: str) -> Path:
    if stage == "base":
        return root / split / "base_detectors" / f"{ds}_{model}.yaml"
    elif stage == "teacher":
        return root / split / "teacher" / "raer_lree" / f"{ds}_{model}.yaml"
    elif stage == "student":
        return root / split / "student" / f"{ds}_{model}.yaml"
    return root / "nonexistent"


def run_job(job: dict, gpu: int) -> dict:
    """Execute a single (config, seed) batch on a specific GPU."""
    config = job["config"]
    dataset = job["dataset"]
    model = job["model"]
    stage = job["stage"]
    seeds = job["seeds"]
    results = []

    for seed in seeds:
        ckpt_base = ROOT / "artifacts" / "checkpoints" / dataset / model

        # Skip if result already exists
        if stage == "base":
            result_path = ckpt_base / "base" / f"seed_{seed}" / "base.pt"
        elif stage in ("teacher", "teacher_hc", "teacher_lree"):
            run_tag = "raer_hc" if "hc" in stage else "raer_lree"
            result_path = ckpt_base / run_tag / f"seed_{seed}" / "raer_teacher.pt"
        elif stage == "student":
            result_path = ckpt_base / "cbr_flash" / f"seed_{seed}" / "cbr_flash_student.pt"
        else:
            result_path = Path("/nonexistent")

        if result_path.exists():
            print(f"  [skip] {dataset}/{model}/{stage} seed={seed} (exists)")
            results.append({"seed": seed, "status": "skipped"})
            continue

        # Build command
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if stage == "base":
            cmd = [PYTHON, "scripts/train_base_detector.py",
                   "--config", config, "--seed", str(seed),
                   "--run_name", "base", "--stratified"]
        elif stage in ("teacher", "teacher_lree"):
            base_ckpt = str(ckpt_base / "base" / f"seed_{seed}" / "base.pt")
            cmd = [PYTHON, "scripts/train_raer_teacher.py",
                   "--config", config, "--seed", str(seed),
                   "--device", "cuda:0", "--run_name", "raer_lree",
                   "--base_ckpt_path", base_ckpt]
        elif stage == "teacher_hc":
            base_ckpt = str(ckpt_base / "base" / f"seed_{seed}" / "base.pt")
            cmd = [PYTHON, "scripts/train_raer_teacher.py",
                   "--config", config, "--seed", str(seed),
                   "--device", "cuda:0", "--run_name", "raer_hc",
                   "--base_ckpt_path", base_ckpt]
        elif stage == "student":
            base_ckpt = str(ckpt_base / "base" / f"seed_{seed}" / "base.pt")
            teacher_ckpt = str(ckpt_base / "raer_lree" / f"seed_{seed}" / "raer_teacher.pt")
            lree_ckpt = str(ckpt_base / "raer_lree" / f"seed_{seed}" / "lree.pt")
            cmd = [PYTHON, "scripts/train_cbr_flash.py",
                   "--config", config, "--seed", str(seed),
                   "--device", "cuda:0", "--run_name", "cbr_flash",
                   "--teacher_ckpt", teacher_ckpt,
                   "--base_ckpt_path", base_ckpt]
            if Path(lree_ckpt).exists():
                cmd.extend(["--teacher_extractor_ckpt", lree_ckpt])
        else:
            results.append({"seed": seed, "status": "unknown_stage"})
            continue

        t0 = time.time()
        print(f"  [GPU {gpu}] {dataset}/{model}/{stage} seed={seed}")
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=1800,
                cwd=str(ROOT),
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                results.append({"seed": seed, "status": "ok", "time": elapsed})
                print(f"    OK ({elapsed:.0f}s)")
            else:
                results.append({"seed": seed, "status": "error", "time": elapsed,
                                "stderr_tail": proc.stderr[-200:] if proc.stderr else ""})
                print(f"    ERROR ({elapsed:.0f}s): {proc.stderr[-100:] if proc.stderr else ''}")
        except subprocess.TimeoutExpired:
            results.append({"seed": seed, "status": "timeout"})
            print(f"    TIMEOUT")

    return {"job": job, "gpu": gpu, "results": results}


def main():
    parser = argparse.ArgumentParser(description="CoVER-FD experiment launcher")
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="Physical GPU IDs to use (e.g., 1 2)")
    parser.add_argument("--experiment", "-e", required=True,
                        choices=["E1", "E2", "E3", "E4", "E6"])
    parser.add_argument("--split", type=str, default=None,
                        help="Split filter (for E1: care_712 or bwgnn_semi)")
    parser.add_argument("--stage", type=str, default=None,
                        help="Stage filter (base, teacher, student)")
    parser.add_argument("--max-per-gpu", type=int, default=1,
                        help="Max concurrent jobs per GPU")
    args = parser.parse_args()

    jobs = make_jobs(args.experiment, args.split, args.stage)
    if not jobs:
        print("No jobs to run.")
        return

    print(f"=== Experiment {args.experiment} ===")
    print(f"GPUs: {args.gpus}")
    print(f"Jobs: {len(jobs)}")

    # Group by stage for sequential execution
    stage_order = {"base": 0, "teacher": 1, "teacher_hc": 1, "teacher_lree": 1, "student": 2}
    stage_groups = {}
    for j in jobs:
        st = j["stage"]
        stage_groups.setdefault(st, []).append(j)

    for st in sorted(stage_groups, key=lambda s: stage_order.get(s, 0)):
        stg_jobs = stage_groups[st]
        print(f"\n--- Stage: {st} ({len(stg_jobs)} jobs) ---")

        # Distribute jobs across GPUs
        gpu_count = len(args.gpus)
        n_ok = 0
        n_err = 0

        with ProcessPoolExecutor(max_workers=gpu_count) as pool:
            futures = {}
            for i, job in enumerate(stg_jobs):
                gpu = args.gpus[i % gpu_count]
                f = pool.submit(run_job, job, gpu)
                futures[f] = job

            for f in as_completed(futures):
                result = f.result()
                for r in result["results"]:
                    if r["status"] == "ok" or r["status"] == "skipped":
                        n_ok += 1
                    else:
                        n_err += 1

        print(f"  Stage {st}: {n_ok} ok, {n_err} errors")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
