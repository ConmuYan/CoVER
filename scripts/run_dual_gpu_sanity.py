"""Dual GPU sanity runner for CoVER-FD.

Supports two modes:
- sequential-split: Stage 1/3 on GPU 2, Qwen Stage 2 on GPU 3
- parallel-datasets: YelpChi on GPU 2, Amazon on GPU 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run_command(cmd: list[str], env: dict[str, str] | None = None, step_name: str = "") -> tuple[bool, float, str]:
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    if env:
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"{'='*60}")

    import os
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.time()
    result = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} failed with return code {result.returncode}")
        print(f"stderr: {result.stderr[-500:]}")
        return False, elapsed, result.stderr

    print(f"\n[OK] {step_name} completed in {elapsed:.2f}s")
    return True, elapsed, result.stdout


def run_sequential_split(yelp_config: str, amazon_config: str, trace_size: int, python: str, project_root: Path):
    print("\n" + "="*60)
    print("Mode: sequential-split")
    print("YelpChi: Stage 1/3 on GPU 2, Stage 2 on GPU 3")
    print("Amazon: Stage 1/3 on GPU 3, Stage 2 on GPU 3")
    print("="*60)

    timings = {}
    results = {}

    train_env_2 = {"CUDA_VISIBLE_DEVICES": "2"}
    train_env_3 = {"CUDA_VISIBLE_DEVICES": "3"}
    llm_env = {"CUDA_VISIBLE_DEVICES": "3"}

    for dataset, config, train_env in [("yelpchi", yelp_config, train_env_2), ("amazon", amazon_config, train_env_3)]:
        print(f"\n{'#'*60}")
        print(f"Processing: {dataset}")
        print(f"{'#'*60}")

        config_path = Path(config)
        if not config_path.exists():
            print(f"Config not found: {config}")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_path = cfg.get("dataset", {}).get("path", "")
        if data_path and not Path(data_path).exists():
            print(f"Data path not found: {data_path}")
            continue

        dataset_timings = {}

        cmd = [python, str(project_root / "scripts" / "train_stage1.py"), "--config", config]
        ok, elapsed, _ = run_command(cmd, train_env, f"Stage 1: {dataset}")
        dataset_timings["stage1"] = elapsed
        if not ok:
            results[dataset] = {"success": False, "error": "stage1 failed"}
            continue

        cmd = [python, str(project_root / "scripts" / "generate_stage2_err.py"),
               "--config", config, "--teacher", "rule", "--trace_size", str(trace_size)]
        ok, elapsed, _ = run_command(cmd, train_env, f"Stage 2: {dataset}")
        dataset_timings["stage2"] = elapsed
        if not ok:
            results[dataset] = {"success": False, "error": "stage2 failed"}
            continue

        cmd = [python, str(project_root / "scripts" / "train_stage3.py"), "--config", config]
        ok, elapsed, _ = run_command(cmd, train_env, f"Stage 3: {dataset}")
        dataset_timings["stage3"] = elapsed
        if not ok:
            results[dataset] = {"success": False, "error": "stage3 failed"}
            continue

        cmd = [python, str(project_root / "scripts" / "evaluate.py"), "--config", config, "--stage", "stage1"]
        ok, elapsed, _ = run_command(cmd, train_env, f"Evaluate Stage 1: {dataset}")
        dataset_timings["eval_stage1"] = elapsed

        cmd = [python, str(project_root / "scripts" / "evaluate.py"), "--config", config, "--stage", "stage3"]
        ok, elapsed, _ = run_command(cmd, train_env, f"Evaluate Stage 3: {dataset}")
        dataset_timings["eval_stage3"] = elapsed

        cmd = [python, str(project_root / "scripts" / "report_evidence_quality.py"), "--config", config]
        run_command(cmd, None, f"Evidence Quality Report: {dataset}")

        timings[dataset] = dataset_timings
        results[dataset] = {"success": True, "timings": dataset_timings}

    return timings, results


def run_parallel_datasets(yelp_config: str, amazon_config: str, trace_size: int, python: str, project_root: Path):
    print("\n" + "="*60)
    print("Mode: parallel-datasets")
    print("YelpChi on GPU 2, Amazon on GPU 3 (concurrent)")
    print("="*60)

    for config in [yelp_config, amazon_config]:
        config_path = Path(config)
        if not config_path.exists():
            print(f"Config not found: {config}")
            return {}, {}

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_path = cfg.get("dataset", {}).get("path", "")
        if data_path and not Path(data_path).exists():
            print(f"Data path not found: {data_path}")
            return {}, {}

    import os

    yelp_env = {"CUDA_VISIBLE_DEVICES": "2"}
    amazon_env = {"CUDA_VISIBLE_DEVICES": "3"}

    yelp_cmd = [
        python, str(project_root / "scripts" / "run_real_sanity.py"),
        "--config", yelp_config, "--teacher", "rule",
        "--trace_size", str(trace_size), "--train_gpus", "2",
    ]

    amazon_cmd = [
        python, str(project_root / "scripts" / "run_real_sanity.py"),
        "--config", amazon_config, "--teacher", "rule",
        "--trace_size", str(trace_size), "--train_gpus", "3",
    ]

    print(f"\nStarting YelpChi on GPU 2...")
    print(f"Starting Amazon on GPU 3...")

    start = time.time()

    yelp_full_env = os.environ.copy()
    yelp_full_env.update(yelp_env)
    amazon_full_env = os.environ.copy()
    amazon_full_env.update(amazon_env)

    yelp_proc = subprocess.Popen(yelp_cmd, env=yelp_full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    amazon_proc = subprocess.Popen(amazon_cmd, env=amazon_full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    yelp_stdout, yelp_stderr = yelp_proc.communicate()
    amazon_stdout, amazon_stderr = amazon_proc.communicate()

    elapsed = time.time() - start

    results = {}
    timings = {"total_parallel": elapsed}

    if yelp_proc.returncode == 0:
        print(f"\n[OK] YelpChi completed in parallel run")
        results["yelpchi"] = {"success": True}
    else:
        print(f"\n[FAILED] YelpChi failed with return code {yelp_proc.returncode}")
        print(f"stderr: {yelp_stderr[-500:]}")
        results["yelpchi"] = {"success": False, "error": yelp_stderr[-500:]}

    if amazon_proc.returncode == 0:
        print(f"\n[OK] Amazon completed in parallel run")
        results["amazon"] = {"success": True}
    else:
        print(f"\n[FAILED] Amazon failed with return code {amazon_proc.returncode}")
        print(f"stderr: {amazon_stderr[-500:]}")
        results["amazon"] = {"success": False, "error": amazon_stderr[-500:]}

    return timings, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp_config", type=str, default="configs/yelpchi_bwgnn.yaml")
    parser.add_argument("--amazon_config", type=str, default="configs/amazon_bwgnn.yaml")
    parser.add_argument("--mode", type=str, default="sequential-split", choices=["sequential-split", "parallel-datasets"])
    parser.add_argument("--trace_size", type=int, default=32)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    python = sys.executable

    start_time = time.time()

    if args.mode == "sequential-split":
        timings, results = run_sequential_split(args.yelp_config, args.amazon_config, args.trace_size, python, project_root)
    elif args.mode == "parallel-datasets":
        timings, results = run_parallel_datasets(args.yelp_config, args.amazon_config, args.trace_size, python, project_root)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    total_time = time.time() - start_time

    report_dir = Path("artifacts") / "reports" / "dual_gpu"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "mode": args.mode,
        "trace_size": args.trace_size,
        "yelp_config": args.yelp_config,
        "amazon_config": args.amazon_config,
        "timings": timings,
        "results": results,
        "total_runtime_seconds": total_time,
    }

    with open(report_dir / "dual_gpu_sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Dual GPU Sanity Report",
        "",
        f"Mode: {args.mode}",
        f"Trace size: {args.trace_size}",
        "",
        "## Results",
    ]

    for dataset, result in results.items():
        status = "PASS" if result.get("success") else "FAIL"
        md_lines.append(f"- {dataset}: {status}")

    md_lines.extend(["", "## Timings"])
    for key, value in timings.items():
        md_lines.append(f"- {key}: {value:.2f}s")
    md_lines.append(f"- Total: {total_time:.2f}s")

    with open(report_dir / "dual_gpu_sanity_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print("Dual GPU Sanity Summary")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    for dataset, result in results.items():
        status = "PASS" if result.get("success") else "FAIL"
        print(f"  {dataset}: {status}")
    print(f"Total: {total_time:.2f}s")
    print(f"\nReport saved to: {report_dir}")

    all_success = all(r.get("success") for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
