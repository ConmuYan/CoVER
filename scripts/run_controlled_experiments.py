"""Run controlled multi-seed experiments for CoVER-FD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import teacher_to_run_name, ensure_dir


def run_command(cmd: list[str], env: dict[str, str] | None = None, step_name: str = "", dry_run: bool = False) -> tuple[bool, float]:
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    if env:
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True, 0.0

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.time()
    result = subprocess.run(cmd, env=full_env, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} failed with return code {result.returncode}")
        return False, elapsed

    print(f"\n[OK] {step_name} completed in {elapsed:.2f}s")
    return True, elapsed


def check_stage1_complete(dataset: str, model: str, seed: int) -> bool:
    from utils.paths import get_base_checkpoint_path, get_stage1_metrics_path
    ckpt = get_base_checkpoint_path(dataset, model, seed)
    metrics = get_stage1_metrics_path(dataset, model, seed)
    return ckpt.exists() and metrics.exists()


def check_stage3_complete(dataset: str, model: str, run_name: str, seed: int) -> bool:
    from utils.paths import get_reasoner_checkpoint_path, get_stage3_metrics_path
    ckpt = get_reasoner_checkpoint_path(dataset, model, run_name, seed)
    metrics = get_stage3_metrics_path(dataset, model, run_name, seed)
    return ckpt.exists() and metrics.exists()


def check_err_complete(dataset: str, model: str, run_name: str, seed: int) -> bool:
    from utils.paths import get_accepted_err_path, get_verifier_stats_path
    accepted = get_accepted_err_path(dataset, model, run_name, seed)
    stats = get_verifier_stats_path(dataset, model, run_name, seed)
    return accepted.exists() and stats.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["yelpchi", "amazon"])
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--teachers", nargs="+", default=["rule", "qwen"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789, 42, 2026])
    parser.add_argument("--rule_trace_size", type=int, default=200)
    parser.add_argument("--qwen_trace_size", type=int, default=32)
    parser.add_argument("--train_gpus", type=str, default="2")
    parser.add_argument("--llm_gpus", type=str, default="3")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_rule", action="store_true")
    parser.add_argument("--skip_qwen", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--confirm_large_qwen_run", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if "qwen" in args.teachers and args.qwen_trace_size > 64 and not args.confirm_large_qwen_run:
        print(f"Error: qwen_trace_size={args.qwen_trace_size} > 64.")
        print(f"Use --confirm_large_qwen_run to proceed.")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    python = sys.executable

    train_env = {"CUDA_VISIBLE_DEVICES": args.train_gpus}
    llm_env = {"CUDA_VISIBLE_DEVICES": args.llm_gpus}

    configs = {
        "yelpchi": "configs/yelpchi_bwgnn.yaml",
        "amazon": "configs/amazon_bwgnn.yaml",
    }

    all_results = {}
    total_start = time.time()

    for dataset in args.datasets:
        config = configs.get(dataset)
        if not config:
            print(f"Unknown dataset: {dataset}")
            continue

        config_path = project_root / config
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_path = cfg.get("dataset", {}).get("path", "")
        if data_path and not Path(data_path).exists():
            print(f"Data path not found: {data_path}")
            continue

        for seed in args.seeds:
            print(f"\n{'#'*60}")
            print(f"Dataset: {dataset}, Seed: {seed}")
            print(f"{'#'*60}")

            seed_results = {"dataset": dataset, "seed": seed, "timings": {}}

            if not args.skip_stage1:
                if check_stage1_complete(dataset, args.model, seed):
                    print(f"[SKIP] Stage 1 already complete")
                else:
                    cmd = [
                        python, str(project_root / "scripts" / "train_stage1.py"),
                        "--config", str(config_path),
                        "--run_name", "base",
                        "--seed", str(seed),
                    ]
                    ok, elapsed = run_command(cmd, train_env, f"Stage 1: {dataset} seed={seed}", args.dry_run)
                    seed_results["timings"]["stage1"] = elapsed
                    if not ok:
                        all_results[f"{dataset}_{seed}"] = seed_results
                        continue

            if "rule" in args.teachers and not args.skip_rule:
                run_name = "rule"
                if not args.skip_stage3 or not check_err_complete(dataset, args.model, run_name, seed):
                    cmd = [
                        python, str(project_root / "scripts" / "generate_stage2_err.py"),
                        "--config", str(config_path),
                        "--teacher", "rule",
                        "--trace_size", str(args.rule_trace_size),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    ok, elapsed = run_command(cmd, train_env, f"Stage 2 Rule: {dataset} seed={seed}", args.dry_run)
                    seed_results["timings"]["stage2_rule"] = elapsed
                    if not ok:
                        all_results[f"{dataset}_{seed}"] = seed_results
                        continue

                if not args.skip_stage3:
                    if check_stage3_complete(dataset, args.model, run_name, seed):
                        print(f"[SKIP] Stage 3 rule already complete")
                    else:
                        cmd = [
                            python, str(project_root / "scripts" / "train_stage3.py"),
                            "--config", str(config_path),
                            "--run_name", run_name,
                            "--seed", str(seed),
                        ]
                        ok, elapsed = run_command(cmd, train_env, f"Stage 3 Rule: {dataset} seed={seed}", args.dry_run)
                        seed_results["timings"]["stage3_rule"] = elapsed
                        if not ok:
                            all_results[f"{dataset}_{seed}"] = seed_results
                            continue

                if not args.skip_eval:
                    cmd = [
                        python, str(project_root / "scripts" / "evaluate.py"),
                        "--config", str(config_path),
                        "--stage", "stage3",
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    ok, elapsed = run_command(cmd, train_env, f"Eval Stage 3 Rule: {dataset} seed={seed}", args.dry_run)
                    seed_results["timings"]["eval_stage3_rule"] = elapsed

                    cmd = [
                        python, str(project_root / "scripts" / "check_run_integrity.py"),
                        "--config", str(config_path),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    run_command(cmd, None, f"Integrity Check Rule: {dataset} seed={seed}", args.dry_run)

                    cmd = [
                        python, str(project_root / "scripts" / "compare_stage1_stage3.py"),
                        "--config", str(config_path),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    run_command(cmd, None, f"Compare Stage1 vs Stage3 Rule: {dataset} seed={seed}", args.dry_run)

            if "qwen" in args.teachers and not args.skip_qwen:
                run_name = "qwen"
                if not args.skip_stage3 or not check_err_complete(dataset, args.model, run_name, seed):
                    cmd = [
                        python, str(project_root / "scripts" / "generate_stage2_err.py"),
                        "--config", str(config_path),
                        "--teacher", "llm",
                        "--trace_size", str(args.qwen_trace_size),
                        "--run_name", run_name,
                        "--seed", str(seed),
                        "--enable_retry",
                    ]
                    ok, elapsed = run_command(cmd, llm_env, f"Stage 2 Qwen: {dataset} seed={seed}", args.dry_run)
                    seed_results["timings"]["stage2_qwen"] = elapsed
                    if not ok:
                        all_results[f"{dataset}_{seed}"] = seed_results
                        continue

                if not args.skip_stage3:
                    if check_stage3_complete(dataset, args.model, run_name, seed):
                        print(f"[SKIP] Stage 3 qwen already complete")
                    else:
                        cmd = [
                            python, str(project_root / "scripts" / "train_stage3.py"),
                            "--config", str(config_path),
                            "--run_name", run_name,
                            "--seed", str(seed),
                        ]
                        ok, elapsed = run_command(cmd, train_env, f"Stage 3 Qwen: {dataset} seed={seed}", args.dry_run)
                        seed_results["timings"]["stage3_qwen"] = elapsed
                        if not ok:
                            all_results[f"{dataset}_{seed}"] = seed_results
                            continue

                if not args.skip_eval:
                    cmd = [
                        python, str(project_root / "scripts" / "evaluate.py"),
                        "--config", str(config_path),
                        "--stage", "stage3",
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    ok, elapsed = run_command(cmd, train_env, f"Eval Stage 3 Qwen: {dataset} seed={seed}", args.dry_run)
                    seed_results["timings"]["eval_stage3_qwen"] = elapsed

                    cmd = [
                        python, str(project_root / "scripts" / "check_run_integrity.py"),
                        "--config", str(config_path),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    run_command(cmd, None, f"Integrity Check Qwen: {dataset} seed={seed}", args.dry_run)

                    cmd = [
                        python, str(project_root / "scripts" / "report_evidence_quality.py"),
                        "--config", str(config_path),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    run_command(cmd, None, f"Evidence Quality Report Qwen: {dataset} seed={seed}", args.dry_run)

                    cmd = [
                        python, str(project_root / "scripts" / "compare_stage1_stage3.py"),
                        "--config", str(config_path),
                        "--run_name", run_name,
                        "--seed", str(seed),
                    ]
                    run_command(cmd, None, f"Compare Stage1 vs Stage3 Qwen: {dataset} seed={seed}", args.dry_run)

            cmd = [
                python, str(project_root / "scripts" / "check_split_sanity.py"),
                "--dataset", dataset,
                "--seed", str(seed),
            ]
            run_command(cmd, None, f"Split Sanity: {dataset} seed={seed}", args.dry_run)

            all_results[f"{dataset}_{seed}"] = seed_results

    total_elapsed = time.time() - total_start

    report_dir = ensure_dir(Path("artifacts") / "reports" / "controlled_experiments")
    report = {
        "datasets": args.datasets,
        "model": args.model,
        "teachers": args.teachers,
        "seeds": args.seeds,
        "rule_trace_size": args.rule_trace_size,
        "qwen_trace_size": args.qwen_trace_size,
        "train_gpus": args.train_gpus,
        "llm_gpus": args.llm_gpus,
        "dry_run": args.dry_run,
        "total_runtime_seconds": total_elapsed,
        "results": all_results,
    }

    with open(report_dir / "controlled_experiments_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("Controlled Experiments Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Report saved to: {report_dir}")


if __name__ == "__main__":
    main()
