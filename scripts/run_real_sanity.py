from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run_command(cmd: list[str], env: dict[str, str] | None = None, step_name: str = "") -> tuple[bool, float]:
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    if env:
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"{'='*60}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="rule", choices=["rule", "llm"])
    parser.add_argument("--trace_size", type=int, default=None)
    parser.add_argument("--train_gpus", type=str, default=None)
    parser.add_argument("--llm_gpus", type=str, default=None)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow_missing_data", action="store_true")
    parser.add_argument("--confirm_large_llm_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = args.seed or config["train"]["seed"]
    data_path = config["dataset"].get("path", "")

    train_gpus = args.train_gpus or config.get("gpu", {}).get("train_visible_devices", "2")
    llm_gpus = args.llm_gpus or config.get("gpu", {}).get("llm_visible_devices", "3")

    if args.trace_size is None:
        if args.teacher == "llm":
            args.trace_size = config.get("llm", {}).get("max_trace_nodes_small", 16)
        else:
            args.trace_size = config.get("evidence", {}).get("trace_size_small", 32)

    if args.teacher == "llm" and args.trace_size > 64 and not args.confirm_large_llm_run:
        print(f"Error: trace_size={args.trace_size} > 64 for LLM teacher.")
        print(f"Use --confirm_large_llm_run to proceed.")
        sys.exit(1)

    if not args.allow_missing_data and data_path and not Path(data_path).exists():
        print(f"Error: Data path not found: {data_path}")
        print(f"Use --allow_missing_data to proceed with synthetic data.")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    python = sys.executable
    timings = {}

    train_env = build_gpu_env(train_gpus)
    llm_env = build_gpu_env(llm_gpus)

    if not args.skip_stage1:
        cmd = [python, str(project_root / "scripts" / "train_stage1.py"), "--config", args.config]
        ok, elapsed = run_command(cmd, train_env, "Stage 1: Train base detector")
        timings["stage1"] = elapsed
        if not ok:
            sys.exit(1)

    if not args.skip_stage2:
        cmd = [python, str(project_root / "scripts" / "generate_stage2_err.py"),
               "--config", args.config, "--teacher", args.teacher,
               "--trace_size", str(args.trace_size)]
        if args.teacher == "llm":
            cmd.extend(["--enable_retry"])
        env = llm_env if args.teacher == "llm" else train_env
        ok, elapsed = run_command(cmd, env, f"Stage 2: Generate ERR ({args.teacher})")
        timings["stage2"] = elapsed
        if not ok:
            sys.exit(1)

    if not args.skip_stage3:
        cmd = [python, str(project_root / "scripts" / "train_stage3.py"), "--config", args.config]
        ok, elapsed = run_command(cmd, train_env, "Stage 3: Train reasoner")
        timings["stage3"] = elapsed
        if not ok:
            sys.exit(1)

    if not args.skip_eval:
        cmd = [python, str(project_root / "scripts" / "evaluate.py"),
               "--config", args.config, "--stage", "stage1"]
        ok, elapsed = run_command(cmd, train_env, "Evaluate Stage 1")
        timings["eval_stage1"] = elapsed

        cmd = [python, str(project_root / "scripts" / "evaluate.py"),
               "--config", args.config, "--stage", "stage3"]
        ok, elapsed = run_command(cmd, train_env, "Evaluate Stage 3")
        timings["eval_stage3"] = elapsed

    cmd = [python, str(project_root / "scripts" / "check_run_integrity.py"), "--config", args.config]
    run_command(cmd, None, "Integrity Check")

    cmd = [python, str(project_root / "scripts" / "compare_stage1_stage3.py"), "--config", args.config]
    run_command(cmd, None, "Compare Stage 1 vs Stage 3")

    cmd = [python, str(project_root / "scripts" / "report_evidence_quality.py"), "--config", args.config]
    run_command(cmd, None, "Evidence Quality Report")

    total_time = sum(timings.values())

    report_dir = Path("artifacts") / "reports" / dataset_name / model_name / f"seed_{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1_metrics = load_metrics(report_dir.parent.parent.parent / "results" / dataset_name / model_name / f"seed_{seed}" / "stage1_metrics.json")
    stage3_metrics = load_metrics(report_dir.parent.parent.parent / "results" / dataset_name / model_name / f"seed_{seed}" / "stage3_metrics.json")
    verifier_stats = load_json(report_dir.parent.parent.parent / "err_cache" / dataset_name / model_name / f"seed_{seed}" / "verifier_stats.json")

    report = {
        "dataset": dataset_name,
        "model": model_name,
        "teacher": args.teacher,
        "trace_size": args.trace_size,
        "seed": seed,
        "split_mode": config["dataset"].get("split_mode", "supervised"),
        "train_ratio": config["dataset"].get("train_ratio", 0.4),
        "val_test_ratio": config["dataset"].get("val_test_ratio", [1, 2]),
        "train_gpus": train_gpus,
        "llm_gpus": llm_gpus,
        "stage1_metrics": stage1_metrics,
        "stage3_metrics": stage3_metrics,
        "verifier_stats": verifier_stats,
        "total_runtime_seconds": total_time,
        "per_stage_runtime_seconds": timings,
    }

    report_name = f"real_sanity_{args.teacher}_trace{args.trace_size}"
    with open(report_dir / f"{report_name}.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Real Sanity Report: {args.teacher} teacher, trace_size={args.trace_size}",
        "",
        f"Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}",
        f"Train GPUs: {train_gpus} | LLM GPUs: {llm_gpus}",
        "",
        "## Stage 1 Metrics",
    ]
    if stage1_metrics:
        for k, v in stage1_metrics.items():
            md_lines.append(f"- {k}: {v:.4f}")

    md_lines.extend(["", "## Stage 3 Metrics"])
    if stage3_metrics:
        for k, v in stage3_metrics.items():
            md_lines.append(f"- {k}: {v:.4f}")

    md_lines.extend(["", "## Runtime"])
    for stage, t in timings.items():
        md_lines.append(f"- {stage}: {t:.2f}s")
    md_lines.append(f"- Total: {total_time:.2f}s")

    with open(report_dir / f"{report_name}.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print("Real Sanity Summary")
    print(f"{'='*60}")
    for stage, t in timings.items():
        print(f"  {stage}: {t:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"\nReport saved to: {report_dir / report_name}.json")


def build_gpu_env(visible_devices: str) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": visible_devices}


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    main()
