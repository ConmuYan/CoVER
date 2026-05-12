from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/yelpchi_gcn.yaml",
    "configs/yelpchi_sage.yaml",
    "configs/yelpchi_gat.yaml",
    "configs/yelpchi_bwgnn.yaml",
]


def run_command(cmd: list[str], continue_on_error: bool = False) -> tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        if not continue_on_error:
            print(f"\n[FAILED] Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return False, result.stderr
    return True, ""


def load_verifier_stats(config_path: Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    seed = config["train"]["seed"]

    stats_path = Path("artifacts") / "err_cache" / dataset / model / f"seed_{seed}" / "verifier_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


def load_metrics(config_path: Path, stage: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    seed = config["train"]["seed"]

    metrics_path = Path("artifacts") / "results" / dataset / model / f"seed_{seed}" / f"{stage}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    python = sys.executable

    results = []

    for config_path in args.configs:
        config_full = project_root / config_path
        if not config_full.exists():
            print(f"Config not found: {config_path}")
            continue

        with open(config_full) as f:
            config = yaml.safe_load(f)

        model_name = config["model"]["name"]
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        start_time = time.time()
        passed = True

        cmd = [python, str(project_root / "scripts" / "run_full_pipeline.py"), "--config", str(config_full)]
        if args.debug:
            cmd.append("--debug")
        ok, err = run_command(cmd, args.continue_on_error)
        if not ok:
            passed = False

        if passed:
            cmd = [python, str(project_root / "scripts" / "check_run_integrity.py"), "--config", str(config_full), "--run_name", "rule"]
            ok, err = run_command(cmd, args.continue_on_error)
            if not ok:
                passed = False

        elapsed = time.time() - start_time
        verifier_stats = load_verifier_stats(config_full) if passed else {}
        stage1_metrics = load_metrics(config_full, "stage1") if passed else {}
        stage3_metrics = load_metrics(config_full, "stage3") if passed else {}

        result = {
            "model": model_name,
            "config": config_path,
            "passed": passed,
            "elapsed": elapsed,
            "verifier_stats": verifier_stats,
            "stage1_metrics": stage1_metrics,
            "stage3_metrics": stage3_metrics,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        acceptance_rate = verifier_stats.get("acceptance_rate", 0)
        print(f"\nResult: {status} | Time: {elapsed:.1f}s | Acceptance: {acceptance_rate:.1%}")

    report_dir = Path("artifacts") / "reports" / "yelpchi"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "smoke_test_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "# Model Smoke Test Summary",
        "",
        "| Model | Status | Time | Acceptance Rate |",
        "|-------|--------|------|-----------------|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        acceptance = r["verifier_stats"].get("acceptance_rate", 0)
        md_lines.append(f"| {r['model']} | {status} | {r['elapsed']:.1f}s | {acceptance:.1%} |")

    with open(report_dir / "smoke_test_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print("Smoke Test Summary")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        acceptance = r["verifier_stats"].get("acceptance_rate", 0)
        print(f"  {r['model']}: {status} ({r['elapsed']:.1f}s, {acceptance:.1%})")

    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
