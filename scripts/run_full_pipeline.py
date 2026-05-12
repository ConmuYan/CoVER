from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], step_name: str) -> float:
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] {step_name} completed in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_stage3", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    python = sys.executable

    timings = {}

    if not args.skip_stage1:
        cmd = [python, str(project_root / "scripts" / "train_stage1.py"), "--config", args.config, "--run_name", "base"]
        if args.debug:
            cmd.append("--debug")
        timings["stage1_train"] = run_command(cmd, "Stage 1: Train base detector")

    if not args.skip_stage2:
        cmd = [python, str(project_root / "scripts" / "generate_stage2_err.py"), "--config", args.config, "--teacher", "rule", "--run_name", "rule"]
        if args.debug:
            cmd.append("--debug")
        timings["stage2_err"] = run_command(cmd, "Stage 2: Generate ERR")

    if not args.skip_stage3:
        cmd = [python, str(project_root / "scripts" / "train_stage3.py"), "--config", args.config, "--run_name", "rule"]
        if args.debug:
            cmd.append("--debug")
        timings["stage3_train"] = run_command(cmd, "Stage 3: Train reasoner")

    if not args.skip_stage1:
        cmd = [python, str(project_root / "scripts" / "evaluate.py"), "--config", args.config, "--stage", "stage1", "--run_name", "base"]
        if args.debug:
            cmd.append("--debug")
        timings["eval_stage1"] = run_command(cmd, "Evaluate Stage 1")

    if not args.skip_stage3:
        cmd = [python, str(project_root / "scripts" / "evaluate.py"), "--config", args.config, "--stage", "stage3", "--run_name", "rule"]
        if args.debug:
            cmd.append("--debug")
        timings["eval_stage3"] = run_command(cmd, "Evaluate Stage 3")

    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    total = 0
    for step, elapsed in timings.items():
        print(f"  {step}: {elapsed:.2f}s")
        total += elapsed
    print(f"  Total: {total:.2f}s")


if __name__ == "__main__":
    main()
