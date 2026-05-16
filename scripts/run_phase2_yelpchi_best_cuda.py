"""Run YelpChi Phase2 best configuration with CUDA and TensorBoard.

The launcher sets thread/worker environment variables inside Python before
spawning the standard trainer. This preserves CUDA visibility in the current
execution environment while bounding CPU usage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SEEDS = [42, 123, 456, 789, 2026]
RUN_NAME = "phase2_yelp_best_lalign1em2_alpha0_cuda_tb"
CONFIG = "configs/cover-rel-gj/phase2_ablations/phase2_yelpchi_E2_judge_residual.yaml"
RUN_ARGS = [
    "--use_judge", "1",
    "--alpha_max", "0",
    "--lambda_align", "1.0e-2",
    "--lambda_trust", "3.0e-3",
    "--lambda_sparse", "1.0e-3",
    "--delta_rel_max", "2.0",
    "--delta_llm_max", "0.75",
    "--tau_gate", "0.7",
    "--lr", "1.0e-3",
    "--patience", "50",
    "--early_stop_metric", "val_auprc",
    "--eval_interval", "1",
]


def configure_worker_env(workers: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "COVER_NUM_THREADS",
        "COVER_INTEROP_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(workers)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cover-fd-yelp-best-cuda")
    return env


def assert_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in launcher process")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def validate_cuda_diagnostics(seed: int, expected_device: str) -> None:
    path = (
        Path("artifacts/logs/yelpchi/bwgnn")
        / RUN_NAME
        / f"seed_{seed}"
        / "phase2_diagnostics.json"
    )
    payload = read_json(path)
    device = payload.get("device", {})
    requested = str(device.get("requested_device", ""))
    cuda_available = bool(device.get("cuda_available", False))
    if requested != expected_device or not cuda_available:
        raise RuntimeError(
            f"Run {RUN_NAME}/seed_{seed} did not use CUDA as expected: "
            f"requested={requested!r}, cuda_available={cuda_available}, path={path}"
        )


def run_train(*, seed: int, device: str, env: dict[str, str], log_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/train_phase2_reasoner.py",
        "--config",
        CONFIG,
        "--seed",
        str(seed),
        "--device",
        device,
        "--run_name",
        RUN_NAME,
        *RUN_ARGS,
    ]
    with log_path.open("a") as log:
        log.write(f"[yelpchi-best-cuda] BEGIN seed={seed} {time.strftime('%F %T')}\n")
        log.flush()
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(
            f"[yelpchi-best-cuda] END seed={seed} rc={proc.returncode} {time.strftime('%F %T')}\n"
        )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    validate_cuda_diagnostics(seed, device)


def aggregate(seeds: list[int]) -> None:
    out_prefix = "artifacts/tables/yelpchi_phase2_best_cuda_tb"
    cmd = [
        sys.executable,
        "scripts/aggregate_phase2_custom_runs.py",
        "--dataset",
        "yelpchi",
        "--runs",
        RUN_NAME,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--out-prefix",
        out_prefix,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--log", default="artifacts/sweeps/_drivers/phase2_yelpchi_best_cuda_tb.log")
    args = parser.parse_args()

    assert_cuda_available()
    env = configure_worker_env(args.workers)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_train(seed=seed, device=args.device, env=env, log_path=log_path)
    aggregate(args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
