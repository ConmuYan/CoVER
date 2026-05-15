"""Run Amazon Phase2 focused search with CUDA and bounded CPU workers.

This launcher intentionally sets thread/worker environment variables inside a
Python process. In the current execution environment, prefixing shell commands
with environment assignments can hide CUDA devices from PyTorch; launching from
the approved Python executable and setting env vars here preserves CUDA access.
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


RELATION_RUNS = [
    (
        "phase2_amz_f1_drel10_tau18_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.0",
            "--tau_gate", "1.8",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f2_drel125_tau18_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.25",
            "--tau_gate", "1.8",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f3_drel10_tau25_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.0",
            "--tau_gate", "2.5",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f4_drel075_tau25_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "0.75",
            "--tau_gate", "2.5",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f5_drel10_tau13_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.0",
            "--tau_gate", "1.3",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f6_drel075_tau13_trust1em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "0.75",
            "--tau_gate", "1.3",
            "--lambda_trust", "1.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f7_drel10_tau18_trust3em2_lsp0_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.0",
            "--tau_gate", "1.8",
            "--lambda_trust", "3.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-3",
            "--patience", "50",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
    (
        "phase2_amz_f8_drel15_tau25_trust5em2_lr1em4_cuda",
        "configs/phase2_amazon_E0_relgate.yaml",
        [
            "--use_judge", "0",
            "--alpha_max", "0",
            "--lambda_align", "0",
            "--delta_rel_max", "1.5",
            "--tau_gate", "2.5",
            "--lambda_trust", "5.0e-2",
            "--lambda_sparse", "0",
            "--lr", "1.0e-4",
            "--patience", "100",
            "--early_stop_metric", "val_auprc",
            "--eval_interval", "1",
        ],
    ),
]


JUDGE_RUNS = [
    (
        "phase2_amz_j1_drel075_align3em3_alpha0_cuda",
        "configs/phase2_amazon_E1_judge_align.yaml",
        ["--delta_rel_max", "0.75", "--lambda_align", "3.0e-3", "--alpha_max", "0"],
    ),
    (
        "phase2_amz_j2_drel075_align1em2_alpha0_cuda",
        "configs/phase2_amazon_E1_judge_align.yaml",
        ["--delta_rel_max", "0.75", "--lambda_align", "1.0e-2", "--alpha_max", "0"],
    ),
    (
        "phase2_amz_j3_drel10_align3em3_alpha0_cuda",
        "configs/phase2_amazon_E1_judge_align.yaml",
        ["--delta_rel_max", "1.0", "--lambda_align", "3.0e-3", "--alpha_max", "0"],
    ),
    (
        "phase2_amz_j4_drel10_align1em2_alpha0_cuda",
        "configs/phase2_amazon_E1_judge_align.yaml",
        ["--delta_rel_max", "1.0", "--lambda_align", "1.0e-2", "--alpha_max", "0"],
    ),
    (
        "phase2_amz_j5_drel075_align3em3_alpha005_cuda",
        "configs/phase2_amazon_E2_judge_residual.yaml",
        ["--delta_rel_max", "0.75", "--lambda_align", "3.0e-3", "--alpha_max", "0.05"],
    ),
    (
        "phase2_amz_j6_drel10_align3em3_alpha005_cuda",
        "configs/phase2_amazon_E2_judge_residual.yaml",
        ["--delta_rel_max", "1.0", "--lambda_align", "3.0e-3", "--alpha_max", "0.05"],
    ),
    (
        "phase2_amz_j7_drel075_align1em2_alpha005_cuda",
        "configs/phase2_amazon_E2_judge_residual.yaml",
        ["--delta_rel_max", "0.75", "--lambda_align", "1.0e-2", "--alpha_max", "0.05"],
    ),
    (
        "phase2_amz_j8_drel10_align1em2_alpha005_cuda",
        "configs/phase2_amazon_E2_judge_residual.yaml",
        ["--delta_rel_max", "1.0", "--lambda_align", "1.0e-2", "--alpha_max", "0.05"],
    ),
]

JUDGE_COMMON = [
    "--use_judge", "1",
    "--tau_gate", "1.8",
    "--lambda_trust", "1.0e-2",
    "--lambda_sparse", "0",
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
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cover-fd-amz-focused-cuda")
    return env


def assert_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in launcher process")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def validate_cuda_diagnostics(run_name: str, seed: int, expected_device: str) -> None:
    path = (
        Path("artifacts/logs/amazon/bwgnn")
        / run_name
        / f"seed_{seed}"
        / "phase2_diagnostics.json"
    )
    payload = read_json(path)
    device = payload.get("device", {})
    requested = str(device.get("requested_device", ""))
    cuda_available = bool(device.get("cuda_available", False))
    if requested != expected_device or not cuda_available:
        raise RuntimeError(
            f"Run {run_name}/seed_{seed} did not use CUDA as expected: "
            f"requested={requested!r}, cuda_available={cuda_available}, path={path}"
        )


def run_train(
    *,
    run_name: str,
    config: str,
    seed: int,
    device: str,
    args: list[str],
    env: dict[str, str],
    log_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/train_phase2_reasoner.py",
        "--config",
        config,
        "--seed",
        str(seed),
        "--device",
        device,
        "--run_name",
        run_name,
        *args,
    ]
    with log_path.open("a") as log:
        log.write(f"[amazon-focused-cuda] BEGIN seed={seed} run={run_name} {time.strftime('%F %T')}\n")
        log.flush()
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(
            f"[amazon-focused-cuda] END seed={seed} run={run_name} "
            f"rc={proc.returncode} {time.strftime('%F %T')}\n"
        )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    validate_cuda_diagnostics(run_name, seed, device)


def aggregate(runs: list[str], seeds: list[int]) -> None:
    out_prefix = "artifacts/tables/amazon_phase2_focused_cuda"
    cmd = [
        sys.executable,
        "scripts/aggregate_phase2_custom_runs.py",
        "--dataset",
        "amazon",
        "--runs",
        *runs,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--out-prefix",
        out_prefix,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--mode",
        choices=["relation", "judge", "all"],
        default="all",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--log", default="artifacts/sweeps/_drivers/phase2_amazon_focused_cuda.log")
    args = parser.parse_args()

    assert_cuda_available()
    env = configure_worker_env(args.workers)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, str, list[str]]] = []
    if args.mode in {"relation", "all"}:
        runs.extend(RELATION_RUNS)
    if args.mode in {"judge", "all"}:
        for run_name, config, specific_args in JUDGE_RUNS:
            runs.append((run_name, config, [*JUDGE_COMMON, *specific_args]))
    selected = runs[args.start_index : args.stop_index]

    for seed in args.seeds:
        for run_name, config, run_args in selected:
            run_train(
                run_name=run_name,
                config=config,
                seed=seed,
                device=args.device,
                args=run_args,
                env=env,
                log_path=log_path,
            )

    aggregate([run_name for run_name, _, _ in runs], args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
