"""Run the YelpChi-BWGNN Full CoVER lambda_sparse x lambda_align sweep.

This is a small execution launcher for the augment.md unit sweep.  It waits
for enough free GPU memory before each run, keeps CPU threads bounded, and
lets the standard Phase2 trainer write TensorBoard, repro configs, and summary
JSON for every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CONFIG = "configs/phase2_reasoner/phase2_yelpchi_bwgnn_full_cover.yaml"
DATASET = "yelpchi"
MODEL = "bwgnn"
SEED = 42
SPARSE_VALUES = [5.0e-4, 1.0e-3, 2.0e-3]
ALIGN_VALUES = [5.0e-3, 1.0e-2, 2.0e-2]


def token(value: float) -> str:
    mapping = {
        5.0e-4: "5em4",
        1.0e-3: "1em3",
        2.0e-3: "2em3",
        5.0e-3: "5em3",
        1.0e-2: "1em2",
        2.0e-2: "2em2",
    }
    return mapping.get(float(value), f"{value:.0e}".replace("-", "m").replace("+", ""))


def run_name(lambda_sparse: float, lambda_align: float) -> str:
    return (
        "phase2_yelpchi_bwgnn_full_cover"
        f"_lsp{token(lambda_sparse)}"
        f"_lalign{token(lambda_align)}"
    )


def base_cache_exists(seed: int) -> bool:
    path = Path("artifacts/base_outputs") / DATASET / MODEL / f"seed_{seed}" / "base_outputs.pt"
    return path.exists()


def gpu_free_memory() -> list[tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows: list[tuple[int, int]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_s, free_s = [part.strip() for part in line.split(",")[:2]]
        rows.append((int(idx_s), int(free_s)))
    return rows


def wait_for_gpu(min_free_mb: int, poll_seconds: int, log) -> int:
    while True:
        rows = gpu_free_memory()
        idx, free = max(rows, key=lambda item: item[1])
        log.write(f"[gpu] best cuda:{idx} free={free}MB need={min_free_mb}MB {time.strftime('%F %T')}\n")
        log.flush()
        if free >= min_free_mb:
            return idx
        time.sleep(poll_seconds)


def configure_env(workers: int) -> dict[str, str]:
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
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cover-fd-full-cover-sweep")
    return env


def summary_path(run: str, seed: int) -> Path:
    return (
        Path("artifacts/results")
        / DATASET
        / MODEL
        / run
        / f"seed_{seed}"
        / "phase2_summary.json"
    )


def validate_summary(run: str, seed: int, expected_device: str) -> None:
    path = summary_path(run, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    payload = json.loads(path.read_text())
    device = payload.get("device", {})
    if str(device.get("requested_device")) != expected_device:
        raise RuntimeError(f"{run} requested_device mismatch: {device}")
    if not bool(device.get("cuda_available", False)):
        raise RuntimeError(f"{run} did not report CUDA availability: {device}")
    tb = payload.get("tensorboard", {})
    if not tb.get("enabled") or not tb.get("path"):
        raise RuntimeError(f"{run} missing TensorBoard metadata: {tb}")


def run_one(
    *,
    run: str,
    seed: int,
    lambda_sparse: float,
    lambda_align: float,
    args: argparse.Namespace,
    env: dict[str, str],
    log,
) -> None:
    if summary_path(run, seed).exists() and not args.force:
        log.write(f"[skip] existing summary for {run}/seed_{seed}\n")
        log.flush()
        return

    min_free = args.min_free_cached_mb if base_cache_exists(seed) else args.min_free_uncached_mb
    gpu_idx = wait_for_gpu(min_free, args.poll_seconds, log)
    device = f"cuda:{gpu_idx}"
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
        run,
        "--two-stage",
        "--lambda_sparse",
        str(lambda_sparse),
        "--lambda_align",
        str(lambda_align),
        "--lambda_int",
        "3.0e-3",
        "--alpha_max",
        "0.10",
        "--eval_interval",
        "1",
    ]
    log.write(f"[begin] {run} seed={seed} device={device} {time.strftime('%F %T')}\n")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"[end] {run} rc={proc.returncode} {time.strftime('%F %T')}\n")
    log.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    validate_summary(run, seed, device)


def aggregate_and_plot(runs: list[str], seed: int, log) -> None:
    out_prefix = "artifacts/tables/yelpchi_bwgnn_full_cover_lsp_lalign_seed42"
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_phase2_custom_runs.py",
            "--dataset",
            DATASET,
            "--model",
            MODEL,
            "--runs",
            *runs,
            "--seeds",
            str(seed),
            "--out-prefix",
            out_prefix,
        ],
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    summary_csv = Path(out_prefix + "_summary.csv")
    best_run = None
    best_auprc = float("-inf")
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                auprc = float(row.get("auprc_mean", "nan"))
            except ValueError:
                continue
            if auprc > best_auprc:
                best_auprc = auprc
                best_run = row["run_name"]

    if best_run is not None:
        subprocess.run(
            [
                sys.executable,
                "scripts/plot_phase2_training_curves.py",
                "--dataset",
                DATASET,
                "--model",
                MODEL,
                "--run",
                best_run,
                "--seeds",
                str(seed),
                "--out-dir",
                f"artifacts/figures/phase2_training_curves/yelpchi_bwgnn_full_cover_lsp_lalign_seed42/{best_run}",
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"[best] {best_run} auprc={best_auprc:.6f}\n")
        log.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-uncached-mb", type=int, default=17000)
    parser.add_argument("--min-free-cached-mb", type=int, default=9000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log", default="artifacts/sweeps/_drivers/phase2_yelpchi_bwgnn_full_cover_lsp_lalign_seed42.log")
    args = parser.parse_args()

    env = configure_env(args.workers)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runs = [run_name(lsp, la) for lsp in SPARSE_VALUES for la in ALIGN_VALUES]
    with log_path.open("a") as log:
        log.write(f"[sweep-start] seed={args.seed} workers={args.workers} {time.strftime('%F %T')}\n")
        log.flush()
        for lambda_sparse in SPARSE_VALUES:
            for lambda_align in ALIGN_VALUES:
                run_one(
                    run=run_name(lambda_sparse, lambda_align),
                    seed=args.seed,
                    lambda_sparse=lambda_sparse,
                    lambda_align=lambda_align,
                    args=args,
                    env=env,
                    log=log,
                )
        aggregate_and_plot(runs, args.seed, log)
        log.write(f"[sweep-done] {time.strftime('%F %T')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
