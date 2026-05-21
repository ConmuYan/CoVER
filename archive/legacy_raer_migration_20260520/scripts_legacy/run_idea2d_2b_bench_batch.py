"""Run Idea-2D speed benchmarks for Idea-2B-teacher distillation in-process.

This avoids CUDA initialization failures seen when launching the same commands
through a nested shell in the current machine environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bench_inference_speed

SEEDS = [42, 123, 456, 789, 2026]
CELLS = [
    ("yelpchi", "bwgnn", "cuda:1"),
    ("yelpchi", "sage", "cuda:3"),
    ("yelpchi", "gat", "cuda:1"),
]
RUN_NAME = "idea2c_distill_adapter_2b"
N_ITERS = 100


def main() -> int:
    for ds, base, device in CELLS:
        config = f"configs/phase2_reasoner/ablation/idea2b_learned_extractor_{ds}_{base}.yaml"
        for seed in SEEDS:
            out = Path(f"artifacts/results/{ds}/{base}/{RUN_NAME}/seed_{seed}/inference_benchmark.json")
            if out.exists():
                print(f"[skip] {ds}-{base} seed={seed}: {out}")
                continue
            argv = [
                "bench_inference_speed.py",
                "--config", config,
                "--seed", str(seed),
                "--device", device,
                "--teacher_ckpt", f"artifacts/checkpoints/{ds}/{base}/idea2b_learned_extractor/seed_{seed}/reasoner.pt",
                "--teacher_extractor_ckpt", f"artifacts/checkpoints/{ds}/{base}/idea2b_learned_extractor/seed_{seed}/evidence_extractor.pt",
                "--adapter_ckpt", f"artifacts/checkpoints/{ds}/{base}/{RUN_NAME}/seed_{seed}/adapter.pt",
                "--base_ckpt_path", f"artifacts/checkpoints/{ds}/{base}/fixed_v1_100ep/seed_{seed}/base.pt",
                "--n_iters", str(N_ITERS),
            ]
            print(f"[run] {ds}-{base} seed={seed} device={device}")
            old_argv = sys.argv
            try:
                sys.argv = argv
                bench_inference_speed.main()
            finally:
                sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
