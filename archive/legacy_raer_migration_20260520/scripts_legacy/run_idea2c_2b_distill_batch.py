"""In-process runner for Idea-2B-teacher distillation.

The local CUDA environment used by this project can fail to initialize CUDA
when Python is launched from a nested shell or with redirected stdout.  This
runner calls ``train_distill_adapter.main()`` in-process for every cell/seed so
that the experiment can be launched as one direct Python command.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import train_distill_adapter


SEEDS = [42, 123, 456, 789, 2026]
RUN_NAME = "idea2c_distill_adapter_2b"

CFG = {
    "yelpchi-bwgnn": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_bwgnn.yaml",
    "yelpchi-sage": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_sage.yaml",
    "yelpchi-gcn": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_gcn.yaml",
    "yelpchi-gat": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_yelpchi_gat.yaml",
    "amazon-bwgnn": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_bwgnn.yaml",
    "amazon-sage": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_sage.yaml",
    "amazon-gcn": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_gcn.yaml",
    "amazon-gat": "configs/phase2_reasoner/ablation/idea2b_learned_extractor_amazon_gat.yaml",
}


def result_path(ds: str, base: str, seed: int) -> Path:
    return (
        Path("artifacts/results")
        / ds
        / base
        / RUN_NAME
        / f"seed_{seed}"
        / "stage3_metrics.json"
    )


def run_one(cell: str, seed: int, device: str) -> None:
    ds, base = cell.split("-", 1)
    if result_path(ds, base, seed).exists():
        print(f"[SKIP] {cell}/seed_{seed}", flush=True)
        return

    teacher_dir = Path("artifacts/checkpoints") / ds / base / "idea2b_learned_extractor" / f"seed_{seed}"
    argv = [
        "train_distill_adapter.py",
        "--config", CFG[cell],
        "--seed", str(seed),
        "--device", device,
        "--run_name", RUN_NAME,
        "--teacher_ckpt", str(teacher_dir / "reasoner.pt"),
        "--teacher_extractor_ckpt", str(teacher_dir / "evidence_extractor.pt"),
        "--base_ckpt_path", f"artifacts/checkpoints/{ds}/{base}/fixed_v1_100ep/seed_{seed}/base.pt",
    ]
    print(f"[RUN] {cell}/seed_{seed} on {device}", flush=True)
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_distill_adapter.main()
    finally:
        sys.argv = old_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    cells = list(CFG)
    devices = ["cuda:1", "cuda:3"]
    i = 0
    for cell in cells:
        for seed in SEEDS:
            run_one(cell, seed, devices[i % len(devices)])
            i += 1
    print("[DONE] idea2c_distill_adapter_2b", flush=True)


if __name__ == "__main__":
    main()
