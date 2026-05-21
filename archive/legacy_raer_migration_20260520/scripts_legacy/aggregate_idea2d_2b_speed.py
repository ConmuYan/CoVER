"""Aggregate Idea-2D speed benchmark for Idea-2B-teacher distillation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 456, 789, 2026]
CELLS = [("yelpchi", "bwgnn"), ("yelpchi", "sage"), ("yelpchi", "gat")]
RUN_NAME = "idea2c_distill_adapter_2b"


def load(ds: str, base: str, seed: int) -> dict:
    path = Path(f"artifacts/results/{ds}/{base}/{RUN_NAME}/seed_{seed}/inference_benchmark.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(path.open())


def mean_sd(vals: list[float]) -> str:
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"


def main() -> int:
    lines = [
        "# Idea-2D Speed — Idea-2B Teacher vs Distill Adapter",
        "",
        "**Run**: `idea2c_distill_adapter_2b`, 3 YelpChi cells × 5 seeds, same seeds `[42, 123, 456, 789, 2026]`.",
        "",
        "Timing is head-level inference on the test split with cached base outputs and cached/online-built relation evidence tensors; it compares the full CoVER-REL reasoner head against the compact adapter head.",
        "",
        "| Cell | N_test | Base-only ms | 2B teacher ms | Adapter ms | Teacher→Adapter speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, base in CELLS:
        rows = [load(ds, base, seed) for seed in SEEDS]
        n_test = int(rows[0]["n_test"])
        base_ms = [float(r["base_only_ms"]) for r in rows]
        teacher_ms = [float(r["teacher_ms"]) for r in rows]
        adapter_ms = [float(r["adapter_ms"]) for r in rows]
        speedup = [float(r["speedup_b_to_c"]) for r in rows]
        lines.append(
            f"| {ds}-{base} | {n_test} | {mean_sd(base_ms)} | {mean_sd(teacher_ms)} | "
            f"{mean_sd(adapter_ms)} | {mean_sd(speedup)}x |"
        )
    lines.append("")
    out = Path("artifacts/tables/idea2d_2b_speed_benchmark.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
