"""T7 - Aggregate CBR-Flash deployment-shift results.

Reads ``phase2_diagnostics.json`` files from the
``g_opd_flash_deployshift_{mode}_v1{seed}_v2{deploy_seed}`` run directories
and reports the deploy-shift Δ AUPRC per mode:

    Δ_deploy(mode, cell, seed) = AUPRC_eval_on_base_v1 - AUPRC_eval_on_base_v2

Hypothesis:
    the full CBR-Flash sampling mode has smaller deploy-shift delta than
    off-policy KD.

Outputs:

* ``artifacts/tables/g_opd_flash_deployshift.md``
* ``artifacts/tables/g_opd_flash_deployshift_summary.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


DATASETS = ("yelpchi", "amazon")
BASES = ("bwgnn", "sage", "gcn", "gat")
SEEDS = (42, 123, 456, 789, 2026)
DEPLOY_SEEDS = (123, 456, 789, 2026, 42)
MODES = ("off_policy", "det_mask", "g_opd_flash")


def _load_one(ds: str, base: str, mode: str, seed: int, deploy_seed: int):
    run_name = f"g_opd_flash_deployshift_{mode}_v1{seed}_v2{deploy_seed}"
    diag_path = Path("artifacts/logs") / ds / base / run_name / f"seed_{seed}" / "phase2_diagnostics.json"
    if not diag_path.exists():
        return None
    with open(diag_path) as f:
        return json.load(f)


def _paired_t_one_sided(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    arr = np.array(a) - np.array(b)
    if np.allclose(arr, 0):
        return 0.0, 1.0
    t, p_two = stats.ttest_1samp(arr, 0.0)
    if np.isnan(t):
        return float("nan"), float("nan")
    return float(t), float((p_two / 2) if t > 0 else (1 - p_two / 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_md", type=str, default="artifacts/tables/g_opd_flash_deployshift.md")
    p.add_argument("--out_summary_json", type=str,
                   default="artifacts/tables/g_opd_flash_deployshift_summary.json")
    args = p.parse_args()

    # Δ per (cell, mode, seed)
    deltas: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    v1_auprcs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    v2_auprcs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for ds in DATASETS:
        for base in BASES:
            for i, seed in enumerate(SEEDS):
                deploy_seed = DEPLOY_SEEDS[i]
                for mode in MODES:
                    d = _load_one(ds, base, mode, seed, deploy_seed)
                    if d is None:
                        continue
                    v1 = float(d.get("test_metrics_student", {}).get("auprc", float("nan")))
                    v2_dict = d.get("test_metrics_deploy_shift") or {}
                    v2 = float(v2_dict.get("auprc", float("nan")))
                    if not (np.isfinite(v1) and np.isfinite(v2)):
                        continue
                    deltas[(ds, base, mode)].append(v1 - v2)
                    v1_auprcs[(ds, base, mode)].append(v1)
                    v2_auprcs[(ds, base, mode)].append(v2)

    # Render table
    lines: list[str] = ["# CBR-Flash - deployment-shift evaluation (T7)\n"]
    lines.append("Δ_deploy(mode, cell) = AUPRC_eval_v1 − AUPRC_eval_v2  (smaller Δ ⇒ more robust)\n")
    lines.append("Paired-t: H1 mode == `g_opd_flash` has smaller Δ_deploy than baseline mode (one-sided)\n\n")
    lines.append("| Dataset | Base | Mode | n | mean Δ_deploy | std | mean AUPRC v1 → v2 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        for base in BASES:
            for mode in MODES:
                k = (ds, base, mode)
                if not deltas.get(k):
                    continue
                ds_arr = np.array(deltas[k])
                v1_mean = float(np.mean(v1_auprcs[k]))
                v2_mean = float(np.mean(v2_auprcs[k]))
                lines.append(
                    f"| {ds} | {base} | `{mode}` | {len(ds_arr)} | "
                    f"{ds_arr.mean():+.4f} | {ds_arr.std(ddof=1) if len(ds_arr) > 1 else 0:.4f} | "
                    f"{v1_mean:.4f} → {v2_mean:.4f} |"
                )

    # ── Tuple-keyed pairing (Critic round-4 CRITICAL fix) ─────────────────
    # Build per-(ds, base, seed, deploy_seed) keys for each mode, then pair
    # by tuple identity.  Eliminates the prior arbitrary index-order pairing
    # that depended on dict iteration order.

    def _build_keyed_deltas(modes_to_include: tuple[str, ...]) -> dict[str, dict[tuple, float]]:
        out: dict[str, dict[tuple, float]] = {m: {} for m in modes_to_include}
        for ds in DATASETS:
            for base in BASES:
                for i, seed in enumerate(SEEDS):
                    deploy_seed = DEPLOY_SEEDS[i]
                    for mode in modes_to_include:
                        d = _load_one(ds, base, mode, seed, deploy_seed)
                        if d is None:
                            continue
                        v1 = float(d.get("test_metrics_student", {}).get("auprc", float("nan")))
                        v2_dict = d.get("test_metrics_deploy_shift") or {}
                        v2 = float(v2_dict.get("auprc", float("nan")))
                        if not (np.isfinite(v1) and np.isfinite(v2)):
                            continue
                        out[mode][(ds, base, seed, deploy_seed)] = v1 - v2
        return out

    keyed_deltas = _build_keyed_deltas(MODES)

    # Pooled paired-t: g_opd_flash Δ vs baseline Δ, paired by tuple key.
    lines.append("\n## Pooled paired-t (one-sided, H1: baseline Δ > g_opd_flash Δ)\n")
    lines.append("Paired by `(dataset, base, seed, deploy_seed)` tuple identity — Critic round-4 CRITICAL fix.\n")
    lines.append("| Baseline mode | n_paired | mean Δ baseline | mean Δ g_opd_flash | t | p | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    summary: dict[str, Any] = {"per_baseline": {}, "pairing_method": "tuple-keyed (ds, base, seed, deploy_seed)"}
    g_dict = keyed_deltas.get("g_opd_flash", {})
    for baseline in ("off_policy", "det_mask"):
        b_dict = keyed_deltas.get(baseline, {})
        common_keys = sorted(set(g_dict.keys()) & set(b_dict.keys()))
        if len(common_keys) < 2:
            continue
        b_arr = [b_dict[k] for k in common_keys]
        g_arr = [g_dict[k] for k in common_keys]
        t, p = _paired_t_one_sided(b_arr, g_arr)
        verdict = "g_opd_flash MORE robust" if (np.isfinite(p) and p < 0.05) else "no sig diff"
        lines.append(
            f"| `{baseline}` | {len(common_keys)} | {np.mean(b_arr):+.4f} | "
            f"{np.mean(g_arr):+.4f} | {t:+.2f} | {p:.4f} | {verdict} |"
        )
        summary["per_baseline"][baseline] = {
            "n_paired": len(common_keys),
            "paired_keys_preview": [str(k) for k in common_keys[:3]],
            "mean_delta_baseline": float(np.mean(b_arr)),
            "mean_delta_g_opd_flash": float(np.mean(g_arr)),
            "t": float(t),
            "p_one_sided": float(p),
            "verdict": verdict,
        }

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[T7/agg] Wrote {args.out_md}")
    print(f"[T7/agg] Wrote {args.out_summary_json}")


if __name__ == "__main__":
    main()
