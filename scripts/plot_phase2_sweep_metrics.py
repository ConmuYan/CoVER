"""Plot Phase2 sweep metrics in a compact Nature-style layout.

The script consumes the summary CSV written by ``aggregate_phase2_custom_runs.py``
and writes both metric and diagnostic line charts as PNG/PDF/SVG.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("auprc", "AUPRC"),
    ("roc_auc", "AUROC"),
    ("macro_f1", "Macro-F1"),
    ("g_means", "G-Means"),
]

DIAGNOSTICS = [
    ("mean_abs_delta_rel", "mean |delta_rel|"),
    ("mean_gate_entropy", "Gate entropy"),
    ("mean_alpha_llm", "mean alpha_llm"),
    ("max_abs_alpha_llm_rejected", "Rejected alpha max"),
]

SUITES = {
    "yelpchi_targeted": {
        "dataset": "YelpChi",
        "groups": [
            (
                "S1 lambda_trust",
                [
                    ("0", "phase2_yelp_s1_ltrust_0"),
                    ("1e-3", "phase2_yelp_s1_ltrust_1em3"),
                    ("3e-3", "phase2_yelp_s1_ltrust_3em3"),
                    ("1e-2", "phase2_yelp_s1_ltrust_1em2"),
                    ("3e-2", "phase2_yelp_s1_ltrust_3em2"),
                ],
            ),
            (
                "S2 lambda_sparse",
                [
                    ("0", "phase2_yelp_s2_lsparse_0"),
                    ("3e-4", "phase2_yelp_s2_lsparse_3em4"),
                    ("1e-3", "phase2_yelp_s2_lsparse_1em3"),
                    ("3e-3", "phase2_yelp_s2_lsparse_3em3"),
                ],
            ),
            (
                "S3 lambda_align",
                [
                    ("0", "phase2_yelp_s3_lalign_0"),
                    ("1e-3", "phase2_yelp_s3_lalign_1em3"),
                    ("3e-3", "phase2_yelp_s3_lalign_3em3"),
                    ("1e-2", "phase2_yelp_s3_lalign_1em2"),
                ],
            ),
            (
                "S3 alpha_max",
                [
                    ("0", "phase2_yelp_s3_alpha_0"),
                    ("0.1", "phase2_yelp_s3_alpha_01"),
                    ("0.3", "phase2_yelp_s3_alpha_03"),
                ],
            ),
        ],
    },
    "amazon_candidates": {
        "dataset": "Amazon",
        "groups": [
            (
                "Amazon default candidates",
                [
                    ("c1 d=0.5", "phase2_amz_c1_drel05_ltrust1em2_tau18_lsp0"),
                    ("c2 d=0.75", "phase2_amz_c2_drel075_ltrust1em2_tau18_lsp0"),
                    ("c3 trust=3e-3", "phase2_amz_c3_drel05_ltrust3em3_tau18_lsp0"),
                    ("c4 sparse", "phase2_amz_c4_drel05_ltrust1em2_tau13_lsp3em4"),
                    ("c5 align", "phase2_amz_c5_drel05_align_only"),
                    ("c6 residual", "phase2_amz_c6_drel05_judge_residual"),
                ],
            ),
        ],
    },
    "amazon_focused_cuda": {
        "dataset": "Amazon",
        "groups": [
            (
                "Relation residual regime",
                [
                    ("d=1.0 tau=1.8", "phase2_amz_f1_drel10_tau18_trust1em2_lsp0_cuda"),
                    ("d=1.25 tau=1.8", "phase2_amz_f2_drel125_tau18_trust1em2_lsp0_cuda"),
                    ("d=1.0 tau=2.5", "phase2_amz_f3_drel10_tau25_trust1em2_lsp0_cuda"),
                    ("d=0.75 tau=2.5", "phase2_amz_f4_drel075_tau25_trust1em2_lsp0_cuda"),
                    ("d=1.0 tau=1.3", "phase2_amz_f5_drel10_tau13_trust1em2_lsp0_cuda"),
                    ("d=0.75 tau=1.3", "phase2_amz_f6_drel075_tau13_trust1em2_lsp0_cuda"),
                    ("trust=3e-2", "phase2_amz_f7_drel10_tau18_trust3em2_lsp0_cuda"),
                    ("lr=1e-4", "phase2_amz_f8_drel15_tau25_trust5em2_lr1em4_cuda"),
                ],
            ),
            (
                "Judge alignment and residual",
                [
                    ("d=.75 a=3e-3", "phase2_amz_j1_drel075_align3em3_alpha0_cuda"),
                    ("d=.75 a=1e-2", "phase2_amz_j2_drel075_align1em2_alpha0_cuda"),
                    ("d=1 a=3e-3", "phase2_amz_j3_drel10_align3em3_alpha0_cuda"),
                    ("d=1 a=1e-2", "phase2_amz_j4_drel10_align1em2_alpha0_cuda"),
                    ("d=.75 res", "phase2_amz_j5_drel075_align3em3_alpha005_cuda"),
                    ("d=1 res", "phase2_amz_j6_drel10_align3em3_alpha005_cuda"),
                    ("d=.75 res hi", "phase2_amz_j7_drel075_align1em2_alpha005_cuda"),
                    ("d=1 res hi", "phase2_amz_j8_drel10_align1em2_alpha005_cuda"),
                ],
            ),
        ],
    },
}


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def to_float(value: str | None) -> float:
    if value in (None, "", "nan", "NaN"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            run = row["run_name"]
            rows[run] = {key: to_float(value) for key, value in row.items() if key != "run_name"}
    return rows


def values_for(
    rows: dict[str, dict[str, float]],
    run_order: list[tuple[str, str]],
    key: str,
) -> tuple[list[str], list[float], list[float]]:
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for label, run in run_order:
        if run not in rows:
            continue
        labels.append(label)
        means.append(rows[run].get(f"{key}_mean", float("nan")))
        stds.append(rows[run].get(f"{key}_std", float("nan")))
    return labels, means, stds


def finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, float) and not math.isnan(v)]


def set_nice_ylim(ax, means: list[float], stds: list[float]) -> None:
    vals = finite(means + [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)])
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-8:
        pad = max(abs(hi) * 0.01, 1e-3)
    else:
        pad = (hi - lo) * 0.16
    ax.set_ylim(lo - pad, hi + pad)


def save_all(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight")


def plot_panel_grid(
    rows: dict[str, dict[str, float]],
    title: str,
    group_name: str,
    run_order: list[tuple[str, str]],
    keys: list[tuple[str, str]],
    out_base: Path,
    color: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))
    fig.suptitle(f"{title}: {group_name}", y=0.995, fontsize=9)
    for ax, (key, label) in zip(axes.ravel(), keys):
        xlabels, means, stds = values_for(rows, run_order, key)
        x = list(range(len(xlabels)))
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            markersize=3.2,
            linewidth=1.15,
            elinewidth=0.75,
            capsize=2.2,
            color=color,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
        ax.set_title(label, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        set_nice_ylim(ax, means, stds)
    fig.tight_layout(pad=1.0)
    save_all(fig, out_base)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    configure_style()
    rows = load_summary(Path(args.summary_csv))
    suite = SUITES[args.suite]
    out_dir = Path(args.out_dir)
    color = "#0072B2" if args.suite.startswith("yelpchi") else "#D55E00"

    for idx, (group_name, run_order) in enumerate(suite["groups"], start=1):
        slug = (
            group_name.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace("-", "_")
        )
        plot_panel_grid(
            rows,
            str(suite["dataset"]),
            group_name,
            run_order,
            METRICS,
            out_dir / f"{idx:02d}_{slug}_metrics",
            color,
        )
        plot_panel_grid(
            rows,
            str(suite["dataset"]),
            group_name,
            run_order,
            DIAGNOSTICS,
            out_dir / f"{idx:02d}_{slug}_diagnostics",
            "#009E73",
        )

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
