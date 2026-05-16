"""Plot Phase2 per-epoch training curves from saved CSV logs.

This script is intentionally log-file based so it works for any completed
Phase2 run, including runs launched before TensorBoard was enabled.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_KEYS = [
    ("loss", "Total loss"),
    ("loss/l_cls", "BCE loss"),
    ("loss/l_intervention", "Intervention loss"),
    ("loss/l_sparse", "Sparse loss"),
    ("loss/l_align", "Align loss"),
]

METRIC_KEYS = [
    ("val/auprc", "Val AUPRC"),
    ("val/roc_auc", "Val AUROC"),
    ("val/macro_f1", "Val Macro-F1"),
    ("val/g_means", "Val G-Means"),
]


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


def read_curve(path: Path) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                curves.setdefault(key, []).append(to_float(value))
    if "loss/l_intervention" not in curves and "loss/l_trust" in curves:
        curves["loss/l_intervention"] = list(curves["loss/l_trust"])
    return curves


def save_all(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight")


def finite(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, float) and not math.isnan(v)]


def set_ylim(ax, series: list[list[float]]) -> None:
    vals: list[float] = []
    for y in series:
        vals.extend(finite(y))
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.14, abs(hi) * 0.005, 1e-4)
    ax.set_ylim(lo - pad, hi + pad)


def mean_curve(seed_curves: list[dict[str, list[float]]], key: str) -> tuple[np.ndarray, np.ndarray]:
    lengths = [len(curves.get(key, [])) for curves in seed_curves if curves.get(key, [])]
    if not lengths:
        return np.array([]), np.array([])
    # Use only the common epoch range. Averaging over "available" runs after
    # early stopping creates artificial jumps when a seed drops out.
    common_len = min(lengths)
    arr = np.full((len(seed_curves), common_len), np.nan, dtype=float)
    for i, curves in enumerate(seed_curves):
        y = np.asarray(curves.get(key, [])[:common_len], dtype=float)
        arr[i, : len(y)] = y
    x = np.arange(1, common_len + 1)
    with np.errstate(invalid="ignore"):
        y_mean = np.nanmean(arr, axis=0)
    return x, y_mean


def plot_grid(
    *,
    seed_curves: list[dict[str, list[float]]],
    seeds: list[int],
    keys: list[tuple[str, str]],
    title: str,
    out_base: Path,
) -> None:
    ncols = 2
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 2.45 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(seeds), 1)))

    for ax, (key, label) in zip(axes_arr, keys):
        plotted: list[list[float]] = []
        for idx, (seed, curves) in enumerate(zip(seeds, seed_curves)):
            y = curves.get(key, [])
            if not y:
                continue
            x = curves.get("epoch", list(range(1, len(y) + 1)))
            ax.plot(x, y, linewidth=0.75, alpha=0.42, color=colors[idx], label=str(seed))
            plotted.append(y)

        mean_x, mean_y = mean_curve(seed_curves, key)
        if len(mean_x):
            ax.plot(mean_x, mean_y, linewidth=1.65, color="black", label="mean")
            plotted.append(mean_y.tolist())

        ax.set_title(label, pad=3)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        set_ylim(ax, plotted)

    for ax in axes_arr[len(keys):]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(len(labels), 6),
        frameon=False,
    )
    fig.suptitle(title, y=0.995, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965), pad=1.0)
    save_all(fig, out_base)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="bwgnn")
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    configure_style()
    seed_curves: list[dict[str, list[float]]] = []
    kept_seeds: list[int] = []
    for seed in args.seeds:
        path = (
            Path("artifacts/logs")
            / args.dataset
            / args.model
            / args.run
            / f"seed_{seed}"
            / "phase2_train_log.csv"
        )
        if not path.exists():
            print(f"skip missing log: {path}")
            continue
        seed_curves.append(read_curve(path))
        kept_seeds.append(seed)

    if not seed_curves:
        raise FileNotFoundError("No Phase2 train logs found")

    out_dir = Path(args.out_dir)
    title_prefix = f"{args.dataset} {args.run}"
    plot_grid(
        seed_curves=seed_curves,
        seeds=kept_seeds,
        keys=LOSS_KEYS,
        title=f"{title_prefix}: train losses",
        out_base=out_dir / "phase2_train_losses",
    )
    plot_grid(
        seed_curves=seed_curves,
        seeds=kept_seeds,
        keys=METRIC_KEYS,
        title=f"{title_prefix}: validation metrics",
        out_base=out_dir / "phase2_val_metrics",
    )
    print(f"Wrote Phase2 training curves to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
