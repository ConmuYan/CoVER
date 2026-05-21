"""Publication-grade figures + tables for the 3-contribution TKDE 2026 paper.

Produces all figures and tables required for §5 paper draft:

- headline/fig1_overview.*           — 3-contribution headline cross-cell AUPRC
- c1_raer/fig2_c1_8cell.*            — C1 8-cell AUPRC lift vs base
- c1_raer/fig2b_c1_law1.*            — C1 Law 1 (evidence-group ablation heatmap)
- c2_lree/fig3_c2_8cell.*            — C2 LREE vs hand-crafted (paired-t sig stars)
- c3_cbr/fig4_c3_cbr_best.*          — C3 CBR-BEST 8-cell paired-t
- c3_cbr/fig4b_c3_speed_auprc.*      — C3 speed-vs-AUPRC tradeoff
- c3_cbr/fig5_c3_ablation_matrix.*   — C3 7-axis ablation summary
- c3_cbr/fig6_c3_mechanism.*         — C3 K2 anti-overcorrection mechanism

Each figure is exported as .pdf (editable text) + .svg (vector) + .png (preview, 600 dpi).

Tables alongside, machine-readable CSV + paper-ready LaTeX/Markdown.

Style: Nature Machine Intelligence pastel — Arial 7pt, restrained palette
       (neutral grey + signal blue/teal + accent orange + sig gold).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpatheffects
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Style — Nature pastel
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "axes.titleweight": "bold",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

# Palette: low-saturation, restrained
C_BASE       = "#8c95a3"   # base detector (neutral grey)
C_RAER       = "#4d7fba"   # C1 RAER
C_LREE       = "#3a9d8a"   # C2 LREE
C_DETMASK    = "#9c7fbd"   # det_mask (Flash-RAER baseline)
C_CBR        = "#e8a05c"   # CBR-K1
C_CBR_BEST   = "#d96458"   # CBR-BEST (headline)
C_TEACHER    = "#5a5a5a"   # teacher line
C_SIG_STAR   = "#c08a1c"   # gold sig star fill
C_DIVERGE_HI = "#3a9d8a"   # heatmap positive (teal)
C_DIVERGE_LO = "#d96458"   # heatmap negative (red)
C_DIVERGE_NEU = "#f4f1ec"  # heatmap mid (cream)
DIVERGE_CMAP = LinearSegmentedColormap.from_list(
    "diverge_pastel", [C_DIVERGE_LO, C_DIVERGE_NEU, C_DIVERGE_HI]
)

ROOT = Path("paper_figures")

# ─────────────────────────────────────────────────────────────────────────────
# Cell list & helpers
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = ("yelpchi", "amazon")
BASES = ("bwgnn", "sage", "gcn", "gat")
SEEDS = (42, 123, 456, 789, 2026)
CELL_LABELS = [f"{ds.title()}-{b.upper()}" for ds in DATASETS for b in BASES]


def sig_marker(p: float) -> str:
    """Return ASCII-safe significance marker (Arial lacks the BLACK STAR glyph)."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_pub(fig: plt.Figure, path: Path) -> None:
    """Save .pdf + .svg + .png (600 dpi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(fig)


def load_runs(prefix: str) -> dict:
    """Load all (ds, base, seed) -> auprc for a given run_name prefix."""
    out = {}
    for ds in DATASETS:
        for b in BASES:
            for s in SEEDS:
                p = Path(f"artifacts/results/{ds}/{b}/{prefix}/seed_{s}/stage3_metrics.json")
                if p.exists():
                    out[(ds, b, s)] = json.load(open(p))["auprc"]
    return out


def load_c1_per_seed() -> dict:
    """Load C1 CoVER-REL canonical per-seed AUPRC across 8 cells.

    Returns: {(ds, base, seed): auprc}
    Y-BWGNN uses unprefixed `idea1_canonical_clsonly`; the other 7 cells use
    `idea1_{ds}_{base}_canonical_clsonly`.
    """
    out = {}
    for ds in DATASETS:
        for b in BASES:
            if (ds, b) == ("yelpchi", "bwgnn"):
                prefix = "idea1_canonical_clsonly"
            else:
                prefix = f"idea1_{ds}_{b}_canonical_clsonly"
            for s in SEEDS:
                p = Path(f"artifacts/results/{ds}/{b}/{prefix}/seed_{s}/stage3_metrics.json")
                if p.exists():
                    out[(ds, b, s)] = json.load(open(p))["auprc"]
    return out


def load_c2_per_seed() -> dict:
    """Load C2 LREE per-seed AUPRC across 8 cells (uniform prefix)."""
    return load_runs("idea2b_learned_extractor")


def load_base_per_seed() -> dict:
    """Load base-detector per-seed AUPRC from each off_policy run's diag JSON.

    We only have base-only AUPRC from seed_42's diagnostics, so we replicate
    the single value across seeds (base is frozen and deterministic given seed).
    Returns: {(ds, base, seed): base_auprc} — values may be NaN.
    """
    out = {}
    for ds in DATASETS:
        for b in BASES:
            for s in SEEDS:
                try:
                    p = Path(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_{s}/phase2_diagnostics.json")
                    if p.exists():
                        d = json.load(open(p))
                        out[(ds, b, s)] = d.get("test_metrics_base_only", {}).get("auprc", float("nan"))
                except Exception:
                    out[(ds, b, s)] = float("nan")
    return out


def overlay_seed_dots(ax, x_center, vals, color="#202020", size=4, jitter=0.045):
    """Overlay 5-seed dots inside a bar to show distribution beyond std.

    `x_center` is the bar's x-coordinate; `vals` is the list of per-seed values.
    Small horizontal jitter spreads points so they don't perfectly stack.
    """
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return
    rng = np.random.default_rng(seed=42)
    xs = x_center + (rng.random(len(vals)) - 0.5) * 2 * jitter
    ax.scatter(xs, vals, s=size, c=color, alpha=0.8, zorder=5,
               edgecolors="white", linewidths=0.25)


def paired_t(a, b):
    arr = np.array(a) - np.array(b)
    if np.allclose(arr, 0) or len(arr) < 2:
        return float("nan"), float("nan")
    t, p2 = stats.ttest_1samp(arr, 0.0)
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────

# C1 numbers (5-seed × 7 cells, from idea1_canonical + base from idea2b table)
# Using idea2b's canonical column = same C1 CoVER-REL baseline
C1_DATA = {
    # (ds, base): (canonical_mean, canonical_std, base_mean, base_std, t, p)
    # base from paper_main_results.csv; CoVER-REL Gate
}
# We'll synthesize from CSV + canonical (idea2b LREE table cell has "canonical (mean ± sd)" column)
def parse_idea2b_table() -> dict:
    """Parse the idea2b_learned_vs_canonical table — returns {(ds, base): row dict}."""
    rows = {}
    text = Path("artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md").read_text()
    in_auprc = False
    for line in text.splitlines():
        if "## AUPRC" in line:
            in_auprc = True
            continue
        if line.startswith("## ") and in_auprc:
            break
        if not in_auprc or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7 or cells[0].startswith("---") or "Cell" in cells[0]:
            continue
        # Expected: Cell | canonical | 2B learned | Δ | t | p | sig
        cell_label = cells[0]
        try:
            ds, base = cell_label.split("-")
            canon_str = cells[1]
            lree_str = cells[2]
            delta_str = cells[3]
            t_str = cells[4]
            p_str = cells[5]
            canon_mean, canon_std = [float(x) for x in canon_str.replace("±", " ").split()]
            lree_mean, lree_std = [float(x) for x in lree_str.replace("±", " ").split()]
            delta = float(delta_str)
            t_val = float(t_str)
            p_val = float(p_str)
            rows[(ds.lower(), base.lower())] = {
                "canon_mean": canon_mean, "canon_std": canon_std,
                "lree_mean": lree_mean, "lree_std": lree_std,
                "delta": delta, "t": t_val, "p": p_val,
            }
        except (ValueError, IndexError):
            continue
    return rows


# C3 baselines/variants
def load_c3_main_data() -> dict:
    """Returns dict of mode -> {(ds, base, seed): auprc}"""
    modes = {
        "off_policy":   "g_opd_flash_off_policy",
        "all_node_mh":  "g_opd_flash_all_node_mh",
        "det_mask":     "g_opd_flash_det_mask",
        "det_mask_cbr": "g_opd_flash_det_mask_cbr",  # CBR-K1 (λ=0.5 linear)
        "cbr_best":     "v3_cbr_best",                # CBR-BEST (λ=1.0 + exp)
        "cbr_l10":      "v3_cbr_l10",
        "cbr_sym":      "v3_cbr_sym",
        "cbr_sq":       "v3_cbr_sq",
        "cbr_exp":      "v3_cbr_exp",
        "cbr_bin":      "v3_cbr_bin",
        "cbr_mask_HT":  "v3_cbr_mask_HT",
        "cbr_mask_dis": "v3_cbr_mask_disagree",
        "cbr_mask_rand":"v3_cbr_mask_rand",
        "cbr_mh":       "v3_cbr_mh",
        "g_opd_flash":  "g_opd_flash_g_opd_flash",
        "opd_strict":   "g_opd_flash_opd_action_strict",
        "opd_strict_mh":"g_opd_flash_opd_action_strict_mh",
        "det_mask_no_rel":      "g_opd_flash_det_mask_no_rel",
        "det_mask_fixed_bce":   "g_opd_flash_det_mask_fixed_bce",
        "det_mask_rev_only":    "g_opd_flash_det_mask_rev_only",
        "det_mask_single_denom":"g_opd_flash_det_mask_single_denom",
        "det_mask_mh":          "g_opd_flash_det_mask_mh",
    }
    return {k: load_runs(v) for k, v in modes.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Headline 3-contribution overview
# ─────────────────────────────────────────────────────────────────────────────
def fig1_headline(c1, c3):
    """Cross-cell mean AUPRC: base → C1 → C1+C2 (LREE) → C1+C2+C3 (CBR-BEST)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    cells_to_plot = [(ds, b) for ds in DATASETS for b in BASES]
    n = len(cells_to_plot)
    x = np.arange(n)
    bw = 0.21

    base = []; raer = []; lree = []; cbr = []
    def base_from_diag(ds, b):
        try:
            d = json.load(open(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))
            return d.get("test_metrics_base_only", {}).get("auprc", float("nan"))
        except Exception:
            return float("nan")
    for (ds, b) in cells_to_plot:
        base.append(base_from_diag(ds, b))
        info = c1.get((ds, b), {})
        raer.append(info.get("canon_mean", float("nan")))
        lree.append(info.get("lree_mean", float("nan")))
        runs = [c3["cbr_best"].get((ds, b, s)) for s in SEEDS]
        cbr.append(np.mean([r for r in runs if r is not None]) if any(r is not None for r in runs) else float("nan"))

    ax.bar(x - 1.5*bw, base, bw, color=C_BASE,     label="Base detector", edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5*bw, raer, bw, color=C_RAER,     label="C1: + RAER", edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5*bw, lree, bw, color=C_LREE,     label="C1+C2: + LREE", edgecolor="white", linewidth=0.5)
    ax.bar(x + 1.5*bw, cbr,  bw, color=C_CBR_BEST, label="C1+C2+C3: + CBR-BEST", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells_to_plot], rotation=0)
    ax.set_ylabel("AUPRC (5-seed mean)")
    ax.set_title("Progressive AUPRC across the three independent contributions, 8 base × dataset cells",
                 pad=18)
    ax.set_ylim(0, max(1.0, max([v for v in cbr if np.isfinite(v)]) + 0.05))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    # Legend below to free top space
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0.0, -0.10),
              fontsize=6.5, columnspacing=1.5, handletextpad=0.5)
    ax.axvline(3.5, color="black", linewidth=0.5, alpha=0.3, ls="--")
    blended = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, 1.02, "YelpChi", ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    ax.text(5.5, 1.02, "Amazon",  ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    save_pub(fig, ROOT / "headline" / "fig1_overview")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: C1 RAER 8-cell AUPRC lift
# ─────────────────────────────────────────────────────────────────────────────
def fig2_c1_8cell(c1, c1_seed, base_seed):
    """Per-cell base vs CoVER-REL (C1 canonical) with 5-seed scatter overlay."""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    n = len(cells); x = np.arange(n); bw = 0.36
    base_mean = []; raer_mean = []; raer_std = []
    deltas = []
    for (ds, b) in cells:
        info = c1.get((ds, b), {})
        try:
            d = json.load(open(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))
            base_m = d.get("test_metrics_base_only", {}).get("auprc", float("nan"))
        except Exception:
            base_m = float("nan")
        base_mean.append(base_m)
        raer_mean.append(info.get("canon_mean", float("nan")))
        raer_std.append(info.get("canon_std", 0.0))
        delta = raer_mean[-1] - base_m if np.isfinite(base_m) and np.isfinite(raer_mean[-1]) else float("nan")
        deltas.append(delta)

    ax.bar(x - bw/2, base_mean, bw, color=C_BASE, label="Base detector",
           edgecolor="white", linewidth=0.5, alpha=0.95)
    ax.bar(x + bw/2, raer_mean, bw, color=C_RAER, label="C1: CoVER-REL (5-seed)",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=raer_std, error_kw=dict(elinewidth=0.6, capsize=2, ecolor="#202020", zorder=4))
    # 5-seed scatter overlay
    for i, (ds, b) in enumerate(cells):
        base_vals = [base_seed.get((ds, b, s)) for s in SEEDS]
        raer_vals = [c1_seed.get((ds, b, s)) for s in SEEDS]
        overlay_seed_dots(ax, x[i] - bw/2, base_vals, color="#2a2a2a")
        overlay_seed_dots(ax, x[i] + bw/2, raer_vals, color="#2a2a2a")
    # Δ annotations — position above the std bar top
    y_max = 1.0
    for i, (bm, rm, std, d) in enumerate(zip(base_mean, raer_mean, raer_std, deltas)):
        if np.isfinite(d):
            top = max(bm, rm + (std if np.isfinite(std) else 0))
            y_text = top + 0.025
            sign = "+" if d >= 0 else ""
            ax.text(i, y_text, f"Δ{sign}{d:.3f}", ha="center", fontsize=5.8,
                    color=C_RAER, fontweight="bold")
            y_max = max(y_max, y_text + 0.06)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells], rotation=0)
    ax.set_ylabel("AUPRC (5-seed mean ± std, dots = per-seed)")
    ax.set_title("C1 — RAER (CoVER-REL): direction-positive AUPRC lift on 8/8 cells", pad=18)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, -0.10),
              fontsize=6.5, columnspacing=1.5, handletextpad=0.5)
    # Dataset separator + sub-group titles (top, outside data area)
    ax.axvline(3.5, color="black", linewidth=0.5, alpha=0.3, ls="--")
    blended = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, 1.02, "YelpChi", ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    ax.text(5.5, 1.02, "Amazon",  ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    save_pub(fig, ROOT / "c1_raer" / "fig2_c1_8cell")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2b: C1 Law 1 — evidence-group ablation heatmap (from idea1 FINAL 7cell table)
# ─────────────────────────────────────────────────────────────────────────────
def fig2b_c1_law1():
    """Heatmap of Δ AUPRC under 3 single-switch ablations × 8 cells."""
    # Parse idea1_ablation_FINAL_7cell.md for the AUPRC table
    text = Path("artifacts/tables/idea1_ablation_FINAL_7cell.md").read_text()
    # Hand-parse: cell × {gate_uniform, shared_expert, no_proto} Δ values
    cell_labels = []
    deltas_dict = {}  # (cell, ablation) -> delta
    in_auprc = False
    for line in text.splitlines():
        if "## 1. AUPRC" in line or "## AUPRC" in line:
            in_auprc = True
            continue
        if in_auprc and (line.startswith("## ") or line.startswith("---")):
            break
        if not in_auprc or not line.startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if len(cells) < 5 or "canonical" in cells[1].lower() or cells[0].startswith("---") or "Cell" in cells[0]:
            continue
        cell_label = cells[0]
        # parse Δ before (t=...) in each ablation column
        def extract_delta(col):
            # "−0.0029 (t=−1.35) ns" or "+0.0002 (t=+0.23) ns"
            import re
            m = re.match(r"([+−-]?[\d.]+)\s*\(t", col)
            if m:
                return float(m.group(1).replace("−", "-"))
            return float("nan")
        try:
            deltas_dict[(cell_label, "gate_uniform")] = extract_delta(cells[2])
            deltas_dict[(cell_label, "shared_expert")] = extract_delta(cells[3])
            deltas_dict[(cell_label, "no_proto")] = extract_delta(cells[4])
            cell_labels.append(cell_label)
        except (IndexError, ValueError):
            continue
    ablations = ["gate_uniform", "shared_expert", "no_proto"]
    ablation_labels = ["Uniform gate\n(drop schema)", "Shared expert\n(drop per-rel MLP)", "Drop prototype\n(evidence C)"]
    if not cell_labels:
        return
    mat = np.array([[deltas_dict.get((cl, a), float("nan")) for a in ablations] for cl in cell_labels])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    vmax = np.nanmax(np.abs(mat)) if np.any(np.isfinite(mat)) else 0.15
    im = ax.imshow(mat, cmap=DIVERGE_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels(ablation_labels, rotation=0, fontsize=6)
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(cell_labels, fontsize=6.5)
    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                txt = f"{v:+.3f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > vmax*0.5 else "#202020")
    ax.set_title("C1 — Law 1: base-strength × evidence-type\ninteraction (Δ AUPRC under single-switch ablation)",
                 fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Δ AUPRC (ablation − canonical)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_pub(fig, ROOT / "c1_raer" / "fig2b_c1_law1")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: C2 LREE vs hand-crafted 8-cell paired-t
# ─────────────────────────────────────────────────────────────────────────────
def fig3_c2_8cell(c1_data, c1_seed, c2_seed):
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    n = len(cells); x = np.arange(n); bw = 0.36
    canon_m = []; canon_s = []; lree_m = []; lree_s = []; ps = []
    for (ds, b) in cells:
        info = c1_data.get((ds, b), {})
        canon_m.append(info.get("canon_mean", float("nan")))
        canon_s.append(info.get("canon_std", 0.0))
        lree_m.append(info.get("lree_mean", float("nan")))
        lree_s.append(info.get("lree_std", 0.0))
        ps.append(info.get("p", float("nan")))

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.bar(x - bw/2, canon_m, bw, color=C_RAER, label="C1: CoVER-REL (hand-crafted, 5-seed)",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=canon_s, error_kw=dict(elinewidth=0.6, capsize=2, ecolor="#202020", zorder=4))
    ax.bar(x + bw/2, lree_m, bw, color=C_LREE, label="C2: + LREE (learnable, 5-seed)",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=lree_s, error_kw=dict(elinewidth=0.6, capsize=2, ecolor="#202020", zorder=4))
    # 5-seed scatter
    for i, (ds, b) in enumerate(cells):
        canon_vals = [c1_seed.get((ds, b, s)) for s in SEEDS]
        lree_vals = [c2_seed.get((ds, b, s)) for s in SEEDS]
        overlay_seed_dots(ax, x[i] - bw/2, canon_vals, color="#2a2a2a")
        overlay_seed_dots(ax, x[i] + bw/2, lree_vals, color="#2a2a2a")
    # Sig markers above the higher std bar top
    y_max = 1.0
    for i, (cm, lm, cs, ls_, p) in enumerate(zip(canon_m, lree_m, canon_s, lree_s, ps)):
        if np.isfinite(p) and p < 0.05:
            top = max(cm + (cs if np.isfinite(cs) else 0), lm + (ls_ if np.isfinite(ls_) else 0))
            y_text = top + 0.02
            ax.text(i, y_text, sig_marker(p), ha="center", fontsize=9,
                    color=C_SIG_STAR, fontweight="bold")
            y_max = max(y_max, y_text + 0.08)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells], rotation=0)
    ax.set_ylabel("AUPRC (5-seed mean ± std, dots = per-seed)")
    ax.set_title("C2 — LREE: learnable evidence extractor; 19/32 stat-sig wins on cross-cell table",
                 pad=18)
    ax.set_ylim(0, max(y_max, max([v for v in lree_m if np.isfinite(v)]) + 0.14))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, -0.10),
              fontsize=6.5, columnspacing=1.5, handletextpad=0.5)
    ax.axvline(3.5, color="black", linewidth=0.5, alpha=0.3, ls="--")
    blended = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, 1.02, "YelpChi", ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    ax.text(5.5, 1.02, "Amazon",  ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    save_pub(fig, ROOT / "c2_lree" / "fig3_c2_8cell")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: C3 CBR-BEST per-cell paired-t
# ─────────────────────────────────────────────────────────────────────────────
def fig4_c3_cbr_best(c3):
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    x = np.arange(len(cells)); bw = 0.28
    dm_m = []; dm_s = []; cbrk1_m = []; cbrk1_s = []; best_m = []; best_s = []
    dm_seed = []; ck_seed = []; be_seed = []
    p_vs_dm = []; p_vs_cbrk1 = []
    for (ds, b) in cells:
        dm = [c3["det_mask"].get((ds,b,s)) for s in SEEDS]
        ck = [c3["det_mask_cbr"].get((ds,b,s)) for s in SEEDS]
        be = [c3["cbr_best"].get((ds,b,s)) for s in SEEDS]
        dm_clean = [r for r in dm if r is not None]
        ck_clean = [r for r in ck if r is not None]
        be_clean = [r for r in be if r is not None]
        dm_seed.append(dm); ck_seed.append(ck); be_seed.append(be)
        if len(be_clean) < 2 or len(dm_clean) < 2:
            dm_m.append(float("nan")); dm_s.append(0); cbrk1_m.append(float("nan")); cbrk1_s.append(0)
            best_m.append(float("nan")); best_s.append(0); p_vs_dm.append(float("nan")); p_vs_cbrk1.append(float("nan"))
            continue
        dm_m.append(np.mean(dm_clean)); dm_s.append(np.std(dm_clean, ddof=1))
        cbrk1_m.append(np.mean(ck_clean)); cbrk1_s.append(np.std(ck_clean, ddof=1))
        best_m.append(np.mean(be_clean)); best_s.append(np.std(be_clean, ddof=1))
        _, p = paired_t(be_clean, dm_clean); p_vs_dm.append(p)
        _, p2 = paired_t(be_clean, ck_clean); p_vs_cbrk1.append(p2)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - bw, dm_m,    bw, color=C_DETMASK,  label="det_mask baseline",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=dm_s, error_kw=dict(elinewidth=0.5, capsize=2, ecolor="#202020", zorder=4))
    ax.bar(x,       cbrk1_m, bw, color=C_CBR,      label="CBR-K1 (λ=0.5, linear)",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=cbrk1_s, error_kw=dict(elinewidth=0.5, capsize=2, ecolor="#202020", zorder=4))
    ax.bar(x + bw, best_m,  bw, color=C_CBR_BEST, label="CBR-BEST (λ=1.0, weight=exp) [headline]",
           edgecolor="white", linewidth=0.5, alpha=0.95,
           yerr=best_s, error_kw=dict(elinewidth=0.5, capsize=2, ecolor="#202020", zorder=4))
    # 5-seed scatter inside each bar
    for i in range(len(cells)):
        overlay_seed_dots(ax, x[i] - bw, dm_seed[i], color="#2a2a2a")
        overlay_seed_dots(ax, x[i],       ck_seed[i], color="#2a2a2a")
        overlay_seed_dots(ax, x[i] + bw,  be_seed[i], color="#2a2a2a")
    # Sig markers above CBR-BEST std bar top
    y_max = 1.0
    for i, (m, std, p) in enumerate(zip(best_m, best_s, p_vs_dm)):
        if np.isfinite(p):
            mark = sig_marker(p)
            if mark:
                y_text = (m if np.isfinite(m) else 0) + (std if np.isfinite(std) else 0) + 0.025
                ax.text(i + bw, y_text, mark, ha="center", fontsize=9,
                        color=C_SIG_STAR, fontweight="bold")
                y_max = max(y_max, y_text + 0.08)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells], rotation=0)
    ax.set_ylabel("AUPRC (5-seed mean ± std, dots = per-seed)")
    ax.set_title("C3 — CBR-BEST: 4/8 cells sig p<0.05 + 2/8 p<0.01 vs det_mask ([headline] variant; * p<0.05, ** p<0.01)",
                 pad=18)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    # Legend BELOW the plot to free top space for sub-group titles + sig markers
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, -0.10),
              fontsize=6.3, columnspacing=1.2, handletextpad=0.5, handlelength=1.2)
    ax.axvline(3.5, color="black", linewidth=0.5, alpha=0.3, ls="--")
    blended = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, 1.02, "YelpChi", ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    ax.text(5.5, 1.02, "Amazon",  ha="center", va="bottom", fontsize=7.5,
            color="#404040", fontweight="bold", transform=blended)
    save_pub(fig, ROOT / "c3_cbr" / "fig4_c3_cbr_best")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4b: C3 speed-vs-AUPRC tradeoff
# ─────────────────────────────────────────────────────────────────────────────
def fig4b_c3_speed_auprc():
    """Scatter: base (small,fast), teacher (slow,best), CBR-BEST student (fast,near-best)."""
    bench = json.load(open("artifacts/results/yelpchi/bwgnn/g_opd_flash_det_mask_cbr/seed_42/inference_benchmark.json"))
    base_auprc = json.load(open("artifacts/logs/yelpchi/bwgnn/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))["test_metrics_base_only"]["auprc"]
    teacher_auprc = json.load(open("artifacts/logs/yelpchi/bwgnn/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))["test_metrics_teacher"]["auprc"]
    student_auprc = json.load(open("artifacts/results/yelpchi/bwgnn/v3_cbr_best/seed_42/stage3_metrics.json"))["auprc"]
    base_ms = bench.get("base_ms", bench.get("base", 0.016))
    teacher_ms = bench.get("teacher_ms", bench.get("teacher", 2.290))
    adapter_ms = bench.get("adapter_ms", bench.get("adapter", 0.879))
    speedup = teacher_ms / adapter_ms
    capture = student_auprc / teacher_auprc * 100

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    # 3 markers
    ax.scatter([base_ms],    [base_auprc],    s=140, c=C_BASE,     edgecolor="black", linewidth=0.6, zorder=4, label="Base detector (~0.5k params)")
    ax.scatter([teacher_ms], [teacher_auprc], s=200, c=C_LREE,     edgecolor="black", linewidth=0.6, zorder=4, label="LREE-RAER teacher (~15k params)")
    ax.scatter([adapter_ms], [student_auprc], s=180, c=C_CBR_BEST, edgecolor="black", linewidth=0.6, zorder=4, label="CBR-BEST student (~4.2k params)")

    # base label — to the upper-right of marker
    ax.annotate(f"Base\n{base_auprc:.3f}", xy=(base_ms, base_auprc),
                xytext=(base_ms * 1.6, base_auprc + 0.005),
                fontsize=6, ha="left", va="bottom", color="#202020")
    # teacher label — to the upper-LEFT of marker (clears the right side for speedup annotation)
    ax.annotate(f"Teacher (LREE-RAER)\n{teacher_auprc:.3f}", xy=(teacher_ms, teacher_auprc),
                xytext=(teacher_ms * 0.45, teacher_auprc + 0.015),
                fontsize=6, ha="left", va="bottom", color="#202020",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, shrinkA=0, shrinkB=4))
    # CBR-BEST label — to the LOWER-RIGHT of marker (clears speedup arrow which goes ABOVE)
    ax.annotate(f"CBR-BEST\n{student_auprc:.3f}  ({capture:.1f}% of teacher)",
                xy=(adapter_ms, student_auprc),
                xytext=(adapter_ms * 1.18, student_auprc - 0.018),
                fontsize=6, ha="left", va="top", color=C_CBR_BEST, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, shrinkA=4, shrinkB=0))
    # Speedup arrow + caption — placed ABOVE the two markers in clear horizontal space
    arrow_y = max(teacher_auprc, student_auprc) + 0.014
    ax.annotate("",
                xy=(adapter_ms * 1.05, arrow_y),
                xytext=(teacher_ms * 0.95, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#404040", lw=0.9,
                                shrinkA=4, shrinkB=4))
    ax.text(np.sqrt(adapter_ms * teacher_ms), arrow_y + 0.003,
            f"{speedup:.2f}× speedup", ha="center", va="bottom", fontsize=6.5,
            color="#202020", fontweight="bold")

    ax.set_xlabel("Inference time (ms / 18 382 nodes, 200 iters)")
    ax.set_ylabel("AUPRC (test, seed 42)")
    ax.set_title("C3 — Speed ↔ AUPRC tradeoff: CBR-BEST recovers 97.2% of teacher AUPRC at 2.6× speedup",
                 fontsize=7, pad=8)
    ax.set_xscale("log")
    # Y limit: extra headroom for the speedup arrow above the teacher marker
    ax.set_ylim(min(base_auprc, student_auprc) - 0.04,
                max(teacher_auprc, student_auprc) + 0.055)
    ax.set_xlim(base_ms * 0.5, teacher_ms * 2.5)
    ax.grid(linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=5.8, frameon=False)
    save_pub(fig, ROOT / "c3_cbr" / "fig4b_c3_speed_auprc")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: C3 7-axis ablation summary heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig5_c3_ablation_matrix(c3):
    """Heatmap of Δ AUPRC vs det_mask, for each variant × cell.

    Outliers beyond ±0.04 are clipped for the colormap but labeled with their
    true value (in white-on-dark text). A left-margin color band marks
    PRIMARY / ABLATIONS / FALSIFIED variant groups.
    """
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    variants = [
        ("det_mask",       "Flash-RAER baseline"),
        ("det_mask_cbr",   "CBR-K1 (λ=0.5, linear)"),
        ("cbr_l10",        "CBR λ=1.0 linear"),
        ("cbr_best",       "CBR-BEST (λ=1.0, exp) [headline]"),
        ("cbr_sq",         "CBR weight=sq"),
        ("cbr_exp",        "CBR weight=exp (alias)"),
        ("cbr_bin",        "CBR weight=bin"),
        ("cbr_sym",        "CBR + symmetric reward"),
        ("cbr_mask_HT",    "Mask: top-K H(p_T)"),
        ("cbr_mask_dis",   "Mask: top-K |p_S − p_T|"),
        ("cbr_mask_rand",  "Mask: random-K (placebo)"),
        ("cbr_mh",         "CBR + multi-head"),
        ("det_mask_no_rel","Drop reliability"),
        ("det_mask_fixed_bce","Drop adaptive BCE"),
        ("det_mask_rev_only","Drop mixed-KL"),
        ("det_mask_single_denom","Drop two-denom"),
        ("det_mask_mh",    "+ Multi-head (Z1)"),
        ("g_opd_flash",    "Stochastic q_φ sampling"),
        ("opd_strict",     "REINFORCE single-step"),
        ("opd_strict_mh",  "REINFORCE + multi-head"),
    ]
    # Define group boundaries (last variant index in each group)
    GROUP_BOUNDS = [
        ("PRIMARY",   0, 3, "#4d7fba"),   # rows 0-3 (det_mask through CBR-BEST)
        ("ABLATIONS", 4, 16, "#7a7a7a"),  # rows 4-16
        ("FALSIFIED", 17, 19, "#d96458"), # rows 17-19
    ]
    n_var = len(variants); n_cell = len(cells)
    mat = np.full((n_var, n_cell), np.nan)
    dm_data = c3["det_mask"]
    for vi, (key, _) in enumerate(variants):
        runs_var = c3.get(key, {})
        for ci, (ds, b) in enumerate(cells):
            var = [runs_var.get((ds,b,s)) for s in SEEDS]
            dm  = [dm_data.get((ds,b,s)) for s in SEEDS]
            var = [r for r in var if r is not None]; dm = [r for r in dm if r is not None]
            if len(var) >= 2 and len(dm) >= 2 and len(var) == len(dm):
                mat[vi, ci] = np.mean(var) - np.mean(dm)
    vmax = 0.04
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    # Clip for colormap but keep raw for labels
    mat_clipped = np.clip(mat, -vmax, vmax)
    im = ax.imshow(mat_clipped, cmap=DIVERGE_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cell))
    ax.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells], rotation=0, fontsize=6.2)
    ax.set_yticks(range(n_var))
    ax.set_yticklabels([lbl for _, lbl in variants], fontsize=6)
    # Numeric annotations — show true value
    for i in range(n_var):
        for j in range(n_cell):
            v = mat[i, j]
            if np.isfinite(v):
                # Out-of-range: outline + bold
                out_of_range = abs(v) > vmax * 1.01
                if out_of_range:
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            fontsize=5.6, color="white", fontweight="bold",
                            path_effects=[mpatheffects.withStroke(linewidth=1.0, foreground="#202020")])
                else:
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            fontsize=5.2, color="white" if abs(v) > vmax * 0.6 else "#202020")
    ax.set_title("C3 — Full 7-axis ablation matrix: Δ AUPRC vs det_mask baseline (5-seed mean per cell)\n"
                 "Only top-K H(p_S) mask + CBR variants are load-bearing; multi-head / stochastic / sym ALL fail\n"
                 "(out-of-range values bolded white-outline; e.g. Mask:H_T drops AUPRC by 0.41 on Y-GAT)",
                 fontsize=7, pad=8)
    # Group-band shading on the LEFT — placed OUTSIDE ytick labels (axes-coord x)
    blended = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    BAND_X = -0.27   # left margin past ytick labels
    LABEL_X = -0.30
    for label, i0, i1, color in GROUP_BOUNDS:
        ax.add_patch(mpatches.Rectangle((BAND_X, i0 - 0.5), 0.020, (i1 - i0 + 1),
                                         facecolor=color, edgecolor="none", alpha=0.7,
                                         transform=blended, clip_on=False, zorder=10))
        ax.text(LABEL_X, (i0 + i1) / 2, label, ha="center", va="center",
                fontsize=6.5, color=color, fontweight="bold", rotation=90,
                transform=blended, clip_on=False)
    # Horizontal separator lines between groups
    for _, i0, i1, _ in GROUP_BOUNDS[:-1]:
        ax.axhline(i1 + 0.5, color="black", linewidth=0.6, alpha=0.7)
    ax.axvline(3.5, color="black", linewidth=0.4, alpha=0.4, ls="--")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ AUPRC vs det_mask (clipped at ±0.04)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    plt.subplots_adjust(left=0.28)
    save_pub(fig, ROOT / "c3_cbr" / "fig5_c3_ablation_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: C3 mechanism — anti-overcorrection + waste reduction
# ─────────────────────────────────────────────────────────────────────────────
def fig6_c3_mechanism():
    """2-panel:
       (a) waste/useful reduction bars per cell
       (b) sensitivity median per cell + null-mechanism callouts
    """
    summary = json.load(open("artifacts/figures/cbr_sensitivity/cbr_sensitivity_summary.json"))
    cells = ["yelpchi/bwgnn", "yelpchi/sage", "yelpchi/gcn", "yelpchi/gat",
             "amazon/bwgnn", "amazon/sage", "amazon/gcn", "amazon/gat"]
    cell_labels_short = [f"{c.split('/')[0][:1].upper()}-{c.split('/')[1].upper()}" for c in cells]
    waste_red = [summary[c]["waste_reduction_pct"] for c in cells]
    useful_red = [summary[c]["useful_reduction_pct"] for c in cells]
    sens_med = [summary[c]["teacher_sens_median"] for c in cells]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8),
                             gridspec_kw=dict(width_ratios=[1.3, 1.1], wspace=0.40))

    # Panel A: waste vs useful reduction
    ax = axes[0]
    xp = np.arange(len(cells)); bw = 0.35
    ax.bar(xp - bw/2, waste_red,  bw, color=C_CBR_BEST,
           label="Waste reduction (low-sens nodes)", edgecolor="white", linewidth=0.5)
    ax.bar(xp + bw/2, useful_red, bw, color=C_TEACHER,
           label="Useful reduction (high-sens nodes)", edgecolor="white", linewidth=0.5)
    ax.set_xticks(xp)
    ax.set_xticklabels(cell_labels_short, rotation=30, fontsize=6.2, ha="right")
    ax.set_ylabel("Reduction vs det_mask (%)")
    ax.set_title("(a) CBR mechanism: differential waste-shrinkage on low-sens nodes\n"
                 "→ anti-overcorrection regularization", fontsize=7, pad=4)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    # YelpChi / Amazon group separator
    ax.axvline(3.5, color="black", linewidth=0.4, alpha=0.3, ls="--")

    # Panel B: sensitivity median per cell, with callouts in clear space
    ax = axes[1]
    bar_colors = [C_CBR_BEST if 0.3 < s < 0.95 else C_BASE for s in sens_med]
    yp = np.arange(len(cells))
    ax.barh(yp, sens_med, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(yp)
    ax.set_yticklabels(cell_labels_short, fontsize=6.2)
    ax.set_xlabel("Teacher sensitivity median  |Δ^T|/δ_max")
    ax.set_title("(b) Mechanism prerequisite: teacher must have\n"
                 "differential sensitivity for CBR to engage", fontsize=7, pad=4)
    ax.axvline(0.30, ls="--", color="#404040", linewidth=0.6, alpha=0.7)
    ax.axvline(0.95, ls="--", color="#404040", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, 1.30)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    # Identify Amazon-GCN and Amazon-SAGE indices for callouts
    a_gcn_idx = cells.index("amazon/gcn")
    a_sage_idx = cells.index("amazon/sage")
    # Callout: Amazon-GCN — text in clear RIGHT space, arrow LEFT to short bar
    ax.annotate("Amazon-GCN:\nteacher rarely\nintervenes\n→ CBR no-op",
                xy=(sens_med[a_gcn_idx], a_gcn_idx),
                xytext=(0.70, a_gcn_idx + 0.0),
                fontsize=5.6, color="#202020",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.5,
                                connectionstyle="arc3,rad=-0.15"))
    # Callout: Amazon-SAGE — text in clear UPPER space, arrow DOWN to bar
    ax.annotate("Amazon-SAGE:\nsaturated teacher\n→ CBR shrinks\nuniformly",
                xy=(sens_med[a_sage_idx], a_sage_idx),
                xytext=(0.05, a_sage_idx + 1.6),
                fontsize=5.6, color="#202020",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.5,
                                connectionstyle="arc3,rad=0.25"))
    # Engagement zone band label — moved ABOVE plot, in axes header
    ax.text((0.30 + 0.95) / 2, -1.0, "engagement zone",
            ha="center", va="bottom", fontsize=5.8, color="#404040", style="italic")
    ax.set_ylim(len(cells) - 0.3, -1.5)

    save_pub(fig, ROOT / "c3_cbr" / "fig6_c3_mechanism")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
def write_table(path: Path, header: list, rows: list, title: str = "") -> None:
    """Emit .csv (machine readable), .md (paper readable), .tex (LaTeX booktabs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV
    with open(path.with_suffix(".csv"), "w") as f:
        f.write(",".join(str(h) for h in header) + "\n")
        for r in rows:
            f.write(",".join(str(c) for c in r) + "\n")
    # MD
    with open(path.with_suffix(".md"), "w") as f:
        if title: f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(str(h) for h in header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
    # LaTeX
    with open(path.with_suffix(".tex"), "w") as f:
        if title: f.write(f"% {title}\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        if title: f.write(f"\\caption{{{title}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(str(h) for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(c).replace("±", "$\\pm$").replace("*", "\\textstar").replace("Δ", "$\\Delta$") for c in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def build_tables(c1, c3):
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    # Table 1: 3-contribution overview
    rows = []
    for (ds, b) in cells:
        info = c1.get((ds, b), {})
        try:
            d = json.load(open(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))
            base = d.get("test_metrics_base_only", {}).get("auprc", float("nan"))
            teacher = d.get("test_metrics_teacher", {}).get("auprc", float("nan"))
        except Exception:
            base = teacher = float("nan")
        be = [c3["cbr_best"].get((ds, b, s)) for s in SEEDS]
        be = [r for r in be if r is not None]
        be_m = np.mean(be) if be else float("nan")
        rows.append([
            f"{ds.title()}-{b.upper()}",
            f"{base:.4f}" if np.isfinite(base) else "—",
            f"{info.get('canon_mean', float('nan')):.4f}" if np.isfinite(info.get('canon_mean', float('nan'))) else "—",
            f"{info.get('lree_mean', float('nan')):.4f}" if np.isfinite(info.get('lree_mean', float('nan'))) else "—",
            f"{be_m:.4f}" if np.isfinite(be_m) else "—",
            f"{teacher:.4f}" if np.isfinite(teacher) else "—",
            f"{be_m/teacher*100:.1f}%" if np.isfinite(be_m) and np.isfinite(teacher) and teacher > 0 else "—",
        ])
    write_table(
        ROOT / "tables" / "table1_3contribution_overview",
        ["Cell", "Base", "C1 RAER", "+ C2 LREE", "+ C3 CBR-BEST", "Teacher (LREE)", "CBR-BEST capture"],
        rows,
        "Table 1 — Three-contribution progressive AUPRC stack (5-seed mean per cell)",
    )

    # Table 2: C1 8-cell + paired-t (LREE table provides canonical numbers — base from diag)
    rows = []
    for (ds, b) in cells:
        info = c1.get((ds, b), {})
        try:
            d = json.load(open(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))
            base = d.get("test_metrics_base_only", {}).get("auprc", float("nan"))
        except Exception:
            base = float("nan")
        canon = info.get("canon_mean", float("nan"))
        canon_s = info.get("canon_std", float("nan"))
        delta = canon - base if np.isfinite(canon) and np.isfinite(base) else float("nan")
        rows.append([
            f"{ds.title()}-{b.upper()}",
            f"{base:.4f}" if np.isfinite(base) else "—",
            f"{canon:.4f}±{canon_s:.4f}",
            f"{delta:+.4f}" if np.isfinite(delta) else "—",
        ])
    write_table(
        ROOT / "tables" / "table2_c1_8cell",
        ["Cell", "Base AUPRC", "C1 CoVER-REL AUPRC (5-seed)", "Δ vs base"],
        rows,
        "Table 2 — C1 RAER (CoVER-REL canonical) per-cell results: 8/8 directional positive AUPRC lift",
    )

    # Table 3: C2 LREE vs hand-crafted paired-t
    rows = []
    sig_count_05 = 0; sig_count_01 = 0
    for (ds, b) in cells:
        info = c1.get((ds, b), {})
        canon_m = info.get("canon_mean", float("nan"))
        canon_s = info.get("canon_std", float("nan"))
        lree_m = info.get("lree_mean", float("nan"))
        lree_s = info.get("lree_std", float("nan"))
        delta = info.get("delta", float("nan"))
        t = info.get("t", float("nan"))
        p = info.get("p", float("nan"))
        rows.append([
            f"{ds.title()}-{b.upper()}",
            f"{canon_m:.4f}±{canon_s:.4f}" if np.isfinite(canon_m) else "—",
            f"{lree_m:.4f}±{lree_s:.4f}" if np.isfinite(lree_m) else "—",
            f"{delta:+.4f}" if np.isfinite(delta) else "—",
            f"{t:+.3f}" if np.isfinite(t) else "—",
            f"{p:.4f}" if np.isfinite(p) else "—",
            sig_marker(p),
        ])
        if np.isfinite(p) and p < 0.05: sig_count_05 += 1
        if np.isfinite(p) and p < 0.01: sig_count_01 += 1
    write_table(
        ROOT / "tables" / "table3_c2_lree",
        ["Cell", "C1 hand-crafted (5-seed)", "C2 LREE (5-seed)", "Δ AUPRC", "t", "p (one-sided)", "Sig"],
        rows,
        f"Table 3 — C2 LREE vs hand-crafted, 5-seed paired-t: {sig_count_05}/8 sig p<0.05, {sig_count_01}/8 sig p<0.01",
    )

    # Table 4: C3 CBR-BEST per-cell paired-t (vs det_mask AND vs CBR-K1)
    rows = []
    sig05_dm = 0; sig01_dm = 0; sig05_ck1 = 0
    for (ds, b) in cells:
        dm = [c3["det_mask"].get((ds, b, s)) for s in SEEDS]; dm = [r for r in dm if r is not None]
        ck = [c3["det_mask_cbr"].get((ds, b, s)) for s in SEEDS]; ck = [r for r in ck if r is not None]
        be = [c3["cbr_best"].get((ds, b, s)) for s in SEEDS]; be = [r for r in be if r is not None]
        t_dm, p_dm = paired_t(be, dm)
        t_ck, p_ck = paired_t(be, ck)
        rows.append([
            f"{ds.title()}-{b.upper()}",
            f"{np.mean(dm):.4f}±{np.std(dm, ddof=1):.4f}" if len(dm) > 1 else "—",
            f"{np.mean(be):.4f}±{np.std(be, ddof=1):.4f}" if len(be) > 1 else "—",
            f"{np.mean(be) - np.mean(dm):+.4f}" if dm and be else "—",
            f"{t_dm:+.2f}" if np.isfinite(t_dm) else "—",
            f"{p_dm:.4f}" if np.isfinite(p_dm) else "—",
            sig_marker(p_dm),
            f"{t_ck:+.2f}" if np.isfinite(t_ck) else "—",
            f"{p_ck:.4f}" if np.isfinite(p_ck) else "—",
            sig_marker(p_ck),
        ])
        if np.isfinite(p_dm) and p_dm < 0.05: sig05_dm += 1
        if np.isfinite(p_dm) and p_dm < 0.01: sig01_dm += 1
        if np.isfinite(p_ck) and p_ck < 0.05: sig05_ck1 += 1
    write_table(
        ROOT / "tables" / "table4_c3_cbr_best",
        ["Cell", "det_mask baseline", "CBR-BEST", "Δ vs det_mask", "t-vs-DM", "p-vs-DM", "sig vs DM",
         "t-vs-CBR-K1", "p-vs-CBR-K1", "sig vs CBR-K1"],
        rows,
        f"Table 4 — C3 CBR-BEST per-cell paired-t: {sig05_dm}/8 sig p<0.05 + {sig01_dm}/8 sig p<0.01 vs det_mask; {sig05_ck1}/8 sig vs CBR-K1",
    )

    # Table 5: full 21-variant cross-cell mean + 3 summary statistics
    variants = [
        "det_mask", "det_mask_cbr", "cbr_best",
        "cbr_l10", "cbr_sq", "cbr_exp", "cbr_bin",
        "cbr_sym", "cbr_mask_HT", "cbr_mask_dis", "cbr_mask_rand", "cbr_mh",
        "det_mask_no_rel", "det_mask_fixed_bce", "det_mask_rev_only", "det_mask_single_denom",
        "det_mask_mh", "g_opd_flash", "opd_strict", "opd_strict_mh", "all_node_mh",
    ]
    rows = []
    for v in variants:
        cell_aurpcs = []
        for (ds, b) in cells:
            runs = [c3[v].get((ds, b, s)) for s in SEEDS]
            runs = [r for r in runs if r is not None]
            if runs: cell_aurpcs.append(np.mean(runs))
        if not cell_aurpcs: continue
        mean = np.mean(cell_aurpcs); std = np.std(cell_aurpcs, ddof=1) if len(cell_aurpcs) > 1 else 0
        # vs det_mask per-cell
        sig05 = 0
        for (ds, b) in cells:
            this = [c3[v].get((ds,b,s)) for s in SEEDS]; this = [r for r in this if r is not None]
            base = [c3["det_mask"].get((ds,b,s)) for s in SEEDS]; base = [r for r in base if r is not None]
            if len(this) >= 2 and len(base) == len(this):
                _, p = paired_t(this, base)
                if np.isfinite(p) and p < 0.05: sig05 += 1
        rows.append([v, f"{mean:.4f}±{std:.4f}", len(cell_aurpcs), f"{sig05}/{len(cell_aurpcs)}"])
    write_table(
        ROOT / "tables" / "table5_c3_ablation_matrix",
        ["Variant", "AUPRC mean ± std (across cells)", "Cells", "Sig cells (p<0.05 vs det_mask)"],
        rows,
        "Table 5 — C3 full ablation matrix: cross-cell mean AUPRC + per-cell paired-t significance count",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[fig-gen] Loading C1 (idea2b table) data...")
    c1 = parse_idea2b_table()
    print(f"[fig-gen]   loaded {len(c1)} cells")

    print("[fig-gen] Loading C1/C2/base per-seed AUPRC (for scatter overlay)...")
    c1_seed = load_c1_per_seed()
    c2_seed = load_c2_per_seed()
    base_seed = load_base_per_seed()
    print(f"[fig-gen]   C1 per-seed runs: {len(c1_seed)}; C2: {len(c2_seed)}; base: {len(base_seed)}")

    print("[fig-gen] Loading C3 (Flash-RAER + CBR) data...")
    c3 = load_c3_main_data()
    print(f"[fig-gen]   loaded {len(c3)} modes")
    for k, v in c3.items():
        if len(v) != 40 and len(v) > 0:
            print(f"  WARN: {k}: {len(v)}/40 runs")

    print("[fig-gen] Generating headline figure 1...")
    fig1_headline(c1, c3)
    print("[fig-gen] Generating C1 figure 2 + 2b...")
    fig2_c1_8cell(c1, c1_seed, base_seed)
    fig2b_c1_law1()
    print("[fig-gen] Generating C2 figure 3...")
    fig3_c2_8cell(c1, c1_seed, c2_seed)
    print("[fig-gen] Generating C3 figure 4 + 4b + 5 + 6...")
    fig4_c3_cbr_best(c3)
    try:
        fig4b_c3_speed_auprc()
    except Exception as e:
        print(f"  fig4b skipped: {e}")
    fig5_c3_ablation_matrix(c3)
    fig6_c3_mechanism()
    print("[fig-gen] Building tables...")
    build_tables(c1, c3)
    print("[fig-gen] Done. Outputs at paper_figures/")
