"""Top-tier additions to paper_figures/ — F3 priority set.

Adds 4 publication-quality figures to complement build_paper_figures.py:

- headline/fig1_hero.{pdf,svg,png}      Nature-style hero composite
                                          (3 schematic panels + 1 quant panel)
- c3_cbr/fig7_violin.{pdf,svg,png}      Per-cell per-mode seed-level distribution
- c3_cbr/fig8_forest.{pdf,svg,png}      Forest plot of CBR-BEST Δ AUPRC + 95% CI
- c3_cbr/fig9_mechanism_scatter.{pdf,svg,png}
                                          Teacher sensitivity × CBR lift scatter
                                          (causal mechanism evidence)

All inherit the build_paper_figures.py style (NMI pastel, Arial 7pt, editable
text). Significance markers use ASCII-safe asterisks. Color story is
consistent across the existing figure set.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

# Reuse styling + palette from build_paper_figures.py
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

# Consistent palette across all figures
C_BASE      = "#8c95a3"
C_RAER      = "#4d7fba"
C_LREE      = "#3a9d8a"
C_DETMASK   = "#9c7fbd"
C_CBR       = "#e8a05c"
C_CBR_BEST  = "#d96458"
C_TEACHER   = "#5a5a5a"
C_YELPCHI   = "#4d7fba"   # YelpChi cells (cool blue)
C_AMAZON    = "#e8a05c"   # Amazon cells (warm orange)
C_GOLD      = "#c08a1c"   # significance gold
C_INK       = "#1a1a1a"
C_PANEL     = "#f4f1ec"   # cream
C_PANEL_DK  = "#e4dfd6"

ROOT = Path("paper_figures")
DATASETS = ("yelpchi", "amazon")
BASES = ("bwgnn", "sage", "gcn", "gat")
SEEDS = (42, 123, 456, 789, 2026)


def sig_marker(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def save_pub(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(fig)


def load_runs(prefix: str) -> dict:
    out = {}
    for ds in DATASETS:
        for b in BASES:
            for s in SEEDS:
                p = Path(f"artifacts/results/{ds}/{b}/{prefix}/seed_{s}/stage3_metrics.json")
                if p.exists():
                    out[(ds, b, s)] = json.load(open(p))["auprc"]
    return out


def paired_t(a, b):
    arr = np.array(a) - np.array(b)
    if np.allclose(arr, 0) or len(arr) < 2:
        return float("nan"), float("nan"), float("nan")
    t, p2 = stats.ttest_1samp(arr, 0.0)
    mean_diff = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2), se


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 (hero) — Nature-style composite: 3 schematics + headline quant
# ─────────────────────────────────────────────────────────────────────────────
def _draw_box(ax, xy, w, h, text, fc=C_PANEL, ec=C_INK, lw=0.7, fontsize=6.5, fw="normal"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=C_INK, fontweight=fw, zorder=3)


def _draw_arrow(ax, x0, y0, x1, y1, color=C_INK, lw=0.7, ls="-", connstyle="arc3,rad=0"):
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle="-|>", mutation_scale=8,
                          linewidth=lw, color=color, zorder=4,
                          linestyle=ls, connectionstyle=connstyle)
    ax.add_patch(arr)


def _schematic_c1(ax):
    """C1 RAER paradigm: frozen base + 4 contract icons + bounded residual."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Title at top — sits in dedicated header strip (y=0.95)
    ax.text(0.5, 0.95, "(a) C1 — RAER paradigm: contract-enforced post-hoc reasoner",
            ha="center", va="top", fontsize=8, fontweight="bold", color=C_INK)
    # [frozen] tag positioned ABOVE base detector box (cleared space)
    ax.text(0.13, 0.80, "[frozen] SHA-256 verified", ha="center", va="center",
            fontsize=5.5, color=C_INK, style="italic")
    # Base detector
    _draw_box(ax, (0.03, 0.50), 0.20, 0.20, "Base detector\n(GNN, frozen)",
              fc=C_BASE, ec=C_INK, fontsize=6.2, fw="bold")
    # base output (logit + embedding) — label ABOVE the arrow
    ax.text(0.285, 0.66, "$b_i, z_i$", ha="center", va="bottom", fontsize=6.2, style="italic")
    _draw_arrow(ax, 0.23, 0.60, 0.34, 0.60)
    # CoVER-REL reasoner
    _draw_box(ax, (0.34, 0.50), 0.22, 0.20, "CoVER-REL\nreasoner",
              fc=C_RAER, ec=C_INK, fontsize=6.2, fw="bold")
    # score-blind input (relation evidence) from below
    _draw_arrow(ax, 0.45, 0.36, 0.45, 0.50)
    ax.text(0.45, 0.30, "score-blind\nrelation evidence", ha="center", va="top",
            fontsize=5.5, color=C_INK, style="italic")
    # bounded residual output (right) toward final logit
    ax.text(0.66, 0.66, "$+ \\delta_{\\max}\\tanh(u)$", ha="center", va="bottom",
            fontsize=6.2, color=C_RAER, fontweight="bold")
    _draw_arrow(ax, 0.56, 0.60, 0.76, 0.60, color=C_RAER, lw=1.0)
    # final logit box
    _draw_box(ax, (0.76, 0.50), 0.20, 0.20, "Final logit\n$s_i$",
              fc="#fff8f0", ec=C_INK, fontsize=6.2, fw="bold")
    # 4 contract chips (bottom)
    contracts = [
        ("C1: Base-freeze\n(SHA-256)", 0.02),
        ("C2: Score-blind\n(input contract)", 0.27),
        ("C3: Train-only\nprototype", 0.52),
        ("C4: $|\\delta| \\leq \\delta_{\\max}$\n(bounded)", 0.77),
    ]
    for txt, x in contracts:
        _draw_box(ax, (x, 0.05), 0.21, 0.14, txt, fc="#f7e6c8", ec="#b07e1c",
                  fontsize=5.6, fw="bold")


def _schematic_c2(ax):
    """C2 LREE: replace hand-crafted 9-dim → per-relation GCN+MLP encoder."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Title
    ax.text(0.5, 0.95, "(b) C2 — LREE: learnable per-relation evidence encoder",
            ha="center", va="top", fontsize=8, fontweight="bold", color=C_INK)
    # Left header (left-aligned, inside dedicated header strip)
    ax.text(0.13, 0.85, "C1 hand-crafted evidence (9-dim per relation)",
            ha="center", va="center", fontsize=6, fontweight="bold", color=C_RAER)
    # 3 hand-crafted boxes (left)
    for i, (g, label) in enumerate([("A", "Structural"), ("B", "Feat. neighbor"), ("C", "Prototype")]):
        _draw_box(ax, (0.04, 0.62 - i * 0.13), 0.18, 0.10, f"{g}: {label}",
                  fc=C_RAER, ec=C_INK, fontsize=5.5)
    # Upgrade arrow
    _draw_arrow(ax, 0.23, 0.47, 0.37, 0.47, lw=1.0, color="#7a7a7a")
    ax.text(0.30, 0.51, "→ upgrade →", ha="center", va="bottom",
            fontsize=5.8, color="#7a7a7a", style="italic")
    # Right header
    ax.text(0.70, 0.85, "C2 LREE encoder (~14k params)",
            ha="center", va="center", fontsize=6, fontweight="bold", color=C_LREE)
    # 3 relation pipelines
    rel_y = [0.62, 0.45, 0.28]
    for i, (rel, ycoord) in enumerate(zip(["RUR", "RSR", "RTR"], rel_y)):
        _draw_box(ax, (0.42, ycoord), 0.10, 0.08, f"GCN\n{rel}", fc=C_LREE, ec=C_INK, fontsize=5.3)
        _draw_arrow(ax, 0.53, ycoord + 0.04, 0.61, ycoord + 0.04, lw=0.6)
        _draw_box(ax, (0.61, ycoord), 0.10, 0.08, "MLP", fc=C_LREE, ec=C_INK, fontsize=5.5)
        _draw_arrow(ax, 0.72, ycoord + 0.04, 0.80, ycoord + 0.04, lw=0.6)
        _draw_box(ax, (0.80, ycoord), 0.13, 0.08, f"$e_r^{{({i+1})}}$",
                  fc="#fff8f0", ec=C_INK, fontsize=5.8, fw="bold")
    # Contracts preserved chip (bottom-left)
    ax.text(0.30, 0.13, "All 4 contracts (C1–C4) preserved",
            ha="center", va="center", fontsize=5.8, fontweight="bold", color="#b07e1c",
            bbox=dict(facecolor="#f7e6c8", edgecolor="#b07e1c", linewidth=0.7, pad=2, boxstyle="round"))
    # Law 3 note (bottom-right)
    ax.text(0.70, 0.13, "Law 3: encoder absorbs explicit prototype\non weak bases (LREE makes C-group redundant)",
            ha="center", va="center", fontsize=5.5, color=C_INK, style="italic")


def _schematic_c3(ax):
    """C3 CBR loss: top-K mask + CBR penalty curve + sensitivity-weighted."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Title
    ax.text(0.5, 0.95, "(c) C3 — Flash-RAER + CBR-BEST loss: contract-budgeted residual regularizer",
            ha="center", va="top", fontsize=8, fontweight="bold", color=C_INK)
    # 3-stage pipeline (top half)
    _draw_box(ax, (0.04, 0.65), 0.17, 0.14, "Train nodes\n(all $N_{train}$)",
              fc="#e8e8e8", ec=C_INK, fontsize=5.6)
    ax.text(0.265, 0.785, "top-K $H(p^S)$", ha="center", va="bottom",
            fontsize=5.5, style="italic")
    _draw_arrow(ax, 0.22, 0.72, 0.32, 0.72)
    _draw_box(ax, (0.32, 0.65), 0.18, 0.14, "Hard-example\nmask $\\mathcal{B}$",
              fc=C_DETMASK, ec=C_INK, fontsize=5.6, fw="bold")
    _draw_arrow(ax, 0.51, 0.72, 0.61, 0.72)
    _draw_box(ax, (0.61, 0.65), 0.23, 0.14, "Student adapter\n$|\\delta^S| \\leq \\delta_{\\max}$",
              fc=C_CBR_BEST, ec=C_INK, fontsize=5.8, fw="bold")
    # Loss equation (right portion, mid-height) — boxed
    ax.text(0.71, 0.38,
            "$\\mathcal{L}_{\\mathrm{CBR}} = \\lambda \\cdot E_{i\\in\\mathcal{B}}\\!\\left[\\frac{|\\delta^S_i|}{\\delta_{\\max}} \\cdot \\exp\\!\\left(-\\frac{|\\Delta^T_i|}{\\delta_{\\max}}\\right)\\right]$",
            ha="center", va="center", fontsize=6.8, color=C_INK,
            bbox=dict(facecolor=C_PANEL, edgecolor=C_INK, linewidth=0.7, boxstyle="round,pad=0.3"))
    ax.text(0.71, 0.17, "Anti-overcorrection regularization\n(K2: differential waste-shrinkage)",
            ha="center", va="center", fontsize=5.5, color=C_INK, style="italic")
    # Inset weight-curve plot (left half, bottom) — fits in clear space
    inset_ax = ax.inset_axes([0.04, 0.06, 0.36, 0.45])
    s = np.linspace(0, 1, 100)
    weight_exp = np.exp(-s)
    weight_lin = 1 - s
    inset_ax.plot(s, weight_exp, color=C_CBR_BEST, linewidth=1.5,
                  label="CBR-BEST $\\exp(-s)$")
    inset_ax.plot(s, weight_lin, color=C_CBR, linewidth=1.0, linestyle="--",
                  label="CBR-K1 $(1-s)$")
    inset_ax.fill_between(s, 0, weight_exp, alpha=0.15, color=C_CBR_BEST)
    inset_ax.set_xlabel("teacher sensitivity $s$", fontsize=5.5)
    inset_ax.set_ylabel("CBR weight", fontsize=5.5)
    inset_ax.tick_params(labelsize=5)
    inset_ax.legend(fontsize=5, loc="upper right", frameon=False)
    inset_ax.set_xlim(0, 1); inset_ax.set_ylim(0, 1.05)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)


def fig1_hero():
    """Nature-style composite hero: 3 schematics (top) + quant headline (bottom).

    Layout: 4 vertically-stacked rows with explicit height ratios.
    Each schematic gets a dedicated row with its own title strip.
    """
    fig = plt.figure(figsize=(7.4, 9.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.3, 1.5, 1.7],
                          hspace=0.45)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    ax_d = fig.add_subplot(gs[3])
    _schematic_c1(ax_a)
    _schematic_c2(ax_b)
    _schematic_c3(ax_c)
    # Quant headline (panel d)
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    n = len(cells); x = np.arange(n); bw = 0.21
    base_v = []; raer_v = []; lree_v = []; cbr_v = []
    for (ds, b) in cells:
        try:
            d = json.load(open(f"artifacts/logs/{ds}/{b}/g_opd_flash_off_policy/seed_42/phase2_diagnostics.json"))
            base_v.append(d.get("test_metrics_base_only", {}).get("auprc", float("nan")))
        except Exception:
            base_v.append(float("nan"))
    import re
    text = Path("artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md").read_text()
    info = {}
    for line in text.splitlines():
        if not line.startswith("|") or "Cell" in line or "---" in line: continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) >= 3 and "-" in c[0]:
            try:
                ds_, base_ = c[0].split("-")
                canon_m = float(c[1].split("±")[0].strip())
                lree_m = float(c[2].split("±")[0].strip())
                info[(ds_.lower(), base_.lower())] = (canon_m, lree_m)
            except (ValueError, IndexError):
                continue
    for (ds, b) in cells:
        cm, lm = info.get((ds, b), (float("nan"), float("nan")))
        raer_v.append(cm); lree_v.append(lm)
    cbr_data = load_runs("v3_cbr_best")
    for (ds, b) in cells:
        runs = [cbr_data.get((ds, b, s)) for s in SEEDS]
        runs = [r for r in runs if r is not None]
        cbr_v.append(np.mean(runs) if runs else float("nan"))
    ax_d.bar(x - 1.5*bw, base_v, bw, color=C_BASE,     edgecolor="white", linewidth=0.5, label="Base detector")
    ax_d.bar(x - 0.5*bw, raer_v, bw, color=C_RAER,     edgecolor="white", linewidth=0.5, label="+ C1 RAER")
    ax_d.bar(x + 0.5*bw, lree_v, bw, color=C_LREE,     edgecolor="white", linewidth=0.5, label="+ C2 LREE")
    ax_d.bar(x + 1.5*bw, cbr_v,  bw, color=C_CBR_BEST, edgecolor="white", linewidth=0.5, label="+ C3 CBR-BEST")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{ds[:1].upper()}-{b.upper()}" for (ds, b) in cells], fontsize=6.5)
    ax_d.set_ylabel("AUPRC (5-seed mean)", fontsize=7)
    ax_d.set_title("(d) Progressive AUPRC across the three contributions on 8 base × dataset cells",
                   fontsize=8, fontweight="bold", color=C_INK, pad=18)
    ax_d.set_ylim(0, max([v for v in cbr_v if np.isfinite(v)]) + 0.10)
    ax_d.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax_d.set_axisbelow(True)
    # Legend below the plot
    ax_d.legend(loc="upper left", ncol=4, bbox_to_anchor=(0.0, -0.15),
                fontsize=6.5, columnspacing=1.5, handletextpad=0.5)
    ax_d.axvline(3.5, color="black", linewidth=0.4, alpha=0.3, ls="--")
    blended = mpl.transforms.blended_transform_factory(ax_d.transData, ax_d.transAxes)
    ax_d.text(1.5, 1.02, "YelpChi", ha="center", va="bottom", fontsize=7,
              color="#404040", fontweight="bold", transform=blended)
    ax_d.text(5.5, 1.02, "Amazon", ha="center", va="bottom", fontsize=7,
              color="#404040", fontweight="bold", transform=blended)
    save_pub(fig, ROOT / "headline" / "fig1_hero")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Per-cell per-mode violin/box plot (seed-level distribution)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_violin():
    """Stacked 2x1 grid: top = YelpChi cells, bottom = Amazon cells.
    Each cell: 4 violins (off_policy / det_mask / CBR-K1 / CBR-BEST)."""
    modes = [
        ("off_policy",    "g_opd_flash_off_policy",    C_BASE,     "Off-policy KL"),
        ("det_mask",      "g_opd_flash_det_mask",      C_DETMASK,  "det_mask baseline"),
        ("det_mask_cbr",  "g_opd_flash_det_mask_cbr",  C_CBR,      "CBR-K1 (λ=0.5, linear)"),
        ("cbr_best",      "v3_cbr_best",                C_CBR_BEST, "CBR-BEST (λ=1.0, exp)"),
    ]
    data_by_mode = {m: load_runs(p) for m, p, *_ in modes}
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), gridspec_kw=dict(hspace=0.35))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        cell_positions = np.arange(len(BASES)) * 1.2  # spacing between cells
        n_modes = len(modes)
        violin_w = 0.18
        offsets = np.linspace(-0.30, 0.30, n_modes)
        for mi, (mname, _, color, _) in enumerate(modes):
            vals_per_cell = []
            for b in BASES:
                v = [data_by_mode[mname].get((ds, b, s)) for s in SEEDS]
                v = [r for r in v if r is not None]
                vals_per_cell.append(v if v else [np.nan])
            positions = cell_positions + offsets[mi]
            parts = ax.violinplot(vals_per_cell, positions=positions, widths=violin_w,
                                   showmeans=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_edgecolor(C_INK)
                pc.set_alpha(0.7); pc.set_linewidth(0.4)
            # overlay seed dots
            for i, vals in enumerate(vals_per_cell):
                if not vals or all(np.isnan(vals)): continue
                ax.scatter([positions[i]] * len(vals), vals, s=4,
                           color=C_INK, alpha=0.55, zorder=3, edgecolors="white", linewidth=0.3)
                # mean horizontal line
                ax.hlines(np.nanmean(vals), positions[i] - 0.08, positions[i] + 0.08,
                          color=color, linewidth=1.2, zorder=4)
        ax.set_xticks(cell_positions)
        ax.set_xticklabels([b.upper() for b in BASES], fontsize=7)
        ax.set_ylabel("AUPRC", fontsize=7)
        ax.set_title(f"({chr(ord('a') + di)}) {ds.title()} — 5-seed per-cell distributions across 4 distillation modes",
                     fontsize=7, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    # shared legend (top-right of first panel)
    handles = [mpatches.Patch(facecolor=c, edgecolor=C_INK, alpha=0.7, label=lbl)
               for _, _, c, lbl in modes]
    axes[0].legend(handles=handles, loc="upper right", ncol=2, fontsize=6, frameon=False)
    save_pub(fig, ROOT / "c3_cbr" / "fig7_violin")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Forest plot of CBR-BEST Δ AUPRC + 95% CI
# ─────────────────────────────────────────────────────────────────────────────
def fig8_forest():
    """Horizontal forest plot: 8 cells sorted by Δ AUPRC vs det_mask,
    each with point estimate + 95% CI; significance star inline; Bonferroni
    critical value as inset dashed line."""
    dm_data = load_runs("g_opd_flash_det_mask")
    cbr_data = load_runs("v3_cbr_best")
    cells = [(ds, b) for ds in DATASETS for b in BASES]
    entries = []
    for (ds, b) in cells:
        dm = [dm_data.get((ds, b, s)) for s in SEEDS]; dm = [r for r in dm if r is not None]
        be = [cbr_data.get((ds, b, s)) for s in SEEDS]; be = [r for r in be if r is not None]
        if len(dm) < 2 or len(be) < 2: continue
        arr = np.array(be) - np.array(dm)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        tc = stats.t.ppf(0.975, df=len(arr) - 1)
        ci_lo = mean - tc * se; ci_hi = mean + tc * se
        t_val, p_two = stats.ttest_1samp(arr, 0.0)
        p_one = float(p_two / 2 if t_val > 0 else 1 - p_two / 2)
        entries.append({
            "cell": f"{ds.title()}-{b.upper()}",
            "dataset": ds,
            "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "t": float(t_val), "p": p_one,
        })
    entries.sort(key=lambda e: e["mean"])
    y_pos = np.arange(len(entries))
    # Wider figure to allow t/p text margin; left margin for cell labels, right margin for t/p
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, e in enumerate(entries):
        color = C_YELPCHI if e["dataset"] == "yelpchi" else C_AMAZON
        sig_color = C_GOLD if e["p"] < 0.05 else color
        sig_alpha = 1.0 if e["p"] < 0.05 else 0.55
        ax.errorbar([e["mean"]], [y_pos[i]],
                    xerr=[[e["mean"] - e["ci_lo"]], [e["ci_hi"] - e["mean"]]],
                    fmt='none', ecolor=color, capsize=2.5, elinewidth=0.8, alpha=sig_alpha)
        ax.scatter([e["mean"]], [y_pos[i]], s=70, c=sig_color, edgecolor=C_INK,
                   linewidth=0.6, alpha=sig_alpha, zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["cell"] for e in entries], fontsize=7)
    ax.set_xlabel("Δ AUPRC vs det_mask baseline (5-seed paired, 95% CI)", fontsize=7)
    ax.axvline(0, color=C_INK, linewidth=0.6, ls="-", alpha=0.7)
    # Bonferroni reference line — label below the top entry, INSIDE the plot
    ax.axvline(0.005, color="#8c95a3", linewidth=0.5, ls=":", alpha=0.6)
    ax.text(0.005, -0.85, "Δ=0.005", ha="center", va="top",
            fontsize=5.5, color="#404040", style="italic")
    # X limit: expand RIGHT so t/p text has clear margin (in data-space, outside CI bars)
    ax.set_xlim(-0.013, 0.045)
    # Place t/p text in dedicated right margin (x=0.034 in data coords)
    label_x = 0.034
    for i, e in enumerate(entries):
        mark = sig_marker(e["p"])
        ax.text(label_x, y_pos[i],
                f"$t$={e['t']:+.2f},  $p$={e['p']:.4f}  {mark}",
                ha="left", va="center", fontsize=6, color=C_INK)
    # Vertical divider between CI plot area and text margin
    ax.axvline(label_x - 0.001, color="#cccccc", linewidth=0.4, alpha=0.6)
    ax.set_title("Forest plot — CBR-BEST per-cell effect size (sorted by Δ AUPRC)\n"
                 "* p<0.05  ** p<0.01  *** p<0.001 (one-sided paired-$t$, df=4)",
                 fontsize=7, fontweight="bold", pad=8)
    yc_handle = ax.scatter([], [], s=70, c=C_YELPCHI, edgecolor=C_INK, label="YelpChi")
    az_handle = ax.scatter([], [], s=70, c=C_AMAZON,  edgecolor=C_INK, label="Amazon")
    sig_handle = ax.scatter([], [], s=70, c=C_GOLD,   edgecolor=C_INK, label="Sig p<0.05")
    ax.legend(handles=[yc_handle, az_handle, sig_handle], loc="lower right",
              fontsize=6, ncol=1, frameon=False)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.2, len(entries) - 0.3)
    save_pub(fig, ROOT / "c3_cbr" / "fig8_forest")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Sensitivity × CBR lift scatter (causal mechanism)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_mechanism_scatter():
    """X = teacher sensitivity median (K2), Y = CBR-BEST Δ AUPRC.
    Cells colored by dataset. Labels manually placed to avoid overlap (small
    leader lines connect them to markers when needed)."""
    sens_summary = json.load(open("artifacts/figures/cbr_sensitivity/cbr_sensitivity_summary.json"))
    dm_data = load_runs("g_opd_flash_det_mask")
    cbr_data = load_runs("v3_cbr_best")
    entries = []
    for (ds, b) in [(d, b) for d in DATASETS for b in BASES]:
        key = f"{ds}/{b}"
        if key not in sens_summary: continue
        sens_median = sens_summary[key].get("teacher_sens_median")
        sens_high_frac = sens_summary[key].get("teacher_sens_high_frac")
        dm = [dm_data.get((ds, b, s)) for s in SEEDS]; dm = [r for r in dm if r is not None]
        be = [cbr_data.get((ds, b, s)) for s in SEEDS]; be = [r for r in be if r is not None]
        if not dm or not be: continue
        delta = float(np.mean(be) - np.mean(dm))
        t_val, p_two = stats.ttest_1samp(np.array(be) - np.array(dm), 0.0)
        p = float(p_two / 2 if t_val > 0 else 1 - p_two / 2)
        entries.append({"cell": f"{ds.title()}-{b.upper()}", "ds": ds,
                        "sens": sens_median, "sens_high_frac": sens_high_frac,
                        "delta": delta, "p": p})

    # Manual label-placement offsets (in axis-data coords): chosen to avoid overlap.
    # Format: cell_name -> (dx, dy, ha, va)
    LABEL_OFFSETS = {
        "Yelpchi-BWGNN": (-0.10,  -0.0030, "right", "center"),
        "Yelpchi-SAGE":  ( 0.06,   0.0025, "left",  "bottom"),
        "Yelpchi-GCN":   ( 0.06,  -0.0030, "left",  "top"),
        "Yelpchi-GAT":   (-0.32,   0.0040, "left",  "bottom"),
        "Amazon-BWGNN":  ( 0.05,   0.0020, "left",  "center"),
        "Amazon-SAGE":   (-0.22,  -0.0030, "left",  "top"),
        "Amazon-GCN":    ( 0.04,   0.0040, "left",  "bottom"),
        "Amazon-GAT":    ( 0.05,  -0.0020, "left",  "center"),
    }
    EXTENDED_NOTES = {
        "Amazon-GCN":   "teacher rarely intervenes\n→ CBR no-op",
        "Yelpchi-GAT":  "weakest base\n→ strongest CBR lift",
        "Amazon-SAGE":  "saturated teacher\n→ uniform shrinkage",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # Engagement zone band — drawn first so markers sit on top
    ax.axvspan(0.30, 0.95, color=C_PANEL, alpha=0.40, zorder=0)
    # Zone label in TOP-LEFT (above scatter), in clear empty space
    ax.text(0.18, 0.0250, "CBR mechanism engagement zone\n(non-uniform sensitivity)",
            ha="left", va="top", fontsize=5.8, color="#7a7a7a", style="italic")
    # Scatter markers
    for e in entries:
        color = C_YELPCHI if e["ds"] == "yelpchi" else C_AMAZON
        size = 100 if e["p"] < 0.05 else 65
        edgecolor = C_GOLD if e["p"] < 0.05 else C_INK
        edge_lw = 1.3 if e["p"] < 0.05 else 0.5
        ax.scatter([e["sens"]], [e["delta"]], s=size, c=color, edgecolor=edgecolor,
                   linewidth=edge_lw, alpha=0.92, zorder=4)
    # Labels with manual offsets + leader lines for offset > threshold
    for e in entries:
        dx, dy, ha, va = LABEL_OFFSETS.get(e["cell"], (0.025, 0, "left", "center"))
        text_x = e["sens"] + dx
        text_y = e["delta"] + dy
        note = EXTENDED_NOTES.get(e["cell"], "")
        if note:
            label = f"{e['cell']}\n{note}"
        else:
            label = e["cell"]
        # Leader line if offset is large enough
        offset_dist = np.sqrt(dx ** 2 + (dy * 50) ** 2)
        if offset_dist > 0.07:
            ax.annotate(label, xy=(e["sens"], e["delta"]),
                        xytext=(text_x, text_y),
                        fontsize=5.6, color=C_INK, ha=ha, va=va,
                        arrowprops=dict(arrowstyle="-", color="#999", lw=0.4))
        else:
            ax.text(text_x, text_y, label,
                    fontsize=5.6, color=C_INK, ha=ha, va=va, alpha=0.95)
    # Reference: 0 delta line
    ax.axhline(0, color=C_INK, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Teacher sensitivity median  $\\mathrm{median}(|\\Delta^T|/\\delta_{\\max})$")
    ax.set_ylabel("Δ AUPRC: CBR-BEST − det_mask (5-seed mean)")
    ax.set_title("CBR mechanism causal evidence: AUPRC lift correlates with teacher sensitivity\n"
                 "(small dots: not sig; large gold-edged: paired-$t$ p<0.05)",
                 fontsize=7, fontweight="bold", pad=8)
    yc_h = ax.scatter([], [], s=80, c=C_YELPCHI, edgecolor=C_INK, label="YelpChi cell")
    az_h = ax.scatter([], [], s=80, c=C_AMAZON,  edgecolor=C_INK, label="Amazon cell")
    sig_h = ax.scatter([], [], s=80, c="white",  edgecolor=C_GOLD, linewidth=1.4, label="Sig p<0.05 (gold edge)")
    ax.legend(handles=[yc_h, az_h, sig_h], loc="lower left", fontsize=6,
              ncol=1, frameon=False)
    # Generous x-range; extra Y headroom for engagement-zone title
    ax.set_xlim(0.00, 1.15)
    ax.set_ylim(-0.014, 0.030)
    ax.grid(linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    save_pub(fig, ROOT / "c3_cbr" / "fig9_mechanism_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[fig-gen-v2] Generating hero composite (Figure 1)...")
    fig1_hero()
    print("[fig-gen-v2] Generating violin distribution (Figure 7)...")
    fig7_violin()
    print("[fig-gen-v2] Generating forest plot (Figure 8)...")
    fig8_forest()
    print("[fig-gen-v2] Generating mechanism scatter (Figure 9)...")
    fig9_mechanism_scatter()
    print("[fig-gen-v2] Done. Top-tier additions at paper_figures/")
