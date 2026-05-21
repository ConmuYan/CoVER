#!/usr/bin/env python3
"""Generate clean per-experiment summary tables from aggregated results.

Outputs markdown tables to artifacts/tables/E{1-8}_summary.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [42, 123, 456, 789, 2026]
KEYS = ["auprc", "roc_auc", "macro_f1"]


def load_metrics(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt(results: list[float]):
    if not results:
        return "—"
    import numpy as np
    arr = np.array(results)
    if len(arr) == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f}±{arr.std():.4f}"


def collect(results_root: Path, dataset: str, model: str, run_name: str, stage: str):
    """Collect metric values across seeds."""
    vals = {k: [] for k in KEYS}
    for seed in SEEDS:
        p = results_root / dataset / model / run_name / f"seed_{seed}"
        if stage == "base":
            p = p / "base_metrics.json"
        else:
            p = p / "test_metrics.json"
        m = load_metrics(p)
        if m:
            for k in KEYS:
                if k in m:
                    vals[k].append(m[k])
    return vals


def table_row(model: str, stage: str, vals: dict):
    auprc = fmt(vals.get("auprc", []))
    auroc = fmt(vals.get("roc_auc", []))
    mf1 = fmt(vals.get("macro_f1", []))
    return f"| {model:<8} | {stage:<12} | {auprc:<20} | {auroc:<20} | {mf1:<20} |"


def save_table(path, header, rows, extra_header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {header}\n\n")
        if extra_header:
            f.write(extra_header + "\n")
            f.write("|" + "|".join(["----------"] * (extra_header.count("|") - 1)) + "|\n")
        else:
            f.write("| Model    | Stage        | AUPRC               | AUROC               | Macro-F1            |\n")
            f.write("|----------|--------------|---------------------|---------------------|---------------------|\n")
        for r in rows:
            f.write(r + "\n")
    print(f"  Saved: {path}")


def main():
    R = Path("artifacts/results")
    T = Path("artifacts/tables")

    # ── E1: Benchmark ──
    print("E1 Benchmark...")
    for split_label, base_rn, teach_rn, stud_rn, ds_list in [
        ("care_712 (7:1:2)", "base", "raer_lree", "cbr_flash", ["yelpchi", "amazon"]),
        ("bwgnn_semi (1:33:66)", "base_semi", "raer_lree_semi", "cbr_flash_semi", ["yelpchi", "amazon"]),
    ]:
        for ds in ds_list:
            rows = []
            for m in ["bwgnn", "sage", "gcn", "gat"]:
                for stage, rn in [("Base", base_rn), ("Teacher", teach_rn), ("Student", stud_rn)]:
                    v = collect(R, ds, m, rn, "base" if stage == "Base" else "other")
                    rows.append(table_row(m, stage, v))
            save_table(T / f"E1_{ds}_{split_label.split()[0]}.md",
                       f"E1 Benchmark — {ds} {split_label}", rows)

    # ── E2: Scarcity ──
    print("E2 Scarcity...")
    rows = []
    for pct in [1, 5, 10, 20, 50]:
        for m in ["bwgnn", "sage", "gcn", "gat"]:
            for stage, rn in [("Base", f"base_scarcity_{pct}pct"),
                              ("Teacher", f"raer_lree_scarcity_{pct}pct"),
                              ("Student", f"cbr_flash_scarcity_{pct}pct")]:
                v = collect(R, "yelpchi", m, rn, "base" if stage == "Base" else "other")
                label = f"{m}@{pct}%"
                auprc = fmt(v.get("auprc", []))
                auroc = fmt(v.get("roc_auc", []))
                mf1 = fmt(v.get("macro_f1", []))
                rows.append(f"| {label:<12} | {stage:<12} | {auprc:<20} | {auroc:<20} | {mf1:<20} |")
    save_table(T / "E2_scarcity.md", "E2 Scarcity — YelpChi care_712", rows)

    # ── E3: Scaling ──
    print("E3 Scaling...")
    for ds in ["yelpnyc", "yelpzip"]:
        rows = []
        for stage, rn in [("Base", "base_neighbor_mb"),
                          ("Teacher", "raer_lree_scalable"),
                          ("Student", "cbr_flash_lree_scalable")]:
            v = collect(R, ds, "sage", rn, "base" if stage == "Base" else "other")
            rows.append(table_row("sage", stage, v))
        save_table(T / f"E3_{ds}.md", f"E3 Scaling — {ds} SAGE (care_712)", rows)

    # ── E4: Generalization ──
    print("E4 Generalization...")
    # TFinance
    rows = []
    for stage, rn in [("Base", "base_bwgnn_424"),
                      ("Teacher", "raer_lree_bwgnn_424"),
                      ("Student", "cbr_flash_bwgnn_424")]:
        v = collect(R, "tfinance", "sage", rn, "base" if stage == "Base" else "other")
        rows.append(table_row("sage", stage, v))
    save_table(T / "E4_tfinance.md", "E4 Generalization — TFinance SAGE (4:2:4)", rows)
    # TSocial (if available)
    rows = []
    for stage, rn in [("Base", "base_neighbor_mb"),
                      ("Teacher", "raer_lree_scalable"),
                      ("Student", "cbr_flash_lree_scalable")]:
        v = collect(R, "tsocial", "sage", rn, "base" if stage == "Base" else "other")
        rows.append(table_row("sage", stage, v))
    save_table(T / "E4_tsocial.md", "E4 Generalization — TSocial SAGE (4:2:4, mini-batch)", rows)

    # ── E5: Efficiency ──
    print("E5 Efficiency...")
    import csv
    e5_path = Path("artifacts/results/E5_efficiency/inference_benchmark.csv")
    if e5_path.exists():
        with open(e5_path) as f:
            reader = csv.DictReader(f)
            e5_data = list(reader)
        rows = []
        for m in ["bwgnn", "sage", "gcn", "gat"]:
            base_row = next((r for r in e5_data if r["model"] == m and r["seed"] == "42"), None)
            if base_row:
                rows.append(f"| {m:<8} | {int(base_row.get('base_params',0)):>10,} | "
                           f"{float(base_row.get('base_latency_ms',0)):.2f} | "
                           f"{int(base_row.get('teacher_params',0)):>10,} | "
                           f"{float(base_row.get('teacher_latency_ms',0)):.2f} | "
                           f"{int(base_row.get('student_params',0)):>10,} | "
                           f"{float(base_row.get('student_latency_ms',0)):.2f} | "
                           f"{float(base_row.get('compression_ratio',0)):.1f}x |")
        save_table(T / "E5_efficiency.md",
                   "E5 Efficiency — YelpChi care_712 (seed=42)", rows,
                   extra_header="| Model | Base# | Base ms | Teacher# | Teacher ms | Student# | Student ms | Compress |")

    # ── E6: Ablation ──
    print("E6 Ablation...")
    rows = []
    for m in ["gcn", "gat", "sage"]:
        for stage, rn in [("Freeze-Base", "base"),
                          ("RAER-HC", "raer_hc"),
                          ("RAER-LREE", "raer_lree")]:
            st = "base" if stage == "Freeze-Base" else "other"
            v = collect(R, "yelpchi", m, rn, st)
            auprc = fmt(v.get("auprc", []))
            auroc = fmt(v.get("roc_auc", []))
            mf1 = fmt(v.get("macro_f1", []))
            rows.append(f"| {m:<8} | {stage:<12} | {auprc:<20} | {auroc:<20} | {mf1:<20} |")
    save_table(T / "E6_ablation.md", "E6 Ablation — YelpChi care_712", rows)

    # E7 loss ablations were invalidated and are intentionally excluded from
    # active summary tables.

    # ── E8: Hyperparameter Sweep ──
    print("E8 Hyperparameter...")
    for m in ["sage", "gcn"]:
        rows = []
        for lam in [0.0, 0.1, 0.5, 1.0]:
            for K in [256, 512, 1024, 2048]:
                rn = f"cbr_flash_cbr_l{lam}_K{K}"
                v = collect(R, "yelpchi", m, rn, "other")
                auprc = fmt(v.get("auprc", []))
                auroc = fmt(v.get("roc_auc", []))
                mf1 = fmt(v.get("macro_f1", []))
                rows.append(f"| {lam:<6} | {K:<6} | {auprc:<20} | {auroc:<20} | {mf1:<20} |")
        save_table(T / f"E8_hyperparam_{m}.md",
                   f"E8 Hyperparameter Sweep — YelpChi care_712 {m.upper()}",
                   rows, extra_header="| λ | K | AUPRC | AUROC | Macro-F1 |")

    print("\nDone. All tables saved to artifacts/tables/")


if __name__ == "__main__":
    main()
