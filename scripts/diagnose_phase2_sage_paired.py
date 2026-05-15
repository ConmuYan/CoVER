"""Paired diagnostic for SAGE Phase2 unified reasoner.

For each dataset/experiment combination, computes:
1. Paired Δ vs SAGE Phase1 base (per-seed, mean ± std, paired t-stat).
2. Paired Δ vs old anchor_gate (per-seed, mean ± std, paired t-stat).
3. mean_abs_delta_rel from phase2_diagnostics (mean across seeds).
4. mean_gate_entropy from phase2_diagnostics.
5. mean_alpha_llm (only meaningful for E1/E2).
6. judge_gate_agreement: cosine similarity between
   (a) judge key_relation empirical distribution from accepted_judge.jsonl, and
   (b) mean gate distribution from phase2_diagnostics gate_weight_rel_*.

Output: artifacts/reports/sage_phase2_paired_diagnostic.md +
artifacts/tables/sage_phase2_paired_diagnostic.csv

Safety: read-only on per-seed JSON; no model re-inference.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["yelpchi", "amazon"]
SEEDS_FULL = [42, 123, 456, 789, 2026]
EXPERIMENTS = [
    ("E0", "phase2_E0_relgate"),
    ("E1", "phase2_E1_judge_align"),
    ("E2", "phase2_E2_judge_residual"),
]

RELATION_NAMES = {
    "yelpchi": ["RUR", "RSR", "RTR"],
    "amazon": ["UPU", "USU", "UVU"],
}


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] {p}: {e}", file=sys.stderr)
        return None


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def base_auprc(ds: str, seed: int) -> float | None:
    p = PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / "base" / f"seed_{seed}" / "stage1_metrics.json"
    d = load_json(p)
    return d.get("auprc") if d else None


def gate_auprc(ds: str, seed: int) -> float | None:
    p = PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / "cover_rel_anchor_gate_nollm" / f"seed_{seed}" / "stage3_metrics.json"
    d = load_json(p)
    return d.get("auprc") if d else None


def phase2_metrics(ds: str, run_name: str, seed: int) -> dict | None:
    p = PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / run_name / f"seed_{seed}" / "stage3_metrics.json"
    return load_json(p)


def phase2_diag(ds: str, run_name: str, seed: int) -> dict | None:
    p = PROJECT_ROOT / "artifacts" / "logs" / ds / "sage" / run_name / f"seed_{seed}" / "phase2_diagnostics.json"
    d = load_json(p)
    return d.get("final_diagnostics") if d else None


def judge_relation_distribution(ds: str, seed: int) -> dict[str, float] | None:
    """Empirical distribution of key_relation in accepted_judge.jsonl."""
    p = PROJECT_ROOT / "artifacts" / "judge_packets" / ds / "sage" / "cover_rel_judge" / f"seed_{seed}" / "accepted_judge.jsonl"
    rows = load_jsonl(p)
    if not rows:
        return None
    rels = RELATION_NAMES[ds]
    counts = {r: 0 for r in rels}
    total = 0
    for r in rows:
        kr = str(r.get("key_relation", "")).upper()
        if kr in counts:
            counts[kr] += 1
            total += 1
    if total == 0:
        return None
    return {r: counts[r] / total for r in rels}


def gate_distribution(diag: dict, ds: str) -> dict[str, float] | None:
    """Mean gate distribution from phase2_diagnostics."""
    rels = RELATION_NAMES[ds]
    out: dict[str, float] = {}
    for i, r in enumerate(rels):
        v = diag.get(f"gate_weight_rel_{i}")
        if not isinstance(v, (int, float)):
            return None
        out[r] = float(v)
    s = sum(out.values())
    if s <= 0:
        return None
    return {r: out[r] / s for r in rels}


def cosine_sim(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    pv = [p.get(k, 0.0) for k in keys]
    qv = [q.get(k, 0.0) for k in keys]
    np_ = math.sqrt(sum(x * x for x in pv))
    nq_ = math.sqrt(sum(x * x for x in qv))
    if np_ <= 0 or nq_ <= 0:
        return 0.0
    return sum(a * b for a, b in zip(pv, qv)) / (np_ * nq_)


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def paired_t(diffs: list[float]) -> tuple[float, float]:
    """Return (paired_t_stat, mean_diff). Returns (nan, mean) if n<2 or zero variance."""
    diffs = [d for d in diffs if isinstance(d, (int, float)) and not math.isnan(d)]
    if not diffs:
        return float("nan"), float("nan")
    m = statistics.mean(diffs)
    if len(diffs) < 2:
        return float("nan"), m
    sd = statistics.stdev(diffs)
    if sd <= 0:
        return float("inf") if m > 0 else (float("-inf") if m < 0 else 0.0), m
    se = sd / math.sqrt(len(diffs))
    return m / se, m


def collect_one(ds: str, exp_id: str, run_name: str) -> dict:
    """Collect all per-seed records for one (dataset, experiment)."""
    seeds_present: list[int] = []
    base_aps: list[float] = []
    gate_aps: list[float] = []
    p2_aps: list[float] = []
    p2_rocs: list[float] = []
    p2_mf1: list[float] = []
    delta_vs_base_per_seed: list[float] = []
    delta_vs_gate_per_seed: list[float] = []
    diag_delta_rel: list[float] = []
    diag_gate_ent: list[float] = []
    diag_alpha_llm: list[float] = []
    diag_alpha_acc: list[float] = []
    diag_alpha_rej_max: list[float] = []
    diag_dominance_rho: list[float] = []
    gate_dist_avg: dict[str, list[float]] = {r: [] for r in RELATION_NAMES[ds]}
    judge_dist_avg: dict[str, list[float]] = {r: [] for r in RELATION_NAMES[ds]}
    judge_gate_cos: list[float] = []

    for s in SEEDS_FULL:
        m = phase2_metrics(ds, run_name, s)
        if m is None or "auprc" not in m:
            continue
        b = base_auprc(ds, s)
        g = gate_auprc(ds, s)
        diag = phase2_diag(ds, run_name, s)
        seeds_present.append(s)
        if b is not None:
            base_aps.append(b)
            delta_vs_base_per_seed.append(float(m["auprc"]) - b)
        if g is not None:
            gate_aps.append(g)
            delta_vs_gate_per_seed.append(float(m["auprc"]) - g)
        p2_aps.append(float(m["auprc"]))
        p2_rocs.append(float(m.get("roc_auc", float("nan"))))
        p2_mf1.append(float(m.get("macro_f1", float("nan"))))
        if diag:
            for k, lst in (
                ("mean_abs_delta_rel", diag_delta_rel),
                ("mean_gate_entropy", diag_gate_ent),
                ("mean_alpha_llm", diag_alpha_llm),
                ("mean_alpha_llm_accepted", diag_alpha_acc),
                ("max_abs_alpha_llm_rejected", diag_alpha_rej_max),
                ("mean_dominance_rho", diag_dominance_rho),
            ):
                v = diag.get(k)
                if isinstance(v, (int, float)):
                    lst.append(float(v))
            gd = gate_distribution(diag, ds)
            jd = judge_relation_distribution(ds, s)
            if gd is not None:
                for r in RELATION_NAMES[ds]:
                    gate_dist_avg[r].append(gd[r])
            if jd is not None:
                for r in RELATION_NAMES[ds]:
                    judge_dist_avg[r].append(jd[r])
            if gd is not None and jd is not None:
                judge_gate_cos.append(cosine_sim(jd, gd))

    return {
        "dataset": ds,
        "experiment": exp_id,
        "run_name": run_name,
        "seeds_present": seeds_present,
        "base_auprc_mean_std": mean_std(base_aps),
        "gate_auprc_mean_std": mean_std(gate_aps),
        "phase2_auprc_mean_std": mean_std(p2_aps),
        "phase2_roc_mean_std": mean_std(p2_rocs),
        "phase2_mf1_mean_std": mean_std(p2_mf1),
        "delta_vs_base_per_seed": delta_vs_base_per_seed,
        "delta_vs_base_mean_std": mean_std(delta_vs_base_per_seed),
        "delta_vs_base_paired_t": paired_t(delta_vs_base_per_seed),
        "delta_vs_gate_per_seed": delta_vs_gate_per_seed,
        "delta_vs_gate_mean_std": mean_std(delta_vs_gate_per_seed),
        "delta_vs_gate_paired_t": paired_t(delta_vs_gate_per_seed),
        "diag_mean_abs_delta_rel": mean_std(diag_delta_rel),
        "diag_gate_entropy": mean_std(diag_gate_ent),
        "diag_alpha_llm": mean_std(diag_alpha_llm),
        "diag_alpha_llm_accepted": mean_std(diag_alpha_acc),
        "diag_max_alpha_rejected_max": max(diag_alpha_rej_max) if diag_alpha_rej_max else float("nan"),
        "diag_dominance_rho": mean_std(diag_dominance_rho),
        "gate_dist_avg": {r: mean_std(gate_dist_avg[r]) for r in RELATION_NAMES[ds]},
        "judge_dist_avg": {r: mean_std(judge_dist_avg[r]) for r in RELATION_NAMES[ds]},
        "judge_gate_agreement_cos_mean_std": mean_std(judge_gate_cos),
    }


def fmt(v, digits: int = 4) -> str:
    if isinstance(v, tuple):
        m, s = v
        if math.isnan(m):
            return "—"
        return f"{m:.{digits}f} ± {s:.{digits}f}"
    if isinstance(v, (int, float)):
        if math.isnan(float(v)):
            return "—"
        return f"{v:.{digits}f}"
    return str(v) if v not in (None, "") else "—"


def write_md(rows: list[dict], path: Path) -> None:
    lines: list[str] = ["# SAGE Phase2 Paired Diagnostic", ""]
    lines.append("Read-only diagnostic over saved per-seed JSON metrics + phase2_diagnostics.")
    lines.append("No model re-inference. Source paths cited below per dataset/experiment.")
    lines.append("")

    for ds in DATASETS:
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Exp | n | Phase2 AUPRC | ΔAUPRC vs base (paired) | t-stat | ΔAUPRC vs anchor_gate (paired) | t-stat | mean\\|Δ_rel\\| | gate H | mean α | judge↔gate cos |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["dataset"] != ds:
                continue
            t_b, _ = r["delta_vs_base_paired_t"]
            t_g, _ = r["delta_vs_gate_paired_t"]
            lines.append(
                f"| {r['experiment']} | {len(r['seeds_present'])} | "
                f"{fmt(r['phase2_auprc_mean_std'])} | "
                f"{fmt(r['delta_vs_base_mean_std'])} | "
                f"{fmt(t_b, 2)} | "
                f"{fmt(r['delta_vs_gate_mean_std'])} | "
                f"{fmt(t_g, 2)} | "
                f"{fmt(r['diag_mean_abs_delta_rel'])} | "
                f"{fmt(r['diag_gate_entropy'])} | "
                f"{fmt(r['diag_alpha_llm'], 6)} | "
                f"{fmt(r['judge_gate_agreement_cos_mean_std'])} |"
            )
        lines.append("")
        lines.append("### Per-seed Δ AUPRC vs SAGE base")
        lines.append("| Exp | " + " | ".join(f"seed {s}" for s in SEEDS_FULL) + " |")
        lines.append("|" + "---|" * (len(SEEDS_FULL) + 1))
        for r in rows:
            if r["dataset"] != ds:
                continue
            seed_to_d = dict(zip(r["seeds_present"], r["delta_vs_base_per_seed"]))
            cells = [fmt(seed_to_d.get(s)) for s in SEEDS_FULL]
            lines.append(f"| {r['experiment']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("### Per-seed Δ AUPRC vs old anchor_gate (Stage3)")
        lines.append("| Exp | " + " | ".join(f"seed {s}" for s in SEEDS_FULL) + " |")
        lines.append("|" + "---|" * (len(SEEDS_FULL) + 1))
        for r in rows:
            if r["dataset"] != ds:
                continue
            seed_to_d = dict(zip(r["seeds_present"], r["delta_vs_gate_per_seed"]))
            cells = [fmt(seed_to_d.get(s)) for s in SEEDS_FULL]
            lines.append(f"| {r['experiment']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"### Gate vs Judge relation distribution ({', '.join(RELATION_NAMES[ds])})")
        lines.append("| Exp | gate distribution | judge distribution | cosine sim |")
        lines.append("|---|---|---|---|")
        for r in rows:
            if r["dataset"] != ds:
                continue
            gate_str = ", ".join(f"{rel}={fmt(r['gate_dist_avg'][rel], 3)}" for rel in RELATION_NAMES[ds])
            judge_str = ", ".join(f"{rel}={fmt(r['judge_dist_avg'][rel], 3)}" for rel in RELATION_NAMES[ds])
            lines.append(f"| {r['experiment']} | {gate_str} | {judge_str} | {fmt(r['judge_gate_agreement_cos_mean_std'])} |")
        lines.append("")
        lines.append("### Safety check (LLM gate leak)")
        for r in rows:
            if r["dataset"] != ds:
                continue
            lines.append(
                f"- **{r['experiment']}**: max_abs_alpha_llm_rejected (max across seeds) = "
                f"{fmt(r['diag_max_alpha_rejected_max'], 8)} (must be < 1e-6 for safety pass)"
            )
        lines.append("")
    lines.append("## Source paths (per dataset/experiment/seed)")
    lines.append("")
    lines.append("- `artifacts/results/{ds}/sage/base/seed_X/stage1_metrics.json` — Phase1 base AUPRC")
    lines.append("- `artifacts/results/{ds}/sage/cover_rel_anchor_gate_nollm/seed_X/stage3_metrics.json` — old anchor_gate AUPRC")
    lines.append("- `artifacts/results/{ds}/sage/{run_name}/seed_X/stage3_metrics.json` — Phase2 reasoner AUPRC")
    lines.append("- `artifacts/logs/{ds}/sage/{run_name}/seed_X/phase2_diagnostics.json` — diagnostics (gate, alpha, delta_rel)")
    lines.append("- `artifacts/judge_packets/{ds}/sage/cover_rel_judge/seed_X/accepted_judge.jsonl` — judge key_relation distribution")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    flat: list[dict] = []
    for r in rows:
        f = {
            "dataset": r["dataset"],
            "experiment": r["experiment"],
            "run_name": r["run_name"],
            "n_seeds": len(r["seeds_present"]),
            "phase2_auprc_mean": r["phase2_auprc_mean_std"][0],
            "phase2_auprc_std": r["phase2_auprc_mean_std"][1],
            "delta_vs_base_mean": r["delta_vs_base_mean_std"][0],
            "delta_vs_base_std": r["delta_vs_base_mean_std"][1],
            "delta_vs_base_t": r["delta_vs_base_paired_t"][0],
            "delta_vs_gate_mean": r["delta_vs_gate_mean_std"][0],
            "delta_vs_gate_std": r["delta_vs_gate_mean_std"][1],
            "delta_vs_gate_t": r["delta_vs_gate_paired_t"][0],
            "mean_abs_delta_rel": r["diag_mean_abs_delta_rel"][0],
            "gate_entropy": r["diag_gate_entropy"][0],
            "mean_alpha_llm": r["diag_alpha_llm"][0],
            "mean_alpha_llm_accepted": r["diag_alpha_llm_accepted"][0],
            "max_abs_alpha_llm_rejected": r["diag_max_alpha_rejected_max"],
            "judge_gate_cos_mean": r["judge_gate_agreement_cos_mean_std"][0],
            "judge_gate_cos_std": r["judge_gate_agreement_cos_mean_std"][1],
        }
        flat.append(f)
    fields = list(flat[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)


def main() -> int:
    rows: list[dict] = []
    for ds in DATASETS:
        for exp_id, run_name in EXPERIMENTS:
            r = collect_one(ds, exp_id, run_name)
            rows.append(r)

    md_path = PROJECT_ROOT / "artifacts" / "reports" / "sage_phase2_paired_diagnostic.md"
    csv_path = PROJECT_ROOT / "artifacts" / "tables" / "sage_phase2_paired_diagnostic.csv"
    write_md(rows, md_path)
    write_csv(rows, csv_path)
    print(json.dumps({"md": str(md_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
