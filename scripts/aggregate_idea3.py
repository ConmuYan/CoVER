"""Aggregate Idea 3 (REL-Curriculum AL) results.

Reads learning_curve.json files from artifacts/results/al/ and produces:
- Per-budget AUPRC comparison table (mean ± std across seeds)
- Paired-t tests: REL-AF vs each baseline AF at each budget
- Markdown output to artifacts/tables/idea3_al_learning_curves.md

Usage::

    python scripts/aggregate_idea3.py \\
        --results_dir artifacts/results/al \\
        --output artifacts/tables/idea3_al_learning_curves.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_all_results(results_dir: Path) -> list[dict]:
    """Recursively load all learning_curve.json files."""
    results = []
    for p in sorted(results_dir.rglob("learning_curve.json")):
        with open(p) as f:
            data = json.load(f)
        results.append(data)
    return results


def build_auprc_table(
    results: list[dict],
    dataset: str,
    base_model: str,
) -> dict:
    """Build a structured table: af -> budget_pct -> list of final AUPRC values across seeds."""
    # Group by (af, seed, budget_pct)
    af_budget_seed_auprc: dict[str, dict[float, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for r in results:
        if r["dataset"] != dataset or r["base_model"] != base_model:
            continue
        af = r["af"]
        seed = r["seed"]
        budget_pct = r["budget_pct"]
        # Get the final entry in the learning curve (highest budget)
        if not r["learning_curve"]:
            continue
        final_entry = r["learning_curve"][-1]
        auprc = final_entry["val_auprc"]
        af_budget_seed_auprc[af][budget_pct][seed] = auprc

    # Also collect all budget points for learning curves
    af_budget_all_points: dict[str, dict[float, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in results:
        if r["dataset"] != dataset or r["base_model"] != base_model:
            continue
        af = r["af"]
        seed = r["seed"]
        for entry in r["learning_curve"]:
            bp = entry["budget_pct"]
            auprc = entry["val_auprc"]
            af_budget_all_points[af][bp][seed].append(auprc)

    # Compute mean ± std for each (af, budget)
    table: dict[str, dict[float, tuple[float, float]]] = {}
    for af in af_budget_seed_auprc:
        table[af] = {}
        for budget_pct in sorted(af_budget_seed_auprc[af].keys()):
            vals = list(af_budget_seed_auprc[af][budget_pct].values())
            if vals:
                table[af][budget_pct] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                table[af][budget_pct] = (0.0, 0.0)

    return {
        "table": table,
        "raw": af_budget_seed_auprc,
    }


def paired_t_tests(
    raw: dict[str, dict[float, dict[int, float]]],
    rel_af: str = "rel_af",
) -> dict:
    """Run paired-t tests: rel_af vs each baseline at each budget."""
    baseline_afs = [af for af in raw if af != rel_af]
    if rel_af not in raw:
        return {"error": f"{rel_af} not found in results"}

    rel_budgets = set(raw[rel_af].keys())
    all_budgets = sorted(rel_budgets)

    results: dict[str, dict[str, dict]] = {}
    for budget_pct in all_budgets:
        results[str(budget_pct)] = {}
        rel_seeds = raw[rel_af].get(budget_pct, {})
        if not rel_seeds:
            continue

        for baseline in baseline_afs:
            base_seeds = raw[baseline].get(budget_pct, {})
            if not base_seeds:
                continue

            # Find common seeds
            common_seeds = sorted(set(rel_seeds.keys()) & set(base_seeds.keys()))
            if len(common_seeds) < 2:
                results[str(budget_pct)][baseline] = {
                    "n_common_seeds": len(common_seeds),
                    "note": "insufficient seeds for paired-t",
                }
                continue

            rel_vals = np.array([rel_seeds[s] for s in common_seeds])
            base_vals = np.array([base_seeds[s] for s in common_seeds])
            diffs = rel_vals - base_vals

            t_stat, p_val = stats.ttest_rel(rel_vals, base_vals)
            results[str(budget_pct)][baseline] = {
                "n_common_seeds": len(common_seeds),
                "rel_mean": round(float(rel_vals.mean()), 6),
                "baseline_mean": round(float(base_vals.mean()), 6),
                "delta": round(float(diffs.mean()), 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_005": bool(p_val < 0.05),
                "significant_001": bool(p_val < 0.01),
            }

    return results


def format_markdown(
    table_data: dict,
    paired_t: dict,
    dataset: str,
    base_model: str,
) -> str:
    """Format the comparison table and paired-t results as Markdown."""
    table = table_data["table"]
    afs = sorted(table.keys())
    all_budgets = sorted(set(b for af_table in table.values() for b in af_table))

    lines: list[str] = []
    lines.append(f"# Idea 3 — REL-Curriculum AL Results")
    lines.append(f"")
    lines.append(f"**Dataset:** {dataset} | **Base model:** {base_model}")
    lines.append(f"")
    lines.append(f"## AUPRC vs Budget (% of train set)")
    lines.append(f"")
    lines.append(f"Values are mean ± std across seeds.")
    lines.append(f"")

    # Header
    header = "| Acquisition | " + " | ".join(f"{b:.0f}%" for b in all_budgets) + " |"
    sep = "|---|" + "|".join(":---:" for _ in all_budgets) + "|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for af in afs:
        cells = []
        for b in all_budgets:
            if b in table[af]:
                mean, std = table[af][b]
                cells.append(f"{mean:.4f} ± {std:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {af} | " + " | ".join(cells) + " |")

    lines.append("")

    # Paired-t section
    lines.append("## Paired-t Tests: REL-AF vs Baselines")
    lines.append("")
    lines.append("Δ = REL-AUPRC − baseline-AUPRC. Positive = REL wins.")
    lines.append("")

    if "error" in paired_t:
        lines.append(f"Error: {paired_t['error']}")
        return "\n".join(lines)

    # Find all baselines across all budgets
    all_baselines = sorted(set(
        baseline
        for budget_data in paired_t.values()
        for baseline in budget_data
    ))

    for baseline in all_baselines:
        lines.append(f"### REL-AF vs {baseline}")
        lines.append("")
        lines.append("| Budget | n_seeds | REL mean | Baseline mean | Δ | t | p | Sig? |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for budget_str in sorted(paired_t.keys(), key=lambda x: float(x)):
            budget_data = paired_t[budget_str]
            if baseline not in budget_data:
                continue
            d = budget_data[baseline]
            if "note" in d:
                lines.append(f"| {budget_str}% | {d['n_common_seeds']} | — | — | — | — | — | {d['note']} |")
            else:
                sig = "**YES**" if d["significant_005"] else "no"
                if d["significant_001"]:
                    sig = "***YES***"
                lines.append(
                    f"| {budget_str}% | {d['n_common_seeds']} "
                    f"| {d['rel_mean']:.4f} | {d['baseline_mean']:.4f} "
                    f"| {d['delta']:+.4f} | {d['t_statistic']:.2f} "
                    f"| {d['p_value']:.4f} | {sig} |"
                )
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    n_sig_wins = 0
    n_total = 0
    for budget_data in paired_t.values():
        for baseline, d in budget_data.items():
            if "t_statistic" in d:
                n_total += 1
                if d.get("significant_005") and d.get("delta", 0) > 0:
                    n_sig_wins += 1

    if n_total > 0:
        lines.append(f"- REL-AF achieves statistically significant wins in **{n_sig_wins}/{n_total}** "
                      f"(budget × baseline) comparisons (p < 0.05).")
    else:
        lines.append("- Insufficient data for paired-t comparisons.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Idea 3 AL results")
    parser.add_argument("--results_dir", type=str, default="artifacts/results/al")
    parser.add_argument("--output", type=str, default="artifacts/tables/idea3_al_learning_curves.md")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset (default: all)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Filter by base model (default: all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} result files")

    if not all_results:
        print("No results found. Exiting.")
        return

    # Group by (dataset, base_model)
    combinations = set()
    for r in all_results:
        ds = r.get("dataset", "unknown")
        bm = r.get("base_model", "unknown")
        if args.dataset and ds != args.dataset:
            continue
        if args.base_model and bm != args.base_model:
            continue
        combinations.add((ds, bm))

    output_lines: list[str] = []
    output_lines.append("# Idea 3 — REL-Curriculum Active Learning")
    output_lines.append("")
    output_lines.append("Generated by `scripts/aggregate_idea3.py`")
    output_lines.append("")

    for ds, bm in sorted(combinations):
        table_data = build_auprc_table(all_results, ds, bm)
        raw = table_data["raw"]

        if not raw:
            print(f"No data for {ds}/{bm}, skipping")
            continue

        # Check if rel_af exists
        if "rel_af" in raw:
            paired_t = paired_t_tests(raw, rel_af="rel_af")
        else:
            paired_t = {"error": "rel_af not found in results"}

        section = format_markdown(table_data, paired_t, ds, bm)
        output_lines.append(section)
        output_lines.append("")
        output_lines.append("---")
        output_lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
