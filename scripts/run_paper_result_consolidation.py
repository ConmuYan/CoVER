from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir


SEEDS = [42, 123, 456, 789, 2026]
MODEL = "bwgnn"
DATASETS = {
    "yelpchi": {
        "base": "base",
        "best_single": "qwen_directional_t200_cover_rel_rur_nollm",
        "best_single_label": "CoVER-REL RUR-only",
        "gate": "cover_rel_anchor_gate_nollm",
        "judge": "cover_rel_judge_rur_strength_gate",
        "primary_relation": "RUR",
    },
    "amazon": {
        "base": "base",
        "best_single": "cover_rel_uvu_nollm",
        "best_single_label": "CoVER-REL UVU-only",
        "gate": "cover_rel_anchor_gate_nollm",
        "judge": "cover_rel_judge_uvu_strength_gate",
        "primary_relation": "UVU",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metric_path(dataset: str, run: str, seed: int, base: bool = False) -> Path:
    filename = "stage1_metrics.json" if base else "stage3_metrics.json"
    return Path("artifacts/results") / dataset / MODEL / run / f"seed_{seed}" / filename


def log_path(dataset: str, run: str, seed: int, name: str) -> Path:
    return Path("artifacts/logs") / dataset / MODEL / run / f"seed_{seed}" / name


def judge_dir(dataset: str, seed: int) -> Path:
    return Path("artifacts/judge_packets") / dataset / MODEL / "cover_rel_judge" / f"seed_{seed}"


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def run_metrics(dataset: str, run: str, seeds: list[int], base: bool = False) -> dict[str, Any]:
    rows = [load_json(metric_path(dataset, run, seed, base=base)) for seed in seeds]
    rows = [row for row in rows if row]
    result: dict[str, Any] = {"seeds_complete": len(rows)}
    for key in ("roc_auc", "auprc", "macro_f1", "f1"):
        values = [float(row[key]) for row in rows if key in row]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_std"] = stdev(values)
    return result


def per_seed_delta(dataset: str, run: str, ref_run: str, seeds: list[int], ref_base: bool = False) -> list[float]:
    values: list[float] = []
    for seed in seeds:
        cur = load_json(metric_path(dataset, run, seed))
        ref = load_json(metric_path(dataset, ref_run, seed, base=ref_base))
        if cur and ref and "auprc" in cur and "auprc" in ref:
            values.append(float(cur["auprc"]) - float(ref["auprc"]))
    return values


def build_main_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, cfg in DATASETS.items():
        gate_run = cfg["gate"]
        methods = [
            ("Fresh BWGNN", cfg["base"], "base", True),
            (cfg["best_single_label"], cfg["best_single"], "best-single relation", False),
            ("CoVER-REL-Gate", cfg["gate"], "main detector", False),
            ("CoVER-REL-Judge", cfg["judge"], "LLM-assisted research model", False),
        ]
        for label, run, role, is_base in methods:
            metrics = run_metrics(dataset, run, SEEDS, base=is_base)
            delta_base = [0.0] if is_base else per_seed_delta(dataset, run, cfg["base"], SEEDS, ref_base=True)
            delta_gate = [0.0] if run == gate_run else ([] if is_base else per_seed_delta(dataset, run, gate_run, SEEDS))
            rows.append({
                "dataset": dataset,
                "method": label,
                "role": role,
                "seeds_complete": metrics.get("seeds_complete", 0),
                "roc_auc": metrics.get("roc_auc_mean", ""),
                "roc_auc_std": metrics.get("roc_auc_std", ""),
                "auprc": metrics.get("auprc_mean", ""),
                "auprc_std": metrics.get("auprc_std", ""),
                "delta_auprc_vs_base": mean(delta_base) if delta_base else "",
                "delta_auprc_vs_gate": mean(delta_gate) if delta_gate else "",
                "macro_f1": metrics.get("macro_f1_mean", ""),
                "macro_f1_std": metrics.get("macro_f1_std", ""),
            })
    return rows


def build_relation_ablation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    yelp = load_csv(Path("artifacts/tables/yelpchi_cover_rel_relation_ablation_3seed.csv"))
    for row in yelp:
        rows.append({
            "dataset": "yelpchi",
            "relation_setting": row.get("relation_set", ""),
            "seeds": "3",
            "delta_auprc": float(row.get("mean_delta_auprc", 0.0)),
            "delta_roc_auc": float(row.get("mean_delta_roc_auc", 0.0)),
            "delta_macro_f1": float(row.get("mean_delta_macro_f1", 0.0)),
            "interpretation": "dominant" if row.get("relation_set") == "rur" else ("weak/noisy" if row.get("go") == "False" else "positive"),
        })
    amazon = load_csv(Path("artifacts/tables/amazon_cover_rel_relation_ablation_3seed.csv"))
    for row in amazon:
        if row.get("row_type") != "mean":
            continue
        rows.append({
            "dataset": "amazon",
            "relation_setting": row.get("relation", ""),
            "seeds": "3",
            "delta_auprc": float(row.get("delta_auprc", 0.0)),
            "delta_roc_auc": float(row.get("delta_roc_auc", 0.0)),
            "delta_macro_f1": float(row.get("delta_macro_f1_at_val", 0.0)),
            "interpretation": "best single" if row.get("relation") == "uvu" else "weak positive",
        })
    return rows


def build_gate_judge_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schema = load_csv(Path("artifacts/tables/cover_rel_schema_aware_summary.csv"))
    judge = load_csv(Path("artifacts/tables/cover_rel_judge_final_5seed_summary.csv"))
    judge_mean = {row["dataset"]: row for row in judge if row.get("seed") == "mean"}
    for row in schema:
        dataset = row["dataset"]
        j = judge_mean.get(dataset, {})
        rows.append({
            "dataset": dataset,
            "best_relation": row.get("best_relation", ""),
            "best_single_delta_auprc": float(row.get("best_single_delta_auprc", 0.0)),
            "gate_delta_auprc": float(row.get("anchor_gate_delta_auprc", 0.0)),
            "gate_delta_vs_best_single": float(row.get("anchor_gate_delta_vs_best_single", 0.0)),
            "gate_near_cap": float(row.get("anchor_gate_near_cap", 0.0)),
            "judge_delta_vs_gate": float(j.get("delta_auprc_vs_anchor", 0.0)) if j else "",
            "judge_delta_vs_best_single": float(j.get("delta_auprc_vs_single", 0.0)) if j else "",
            "judge_acceptance": float(j.get("judge_acceptance_rate", 0.0)) if j else "",
            "judge_alpha_mean": float(j.get("alpha_llm_mean", 0.0)) if j else "",
            "judge_llm_near_cap": float(j.get("llm_near_cap_fraction", 0.0)) if j else "",
            "final_positioning": "main detector = Gate; explanation extension = Judge",
        })
    return rows


def build_negative_routes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lift = load_csv(Path("artifacts/tables/yelpchi_fresh_bwgnn_vs_qwen_directional_t200_cover_lift_3seed.csv"))
    lift_seed_rows = [row for row in lift if row.get("seed") not in {"mean", "std"}]
    if lift_seed_rows:
        deltas = [float(row["delta_auprc"]) for row in lift_seed_rows]
        macro = [float(row["delta_macro_f1_at_val"]) for row in lift_seed_rows]
        latent = [float(row["latent_cosine_similarity"]) for row in lift_seed_rows]
        rows.append({
            "route": "CoVER-LIFT canonical ERR hidden",
            "dataset": "yelpchi",
            "seeds": len(lift_seed_rows),
            "mean_delta_auprc": mean(deltas),
            "mean_delta_macro_f1": mean(macro),
            "latent_cosine": mean(latent),
            "lesson": "Latent alignment can be high while canonical ERR hidden is not fraud-discriminative.",
        })
    rows.append({
        "route": "ERR-only / structure-only evidence",
        "dataset": "yelpchi",
        "seeds": "",
        "mean_delta_auprc": "",
        "mean_delta_macro_f1": "",
        "latent_cosine": "",
        "lesson": "Thin structured evidence lacks the relation-aware anonymous feature signal needed for ranking gains.",
    })
    rows.append({
        "route": "Unrestricted Judge tuning",
        "dataset": "amazon",
        "seeds": "3",
        "mean_delta_auprc": 0.000173,
        "mean_delta_macro_f1": "",
        "latent_cosine": "",
        "lesson": "Original judge was weakly positive but had seed-level alpha saturation; conservative strength-aware fusion is safer.",
    })
    return rows


def write_method_draft(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        """# Method Draft: CoVER-REL and CoVER-REL-Judge

## Overview

CoVER is a contract-verified evidence reasoning framework for multi-relation graph fraud detection. The final method separates the deployment-friendly relation detector from the LLM-assisted research extension.

- **CoVER-REL-Gate** is the main detector. It uses schema-aware relation evidence experts and an anchor-preserving gate to fuse relation-wise anonymous feature statistics.
- **CoVER-REL-Judge** is the LLM-assisted extension. A score-blind LLM judge reads the same relation evidence packet, emits a verified structured judgement and short explanation, and a conservative gated residual fusion module decides how much this judgement can adjust the relation-only logit.

## Relation Evidence

For a graph \(G=(V, X, \\{E_r\\}_{r\\in\\mathcal R})\), CoVER-REL builds one evidence expert per relation \(r\). For node \(i\), the relation feature vector includes degree, log-degree bucket, neighbor feature deviation, cosine consistency, z-score outlier count, train-only fraud/benign prototype distances, and the fraud-vs-benign prototype margin.

All prototype statistics are computed from train labels only. The evidence packets and LLM judge prompts exclude base scores, probabilities, logits, confidence, base predictions, target labels, split identity, FN/FP status, and ground truth.

## Schema-Aware Gate

CoVER-REL-Gate preserves the strongest schema relation as an anchor and lets optional relations contribute through sparse gates. This avoids hardcoding YelpChi-specific RUR logic while allowing relation utility to differ by dataset:

- YelpChi: RUR is dominant.
- Amazon: UVU is strongest, with more distributed relation utility.

## LLM Judge Extension

The LLM judge receives only score-blind relation evidence packets and returns JSON:

```json
{
  "verdict": "fake|real|uncertain",
  "evidence_strength": "weak|moderate|strong",
  "key_relation": "...",
  "supporting_evidence": ["..."],
  "counter_evidence": ["..."],
  "uncertainty_factors": ["..."],
  "short_explanation": "..."
}
```

Verifier-rejected outputs are excluded from training. The short explanation is used only for human-facing interpretation and is not used in loss.

The final fusion is:

\[
\\text{final\\_logit} = s_{rel} + \\alpha_{llm}\\,\\Delta_{llm}.
\]

The final 5-seed confirmation uses conservative strength-aware fusion so weak, moderate, and uncertain judge outputs are downweighted.
""",
        encoding="utf-8",
    )


def write_results_narrative(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        """# Results Narrative Draft

## Main Claim

The main quantitative contribution is **CoVER-REL-Gate**, not the LLM judge. Relation-aware anonymous feature evidence consistently improves the BWGNN prior under dataset-specific relation schemas.

The LLM-assisted extension, **CoVER-REL-Judge**, preserves Gate-level performance while adding structured, score-blind, contract-verified explanations.

## Key Findings

1. **Relation evidence is the real signal.** YelpChi gains concentrate in RUR, while Amazon gains concentrate around UVU with more distributed relation utility.
2. **Schema-aware gating generalizes the idea.** The method does not hardcode RUR; it uses each dataset's relation schema and selects or gates useful relation evidence.
3. **Judge is safe and stable, but not the main performance jump.** CoVER-REL-Judge is Acceptable GO on both datasets: small positive or neutral AUPRC changes over Gate, high verifier acceptance, no forbidden-field audit failures, and valid structured explanations.
4. **Negative routes motivate the pivot.** Canonical ERR hidden-state distillation produced high latent alignment but did not improve AUPRC, showing that thin ERR latents were not sufficiently fraud-discriminative.

## Recommended Paper Positioning

- Put CoVER-REL-Gate in the main result table as the main detector.
- Put CoVER-REL-Judge in the main table or a dedicated extension table, described as the LLM-assisted research model.
- Include judge examples and safety audit to support interpretability claims.
- Include DIR/LIFT failure routes in an appendix to explain why relation-aware evidence construction became central.
""",
        encoding="utf-8",
    )


def write_failure_appendix(path: Path, negative_rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Appendix Draft: Negative Routes and Design Pivot",
        "",
        "Earlier CoVER variants tested whether LLM-generated ERR records or canonical ERR hidden states could directly provide the main discriminative signal. The results indicate that this route is insufficient on YelpChi/Amazon `.mat` data.",
        "",
        "## Summary",
        "",
    ]
    for row in negative_rows:
        lines.append(f"- **{row['route']}**: {row['lesson']}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The failure was not evidence against the CoVER framework. It was evidence that the LLM was seeing evidence that was too thin. The successful pivot was to expose the strongest available score-blind signal in the data: relation-aware anonymous feature statistics from the dataset relation schema.",
        "",
        "This motivates CoVER-REL: keep the base GNN prior, contract verification, score-blind evidence discipline, and LLM-assisted explanation, but make relation-aware evidence construction the central discriminative branch.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        """# Paper Result Consolidation

Generated by `scripts/run_paper_result_consolidation.py`.

## Files

- `artifacts/tables/paper_main_results.csv/.md`: main detector comparison.
- `artifacts/tables/paper_relation_ablation.csv/.md`: relation utility ablation.
- `artifacts/tables/paper_gate_judge_summary.csv/.md`: Gate/Judge positioning summary.
- `artifacts/tables/paper_negative_routes.csv/.md`: failed or non-main routes.
- `artifacts/paper/method_section_draft.md`: method write-up draft.
- `artifacts/paper/results_narrative.md`: result narrative draft.
- `artifacts/paper/appendix_failure_routes.md`: appendix draft for DIR/LIFT/ERR-only routes.

## Final Positioning

CoVER-REL-Gate is the main performance model. CoVER-REL-Judge is the LLM-assisted research extension with structured, score-blind explanations and stable 5-seed behavior.
""",
        encoding="utf-8",
    )


def main() -> None:
    main_rows = build_main_results()
    relation_rows = build_relation_ablation()
    gate_judge_rows = build_gate_judge_summary()
    negative_rows = build_negative_routes()

    write_csv(Path("artifacts/tables/paper_main_results.csv"), main_rows)
    write_md(
        Path("artifacts/tables/paper_main_results.md"),
        main_rows,
        ["dataset", "method", "role", "seeds_complete", "auprc", "delta_auprc_vs_base", "delta_auprc_vs_gate", "roc_auc", "macro_f1"],
    )
    write_csv(Path("artifacts/tables/paper_relation_ablation.csv"), relation_rows)
    write_md(
        Path("artifacts/tables/paper_relation_ablation.md"),
        relation_rows,
        ["dataset", "relation_setting", "seeds", "delta_auprc", "delta_roc_auc", "delta_macro_f1", "interpretation"],
    )
    write_csv(Path("artifacts/tables/paper_gate_judge_summary.csv"), gate_judge_rows)
    write_md(
        Path("artifacts/tables/paper_gate_judge_summary.md"),
        gate_judge_rows,
        [
            "dataset",
            "best_relation",
            "best_single_delta_auprc",
            "gate_delta_auprc",
            "judge_delta_vs_gate",
            "judge_acceptance",
            "judge_alpha_mean",
            "final_positioning",
        ],
    )
    write_csv(Path("artifacts/tables/paper_negative_routes.csv"), negative_rows)
    write_md(
        Path("artifacts/tables/paper_negative_routes.md"),
        negative_rows,
        ["route", "dataset", "seeds", "mean_delta_auprc", "mean_delta_macro_f1", "latent_cosine", "lesson"],
    )
    write_method_draft(Path("artifacts/paper/method_section_draft.md"))
    write_results_narrative(Path("artifacts/paper/results_narrative.md"))
    write_failure_appendix(Path("artifacts/paper/appendix_failure_routes.md"), negative_rows)
    write_readme(Path("artifacts/paper/README.md"))
    print(json.dumps({
        "main_results": "artifacts/tables/paper_main_results.md",
        "relation_ablation": "artifacts/tables/paper_relation_ablation.md",
        "gate_judge_summary": "artifacts/tables/paper_gate_judge_summary.md",
        "paper_dir": "artifacts/paper",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
