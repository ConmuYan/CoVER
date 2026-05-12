from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = config["train"]["seed"]

    err_cache_dir = Path("artifacts") / "err_cache" / dataset_name / model_name / f"seed_{seed}"
    report_dir = Path("artifacts") / "reports" / dataset_name / model_name / f"seed_{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

    rejected = load_jsonl(err_cache_dir / "rejected_err.jsonl")
    evidence_cards = load_jsonl(err_cache_dir / "evidence_cards.jsonl")
    rule_errs = load_jsonl(err_cache_dir / "rule_err.jsonl")

    with open(err_cache_dir / "verifier_stats.json") as f:
        verifier_stats = json.load(f)

    card_map = {card["node_id"]: card for card in evidence_cards}

    reject_reason_counts = Counter()
    rejected_risk_types = Counter()
    rejected_signals = Counter()
    rejected_strengths = Counter()

    examples_by_reason: dict[str, list[dict]] = {}

    for item in rejected:
        reasons = item.get("reject_reasons", [])
        for r in reasons:
            reject_reason_counts[r] += 1
            if r not in examples_by_reason:
                examples_by_reason[r] = []
            if len(examples_by_reason[r]) < 3:
                examples_by_reason[r].append(item)

        rejected_risk_types[item["risk_type"]] += 1

        card = card_map.get(item["node_id"])
        if card:
            rejected_signals[card["reasoning"]["detector_signal"]] += 1
            rejected_strengths[card["reasoning"]["detector_signal_strength"]] += 1

    weak_spectral = 0
    for item in rejected:
        if item["risk_type"] == "spectral_anomaly":
            card = card_map.get(item["node_id"])
            if card and card["reasoning"]["detector_signal_strength"] == "weak":
                weak_spectral += 1

    missing_field_refs = 0
    for item in rejected:
        card = card_map.get(item["node_id"])
        if not card:
            continue
        reasoning = card["reasoning"]
        if item["risk_type"] == "spectral_anomaly":
            if "detector_signal" not in item["supporting_evidence"]:
                missing_field_refs += 1
        elif item["risk_type"] == "feature_structure_conflict":
            if "feature_neighbor_discrepancy" not in item["supporting_evidence"]:
                missing_field_refs += 1

    report = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "verifier_stats": verifier_stats,
        "reject_reason_counts": dict(reject_reason_counts),
        "rejected_risk_type_distribution": dict(rejected_risk_types),
        "rejected_detector_signal_distribution": dict(rejected_signals),
        "rejected_detector_signal_strength_distribution": dict(rejected_strengths),
        "weak_spectral_anomaly_count": weak_spectral,
        "missing_field_reference_count": missing_field_refs,
        "examples_by_reason": {k: v[:3] for k, v in examples_by_reason.items()},
    }

    with open(report_dir / "verifier_rejection_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Verifier Rejection Report",
        f"",
        f"Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}",
        "",
        "## Summary",
        f"- Total rejected: {len(rejected)}",
        f"- Acceptance rate: {verifier_stats['acceptance_rate']:.2%}",
        "",
        "## Reject Reason Counts",
    ]
    for reason, count in reject_reason_counts.most_common():
        md_lines.append(f"- {reason}: {count}")

    md_lines.extend([
        "",
        "## Rejected Risk Type Distribution",
    ])
    for rt, count in rejected_risk_types.most_common():
        md_lines.append(f"- {rt}: {count}")

    md_lines.extend([
        "",
        "## Rejected Detector Signal Distribution",
    ])
    for sig, count in rejected_signals.most_common():
        md_lines.append(f"- {sig}: {count}")

    md_lines.extend([
        "",
        "## Issues Found",
        f"- Weak spectral_anomaly count: {weak_spectral}",
        f"- Missing field reference count: {missing_field_refs}",
    ])

    with open(report_dir / "verifier_rejection_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Report saved to: {report_dir}")
    print(f"\nReject reason counts:")
    for reason, count in reject_reason_counts.most_common():
        print(f"  {reason}: {count}")
    print(f"\nRejected risk_type distribution:")
    for rt, count in rejected_risk_types.most_common():
        print(f"  {rt}: {count}")
    print(f"\nIssues:")
    print(f"  Weak spectral_anomaly: {weak_spectral}")
    print(f"  Missing field references: {missing_field_refs}")


if __name__ == "__main__":
    main()
