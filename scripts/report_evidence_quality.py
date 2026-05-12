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
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = args.seed or config["train"]["seed"]
    run_name = args.run_name or config.get("run", {}).get("run_name", "base")

    err_cache_dir = Path("artifacts") / "err_cache" / dataset_name / model_name / run_name / f"seed_{seed}"
    report_dir = Path("artifacts") / "reports" / dataset_name / model_name / run_name / f"seed_{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

    evidence_cards = load_jsonl(err_cache_dir / "evidence_cards.jsonl")
    accepted = load_jsonl(err_cache_dir / "accepted_err.jsonl")
    rejected = load_jsonl(err_cache_dir / "rejected_err.jsonl")
    raw_outputs = load_jsonl(err_cache_dir / "raw_llm_outputs.jsonl")

    verifier_stats = {}
    if (err_cache_dir / "verifier_stats.json").exists():
        with open(err_cache_dir / "verifier_stats.json") as f:
            verifier_stats = json.load(f)

    stage2_stats = {}
    if (err_cache_dir / "stage2_stats.json").exists():
        with open(err_cache_dir / "stage2_stats.json") as f:
            stage2_stats = json.load(f)

    risk_type_dist = Counter()
    supporting_dist = Counter()
    counter_dist = Counter()
    signal_dist = Counter()
    strength_dist = Counter()

    for err in accepted + rejected:
        risk_type_dist[err.get("risk_type", "unknown")] += 1
        for s in err.get("supporting_evidence", []):
            supporting_dist[s] += 1
        for c in err.get("counter_evidence", []):
            counter_dist[c] += 1

    for card in evidence_cards:
        reasoning = card.get("reasoning", {})
        signal_dist[reasoning.get("detector_signal", "unknown")] += 1
        strength_dist[reasoning.get("detector_signal_strength", "unknown")] += 1

    weak_count = sum(1 for rt in risk_type_dist if "weak" in rt.lower() or "uncertain" in rt.lower())
    weak_total = sum(risk_type_dist[rt] for rt in risk_type_dist if "weak" in rt.lower() or "uncertain" in rt.lower())
    total_err = len(accepted) + len(rejected)

    num_llm_calls = len(raw_outputs)
    num_parse_success = sum(1 for r in raw_outputs if r.get("parsed_ok", False))
    num_parse_failed = num_llm_calls - num_parse_success

    report = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "num_cards": len(evidence_cards),
        "num_llm_calls": num_llm_calls,
        "num_parse_success": num_parse_success,
        "num_parse_failed": num_parse_failed,
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "acceptance_rate": len(accepted) / total_err if total_err > 0 else 0,
        "reject_reason_counts": verifier_stats.get("reject_reason_counts", {}),
        "accepted_after_initial": stage2_stats.get("num_accepted_after_initial", 0),
        "accepted_after_retry": stage2_stats.get("num_accepted_after_retry", 0),
        "risk_type_distribution": dict(risk_type_dist),
        "weak_or_uncertain_count": weak_total,
        "weak_or_uncertain_ratio": weak_total / total_err if total_err > 0 else 0,
        "supporting_evidence_distribution": dict(supporting_dist.most_common(20)),
        "counter_evidence_distribution": dict(counter_dist.most_common(20)),
        "detector_signal_distribution": dict(signal_dist),
        "detector_signal_strength_distribution": dict(strength_dist),
        "accepted_examples": accepted[:5],
        "rejected_examples": rejected[:5],
    }

    with open(report_dir / "evidence_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Evidence Quality Report",
        "",
        f"Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}",
        "",
        "## Summary",
        f"- Cards: {len(evidence_cards)}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        f"- Acceptance rate: {report['acceptance_rate']:.1%}",
        f"- Weak/uncertain ratio: {report['weak_or_uncertain_ratio']:.1%}",
        "",
        "## Risk Type Distribution",
    ]
    for rt, count in risk_type_dist.most_common():
        md_lines.append(f"- {rt}: {count}")

    md_lines.extend(["", "## Top Supporting Evidence"])
    for s, count in supporting_dist.most_common(10):
        md_lines.append(f"- {s}: {count}")

    with open(report_dir / "evidence_quality_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Evidence quality report saved to: {report_dir}")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Acceptance rate: {report['acceptance_rate']:.1%}")


if __name__ == "__main__":
    main()
