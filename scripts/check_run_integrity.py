from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

SCORE_LEAKAGE_PATTERN = re.compile(
    r"\b(base_score|score|logit|logits|prob|probs|probability|probabilities|confidence)\b",
    re.IGNORECASE,
)


def check_file_exists(path: Path, required: bool = True) -> tuple[bool, str]:
    if path.exists():
        return True, f"OK: {path.name}"
    if required:
        return False, f"MISSING: {path.name}"
    return True, f"WARNING: {path.name} not found (optional)"


def check_score_leakage(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return True, f"SKIP: {path.name} not found"

    with open(path) as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            text = json.dumps(data)
            if SCORE_LEAKAGE_PATTERN.search(text):
                return False, f"SCORE LEAKAGE in {path.name} line {i}"

    return True, f"OK: {path.name} score-blind"


def check_accepted_plus_rejected(err_cache_dir: Path) -> tuple[bool, str]:
    accepted_path = err_cache_dir / "accepted_err.jsonl"
    rejected_path = err_cache_dir / "rejected_err.jsonl"
    rule_err_path = err_cache_dir / "rule_err.jsonl"

    if not all(p.exists() for p in [accepted_path, rejected_path, rule_err_path]):
        return True, "SKIP: ERR files not complete"

    num_accepted = sum(1 for _ in open(accepted_path))
    num_rejected = sum(1 for _ in open(rejected_path))
    num_total = sum(1 for _ in open(rule_err_path))

    if num_accepted + num_rejected == num_total:
        return True, f"OK: accepted({num_accepted}) + rejected({num_rejected}) == total({num_total})"
    return False, f"MISMATCH: accepted({num_accepted}) + rejected({num_rejected}) != total({num_total})"


def check_verifier_stats(err_cache_dir: Path) -> tuple[bool, str]:
    stats_path = err_cache_dir / "verifier_stats.json"
    if not stats_path.exists():
        return True, "SKIP: verifier_stats.json not found"

    with open(stats_path) as f:
        stats = json.load(f)

    rate = stats.get("acceptance_rate", -1)
    if not (0 <= rate <= 1):
        return False, f"INVALID acceptance_rate: {rate}"

    return True, f"OK: acceptance_rate={rate:.2%}"


def check_metrics_keys(results_dir: Path, stage: str) -> tuple[bool, str]:
    metrics_path = results_dir / f"{stage}_metrics.json"
    if not metrics_path.exists():
        return True, f"SKIP: {metrics_path.name} not found"

    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = ["roc_auc", "auprc", "f1"]
    missing = [k for k in required_keys if k not in metrics]
    if missing:
        return False, f"MISSING keys in {metrics_path.name}: {missing}"

    return True, f"OK: {metrics_path.name} has {required_keys}"


def check_checkpoint_order(checkpoint_dir: Path) -> tuple[bool, str]:
    base_path = checkpoint_dir / "base.pt"
    reasoner_path = checkpoint_dir / "reasoner.pt"

    if not base_path.exists() or not reasoner_path.exists():
        return True, "SKIP: checkpoints not complete"

    base_mtime = base_path.stat().st_mtime
    reasoner_mtime = reasoner_path.stat().st_mtime

    if reasoner_mtime >= base_mtime:
        return True, "OK: reasoner.pt newer than base.pt"
    return False, "WARNING: reasoner.pt older than base.pt"


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

    checkpoint_dir = Path("artifacts") / "checkpoints" / dataset_name / model_name / run_name / f"seed_{seed}"
    log_dir = Path("artifacts") / "logs" / dataset_name / model_name / run_name / f"seed_{seed}"
    err_cache_dir = Path("artifacts") / "err_cache" / dataset_name / model_name / run_name / f"seed_{seed}"
    results_dir = Path("artifacts") / "results" / dataset_name / model_name / run_name / f"seed_{seed}"

    base_checkpoint_dir = Path("artifacts") / "checkpoints" / dataset_name / model_name / "base" / f"seed_{seed}"
    base_results_dir = Path("artifacts") / "results" / dataset_name / model_name / "base" / f"seed_{seed}"

    checks = []

    print(f"Checking run integrity for: {dataset_name}/{model_name}/{run_name}/seed_{seed}")
    print("=" * 60)

    files_to_check = [
        (base_checkpoint_dir / "base.pt", True),
        (checkpoint_dir / "reasoner.pt", True),
        (base_results_dir / "stage1_metrics.json", False),
        (results_dir / "stage3_metrics.json", True),
        (err_cache_dir / "evidence_cards.jsonl", True),
        (err_cache_dir / "teacher_payloads.jsonl", True),
        (err_cache_dir / "rule_err.jsonl", True),
        (err_cache_dir / "accepted_err.jsonl", True),
        (err_cache_dir / "rejected_err.jsonl", True),
        (err_cache_dir / "verifier_stats.json", True),
    ]

    for path, required in files_to_check:
        ok, msg = check_file_exists(path, required)
        checks.append((ok, msg))
        print(f"  {msg}")

    ok, msg = check_score_leakage(err_cache_dir / "teacher_payloads.jsonl")
    checks.append((ok, msg))
    print(f"  {msg}")

    ok, msg = check_accepted_plus_rejected(err_cache_dir)
    checks.append((ok, msg))
    print(f"  {msg}")

    ok, msg = check_verifier_stats(err_cache_dir)
    checks.append((ok, msg))
    print(f"  {msg}")

    ok, msg = check_metrics_keys(results_dir, "stage1")
    checks.append((ok, msg))
    print(f"  {msg}")

    ok, msg = check_metrics_keys(results_dir, "stage3")
    checks.append((ok, msg))
    print(f"  {msg}")

    ok, msg = check_checkpoint_order(checkpoint_dir)
    checks.append((ok, msg))
    print(f"  {msg}")

    print("=" * 60)
    all_ok = all(ok for ok, _ in checks)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
        for ok, msg in checks:
            if not ok:
                print(f"  FAIL: {msg}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
