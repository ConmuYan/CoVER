from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--enable_retry", action="store_true", default=True)
    parser.add_argument("--disable_retry", action="store_true")
    parser.add_argument("--max_verifier_retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = args.seed

    model_path = args.model_path or config.get("llm", {}).get("model_name_or_path")
    if not model_path or not Path(model_path).exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    enable_retry = args.enable_retry and not args.disable_retry

    temp_config = config.copy()
    temp_config["llm"] = {
        "backend": "transformers_local",
        "model_name_or_path": model_path,
        "temperature": 0.0,
        "max_retries": 1,
        "max_new_tokens": 256,
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "max_trace_nodes_debug": args.num_samples,
        "max_trace_nodes": args.num_samples,
        "enable_verifier_retry": enable_retry,
        "max_verifier_retries": args.max_verifier_retries,
        "max_parse_retries": 1,
    }
    temp_config["train"]["seed"] = seed

    temp_config_path = Path("/tmp") / f"qwen_calibration_{dataset_name}_{model_name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f)

    print(f"Running Qwen calibration with {args.num_samples} samples...")
    print(f"Model: {model_path}")
    print(f"Retry: {'enabled' if enable_retry else 'disabled'}")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_stage2_err.py"),
        "--config", str(temp_config_path),
        "--teacher", "llm",
        "--debug",
    ]
    if enable_retry:
        cmd.extend(["--enable_retry", "--max_verifier_retries", str(args.max_verifier_retries)])

    result = subprocess.run(cmd, capture_output=False)

    temp_config_path.unlink(missing_ok=True)

    err_cache_dir = Path("artifacts") / "err_cache" / dataset_name / model_name / f"seed_{seed}"
    report_dir = Path("artifacts") / "reports" / dataset_name / model_name / f"seed_{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    if (err_cache_dir / "stage2_stats.json").exists():
        with open(err_cache_dir / "stage2_stats.json") as f:
            stats = json.load(f)

    raw_outputs = []
    if (err_cache_dir / "raw_llm_outputs.jsonl").exists():
        with open(err_cache_dir / "raw_llm_outputs.jsonl") as f:
            for line in f:
                raw_outputs.append(json.loads(line))

    accepted = []
    if (err_cache_dir / "accepted_err.jsonl").exists():
        with open(err_cache_dir / "accepted_err.jsonl") as f:
            for line in f:
                accepted.append(json.loads(line))

    rejected = []
    if (err_cache_dir / "rejected_err.jsonl").exists():
        with open(err_cache_dir / "rejected_err.jsonl") as f:
            for line in f:
                rejected.append(json.loads(line))

    risk_type_dist = {}
    for err in accepted + rejected:
        rt = err.get("risk_type", "unknown")
        risk_type_dist[rt] = risk_type_dist.get(rt, 0) + 1

    report = {
        "model_path": model_path,
        "num_samples": args.num_samples,
        "num_initial_calls": stats.get("num_initial_calls", 0),
        "num_total_llm_calls": stats.get("num_total_llm_calls", 0),
        "parse_success": stats.get("num_parse_success", 0),
        "parse_failed": stats.get("num_parse_failed", 0),
        "accepted_after_initial": stats.get("num_accepted_after_initial", 0),
        "accepted_after_retry": stats.get("num_accepted_after_retry", 0),
        "final_accepted": stats.get("num_final_accepted", 0),
        "final_rejected": stats.get("num_final_rejected", 0),
        "final_acceptance_rate": stats.get("final_acceptance_rate", 0),
        "reject_reason_counts": stats.get("reject_reason_counts", {}),
        "risk_type_distribution": risk_type_dist,
        "avg_attempts_per_node": stats.get("num_total_llm_calls", 0) / max(stats.get("num_initial_calls", 1), 1),
        "score_blind_check_passed": stats.get("score_blind_check_passed", False),
        "accepted_examples": accepted[:3],
        "rejected_examples": rejected[:3],
    }

    with open(report_dir / "qwen_calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Qwen Calibration Report",
        "",
        f"Model: {model_path}",
        f"Samples: {args.num_samples}",
        f"Retry: {'enabled' if enable_retry else 'disabled'}",
        "",
        "## Results",
        f"- Parse success: {report['parse_success']}",
        f"- Parse failed: {report['parse_failed']}",
        f"- Accepted after initial: {report['accepted_after_initial']}",
        f"- Accepted after retry: {report['accepted_after_retry']}",
        f"- Final accepted: {report['final_accepted']}",
        f"- Final rejected: {report['final_rejected']}",
        f"- Final acceptance rate: {report['final_acceptance_rate']:.1%}",
        "",
        "## Risk Type Distribution",
    ]
    for rt, count in risk_type_dist.items():
        md_lines.append(f"- {rt}: {count}")

    with open(report_dir / "qwen_calibration_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n=== Qwen Calibration Report ===")
    print(f"  Parse success: {report['parse_success']}")
    print(f"  Final accepted: {report['final_accepted']}")
    print(f"  Final rejected: {report['final_rejected']}")
    print(f"  Acceptance rate: {report['final_acceptance_rate']:.1%}")
    print(f"\nReport saved to: {report_dir}")


if __name__ == "__main__":
    main()
