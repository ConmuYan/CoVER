from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.llm_teacher import OfflineLLMTeacher
from evidence.verifier import EvidenceContractVerifier, load_contracts


def _make_sample_payloads(num_samples: int = 2) -> list[dict]:
    return [
        {
            "node_id": i,
            "detector_name": "bwgnn",
            "reasoning": {
                "degree_level": "high" if i % 2 == 0 else "medium",
                "neighbor_consistency": "low" if i % 3 == 0 else "high",
                "feature_neighbor_discrepancy": "high" if i % 4 == 0 else "low",
                "detector_signal": "high_frequency_response_high" if i % 2 == 0 else "normal",
                "detector_signal_strength": "strong" if i % 2 == 0 else "weak",
                "counter_signal": "benign_neighbor_signal_low",
                "allowed_support_ids": ["degree_level", "detector_signal", "detector_signal_strength"],
                "allowed_counter_ids": ["counter_signal"],
            },
        }
        for i in range(num_samples)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    llm_config = config.get("llm", {})
    model_path = llm_config.get("model_name_or_path")

    if not model_path or not Path(model_path).exists():
        print(f"Model path not found: {model_path}")
        print("Skipping local LLM smoke test")
        sys.exit(0)

    print(f"Loading model from: {model_path}")
    teacher = OfflineLLMTeacher(
        backend="transformers_local",
        model_name_or_path=model_path,
        temperature=llm_config.get("temperature", 0.0),
        max_new_tokens=llm_config.get("max_new_tokens", 256),
        device_map=llm_config.get("device_map", "auto"),
        torch_dtype=llm_config.get("torch_dtype", "auto"),
        trust_remote_code=llm_config.get("trust_remote_code", True),
    )

    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts, enable_label_compatibility=False)

    payloads = _make_sample_payloads(args.num_samples)
    results = []

    for payload in payloads:
        print(f"\nProcessing node {payload['node_id']}...")
        err, metadata = teacher.generate(payload)

        result = {
            "node_id": payload["node_id"],
            "parsed_ok": metadata["parsed_ok"],
            "raw_output": metadata["raw_output"],
            "parse_error": metadata.get("parse_error"),
        }

        if err is not None:
            accepted, reasons = verifier.verify(err, None)
            result["err"] = {
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
            }
            result["accepted"] = accepted
            result["reject_reasons"] = reasons if not accepted else []
        else:
            result["accepted"] = False
            result["reject_reasons"] = ["parse_failed"]

        results.append(result)
        print(f"  Parsed: {metadata['parsed_ok']}, Accepted: {result.get('accepted', False)}")

    num_parsed = sum(1 for r in results if r["parsed_ok"])
    num_accepted = sum(1 for r in results if r.get("accepted", False))

    summary = {
        "num_samples": len(payloads),
        "num_parsed": num_parsed,
        "num_accepted": num_accepted,
        "acceptance_rate": num_accepted / len(payloads) if payloads else 0,
        "results": results,
    }

    report_dir = Path("artifacts") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "local_llm_smoke_test.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Local LLM Smoke Test ===")
    print(f"  Samples: {len(payloads)}")
    print(f"  Parsed: {num_parsed}")
    print(f"  Accepted: {num_accepted}")
    print(f"  Acceptance rate: {summary['acceptance_rate']:.1%}")
    print(f"\nReport saved to: {report_dir / 'local_llm_smoke_test.json'}")


if __name__ == "__main__":
    main()
