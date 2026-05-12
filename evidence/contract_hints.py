from __future__ import annotations

CONTRACT_HINTS = {
    "structural_discrepancy": (
        "structural_discrepancy:\n"
        "  supporting_evidence must include at least one of:\n"
        "  - degree_level (when degree is high)\n"
        "  - neighbor_consistency (when consistency is low)\n"
        "  - detector_signal (when signal indicates discrepancy)\n"
        "  Choose this when structural features strongly indicate fraud."
    ),
    "camouflage_neighbor": (
        "camouflage_neighbor:\n"
        "  supporting_evidence must include neighbor_consistency.\n"
        "  Do not choose camouflage_neighbor if neighbor_consistency is high.\n"
        "  Choose this when neighbors show inconsistent patterns suggesting camouflage."
    ),
    "spectral_anomaly": (
        "spectral_anomaly:\n"
        "  supporting_evidence must include detector_signal.\n"
        "  If detector_signal_strength is available and not weak, include detector_signal_strength.\n"
        "  Do not choose spectral_anomaly when detector_signal_strength is weak.\n"
        "  Choose this when spectral/frequency analysis reveals anomalous patterns."
    ),
    "feature_structure_conflict": (
        "feature_structure_conflict:\n"
        "  supporting_evidence must include feature_neighbor_discrepancy.\n"
        "  Do not choose feature_structure_conflict if feature_neighbor_discrepancy is low.\n"
        "  Choose this when node features conflict with neighborhood structure."
    ),
    "relation_or_burst_anomaly": (
        "relation_or_burst_anomaly:\n"
        "  supporting_evidence must include degree_level or a relation/burst-related detector signal if available.\n"
        "  Choose this when relational patterns or burst activity indicates fraud."
    ),
    "weak_or_uncertain_evidence": (
        "weak_or_uncertain_evidence:\n"
        "  Choose this when evidence is weak, conflicting, insufficient,\n"
        "  or no other risk_type contract can be satisfied.\n"
        "  This is the safe default when uncertain."
    ),
}


def get_contract_hint_for_reason_type(risk_type: str) -> str:
    return CONTRACT_HINTS.get(risk_type, "")


def get_all_contract_hints() -> str:
    return "\n\n".join(CONTRACT_HINTS.values())


def get_retry_hint(reject_reasons: list[str], err: ERR | None, payload: dict) -> str:
    hints = []

    if "contract_required_not_satisfied" in reject_reasons:
        if err and err.risk_type in CONTRACT_HINTS:
            hints.append(f"Your previous risk_type '{err.risk_type}' did not satisfy its contract.")
            hints.append(CONTRACT_HINTS[err.risk_type])
        else:
            hints.append("Your previous ERR did not satisfy the evidence contract.")

    if "unavailable_evidence" in reject_reasons:
        reasoning = payload.get("reasoning", {})
        allowed_support = reasoning.get("allowed_support_ids", [])
        allowed_counter = reasoning.get("allowed_counter_ids", [])
        hints.append(f"You cited evidence fields that are not available.")
        hints.append(f"allowed_support_ids: {allowed_support}")
        hints.append(f"allowed_counter_ids: {allowed_counter}")

    if "invalid_support_role" in reject_reasons:
        hints.append("Your supporting_evidence contains fields not in allowed_support_ids.")

    if "invalid_counter_role" in reject_reasons:
        hints.append("Your counter_evidence contains fields not in allowed_counter_ids.")

    if "support_counter_overlap" in reject_reasons:
        hints.append("supporting_evidence and counter_evidence must not overlap.")

    if "score_leakage" in reject_reasons:
        hints.append("Your output contained references to scores or predictions. Remove all such references.")

    if not hints:
        hints.append("Your previous output did not pass verification. Please follow the contract rules more carefully.")

    hints.append("\nIf you cannot satisfy the required evidence for any risk_type, choose weak_or_uncertain_evidence.")

    return "\n".join(hints)
