# qwen_directional_t200 Stage2 Quality, 5 Seeds

- Generated git hash: `725bd98`
- Mean accepted: 193.8/200
- Mean acceptance rate: 96.900%
- Mean LLM calls: 225.4
- Mean verifier retries: 19.2

## Per-Seed Table

| seed | accepted | rejected | acceptance_rate | llm_calls | verifier_retries | supporting_evidence_entropy | counter_evidence_entropy | evidence_direction_distribution | risk_type_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 193 | 7 | 0.965 | 223 | 16 | 3.1984129314474847 | 0.22671215594730051 | {"decrease_risk": 63, "increase_risk": 71, "uncertain": 66} | {"feature_structure_conflict": 71, "weak_or_uncertain_evidence": 129} |
| 123 | 194 | 6 | 0.97 | 223 | 17 | 3.213024335204873 | 0.205592508185083 | {"decrease_risk": 59, "increase_risk": 79, "uncertain": 62} | {"feature_structure_conflict": 79, "weak_or_uncertain_evidence": 121} |
| 456 | 195 | 5 | 0.975 | 221 | 16 | 3.1553235015883727 | 0.17484673724763006 | {"decrease_risk": 62, "increase_risk": 81, "uncertain": 57} | {"feature_structure_conflict": 81, "weak_or_uncertain_evidence": 119} |
| 789 | 194 | 6 | 0.97 | 225 | 19 | 3.026440037363147 | 0.23456950802708504 | {"decrease_risk": 57, "increase_risk": 97, "uncertain": 46} | {"feature_structure_conflict": 97, "weak_or_uncertain_evidence": 103} |
| 2026 | 193 | 7 | 0.965 | 235 | 28 | 2.95608511487934 | 0.30547186246445523 | {"decrease_risk": 54, "increase_risk": 99, "uncertain": 47} | {"feature_structure_conflict": 99, "weak_or_uncertain_evidence": 101} |
| mean | 193.8 | 6.2 | 0.969 | 225.4 | 19.2 | 3.1098571840966436 | 0.22943855437431077 |  |  |


Stage2 quality gate passed: all seeds have accepted ERRs, accepted+rejected=200, score-blind passed, and non-degenerate evidence_direction.
