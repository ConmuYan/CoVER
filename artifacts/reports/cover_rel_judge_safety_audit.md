# CoVER-REL-Judge Safety Audit

## yelpchi

- Judge acceptance mean: 0.923333
- Forbidden audit passed all seeds: True
- Accepted outputs passed all seeds: True
- Prompt/packet audit passed all seeds: True
- Rejected judge outputs are excluded from judge features and fusion training.
- `short_explanation` is human-facing only and is not used in loss.
- Stage3 training consumes accepted judge features only and does not call Qwen.

- seed 42: accepted=52, rejected=8, acceptance=0.866667, audit_passed=True
- seed 123: accepted=56, rejected=4, acceptance=0.933333, audit_passed=True
- seed 456: accepted=57, rejected=3, acceptance=0.950000, audit_passed=True
- seed 789: accepted=54, rejected=6, acceptance=0.900000, audit_passed=True
- seed 2026: accepted=58, rejected=2, acceptance=0.966667, audit_passed=True

## amazon

- Judge acceptance mean: 0.940000
- Forbidden audit passed all seeds: True
- Accepted outputs passed all seeds: True
- Prompt/packet audit passed all seeds: True
- Rejected judge outputs are excluded from judge features and fusion training.
- `short_explanation` is human-facing only and is not used in loss.
- Stage3 training consumes accepted judge features only and does not call Qwen.

- seed 42: accepted=56, rejected=4, acceptance=0.933333, audit_passed=True
- seed 123: accepted=56, rejected=4, acceptance=0.933333, audit_passed=True
- seed 456: accepted=56, rejected=4, acceptance=0.933333, audit_passed=True
- seed 789: accepted=56, rejected=4, acceptance=0.933333, audit_passed=True
- seed 2026: accepted=58, rejected=2, acceptance=0.966667, audit_passed=True

