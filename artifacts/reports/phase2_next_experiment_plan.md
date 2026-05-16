# Phase2 Next Experiment Plan

## Starting Point

The fixed Phase2 runs show two different regimes:

- YelpChi: E0/E1/E2 are all strong. E2 reaches AUPRC 0.5676 and the relation gate is RUR-concentrated. The next work should be sensitivity and ablation, not broad tuning.
- Amazon: E0/E1/E2 stay near BWGNN base. The diagnostics show large average relation residuals in v1 and near-identity behavior in v2. The next default should search for small bounded residuals and a distributed gate before enabling judge residuals.

## YelpChi

Keep the theory-guided default fixed:

```yaml
lr: 1e-3
weight_decay: 1e-4
epochs: 300
patience: 50
early_stop_metric: val_auprc
rel_hidden_dim: 64
rel_num_layers: 2
rel_dropout: 0.30
tau_gate: 0.7
delta_rel_max: 2.0
delta_llm_max: 0.75
alpha_bias_init: -3.0
use_short_explanation_text: false
accept_only_verified_judge: true
```

Run only targeted sensitivity:

```bash
bash scripts/run_phase2_yelpchi_targeted_sensitivity.sh 0 42 123 456 789 2026
```

The planned knobs are:

- `lambda_trust`: 0, 1e-3, 3e-3, 1e-2, 3e-2
- `lambda_sparse`: 0, 3e-4, 1e-3, 3e-3
- `lambda_align`: 0, 1e-3, 3e-3, 1e-2 with `alpha_max=0`
- `alpha_max`: 0, 0.1, 0.3 with `lambda_align=1e-3`

The expected paper evidence is not just AUPRC. Report AUPRC, mean absolute relation residual, gate entropy, RUR gate weight, mean accepted alpha, and rejected/missing alpha.

## Amazon

Do not start from the LLM branch. The current evidence says the main problem is relation residual calibration:

- v1 can learn large relation residuals with no AUPRC gain.
- v2 constrains the residual but often stops near identity.
- The legacy gate's gain is small, so the new default should search for small nudges rather than large intervention.

Run the candidate search:

```bash
bash scripts/run_phase2_amazon_default_candidates.sh 0 42 123 456 789 2026
```

Candidate logic:

- `c1`: `delta_rel_max=0.5`, `lambda_trust=1e-2`, `tau_gate=1.8`, `lambda_sparse=0`
- `c2`: `delta_rel_max=0.75`, `lambda_trust=1e-2`, `tau_gate=1.8`, `lambda_sparse=0`
- `c3`: `delta_rel_max=0.5`, `lambda_trust=3e-3`, `tau_gate=1.8`, `lambda_sparse=0`
- `c4`: `delta_rel_max=0.5`, `lambda_trust=1e-2`, `tau_gate=1.3`, `lambda_sparse=3e-4`
- `c5`: c1 plus judge alignment only
- `c6`: c1 plus judge alignment and `alpha_max=0.05`

Selection rule:

1. Pick the best relation-only candidate by 5-seed AUPRC, but reject it if it wins by a tiny amount while using saturated residuals.
2. Prefer candidates with `mean|delta_rel|` materially below the v1 default and with non-collapsed gate entropy.
3. Only after selecting the relation-only default, compare c5/c6 to check whether judge alignment or residual adds anything.
4. If none exceeds legacy gate, report Amazon as a near-saturated base case and keep the legacy gate as a reference line.
