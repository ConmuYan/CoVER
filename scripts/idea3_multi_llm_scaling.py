"""Idea 3 Multi-LLM Scaling Experiment.

Tests whether larger LLMs design better fraud feature composites (scaling law).
Models tested:
  - Qwen3-0.6B (0.6B params)
  - Qwen3-4B (4B params, base)
  - Qwen3-4B-Instruct-2507 (4B params, instruct)
  - Qwen3-8B (8B params)
  - Qwen3-32B (32B params, if available)
  - gemma-4-E2B-it (2B params)
  - BERT-base (110M params, PLM adapter)
  - RoBERTa-base (125M params, PLM adapter)

Phase 1: Generate 5 composite formulas from each LLM (unified prompt)
Phase 2: Evaluate all formulas on YelpChi-BWGNN seed_42 (quick screen)
Phase 3: Top-2 models expanded to 8 cells × 5 seeds
Phase 4: PLM adapter (BERT/RoBERTa) reward-weighted regression
Phase 5: Scaling plot + comparison table

Usage:
    python scripts/idea3_multi_llm_scaling.py --phase all --device cuda:1
    python scripts/idea3_multi_llm_scaling.py --phase 1 --device cuda:1   # just generate
    python scripts/idea3_multi_llm_scaling.py --phase 2 --device cuda:1   # just evaluate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import load_relation_stats, RELATION_SCHEMAS
from models.cover_rel_reasoner import CoVERRelReasoner

SEEDS = [42, 123, 456, 789, 2026]
CELLS = [
    ("yelpchi", "bwgnn"), ("yelpchi", "sage"), ("yelpchi", "gcn"), ("yelpchi", "gat"),
    ("amazon", "bwgnn"), ("amazon", "sage"), ("amazon", "gcn"), ("amazon", "gat"),
]

# ─── Unified prompt ──────────────────────────────────────────────────────────
FEATURE_PROMPT = """You are a fraud detection expert. I have a graph of user-product interactions with fraud labels.

For each node (user/product), I have 6 base features computed from 3 relation types (RUR=same-user review, RSR=same-product same-star, RTR=same-product same-month):

- f1 = fraud neighbor rate (RUR): fraction of RUR-neighbors labeled fraud in training set
- f2 = fraud neighbor rate (RSR): fraction of RSR-neighbors labeled fraud in training set
- f3 = fraud neighbor rate (RTR): fraction of RTR-neighbors labeled fraud in training set
- f4 = log(1 + degree in RUR)
- f5 = log(1 + degree in RSR)
- f6 = log(1 + degree in RTR)

Design exactly 5 composite features that combine these base features to better detect fraud.
Each formula should use f1-f6 and basic operations (+, -, *, /, log, sqrt, max, min).
Each formula must produce a single scalar value per node.

Output ONLY a JSON array of 5 objects, each with:
- "formula": the mathematical expression using f1-f6
- "reason": one sentence explaining why this helps detect fraud

Example format:
[{"formula": "f1*f3", "reason": "Interaction of fraud rates across review and temporal relations."}]

Output the JSON array now:"""

# ─── LLM registry ────────────────────────────────────────────────────────────
LLM_MODELS = {
    "qwen3-0.6b": {"path": "/data1/mq/models/Qwen3-0.6B", "params": 0.6e9},
    "qwen3-4b": {"path": "/data1/mq/models/Qwen3-4B", "params": 4e9},
    "qwen3-4b-instruct": {"path": "/data1/mq/models/Qwen3-4B-Instruct-2507", "params": 4e9},
    "qwen3-8b": {"path": "/data1/mq/models/Qwen3-8B", "params": 8e9},
    "qwen3-32b": {"path": "/data1/mq/models/Qwen3-32B", "params": 32e9},
    "gemma-4-e2b-it": {"path": "/data1/mq/models/gemma-4-E2B-it", "params": 2e9},
}

PLM_MODELS = {
    "bert-base": {"model_id": "bert-base-uncased", "params": 110e6},
    "roberta-base": {"model_id": "roberta-base", "params": 125e6},
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_cell(dataset: str, base: str, seed: int, device: str = "cuda:1"):
    """Load data, base outputs, relation features, and CoVER-REL delta_rel."""
    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    data = load_fraud_dataset(
        name=dataset, path=f"datasets/{mat_name}.mat", format="mat",
        seed=seed, train_ratio=0.4, val_test_ratio=[1, 2], stratified=True,
    )
    base_path = f"artifacts/base_outputs/{dataset}/{base}/seed_{seed}/_override_seed_{seed}_base.pt"
    base_outputs = torch.load(base_path, weights_only=True, map_location="cpu")
    base_logit = base_outputs["base_logits"].numpy()
    base_z = base_outputs["base_z"].numpy()

    canon_run = "idea1_canonical_clsonly" if (dataset == "yelpchi" and base == "bwgnn") else f"idea1_{dataset}_{base}_canonical_clsonly"
    rel_path = f"artifacts/relation_features/{dataset}/{base}/seed_{seed}/all/rel_stats.pt"
    rel_stats, _ = load_relation_stats(rel_path, num_nodes=data.x.shape[0])

    reasoner = CoVERRelReasoner(
        base_z_dim=base_z.shape[1],
        relation_names=list(RELATION_SCHEMAS[dataset].keys()),
        anchor_relation=list(RELATION_SCHEMAS[dataset].keys())[0],
        rel_stat_dim=9, rel_hidden_dim=64, rel_num_layers=2,
        rel_dropout=0.3, tau_gate=0.7, delta_rel_max=2.0,
    )
    ckpt = torch.load(
        f"artifacts/checkpoints/{dataset}/{base}/{canon_run}/seed_{seed}/reasoner.pt",
        weights_only=False, map_location="cpu",
    )
    reasoner.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    reasoner.eval()
    dev = torch.device(device)
    reasoner = reasoner.to(dev)
    with torch.no_grad():
        out = reasoner(
            torch.tensor(base_z, dtype=torch.float32).to(dev),
            torch.tensor(base_logit, dtype=torch.float32).to(dev),
            rel_stats.to(dev),
        )
    delta_rel = out["delta_rel"].cpu().numpy()
    return data, base_logit, base_z, delta_rel


def compute_graph_stats(dataset: str, data, adjs: dict):
    """Compute fraud_nbr_rates and log_degrees for all relations."""
    N = data.x.shape[0]
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    train_fraud = np.zeros(N)
    train_fraud[train_mask & (y == 1)] = 1
    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())
    features = {}
    for name in rel_names:
        A = adjs[name]
        deg = np.array(A.sum(axis=1)).flatten()
        rs = deg.copy()
        rs[rs == 0] = 1
        fraud_nbr = A.dot(train_fraud) / rs
        features[f"fraud_nbr_{name}"] = fraud_nbr
        features[f"log_deg_{name}"] = np.log1p(deg)
    return features, rel_names


def build_composite_features(features: dict, rel_names: list, exprs: list) -> np.ndarray:
    """Evaluate composite feature expressions."""
    f_map = {}
    for i, name in enumerate(rel_names):
        f_map[f"f{i+1}"] = features[f"fraud_nbr_{name}"]
        f_map[f"f{i+1+len(rel_names)}"] = features[f"log_deg_{name}"]
    composites = []
    for expr in exprs:
        try:
            # Sanitize: only allow safe operations
            safe_dict = {
                **f_map,
                "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                "max": np.maximum, "min": np.minimum,
            }
            feat = eval(expr, {"__builtins__": {}}, safe_dict)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
            composites.append(feat.reshape(-1, 1))
        except Exception as e:
            print(f"    Warning: expression '{expr}' failed: {e}")
            composites.append(np.zeros((len(next(iter(features.values()))), 1)))
    return np.hstack(composites) if composites else np.zeros((len(next(iter(features.values()))), 0))


def eval_lr(X_train, y_train, X_test, y_test, seed: int) -> dict:
    """Train LR and evaluate."""
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    lr.fit(X_train, y_train)
    prob = lr.predict_proba(X_test)[:, 1]
    return {
        "auprc": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }


def parse_llm_formulas(response: str) -> list[str]:
    """Extract formula strings from LLM JSON response."""
    # Try JSON parse
    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            items = json.loads(match.group())
            formulas = [item.get("formula", item.get("expr", "")) for item in items]
            return [f for f in formulas if f]
    except json.JSONDecodeError:
        pass
    # Fallback: extract lines that look like formulas
    formulas = []
    for line in response.split("\n"):
        line = line.strip()
        if any(op in line for op in ["f1", "f2", "f3", "f4", "f5", "f6"]) and \
           any(op in line for op in ["+", "-", "*", "/", "log", "sqrt", "max", "min"]):
            # Clean up
            f = re.sub(r'^[\d\.\-\s]*[\"\']*formula[\"\']*\s*[:=]\s*[\"\']*', '', line)
            f = re.sub(r'[\"\']\s*,?\s*$', '', f).strip()
            if f and len(f) < 100:
                formulas.append(f)
    return formulas[:5]


# ─── Phase 1: LLM Inference ─────────────────────────────────────────────────
def run_llm_inference(model_name: str, model_info: dict, device: str,
                      max_new_tokens: int = 1024) -> dict:
    """Run a single LLM to generate 5 composite formulas."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = model_info["path"]
    print(f"\n  Loading {model_name} from {model_path} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype and device_map based on model size
    params = model_info["params"]
    if params >= 16e9:
        # Large model: use 4-bit quantization
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    elif params >= 8e9:
        # 8B model: bf16, fits on single 3090
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Format prompt using chat template
    messages = [{"role": "user", "content": FEATURE_PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
        )
    gen_time = time.time() - t0

    # Decode only new tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    formulas = parse_llm_formulas(response)

    print(f"  Generated in {gen_time:.1f}s, got {len(formulas)} formulas")
    print(f"  Response preview: {response[:300]}...")
    print(f"  Parsed formulas: {formulas}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "response": response,
        "formulas": formulas,
        "load_time": load_time,
        "gen_time": gen_time,
        "params": params,
    }


def phase1_generate_all(device: str) -> dict:
    """Phase 1: Generate formulas from all LLMs."""
    print("=" * 60)
    print("PHASE 1: LLM Inference — Generate composite formulas")
    print("=" * 60)

    results = {}
    # Sort by size (small first for quick feedback)
    sorted_models = sorted(LLM_MODELS.items(), key=lambda x: x[1]["params"])

    for model_name, model_info in sorted_models:
        model_path = model_info["path"]
        if not Path(model_path).exists():
            print(f"\n  SKIP {model_name}: {model_path} not found")
            continue
        # Check it's not just config files
        model_files = list(Path(model_path).glob("*.safetensors")) + \
                      list(Path(model_path).glob("*.bin")) + \
                      list(Path(model_path).glob("model*.gguf"))
        if not model_files:
            print(f"\n  SKIP {model_name}: no model weights found in {model_path}")
            continue

        try:
            result = run_llm_inference(model_name, model_info, device)
            results[model_name] = result
        except Exception as e:
            print(f"\n  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"model": model_name, "error": str(e), "formulas": []}

    # Save raw results
    out_dir = Path("artifacts/results/idea3_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "llm_formulas.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved LLM results to {out_dir / 'llm_formulas.json'}")
    return results


# ─── Phase 2: Evaluate on YelpChi-BWGNN seed_42 ─────────────────────────────
def phase2_evaluate_quick(llm_results: dict, device: str) -> dict:
    """Phase 2: Evaluate all model formulas on YelpChi-BWGNN seed_42."""
    print("\n" + "=" * 60)
    print("PHASE 2: Quick Evaluation — YelpChi-BWGNN seed_42")
    print("=" * 60)

    dataset, base, seed = "yelpchi", "bwgnn", 42

    # Load data
    mat = loadmat("datasets/YelpChi.mat")
    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())
    data, base_logit, base_z, delta_rel = load_cell(dataset, base, seed, device)
    N = data.x.shape[0]
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    # Build adj CSR
    adjs = {}
    for name in rel_names:
        key = schema[name]["mat_key"]
        sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
        adjs[name] = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))

    features, _ = compute_graph_stats(dataset, data, adjs)
    X_base = base_logit.reshape(-1, 1)

    eval_results = {}
    for model_name, result in llm_results.items():
        formulas = result.get("formulas", [])
        if not formulas or len(formulas) < 3:
            print(f"  {model_name}: skip ({len(formulas)} formulas)")
            eval_results[model_name] = {"auprc": 0.0, "n_formulas": len(formulas), "formulas": formulas}
            continue

        llm_feats = build_composite_features(features, rel_names, formulas)
        X_combo = np.column_stack([X_base, llm_feats])
        metrics = eval_lr(X_combo[train_mask], y[train_mask], X_combo[test_mask], y[test_mask], seed)
        eval_results[model_name] = {**metrics, "n_formulas": len(formulas), "formulas": formulas}
        print(f"  {model_name}: AUPRC={metrics['auprc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  ({len(formulas)} formulas)")

    # Add baselines
    raw_feats = np.column_stack([features[f"fraud_nbr_{name}"] for name in rel_names] +
                                 [features[f"log_deg_{name}"] for name in rel_names])
    base_only = eval_lr(X_base[train_mask], y[train_mask], X_base[test_mask], y[test_mask], seed)
    base_raw = eval_lr(np.column_stack([X_base, raw_feats])[train_mask], y[train_mask],
                       np.column_stack([X_base, raw_feats])[test_mask], y[test_mask], seed)
    base_rel = eval_lr(np.column_stack([X_base, delta_rel])[train_mask], y[train_mask],
                       np.column_stack([X_base, delta_rel])[test_mask], y[test_mask], seed)

    eval_results["__base_only"] = base_only
    eval_results["__base_raw"] = base_raw
    eval_results["__base_rel"] = base_rel

    print(f"\n  Baselines: base_only={base_only['auprc']:.4f}  base_raw={base_raw['auprc']:.4f}  base_rel={base_rel['auprc']:.4f}")

    # Save
    out_dir = Path("artifacts/results/idea3_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "quick_eval_yelpchi_bwgnn_seed42.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    return eval_results


# ─── Phase 3: Full 8-cell benchmark for top-2 ────────────────────────────────
def phase3_full_benchmark(top_models: list[str], llm_results: dict, device: str) -> dict:
    """Phase 3: Run top-2 models on 8 cells × 5 seeds."""
    print("\n" + "=" * 60)
    print(f"PHASE 3: Full 8-cell benchmark — {top_models}")
    print("=" * 60)

    all_results = {}

    for dataset, base in CELLS:
        cell_key = f"{dataset}-{base}"
        print(f"\n{'='*50}")
        print(f"Cell: {cell_key}")
        print(f"{'='*50}")

        mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
        mat = loadmat(f"datasets/{mat_name}.mat")
        schema = RELATION_SCHEMAS[dataset]
        rel_names = list(schema.keys())

        cell_results = {}
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            data, base_logit, base_z, delta_rel = load_cell(dataset, base, seed, device)
            N = data.x.shape[0]
            y = data.y.numpy()
            train_mask = data.train_mask.numpy()
            test_mask = data.test_mask.numpy()

            adjs = {}
            for name in rel_names:
                key = schema[name]["mat_key"]
                sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
                adjs[name] = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))

            features, _ = compute_graph_stats(dataset, data, adjs)
            X_base = base_logit.reshape(-1, 1)

            r = {}
            r["base_only"] = eval_lr(X_base[train_mask], y[train_mask], X_base[test_mask], y[test_mask], seed)
            r["base_rel"] = eval_lr(np.column_stack([X_base, delta_rel])[train_mask], y[train_mask],
                                    np.column_stack([X_base, delta_rel])[test_mask], y[test_mask], seed)

            for model_name in top_models:
                formulas = llm_results.get(model_name, {}).get("formulas", [])
                if not formulas:
                    continue
                llm_feats = build_composite_features(features, rel_names, formulas)
                X_combo = np.column_stack([X_base, llm_feats])
                metrics = eval_lr(X_combo[train_mask], y[train_mask], X_combo[test_mask], y[test_mask], seed)
                r[f"base_{model_name}"] = metrics

                # REL + model
                X_full = np.column_stack([X_base, delta_rel, llm_feats])
                metrics_full = eval_lr(X_full[train_mask], y[train_mask], X_full[test_mask], y[test_mask], seed)
                r[f"base_rel_{model_name}"] = metrics_full

            cell_results[seed] = r
            # Print summary
            parts = [f"base={r['base_only']['auprc']:.4f}", f"rel={r['base_rel']['auprc']:.4f}"]
            for model_name in top_models:
                key = f"base_{model_name}"
                if key in r:
                    parts.append(f"{model_name}={r[key]['auprc']:.4f}")
            print(f"    {' | '.join(parts)}")

        all_results[cell_key] = cell_results

    # Save
    out_dir = Path("artifacts/results/idea3_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_benchmark_top2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ─── Phase 4: PLM Adapter (BERT/RoBERTa) ────────────────────────────────────
class PLMFeatureAdapter(torch.nn.Module):
    """BERT/RoBERTa + selection head for candidate formula selection.

    Input : textual description of graph statistics.
    Output: selection logits over a fixed 20-candidate pool.
    Training: REINFORCE with Gumbel-top-k sampling. Reward = val AUPRC −
        running baseline. This makes the PLM actually learn which composites
        to select for each dataset — gradients flow from val AUPRC back into
        the selection head and the PLM encoder.

    The previous implementation hard-coded `selected_indices = [0..4]` and
    used a detached numpy weight vector to "weight" them, so the PLM never
    influenced inference. Two different PLMs converged to identical test
    AUPRC because the LR head silently re-learned weights over the same 5
    fixed features. This version makes selection itself the learned object.
    """
    def __init__(self, plm_name: str, n_candidates: int = 20, n_select: int = 5,
                 freeze_plm: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.plm = AutoModel.from_pretrained(plm_name)
        hidden_size = self.plm.config.hidden_size
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        self.select_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, n_candidates),
        )
        self.n_candidates = n_candidates
        self.n_select = n_select

    def forward(self, input_ids, attention_mask):
        out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.select_head(cls)  # (B, n_candidates)


def make_stat_description(dataset: str, features: dict, rel_names: list, train_mask: np.ndarray, y: np.ndarray) -> str:
    """Create a textual description of graph statistics for PLM input."""
    lines = [f"Fraud detection graph with {len(rel_names)} relation types: {', '.join(rel_names)}."]
    lines.append(f"Total nodes: {len(y)}, fraud rate: {y[train_mask].mean():.3f} (train set).")
    for name in rel_names:
        f_key = f"fraud_nbr_{name}"
        d_key = f"log_deg_{name}"
        if f_key in features:
            vals = features[f_key]
            lines.append(f"Relation {name}: fraud_nbr_rate mean={vals.mean():.4f} std={vals.std():.4f} "
                         f"max={vals.max():.4f}.")
        if d_key in features:
            vals = features[d_key]
            lines.append(f"Relation {name}: log_degree mean={vals.mean():.4f} std={vals.std():.4f} "
                         f"max={vals.max():.4f}.")
    lines.append("Design 5 composite features using f1-f6 to detect fraud.")
    return " ".join(lines)


def generate_candidate_formulas():
    """Generate a fixed pool of candidate composite formulas for PLM to weight."""
    # Pool of 20 candidate formulas covering various patterns
    candidates = [
        "f1*f3",           # cross-relation fraud interaction
        "f2/(f5+0.1)",     # fraud/degree ratio
        "f4*log(f6+1)",    # degree × log degree
        "sqrt(f1+f2)",     # sqrt fraud aggregation
        "max(f3,f5)-f4",   # max of fraud/degree minus degree
        "f1+f2+f3",        # sum of all fraud rates
        "f1*f4",           # fraud × own degree
        "f2*f5",           # fraud × own degree (RSR)
        "f3*f6",           # fraud × own degree (RTR)
        "f1/(f4+0.1)",     # fraud per log-degree
        "f2/(f6+0.1)",     # RSR fraud per RTR degree
        "sqrt(f4*f5)",     # geometric mean of degrees
        "f1*f2*f3",        # triple fraud interaction
        "log(f4+f5+f6+1)", # log total degree
        "f1-f2",           # fraud rate contrast
        "f3-f1",           # temporal vs user fraud
        "max(f1,f2,f3)",   # max fraud rate
        "min(f1,f2,f3)",   # min fraud rate
        "(f1+f2+f3)/(f4+f5+f6+0.1)",  # avg fraud per avg degree
        "f1*sqrt(f4)+f2*sqrt(f5)",     # weighted fraud-degree
    ]
    return candidates


def phase4_plm_adapter(plm_name: str, model_info: dict, device: str, n_epochs: int = 80,
                        rl_seed: int = 0) -> dict:
    """Train PLM adapter via REINFORCE candidate selection.

    Pipeline (per epoch):
      1. PLM(text) → 20-d selection logits
      2. Gumbel-top-k sample → 5 distinct candidate indices (no replacement)
      3. LR(base_logit, selected 5 features) on train → val AUPRC = reward
      4. REINFORCE update: loss = −(reward − running_baseline) · Σ log_prob[selected]

    Final result is the val-best selection re-evaluated on the test set. Two
    different PLMs that converge to different selections now produce
    genuinely different test AUPRC (previously they collided because
    selection was hard-coded and the LR head silently re-fit over the same
    5 fixed features).
    """
    print(f"\n  Training PLM adapter (REINFORCE-select): {plm_name}")

    dataset, base, seed = "yelpchi", "bwgnn", 42
    mat = loadmat("datasets/YelpChi.mat")
    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())
    data, base_logit, base_z, delta_rel = load_cell(dataset, base, seed, device)
    N = data.x.shape[0]
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    adjs = {}
    for name in rel_names:
        key = schema[name]["mat_key"]
        sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
        adjs[name] = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))
    features, _ = compute_graph_stats(dataset, data, adjs)

    desc = make_stat_description(dataset, features, rel_names, train_mask, y)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_id"])
    enc = tokenizer(desc, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    candidate_formulas = generate_candidate_formulas()
    n_candidates = len(candidate_formulas)
    n_select = 5

    plm_adapter = PLMFeatureAdapter(
        model_info["model_id"],
        n_candidates=n_candidates,
        n_select=n_select,
        freeze_plm=False,
    ).to(device)
    optimizer = torch.optim.AdamW(plm_adapter.parameters(), lr=1e-4, weight_decay=0.01)

    X_base = base_logit.reshape(-1, 1)
    candidate_feats = build_composite_features(features, rel_names, candidate_formulas)
    # Shape: (N, n_candidates)

    # Deterministic RL seed so REINFORCE rollouts are reproducible across runs.
    rng = torch.Generator(device=device).manual_seed(int(rl_seed))

    best_val = 0.0
    best_indices = None
    best_test = {"auprc": 0.0, "roc_auc": 0.0}
    baseline = 0.5
    baseline_momentum = 0.9

    for epoch in range(n_epochs):
        plm_adapter.train()
        logits = plm_adapter(input_ids, attention_mask).squeeze(0)  # (n_candidates,)
        # Gumbel-top-k for sampling 5 distinct indices without replacement.
        gumbel = -torch.log(
            -torch.log(torch.rand(n_candidates, generator=rng, device=device) + 1e-20) + 1e-20
        )
        _, selected_idx = (logits + gumbel).topk(n_select)
        selected_np = selected_idx.detach().cpu().numpy()

        # Evaluate selection: train LR on (base_logit, 5 selected features), report val AUPRC.
        sel_feats = candidate_feats[:, selected_np]
        X_combo = np.column_stack([X_base, sel_feats])
        val_metrics = eval_lr(
            X_combo[train_mask], y[train_mask],
            X_combo[val_mask], y[val_mask], seed,
        )
        val_auprc = val_metrics["auprc"]

        # REINFORCE update using running-mean baseline (variance reduction).
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_sum = log_probs[selected_idx].sum()
        advantage = val_auprc - baseline
        loss = -advantage * log_prob_sum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * val_auprc

        if val_auprc > best_val:
            best_val = val_auprc
            best_indices = selected_np.tolist()
            best_test = eval_lr(
                X_combo[train_mask], y[train_mask],
                X_combo[test_mask], y[test_mask], seed,
            )

        if epoch % 10 == 0:
            print(
                f"    Epoch {epoch}: val_AUPRC={val_auprc:.4f}  "
                f"baseline={baseline:.4f}  adv={advantage:+.4f}  sel={selected_np.tolist()}"
            )

    best_formulas = [candidate_formulas[i] for i in best_indices] if best_indices else None
    print(
        f"  {plm_name} best: val_AUPRC={best_val:.4f}  test_AUPRC={best_test['auprc']:.4f}  "
        f"selected_indices={best_indices}"
    )

    return {
        "model": plm_name,
        "params": model_info["params"],
        "best_formulas": best_formulas,
        "best_indices": best_indices,
        "best_weights": None,  # kept for back-compat with prior schema
        "val_auprc": best_val,
        "test_auprc": best_test["auprc"],
        "test_roc_auc": best_test["roc_auc"],
    }


# ─── Phase 5: Aggregate & Plot ───────────────────────────────────────────────
def phase5_scaling_plot(quick_eval: dict, full_results: dict = None, plm_results: list = None):
    """Phase 5: Generate scaling plot and comparison table."""
    print("\n" + "=" * 60)
    print("PHASE 5: Scaling Plot & Comparison Table")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect data points: model_name -> (params, auprc)
    data_points = []
    model_formulas = {}

    # From quick eval
    for model_name, result in quick_eval.items():
        if model_name.startswith("__"):
            continue
        if "auprc" not in result or result.get("auprc", 0) == 0:
            continue
        params = LLM_MODELS.get(model_name, {}).get("params", 0)
        if params == 0:
            continue
        data_points.append({
            "name": model_name,
            "params": params,
            "auprc": result["auprc"],
            "type": "LLM",
            "formulas": result.get("formulas", []),
        })
        model_formulas[model_name] = result.get("formulas", [])

    # From PLM results
    if plm_results:
        for r in plm_results:
            data_points.append({
                "name": r["model"],
                "params": r["params"],
                "auprc": r["test_auprc"],
                "type": "PLM",
                "formulas": r.get("best_formulas", []),
            })

    if not data_points:
        print("  No data points to plot!")
        return

    # Sort by params
    data_points.sort(key=lambda x: x["params"])

    # ── Scaling plot ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    llm_pts = [d for d in data_points if d["type"] == "LLM"]
    plm_pts = [d for d in data_points if d["type"] == "PLM"]

    if llm_pts:
        x = [d["params"] / 1e9 for d in llm_pts]
        y = [d["auprc"] for d in llm_pts]
        names = [d["name"] for d in llm_pts]
        ax.scatter(x, y, s=120, c="royalblue", zorder=5, label="LLM (causal)")
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), textcoords="offset points",
                       xytext=(8, 5), fontsize=8)
        # Trend line
        if len(x) > 2:
            log_x = np.log10(np.array(x) * 1e9)
            z = np.polyfit(log_x, y, 1)
            x_line = np.logspace(np.log10(min(x) * 0.5), np.log10(max(x) * 2), 100)
            log_x_line = np.log10(x_line * 1e9)
            y_line = np.polyval(z, log_x_line)
            ax.plot(x_line, y_line, '--', color='royalblue', alpha=0.5, label='LLM trend (log-fit)')

    if plm_pts:
        x = [d["params"] / 1e9 for d in plm_pts]
        y = [d["auprc"] for d in plm_pts]
        names = [d["name"] for d in plm_pts]
        ax.scatter(x, y, s=120, c="coral", marker="s", zorder=5, label="PLM (encoder)")
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), textcoords="offset points",
                       xytext=(8, 5), fontsize=8)

    # Baselines
    base_auprc = quick_eval.get("__base_only", {}).get("auprc", 0)
    rel_auprc = quick_eval.get("__base_rel", {}).get("auprc", 0)
    if base_auprc > 0:
        ax.axhline(y=base_auprc, color='gray', linestyle=':', alpha=0.7, label=f'Base only ({base_auprc:.4f})')
    if rel_auprc > 0:
        ax.axhline(y=rel_auprc, color='green', linestyle=':', alpha=0.7, label=f'Base+REL ({rel_auprc:.4f})')

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (B)", fontsize=12)
    ax.set_ylabel("AUPRC (YelpChi-BWGNN, seed_42)", fontsize=12)
    ax.set_title("Idea 3: Scaling Law — LLM Feature Design Quality vs Model Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_dir = Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "idea3_scaling_plot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved scaling plot to {out_dir / 'idea3_scaling_plot.png'}")
    plt.close(fig)

    # ── Comparison table ──
    lines = []
    lines.append("# Idea 3 Multi-LLM Scaling Comparison\n")
    lines.append("**Task**: Design 5 composite fraud features from graph statistics.")
    lines.append("**Evaluation**: LR(base_logit + 5 features) on YelpChi-BWGNN seed_42.\n")
    lines.append("## Quick Screen Results\n")
    lines.append("| Model | Params | AUPRC | ROC-AUC | n_formulas |")
    lines.append("|---|---:|---:|---:|---:|")

    for d in data_points:
        params_str = f"{d['params']/1e9:.1f}B" if d['params'] >= 1e9 else f"{d['params']/1e6:.0f}M"
        lines.append(f"| {d['name']} | {params_str} | {d['auprc']:.4f} | — | {len(d.get('formulas', []))} |")

    # Add baselines
    for bk, bv in quick_eval.items():
        if bk.startswith("__") and "auprc" in bv:
            label = bk.replace("__", "").replace("_", "+")
            lines.append(f"| *{label}* | — | {bv['auprc']:.4f} | — | — |")

    lines.append("\n## Model-Designed Formulas\n")
    for d in data_points:
        if d.get("formulas"):
            lines.append(f"### {d['name']} ({d['params']/1e9:.1f}B)")
            for i, f in enumerate(d["formulas"], 1):
                lines.append(f"  {i}. `{f}`")
            lines.append("")

    # Scaling analysis
    lines.append("## Scaling Analysis\n")
    if len(llm_pts) > 2:
        log_params = np.log10([d["params"] for d in llm_pts])
        auprcs = [d["auprc"] for d in llm_pts]
        corr = np.corrcoef(log_params, auprcs)[0, 1]
        lines.append(f"- Pearson correlation (log10(params) vs AUPRC): r={corr:.3f}")
        lines.append(f"- Smallest ({llm_pts[0]['name']}, {llm_pts[0]['params']/1e9:.1f}B) AUPRC: {llm_pts[0]['auprc']:.4f}")
        lines.append(f"- Largest ({llm_pts[-1]['name']}, {llm_pts[-1]['params']/1e9:.1f}B) AUPRC: {llm_pts[-1]['auprc']:.4f}")
        lines.append(f"- Δ = {llm_pts[-1]['auprc'] - llm_pts[0]['auprc']:+.4f}")

    table_path = Path("artifacts/tables/idea3_multi_llm_comparison.md")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(f"  Saved comparison table to {table_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", help="1=generate, 2=evaluate, 3=full_bench, 4=plm, 5=plot, all=all")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    phases = ["1", "2", "3", "4", "5"] if args.phase == "all" else [args.phase]
    out_dir = Path("artifacts/results/idea3_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_results = None
    quick_eval = None

    # Phase 1: Generate
    if "1" in phases:
        llm_results = phase1_generate_all(args.device)
        # Save
        with open(out_dir / "llm_formulas.json", "w") as f:
            json.dump(llm_results, f, indent=2, default=str)

    # Phase 2: Quick evaluate
    if "2" in phases:
        if llm_results is None:
            with open(out_dir / "llm_formulas.json") as f:
                llm_results = json.load(f)
        quick_eval = phase2_evaluate_quick(llm_results, args.device)

    # Phase 3: Full benchmark (top-2)
    if "3" in phases:
        if llm_results is None:
            with open(out_dir / "llm_formulas.json") as f:
                llm_results = json.load(f)
        if quick_eval is None:
            with open(out_dir / "quick_eval_yelpchi_bwgnn_seed42.json") as f:
                quick_eval = json.load(f)
        # Select top-2 LLM models by AUPRC
        llm_only = {k: v for k, v in quick_eval.items() if not k.startswith("__") and v.get("auprc", 0) > 0}
        sorted_models = sorted(llm_only.items(), key=lambda x: x[1]["auprc"], reverse=True)
        top2 = [m[0] for m in sorted_models[:2]]
        print(f"\n  Top-2 models for full benchmark: {top2}")
        full_results = phase3_full_benchmark(top2, llm_results, args.device)
        with open(out_dir / "full_benchmark_top2.json", "w") as f:
            json.dump(full_results, f, indent=2, default=str)

    # Phase 4: PLM adapters
    if "4" in phases:
        plm_results = []
        for plm_name, plm_info in PLM_MODELS.items():
            try:
                # REINFORCE seed mirrors PLM ordering so BERT/RoBERTa rollouts
                # are reproducible but distinct.
                rl_seed = 42 + abs(hash(plm_name)) % 1000
                result = phase4_plm_adapter(
                    plm_name, plm_info, args.device, n_epochs=80, rl_seed=rl_seed
                )
                plm_results.append(result)
            except Exception as e:
                print(f"  ERROR {plm_name}: {e}")
                import traceback
                traceback.print_exc()
        with open(out_dir / "plm_adapter_results.json", "w") as f:
            json.dump(plm_results, f, indent=2, default=str)

    # Phase 5: Scaling plot
    if "5" in phases:
        if quick_eval is None:
            p = out_dir / "quick_eval_yelpchi_bwgnn_seed42.json"
            if p.exists():
                with open(p) as f:
                    quick_eval = json.load(f)
            else:
                print("  No quick eval results found. Run phase 2 first.")
                return

        plm_results = None
        p = out_dir / "plm_adapter_results.json"
        if p.exists():
            with open(p) as f:
                plm_results = json.load(f)

        full_results = None
        p = out_dir / "full_benchmark_top2.json"
        if p.exists():
            with open(p) as f:
                full_results = json.load(f)

        phase5_scaling_plot(quick_eval, full_results, plm_results)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
