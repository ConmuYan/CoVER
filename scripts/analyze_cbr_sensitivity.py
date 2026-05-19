"""K2: CBR sensitivity mechanism analysis (MF-3 reproduction script).

Replaces the inline diagnostic that produced the K2 figure in commit
`ac86df2`. Extends analysis to all 8 cells (YelpChi + Amazon × {bwgnn,
sage, gcn, gat}) — addressing Opus critic round-9 MF-3 (Amazon mechanism
untested in original K2).

For each cell:
1. Loads frozen base + frozen LREE-RAER teacher + trained students
   (det_mask AND det_mask_cbr) at seed 42.
2. Computes per-train-node:
   - sensitivity_i = |teacher_logit_i - base_logit_i| / δ_max ∈ [0, 1]
   - student |δ| under det_mask vs det_mask_cbr
   - waste_i = student|δ| · (1 - sensitivity_i)   ← CBR penalises this
   - useful_i = student|δ| · sensitivity_i        ← CBR does NOT touch
3. Reports waste reduction (%) and useful reduction (%) per cell.
4. Emits 2 figures:
   - `yelpchi_sensitivity_distributions.png` — 4-cell sensitivity histograms
   - `amazon_sensitivity_distributions.png`  — 4-cell sensitivity histograms
5. Writes summary JSON for downstream paper-table use.

Usage::

    PYTHONPATH=. python scripts/analyze_cbr_sensitivity.py

The script is deterministic given the listed seeds + frozen checkpoints.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import RELATION_SCHEMAS
from models.cover_rel_reasoner import CoVERRelReasoner
from models.flash_adapter import FlashAdapter
from scripts.train_distill_adapter import build_learned_teacher_features, load_frozen_base
from scripts.train_g_opd_flash import generate_teacher_cache_with_heads


def _config_for(ds: str, base: str) -> Path:
    if ds == "yelpchi" and base == "bwgnn":
        return Path("configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml")
    return Path(f"configs/phase2_reasoner/ablation/idea1_{ds}_{base}_canonical_clsonly.yaml")


@torch.no_grad()
def diagnose_cell(
    ds: str,
    base: str,
    seed: int,
    device: torch.device,
    delta_max: float = 2.0,
) -> dict:
    cfg_path = _config_for(ds, base)
    if not cfg_path.exists():
        return {"skip": True, "reason": f"missing config {cfg_path}"}
    cfg = yaml.safe_load(open(cfg_path))
    data = load_fraud_dataset(
        name=ds,
        path=cfg["dataset"].get("path"),
        format=cfg["dataset"].get("format"),
        seed=seed,
        scarcity_ratio=1.0,
        split_mode="supervised",
        train_ratio=0.4,
        val_test_ratio=[1, 2],
        stratified=True,
    )
    bl, bz, _ = load_frozen_base(
        cfg, ds, base, seed, data, device,
        ckpt_override=f"artifacts/checkpoints/{ds}/{base}/fixed_v1_100ep/seed_{seed}/base.pt",
    )
    rels = list(RELATION_SCHEMAS[ds].keys())
    extractor_path = Path(
        f"artifacts/checkpoints/{ds}/{base}/idea2b_learned_extractor/seed_{seed}/evidence_extractor.pt"
    )
    if not extractor_path.exists():
        return {"skip": True, "reason": f"missing extractor {extractor_path}"}
    rf = build_learned_teacher_features(
        config=cfg, data=data, relation_names=rels,
        extractor_ckpt_path=extractor_path, device=device,
    )
    teacher = CoVERRelReasoner(
        base_z_dim=int(bz.shape[1]),
        relation_names=rels,
        anchor_relation=rels[0],
        rel_stat_dim=9,
        rel_hidden_dim=64,
    ).to(device)
    raw = torch.load(
        f"artifacts/checkpoints/{ds}/{base}/idea2b_learned_extractor/seed_{seed}/reasoner.pt",
        map_location=device, weights_only=False,
    )
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    teacher.load_state_dict(state)
    teacher.eval()
    cache = generate_teacher_cache_with_heads(teacher, bz, bl, rf, device)
    # sensitivity = |teacher_logit - base_logit| / δ_max
    sensitivity = ((cache["logit"] - bl.view(-1)).abs() / delta_max).clamp(0, 1)
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
    sens_train = sensitivity[train_idx].cpu().numpy()
    out = {
        "ds": ds, "base": base, "seed": seed,
        "n_train": int(len(sens_train)),
        "teacher_sens_mean": float(sens_train.mean()),
        "teacher_sens_median": float(np.median(sens_train)),
        "teacher_sens_high_frac": float((sens_train > 0.5).sum() / len(sens_train)),
        "teacher_sens_low_frac": float((sens_train < 0.1).sum() / len(sens_train)),
    }
    out_students: dict[str, dict] = {}
    for mode in ("g_opd_flash_det_mask", "g_opd_flash_det_mask_cbr"):
        st_path = Path(f"artifacts/checkpoints/{ds}/{base}/{mode}/seed_{seed}/student.pt")
        if not st_path.exists():
            continue
        st = FlashAdapter(
            base_z_dim=int(bz.shape[1]),
            num_relations=len(rels),
            hidden_dim=32,
            delta_max=delta_max,
        ).to(device)
        st.load_state_dict(torch.load(st_path, map_location=device, weights_only=True))
        st.eval()
        s_out = st(bz, bl, rf, return_heads=False)
        s_delta = (s_out["final_logit"] - bl.view(-1)).abs() / delta_max
        s_delta_train = s_delta[train_idx].cpu().numpy()
        waste = s_delta_train * (1.0 - sens_train)
        useful = s_delta_train * sens_train
        out_students[mode] = {
            "s_delta": s_delta_train,  # kept for plotting
            "s_delta_mean": float(s_delta_train.mean()),
            "waste_mean": float(waste.mean()),
            "useful_mean": float(useful.mean()),
        }
    if "g_opd_flash_det_mask" in out_students and "g_opd_flash_det_mask_cbr" in out_students:
        det = out_students["g_opd_flash_det_mask"]
        cbr = out_students["g_opd_flash_det_mask_cbr"]
        out["waste_reduction_pct"] = (
            (det["waste_mean"] - cbr["waste_mean"]) / max(det["waste_mean"], 1e-9) * 100
        )
        out["useful_reduction_pct"] = (
            (det["useful_mean"] - cbr["useful_mean"]) / max(det["useful_mean"], 1e-9) * 100
        )
        out["delta_reduction_pct"] = (
            (det["s_delta_mean"] - cbr["s_delta_mean"]) / max(det["s_delta_mean"], 1e-9) * 100
        )
    out["_sens_train"] = sens_train
    out["_students"] = out_students
    return out


def plot_dataset(diag_rows: list[dict], dataset: str, out_path: Path) -> None:
    bases = ["bwgnn", "sage", "gcn", "gat"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, base in enumerate(bases):
        row = next((r for r in diag_rows if r.get("base") == base and not r.get("skip")), None)
        if row is None:
            for ax in (axes[0, i], axes[1, i]):
                ax.set_title(f"{dataset}-{base.upper()}: SKIPPED")
                ax.axis("off")
            continue
        sens = row["_sens_train"]
        students = row["_students"]
        # Top row: teacher sensitivity distribution
        ax = axes[0, i]
        ax.hist(sens, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_title(f"{dataset}-{base.upper()}: teacher |Δ^T|/δ_max")
        ax.set_xlabel("sensitivity")
        ax.set_ylabel("count")
        ax.axvline(0.5, color="red", ls="--", alpha=0.5, label="high-sens threshold")
        ax.legend(fontsize=8)
        # Bottom row: student |δ| distribution det_mask vs det_mask_cbr
        ax = axes[1, i]
        if "g_opd_flash_det_mask" in students:
            ax.hist(students["g_opd_flash_det_mask"]["s_delta"], bins=50,
                    color="orange", alpha=0.5, label="det_mask")
        if "g_opd_flash_det_mask_cbr" in students:
            ax.hist(students["g_opd_flash_det_mask_cbr"]["s_delta"], bins=50,
                    color="green", alpha=0.5, label="det_mask_cbr")
        ax.set_title(f"{dataset}-{base.upper()}: student |Δ^S|/δ_max")
        ax.set_xlabel("residual magnitude")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fig_root = Path("artifacts/figures/cbr_sensitivity")
    summary: dict = {}
    for ds in ("yelpchi", "amazon"):
        rows: list[dict] = []
        for base in ("bwgnn", "sage", "gcn", "gat"):
            row = diagnose_cell(ds, base, seed=42, device=device)
            rows.append(row)
            if row.get("skip"):
                print(f"[K2] {ds}/{base}: SKIP — {row['reason']}")
                continue
            key = f"{ds}/{base}"
            summary[key] = {k: v for k, v in row.items() if not k.startswith("_")}
            # Strip _students from JSON, keep summary
            students = row["_students"]
            for mode, st in students.items():
                if "s_delta" in st:
                    st_copy = {k: v for k, v in st.items() if k != "s_delta"}
                    summary[key].setdefault("students", {})[mode] = st_copy
            summary[key].pop("_students", None)
            summary[key].pop("_sens_train", None)
            print(
                f"[K2] {ds}/{base}: sens median={row['teacher_sens_median']:.3f}, "
                f"high-sens frac={row['teacher_sens_high_frac']:.3f}, "
                f"waste ↓ {row.get('waste_reduction_pct', float('nan')):+.1f}%, "
                f"useful ↓ {row.get('useful_reduction_pct', float('nan')):+.1f}%"
            )
        plot_dataset(rows, ds, fig_root / f"{ds}_sensitivity_distributions.png")
        print(f"[K2] Figure written → {fig_root / f'{ds}_sensitivity_distributions.png'}")

    out_json = Path("artifacts/figures/cbr_sensitivity/cbr_sensitivity_summary.json")
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[K2] Summary JSON → {out_json}")


if __name__ == "__main__":
    main()
