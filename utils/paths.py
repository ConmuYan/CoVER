"""Artifact path management for CoVER-FD experiments."""

from __future__ import annotations

from pathlib import Path


ARTIFACTS_ROOT = Path("artifacts")


def _stratified_dir_name(stratified: bool) -> str:
    return "stratified_true" if stratified else "stratified_false"


def get_checkpoint_dir(dataset: str, model: str, run_name: str = "base", seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "checkpoints" / dataset / model / run_name / f"seed_{seed}"


def get_err_cache_dir(dataset: str, model: str, run_name: str = "rule", seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "err_cache" / dataset / model / run_name / f"seed_{seed}"


def get_logs_dir(dataset: str, model: str, run_name: str = "base", seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "logs" / dataset / model / run_name / f"seed_{seed}"


def get_results_dir(dataset: str, model: str, run_name: str = "base", seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "results" / dataset / model / run_name / f"seed_{seed}"


def get_reports_dir(dataset: str, model: str, seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "reports" / dataset / model / f"seed_{seed}"


def get_split_dir(dataset: str, seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "splits" / dataset / f"seed_{seed}"


def get_base_checkpoint_path(dataset: str, model: str, seed: int = 0) -> Path:
    return get_checkpoint_dir(dataset, model, "base", seed) / "base.pt"


def get_reasoner_checkpoint_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_checkpoint_dir(dataset, model, run_name, seed) / "reasoner.pt"


def get_stage1_metrics_path(dataset: str, model: str, seed: int = 0) -> Path:
    return get_results_dir(dataset, model, "base", seed) / "stage1_metrics.json"


def get_stage3_metrics_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_results_dir(dataset, model, run_name, seed) / "stage3_metrics.json"


def get_evidence_cards_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_err_cache_dir(dataset, model, run_name, seed) / "evidence_cards.jsonl"


def get_accepted_err_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_err_cache_dir(dataset, model, run_name, seed) / "accepted_err.jsonl"


def get_rejected_err_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_err_cache_dir(dataset, model, run_name, seed) / "rejected_err.jsonl"


def get_verifier_stats_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_err_cache_dir(dataset, model, run_name, seed) / "verifier_stats.json"


def get_stage2_stats_path(dataset: str, model: str, run_name: str, seed: int = 0) -> Path:
    return get_err_cache_dir(dataset, model, run_name, seed) / "stage2_stats.json"


def get_split_path(dataset: str, seed: int = 0) -> Path:
    return get_stratified_split_path(dataset, False, seed)


def get_split_meta_path(dataset: str, seed: int = 0) -> Path:
    return get_stratified_split_meta_path(dataset, False, seed)


def teacher_to_run_name(teacher: str) -> str:
    if teacher == "llm":
        return "qwen"
    return teacher


def get_stratified_split_dir(dataset: str, stratified: bool, seed: int = 0) -> Path:
    """Return split directory that distinguishes stratified vs non-stratified."""
    return ARTIFACTS_ROOT / "splits" / dataset / _stratified_dir_name(stratified) / f"seed_{seed}"


def get_stratified_split_path(dataset: str, stratified: bool, seed: int = 0) -> Path:
    return get_stratified_split_dir(dataset, stratified, seed) / "split.pt"


def get_stratified_split_meta_path(dataset: str, stratified: bool, seed: int = 0) -> Path:
    return get_stratified_split_dir(dataset, stratified, seed) / "split_meta.json"


def get_reports_dir_stratified(dataset: str, model: str, stratified: bool, seed: int = 0) -> Path:
    return ARTIFACTS_ROOT / "reports" / dataset / model / _stratified_dir_name(stratified) / f"seed_{seed}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
