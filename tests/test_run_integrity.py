import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_integrity_check_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parent.parent)

    import subprocess
    err_dir = Path("artifacts") / "err_cache" / "yelpchi" / "bwgnn" / "rule_safe" / "seed_123"
    if not err_dir.exists():
        pytest.skip("No rule_safe artifacts (fresh restart)")

    result = subprocess.run(
        ["python", "scripts/check_run_integrity.py", "--config", "configs/yelpchi_bwgnn.yaml", "--run_name", "rule_safe", "--seed", "123"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Integrity check failed: {result.stderr}"


def test_compare_stage1_stage3(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parent.parent)

    err_dir = Path("artifacts") / "err_cache" / "yelpchi" / "bwgnn" / "rule_safe" / "seed_123"
    if not err_dir.exists():
        pytest.skip("No rule_safe artifacts (fresh restart)")

    import subprocess
    result = subprocess.run(
        ["python", "scripts/compare_stage1_stage3.py", "--config", "configs/yelpchi_bwgnn.yaml", "--run_name", "rule_safe", "--seed", "123"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Compare failed: {result.stderr}"
    assert "Stage 1 vs Stage 3" in result.stdout


def test_score_leakage_detection():
    from scripts.check_run_integrity import check_score_leakage
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"reasoning": {"base_score": 0.9}}) + "\n")
        f.flush()
        ok, msg = check_score_leakage(Path(f.name))
        assert not ok
        assert "SCORE LEAKAGE" in msg

    Path(f.name).unlink()


def test_accepted_plus_rejected_check():
    from scripts.check_run_integrity import check_accepted_plus_rejected

    err_cache_dir = Path("artifacts") / "err_cache" / "yelpchi" / "gcn" / "seed_0"
    if err_cache_dir.exists():
        ok, msg = check_accepted_plus_rejected(err_cache_dir)
        assert ok
