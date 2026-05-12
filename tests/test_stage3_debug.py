import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stage3_debug_creates_artifacts(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(Path(__file__).parent.parent)

    import subprocess
    result = subprocess.run(
        ["python", "scripts/train_stage3.py", "--config", "configs/yelpchi_gcn.yaml", "--debug"],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    reasoner_path = Path("artifacts/checkpoints/yelpchi/gcn/seed_0/reasoner.pt")
    assert reasoner_path.exists(), "reasoner.pt not created"

    metrics_path = Path("artifacts/logs/yelpchi/gcn/seed_0/stage3.json")
    assert metrics_path.exists(), "stage3.json not created"
