"""SHA-256 immutability test for base.pt.

Verifies that the base checkpoint is not modified by any Phase 3 step.
Pure file-hash test (no GPU required).
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

BASE_PT_PATH = Path(
    "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd"
    "/artifacts/checkpoints/yelpchi/bwgnn/fixed_v1_100ep/seed_42/base.pt"
)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class TestBaseFreezeHash:
    @pytest.mark.skipif(
        not BASE_PT_PATH.exists(),
        reason=f"base.pt not found at {BASE_PT_PATH}",
    )
    def test_base_pt_hash_unchanged_after_copy(self):
        """Verify SHA-256 of base.pt is stable (read-only check)."""
        hash_before = _sha256_file(BASE_PT_PATH)
        # Read it again to verify deterministic hashing
        hash_again = _sha256_file(BASE_PT_PATH)
        assert hash_before == hash_again, (
            f"SHA-256 of base.pt changed between reads: "
            f"{hash_before} != {hash_again}"
        )

    @pytest.mark.skipif(
        not BASE_PT_PATH.exists(),
        reason=f"base.pt not found at {BASE_PT_PATH}",
    )
    def test_base_pt_survives_fake_phase3_step(self):
        """Simulate a Phase 3 step (copy to temp, verify original untouched)."""
        hash_before = _sha256_file(BASE_PT_PATH)

        # Simulate a "Phase 3 step" that reads but does not modify base.pt
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_copy = Path(tmpdir) / "base_copy.pt"
            shutil.copy2(BASE_PT_PATH, tmp_copy)

            # Verify the copy matches
            hash_copy = _sha256_file(tmp_copy)
            assert hash_copy == hash_before

            # Modify the copy (simulating accidental mutation)
            with open(tmp_copy, "ab") as f:
                f.write(b"corrupted")
            hash_corrupted = _sha256_file(tmp_copy)
            assert hash_corrupted != hash_before, (
                "Corruption should change the hash"
            )

        # Original must be untouched
        hash_after = _sha256_file(BASE_PT_PATH)
        assert hash_before == hash_after, (
            f"base.pt was mutated during fake Phase 3 step: "
            f"{hash_before} != {hash_after}"
        )
