"""Compatibility entry point for the old G-OPD-Flash script.

Canonical script: :mod:`scripts.train_cbr_flash`.
"""

from __future__ import annotations

from scripts.train_cbr_flash import *  # noqa: F401,F403
from scripts.train_cbr_flash import main


if __name__ == "__main__":
    main()
