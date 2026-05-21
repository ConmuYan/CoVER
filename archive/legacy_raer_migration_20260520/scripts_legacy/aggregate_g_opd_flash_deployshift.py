"""Compatibility entry point for the old CBR-Flash deploy-shift aggregator.

Canonical script: :mod:`scripts.aggregate_cbr_flash_deployshift`.
"""

from __future__ import annotations

from scripts.aggregate_cbr_flash_deployshift import *  # noqa: F401,F403
from scripts.aggregate_cbr_flash_deployshift import main


if __name__ == "__main__":
    main()
