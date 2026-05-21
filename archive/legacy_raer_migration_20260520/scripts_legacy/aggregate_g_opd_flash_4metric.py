"""Compatibility entry point for the old CBR-Flash 4-metric aggregator name.

Canonical script: :mod:`scripts.aggregate_cbr_flash_4metric`.
"""

from __future__ import annotations

from scripts.aggregate_cbr_flash_4metric import *  # noqa: F401,F403
from scripts.aggregate_cbr_flash_4metric import main


if __name__ == "__main__":
    main()
