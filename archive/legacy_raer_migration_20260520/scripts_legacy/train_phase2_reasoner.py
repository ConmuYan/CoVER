"""Compatibility entry point for the old phase-2 teacher script.

Canonical script: :mod:`scripts.train_raer_teacher`.
"""

from __future__ import annotations

from scripts.train_raer_teacher import *  # noqa: F401,F403
from scripts.train_raer_teacher import main


if __name__ == "__main__":
    main()
