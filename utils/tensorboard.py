"""TensorBoard logging utilities for CoVER-FD training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TensorBoardLogger:
    """TensorBoard logger for training metrics."""

    def __init__(
        self,
        log_dir: str | Path,
        enabled: bool = True,
    ):
        self.enabled = enabled and HAS_TENSORBOARD
        self.writer = None

        if self.enabled:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = ""):
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def close(self):
        if self.writer:
            self.writer.close()


def create_logger(
    dataset_name: str,
    model_name: str,
    seed: int,
    stage: str,
    enabled: bool = True,
) -> TensorBoardLogger:
    log_dir = Path("artifacts") / "tensorboard" / dataset_name / model_name / f"seed_{seed}" / stage
    return TensorBoardLogger(log_dir, enabled=enabled)
