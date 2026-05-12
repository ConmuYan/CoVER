from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class BaseModelOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.logits.ndim > 1:
            self.logits = self.logits.view(-1)
