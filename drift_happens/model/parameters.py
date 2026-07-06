"""Parameter counting helpers for benchmark audits."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ParameterCounts:
    """Trainable/frozen/total parameter counts for a module."""

    trainable: int
    frozen: int

    @property
    def total(self) -> int:
        return self.trainable + self.frozen

    def as_dict(self) -> dict[str, int]:
        return {
            "frozen_parameters": self.frozen,
            "total_parameters": self.total,
            "trainable_parameters": self.trainable,
        }


def count_parameters(module: nn.Module) -> ParameterCounts:
    """Count trainable and frozen parameters separately."""
    trainable = 0
    frozen = 0
    for parameter in module.parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
    return ParameterCounts(trainable=trainable, frozen=frozen)
