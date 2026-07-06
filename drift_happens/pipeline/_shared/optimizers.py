"""
Single source of truth for building pipeline optimizer factories.

Each trainer builder turns a configured optimizer name, learning rate, and weight decay
into the ``optimizer_factory`` callable that ``PytorchTrainer`` invokes. The
hyperparameters are bound here as arguments so that every factory stays independent of
the builder's loop variable; a closure over the loop ``config`` would instead share the
last iteration's values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

OptimizerName = Literal["adam", "adamw"]
_OptimizerClass = type[torch.optim.Adam] | type[torch.optim.AdamW]

_OPTIMIZER_CLASSES: dict[OptimizerName, _OptimizerClass] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def make_optimizer_factory(
    optimizer: OptimizerName,
    *,
    learning_rate: float,
    weight_decay: float,
) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """Bind the optimizer name and hyperparameters into a fresh factory."""
    optimizer_cls = _OPTIMIZER_CLASSES[optimizer]

    def optimizer_factory(module: torch.nn.Module) -> torch.optim.Optimizer:
        return optimizer_cls(
            module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return optimizer_factory
