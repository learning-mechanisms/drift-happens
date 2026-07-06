"""Typed entries for Python-defined experiment presets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from drift_happens.configs import ExperimentConfig

ComparisonRole = Literal["smoke", "headline"]


@dataclass(frozen=True, slots=True)
class PresetEntry:
    """
    One materializable experiment preset.

    ``factory`` returns one validated ``ExperimentConfig`` with a scalar seed. ``seeds``
    stays on the preset envelope so sweep expansion can launch one process per seed
    without changing the root run contract.
    """

    group: str
    name: str
    factory: Callable[[], ExperimentConfig]
    seeds: tuple[int, ...]
    description: str = ""
    tags: tuple[str, ...] = ()
    comparison_group: str | None = None
    comparison_role: ComparisonRole = "headline"
    variant_fields: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.group, self.name

    def build(self) -> ExperimentConfig:
        return self.factory()
