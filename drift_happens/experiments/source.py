"""Load experiment configs from plain files or materialized snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drift_happens.configs import ExperimentConfig, apply_overrides, load_config_data
from drift_happens.utils.snapshot import SNAPSHOT_KIND
from drift_happens.utils.wandb_identity import source_wandb_identity


@dataclass(frozen=True, slots=True)
class ExperimentSource:
    """Resolved one-run config plus source-envelope metadata."""

    config: ExperimentConfig
    path: Path
    seeds: tuple[int, ...]
    source_identity: str | None
    is_preset_snapshot: bool


def load_experiment_source(
    path: Path,
    *,
    overrides: tuple[str, ...] = (),
) -> ExperimentSource:
    """Load a config file or materialized preset snapshot."""
    resolved = Path(path).resolve()
    data = load_config_data(resolved)
    if _is_preset_snapshot(data):
        config_data = data.get("config")
        if not isinstance(config_data, dict):
            raise ValueError(f"snapshot {resolved} has a non-object config")
        if overrides:
            config_data = apply_overrides(config_data, list(overrides))
        return ExperimentSource(
            config=ExperimentConfig.model_validate(config_data),
            path=resolved,
            seeds=_snapshot_seeds(data, resolved),
            source_identity=source_wandb_identity(resolved),
            is_preset_snapshot=True,
        )
    if overrides:
        data = apply_overrides(data, list(overrides))
    cfg = ExperimentConfig.model_validate(data)
    return ExperimentSource(
        config=cfg,
        path=resolved,
        seeds=(cfg.seed,),
        source_identity=source_wandb_identity(resolved),
        is_preset_snapshot=False,
    )


def _is_preset_snapshot(data: dict[str, Any]) -> bool:
    return data.get("kind") == SNAPSHOT_KIND


def _snapshot_seeds(data: dict[str, Any], resolved: Path) -> tuple[int, ...]:
    # A materialized snapshot always carries a non-empty integer seeds list.
    raw = data.get("seeds")
    if not isinstance(raw, list | tuple):
        raise ValueError(f"snapshot {resolved} has missing or non-list seeds")
    seeds: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"snapshot {resolved} has a non-integer seed: {item!r}")
        seeds.append(item)
    if not seeds:
        raise ValueError(f"snapshot {resolved} has an empty seeds list")
    return tuple(seeds)
