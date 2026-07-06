"""Read frozen results and dataset statistics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets import schema
from drift_happens.analysis.datasets.locations import (
    DATASET_STATS_PARQUET,
    PARAMS_PARQUET,
    RESULTS_PARQUET,
)


def results() -> pl.DataFrame:
    """Frozen results table."""
    return _load(RESULTS_PARQUET, schema.check)


def dataset_stats() -> pl.DataFrame:
    """Frozen dataset-statistics table."""
    return _load(DATASET_STATS_PARQUET, schema.check_stats)


def params() -> pl.DataFrame:
    """Frozen model-parameter table."""
    return _load(PARAMS_PARQUET, schema.check_params)


def _load(path: Path, validate: Callable[[pl.DataFrame], pl.DataFrame]) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `drift analysis export` first")
    return validate(pl.read_parquet(path))
