"""Frozen schemas and validation for the analysis parquets."""

from __future__ import annotations

import polars as pl

SCHEMA: dict[str, pl.DataType] = {
    "experiment": pl.Utf8(),
    "dataset": pl.Utf8(),
    "dataset_variant": pl.Utf8(),
    "trainer": pl.Utf8(),
    "trainer_family": pl.Utf8(),
    "seed": pl.Int64(),
    "phase": pl.Utf8(),
    "metric": pl.Utf8(),
    "value": pl.Float64(),
    "train_slice": pl.Utf8(),
    "eval_slice": pl.Utf8(),
    "step": pl.Int64(),
    "epoch": pl.Int64(),
    "config_hash": pl.Utf8(),
    "snapshot_sha256": pl.Utf8(),
    "timestamp": pl.Utf8(),
}

STATS_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.Utf8(),
    "slice_kind": pl.Utf8(),
    "slice": pl.Int64(),
    "group_kind": pl.Utf8(),
    "group": pl.Utf8(),
    "count": pl.Int64(),
}

PARAMS_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.Utf8(),
    "trainer": pl.Utf8(),
    "trainer_family": pl.Utf8(),
    "trainable": pl.Int64(),
    "total": pl.Int64(),
}

COLUMNS: tuple[str, ...] = tuple(SCHEMA)


def check(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a results frame against the long-format schema."""
    return _validate(frame, SCHEMA, "results")


def check_stats(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a dataset-statistics frame against the stats schema."""
    return _validate(frame, STATS_SCHEMA, "stats")


def check_params(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a model-parameter frame against the params schema."""
    return _validate(frame, PARAMS_SCHEMA, "params")


def _validate(
    frame: pl.DataFrame, schema: dict[str, pl.DataType], name: str
) -> pl.DataFrame:
    missing = [column for column in schema if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} frame is missing columns: {missing}")
    selected = frame.select(list(schema))
    casts = []
    for column, dtype in schema.items():
        actual = selected.schema[column]
        if actual == dtype:
            continue
        # an all-null column infers as Null; cast it to the declared dtype
        if actual == pl.Null:
            casts.append(pl.col(column).cast(dtype))
        else:
            raise ValueError(f"column {column!r} has dtype {actual}, expected {dtype}")
    return selected.with_columns(casts) if casts else selected
