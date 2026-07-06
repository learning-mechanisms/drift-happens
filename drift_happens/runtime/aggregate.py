"""
Aggregate per-run JSONL metric ledgers into one long-format results table.

Each run streams scalar metrics to ``<run_dir>/metrics/<phase>.jsonl`` (the canonical,
crash-safe, resume-friendly store). For analysis and publication those ledgers are
compacted into a single long-format Parquet so plots and queries read one columnar file
instead of walking thousands of per-cell artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from drift_happens.utils import paths
from drift_happens.utils.log import get_logger

logger = get_logger()

RESULTS_SCHEMA: dict[str, pl.DataType] = {
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


def _is_optional_int(value: object) -> bool:
    """Whether ``value`` fits a nullable Int64 column (bool is not an int here)."""
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _iter_ledger_rows(runs_root: Path) -> Iterator[dict[str, object]]:
    for ledger in sorted(runs_root.glob("**/metrics/*.jsonl")):
        with ledger.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping unparsable line {line_number} in {ledger}"
                    )
                    continue
                if not isinstance(row, dict):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: not a JSON object"
                    )
                    continue
                identity = row.get("run_identity") or {}
                if not isinstance(identity, dict):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: "
                        "run_identity is not an object"
                    )
                    continue
                seed = row.get("seed")
                if not _is_optional_int(seed):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: "
                        f"seed {seed!r} is not an integer"
                    )
                    continue
                value = row.get("value")
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int | float)
                ):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: "
                        f"value {value!r} is not numeric"
                    )
                    continue
                step = row.get("step")
                if not _is_optional_int(step):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: "
                        f"step {step!r} is not an integer"
                    )
                    continue
                epoch = row.get("epoch")
                if not _is_optional_int(epoch):
                    logger.warning(
                        f"Skipping line {line_number} in {ledger}: "
                        f"epoch {epoch!r} is not an integer"
                    )
                    continue
                out: dict[str, object] = {k: row.get(k) for k in RESULTS_SCHEMA}
                out["config_hash"] = identity.get("config_hash")
                out["snapshot_sha256"] = identity.get("snapshot_sha256")
                # validated fields kept from earlier checks
                out["seed"] = seed
                out["value"] = value
                out["step"] = step
                out["epoch"] = epoch
                yield out


def aggregate_metric_ledgers(runs_root: Path) -> pl.DataFrame:
    """
    Read every ``metrics/*.jsonl`` ledger under ``runs_root`` into one frame.

    Args:
        runs_root: Directory holding run directories (recursively searched).

    Returns:
        A long-format frame with one row per scalar metric observation. Empty (but
        correctly typed) when no ledgers are present.
    """
    return pl.DataFrame(list(_iter_ledger_rows(runs_root)), schema=RESULTS_SCHEMA)


def write_results_parquet(
    runs_root: Path | None = None, output: Path | None = None
) -> Path:
    """
    Compact all run ledgers under ``runs_root`` into a single Parquet file.

    Args:
        runs_root: Runs directory to scan. Defaults to ``paths.RUNS_DIR``.
        output: Destination Parquet path. Defaults to ``runs_root/results.parquet``.

    Returns:
        The path the Parquet file was written to.
    """
    runs_root = runs_root or paths.RUNS_DIR
    output = output or (runs_root / "results.parquet")
    frame = aggregate_metric_ledgers(runs_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    return output
