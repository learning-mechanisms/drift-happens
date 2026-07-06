"""Build the analysis results parquet from drift-matrix files and write the run lock."""

from __future__ import annotations

import polars as pl

from drift_happens.analysis.datasets.locations import RUNS_LOCK
from drift_happens.analysis.export.from_matrices import build_results_from_matrices
from drift_happens.analysis.export.runs import Lock, lock, write_lock


def export() -> Lock:
    output = build_results_from_matrices()
    locked = lock(pl.read_parquet(output))
    write_lock(locked, RUNS_LOCK)
    return locked


if __name__ == "__main__":
    result = export()
    print(f"froze {len(result['runs'])} runs, {len(result['missing'])} missing")
