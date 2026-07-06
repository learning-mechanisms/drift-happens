"""
Build the frozen results table from the eval drift-matrix files.

Each eval run writes ``results/drift_matrix.json``, the ``{train_slice: {eval_slice:
{metric: value}}}`` matrix; reshape those into the long-format results parquet the site
and figures read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets import DATASETS, schema
from drift_happens.analysis.datasets.locations import RESULTS_PARQUET
from drift_happens.analysis.plots.names import trainer_family
from drift_happens.utils.paths import RUNS_DIR

# run dir name: "<prefix>-conference__<trainer>__seed=<n>__eval"
_RUN = re.compile(
    r"^(?P<prefix>.+?)-conference__(?P<trainer>.+?)__seed=(?P<seed>\d+)__eval$"
)
_DATASET_OF_PREFIX = {
    "arxiv": "arxiv",
    "amazon-reviews-23": "amazon_reviews_23",
    "yearbook": "yearbook",
}

_EXCLUDED_TRAINERS: set[tuple[str, str]] = {("amazon_reviews_23", "minilm_l6_frozen")}


def _rows_from_run(run_dir: Path) -> list[dict[str, object]]:
    match = _RUN.match(run_dir.name)
    if match is None:
        return []
    dataset = _DATASET_OF_PREFIX.get(match["prefix"])
    if dataset not in DATASETS:
        return []
    matrix_path = run_dir / "results" / "drift_matrix.json"
    if not matrix_path.exists():
        return []
    metric = DATASETS[dataset].metric
    trainer = match["trainer"]
    if (dataset, trainer) in _EXCLUDED_TRAINERS:
        return []
    seed = int(match["seed"])
    family = trainer_family(dataset, trainer)
    try:
        matrix = json.loads(matrix_path.read_text())
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, object]] = []
    for train_slice, evals in matrix.items():
        for eval_slice, metrics in evals.items():
            value = metrics.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "experiment": None,
                    "dataset": dataset,
                    "dataset_variant": None,
                    "trainer": trainer,
                    "trainer_family": family,
                    "seed": seed,
                    "phase": "eval",
                    "metric": metric,
                    "value": float(value),
                    "train_slice": str(train_slice),
                    "eval_slice": str(eval_slice),
                    "step": None,
                    "epoch": None,
                    "config_hash": None,
                    "snapshot_sha256": None,
                    "timestamp": None,
                }
            )
    return rows


def build_results_from_matrices(
    runs_root: Path = RUNS_DIR, output: Path = RESULTS_PARQUET
) -> Path:
    """Reshape every run's drift-matrix file into the frozen results parquet."""
    rows: list[dict[str, object]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir():
            rows.extend(_rows_from_run(run_dir))
    if not rows:
        raise FileNotFoundError(f"no drift_matrix.json files under {runs_root}")
    frame = schema.check(pl.DataFrame(rows)).sort(schema.COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    return output
