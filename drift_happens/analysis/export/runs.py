"""Pin the runs feeding the frozen results and record the missing cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import polars as pl

from drift_happens.analysis.datasets import DATASETS
from drift_happens.utils.paths import CONFIGS_DIR


class RunEntry(TypedDict):
    dataset: str
    trainer: str
    seed: int
    config_hash: str | None
    snapshot_sha256: str | None


class MissingCell(TypedDict):
    dataset: str
    trainer: str
    seed: int


class Lock(TypedDict):
    datasets: list[str]
    runs: list[RunEntry]
    missing: list[MissingCell]


_PRESET_INDEX = CONFIGS_DIR / "snapshots" / "presets" / "index.json"
_GROUP_TO_DATASET = {
    "yearbook-conference": "yearbook",
    "arxiv-conference": "arxiv",
    "amazon-reviews-23-conference": "amazon_reviews_23",
}


def expected_cells(index_path: Path = _PRESET_INDEX) -> list[tuple[str, str, int]]:
    presets = json.loads(index_path.read_text())["presets"]
    cells: list[tuple[str, str, int]] = []
    for preset in presets:
        dataset = _GROUP_TO_DATASET.get(preset["group"])
        if dataset is None:
            continue
        cells.extend((dataset, preset["name"], seed) for seed in preset["seeds"])
    return sorted(cells)


def expected_trainers(index_path: Path = _PRESET_INDEX) -> dict[str, list[str]]:
    """Planned conference trainers per dataset, from the preset index."""
    trainers: dict[str, set[str]] = {}
    for dataset, trainer, _ in expected_cells(index_path):
        trainers.setdefault(dataset, set()).add(trainer)
    return {dataset: sorted(names) for dataset, names in trainers.items()}


def lock(frame: pl.DataFrame, index_path: Path = _PRESET_INDEX) -> Lock:
    present = (
        frame.filter(pl.col("dataset").is_in(list(DATASETS)))
        .sort("timestamp")
        .group_by("dataset", "trainer", "seed")
        .agg(
            pl.col("config_hash").last(),
            pl.col("snapshot_sha256").last(),
        )
        .sort("dataset", "trainer", "seed")
    )
    runs: list[RunEntry] = [
        {
            "dataset": record["dataset"],
            "trainer": record["trainer"],
            "seed": record["seed"],
            "config_hash": record["config_hash"],
            "snapshot_sha256": record["snapshot_sha256"],
        }
        for record in present.iter_rows(named=True)
    ]
    seen = {(run["dataset"], run["trainer"], run["seed"]) for run in runs}
    missing: list[MissingCell] = [
        {"dataset": dataset, "trainer": trainer, "seed": seed}
        for dataset, trainer, seed in expected_cells(index_path)
        if (dataset, trainer, seed) not in seen
    ]
    return {"datasets": sorted(DATASETS), "runs": runs, "missing": missing}


def write_lock(lock_data: Lock, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock_data, indent=2, sort_keys=True) + "\n")
    return path
