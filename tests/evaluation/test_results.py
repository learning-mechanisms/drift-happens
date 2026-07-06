from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drift_happens.evaluation.results import (
    build_metric_dataframe,
    discover_result_runs,
    resolve_metric,
)


def _write_run(
    root: Path,
    *,
    dataset: str = "yearbook",
    trainer: str = "cnn_s",
    experiment: str = "cnn-small",
    source_identity: str = "yearbook__cnn-small",
    seed: int = 0,
    matrix: dict | None = None,
) -> Path:
    run_dir = (
        root
        / dataset
        / trainer
        / experiment
        / f"seed={seed}"
        / f"{source_identity}__cfg-abc"
    )
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "dataset": {"name": dataset},
                "evaluation": {"metric": "accuracy"},
                "name": experiment,
                "seed": seed,
                "tags": ["conference", dataset],
                "trainer": {"key": trainer},
            }
        )
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_identity": {"source_identity": source_identity}})
    )
    (run_dir / "results" / "summary.json").write_text(
        json.dumps({"primary_metric": "accuracy", "seed": seed})
    )
    (run_dir / "results" / "drift_matrix.json").write_text(
        json.dumps(
            matrix
            or {
                "2000": {
                    "2000": {"accuracy": 0.8, "loss": 0.4},
                    "2001": {"accuracy": 0.7},
                },
                "2001": {"2001": {"accuracy": 0.9}},
            }
        )
    )
    return run_dir


def test_discover_result_runs_loads_runtime_matrix_metadata(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "runs", seed=0)

    runs = discover_result_runs(runs_root=tmp_path / "runs")

    assert [run.run_dir for run in runs] == [run_dir]
    assert runs[0].dataset == "yearbook"
    assert runs[0].trainer_key == "cnn_s"
    assert runs[0].source_identity == "yearbook__cnn-small"
    assert runs[0].primary_metric == "accuracy"


def test_resolve_metric_accepts_unprefixed_request_for_prefixed_runtime_metric(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path / "runs",
        dataset="synthetic",
        matrix={"synthetic": {"synthetic": {"eval/accuracy": 0.75}}},
    )
    run = discover_result_runs(runs_root=tmp_path / "runs")[0]

    assert resolve_metric(run, requested="accuracy") == "eval/accuracy"
    assert (
        build_metric_dataframe(run.matrix, "accuracy").loc["synthetic", "synthetic"]
        == 0.75
    )


def test_build_metric_dataframe_sorts_numeric_slices_and_keeps_missing_cells() -> None:
    df = build_metric_dataframe(
        {
            "10": {"11": {"accuracy": 0.8}},
            "2": {"2": {"accuracy": 0.7}},
        },
        "accuracy",
    )

    assert df.index.tolist() == ["2", "10"]
    assert df.columns.tolist() == ["2", "11"]
    assert pd.isna(df.loc["10", "2"])
