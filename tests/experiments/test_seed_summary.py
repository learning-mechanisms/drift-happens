from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drift_happens.experiments.seed_summary import _metric_summaries, summarize_seeds
from drift_happens.experiments.source import load_experiment_source
from drift_happens.runtime.local import worker_main


def _seed_results(*values: float) -> list[dict[str, Any]]:
    return [{"metrics": {"eval/accuracy": value}} for value in values]


def test_seed_summary_aggregates_completed_local_runs(tmp_artifacts: Path) -> None:
    source_path = Path(
        "configs/snapshots/presets/smoke/synthetic-classification-cpu.json"
    )
    source = load_experiment_source(
        source_path,
        overrides=("trainer.training.num_epochs=1",),
    )
    for seed in (0, 1):
        worker_main(
            source.config,
            seed=seed,
            runs_root=tmp_artifacts / "runs",
            source_path=source_path,
        )

    report_path = summarize_seeds(
        source.config,
        source_path=source_path,
        seeds=(0, 1, 2),
        out_dir=tmp_artifacts / "reports" / "seeds",
        runs_root=tmp_artifacts / "runs",
    )

    report = json.loads(report_path.read_text())
    assert report["successful_seeds"] == [0, 1]
    assert report["missing_or_failed"] == [
        {
            "eval": "missing",
            "run_dir": None,
            "seed": 2,
            "status": "missing",
            "train": "missing",
        }
    ]
    train_accuracy = report["metric_summaries"]["train/accuracy"]
    assert report["failed_or_corrupt_count"] == 1
    assert train_accuracy["count"] == 2

    # Verify min/max against the actual per-seed values, not just their ordering.
    summary_files = sorted((tmp_artifacts / "runs").glob("**/results/summary.json"))
    assert len(summary_files) == 2
    seed_train_values = [
        json.loads(f.read_text())["metrics"]["train/accuracy"] for f in summary_files
    ]
    assert train_accuracy["min"] == min(seed_train_values)
    assert train_accuracy["max"] == max(seed_train_values)

    # A completed run whose summary.json no longer parses must not count as a
    # successful seed with empty metrics.
    corrupted = summary_files
    assert len(corrupted) == 2
    corrupted[0].write_text("{not json")

    report_path = summarize_seeds(
        source.config,
        source_path=source_path,
        seeds=(0, 1, 2),
        out_dir=tmp_artifacts / "reports" / "seeds",
        runs_root=tmp_artifacts / "runs",
    )
    report = json.loads(report_path.read_text())

    assert len(report["successful_seeds"]) == 1
    assert report["failed_or_corrupt_count"] == 2
    corrupt_rows = [
        row for row in report["missing_or_failed"] if row["status"] == "corrupt_summary"
    ]
    assert len(corrupt_rows) == 1
    assert corrupt_rows[0]["run_dir"] is not None
    assert report["metric_summaries"]["train/accuracy"]["count"] == 1


def test_metric_summaries_use_student_t_interval_for_three_seeds() -> None:
    summary = _metric_summaries(_seed_results(1.0, 2.0, 3.0))["eval/accuracy"]

    # n=3: mean 2, std 1, stderr 1/sqrt(3); t(0.975, df=2) = 4.302653 gives a
    # half-width of 2.4841377.
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)
    assert summary["stderr"] == pytest.approx(0.5773503)
    assert summary["ci95_low"] == pytest.approx(-0.4841377)
    assert summary["ci95_high"] == pytest.approx(4.4841377)


def test_metric_summaries_widen_the_interval_for_two_seeds() -> None:
    summary = _metric_summaries(_seed_results(1.0, 3.0))["eval/accuracy"]

    # n=2: mean 2, std sqrt(2), stderr 1; t(0.975, df=1) = 12.706205.
    assert summary["stderr"] == pytest.approx(1.0)
    assert summary["ci95_low"] == pytest.approx(2.0 - 12.706205)
    assert summary["ci95_high"] == pytest.approx(2.0 + 12.706205)


def test_metric_summaries_collapse_the_interval_for_a_single_seed() -> None:
    summary = _metric_summaries(_seed_results(0.75))["eval/accuracy"]

    assert summary["std"] == 0.0
    assert summary["ci95_low"] == 0.75
    assert summary["ci95_high"] == 0.75
