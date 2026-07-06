from __future__ import annotations

import csv
import json
from pathlib import Path

from drift_happens.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainerConfig,
)
from drift_happens.runtime.dataset_pipeline import (
    _log_pipeline_summary_metrics,
    _read_eval_matrix,
    _write_pipeline_summary,
)
from drift_happens.runtime.metrics import MetricRecord


def _config(metric: str = "accuracy") -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        seed=3,
        dataset=DatasetConfig(name="arxiv"),
        trainer=TrainerConfig(key="text", model={}),
        evaluation=EvaluationConfig(metric=metric),
    )


def _completion_path(root: Path, train: str, eval_: str) -> Path:
    return (
        root
        / "stages"
        / "eval"
        / "text"
        / f"train_slice_{train}"
        / f"eval_slice={eval_}.completion.json"
    )


def _completion(root: Path, train: str, eval_: str, metrics: object) -> None:
    path = _completion_path(root, train, eval_)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics}))


class RecordingMetricSink:
    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)

    def close(self, exit_code: int | None = None) -> None:
        return None


def test_read_eval_matrix_ignores_bad_json_and_non_numeric_metrics(
    tmp_path: Path,
) -> None:
    _completion(tmp_path, "2000", "2001", {"accuracy": 0.8, "flag": True, "note": "x"})
    _completion(tmp_path, "2001", "2002", None)
    bad = _completion_path(tmp_path, "bad", "bad")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{bad")

    matrix = _read_eval_matrix(tmp_path, _config())

    assert matrix == {"2000": {"2001": {"accuracy": 0.8}}}


def test_write_pipeline_summary_writes_json_csv_and_primary_mean(
    tmp_path: Path,
) -> None:
    _completion(tmp_path, "2000", "2000", {"accuracy": 0.5, "loss": 2.0})
    _completion(tmp_path, "2000", "2001", {"accuracy": 0.75})

    summary_metrics = _write_pipeline_summary(tmp_path, _config(metric="accuracy"))

    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    matrix = json.loads((tmp_path / "results" / "drift_matrix.json").read_text())
    with (tmp_path / "results" / "drift_matrix.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert summary["primary_value"] == 0.625
    # Seed summaries aggregate the metrics dict, so the primary metric must
    # appear there too.
    assert summary["metrics"] == {"accuracy": 0.625}
    assert matrix["2000"]["2000"]["loss"] == 2.0
    assert summary_metrics == {"accuracy": 0.625}
    assert rows == [
        {"train_slice": "2000", "eval_slice": "2000", "accuracy": "0.5", "loss": "2.0"},
        {"train_slice": "2000", "eval_slice": "2001", "accuracy": "0.75", "loss": ""},
    ]


def test_write_pipeline_summary_without_primary_values_leaves_metrics_empty(
    tmp_path: Path,
) -> None:
    _completion(tmp_path, "2000", "2000", {"loss": 2.0})

    _write_pipeline_summary(tmp_path, _config(metric="accuracy"))

    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["primary_value"] is None
    assert summary["metrics"] == {}


def test_log_pipeline_summary_metrics_emits_named_and_primary_rows() -> None:
    cfg = _config(metric="accuracy")
    sink = RecordingMetricSink()

    _log_pipeline_summary_metrics(sink, cfg, {"accuracy": 0.75, "loss": 2.0})

    by_metric = {record.metric: record for record in sink.records}
    assert by_metric["summary/accuracy"].value == 0.75
    assert by_metric["summary/loss"].value == 2.0
    assert by_metric["summary/primary_metric"].value == 0.75
    assert (
        by_metric["summary/primary_metric"].context["summary/primary_metric_name"]
        == "accuracy"
    )
