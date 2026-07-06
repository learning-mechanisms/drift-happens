from __future__ import annotations

import json
from pathlib import Path

import pytest

from drift_happens.experiments.registry import preset
from drift_happens.runtime.metrics import (
    CompositeMetricSink,
    JsonlMetricSink,
    MetricRecord,
    NoopMetricSink,
)
from drift_happens.utils.wandb_identity import build_run_identity


def _reject_json_constant(token: str) -> object:
    raise AssertionError(f"non-finite JSON token written: {token}")


def test_jsonl_metric_sink_sanitizes_non_finite_values(tmp_path: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    sink = JsonlMetricSink(run_dir=tmp_path)

    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/loss",
            value=float("nan"),
            step=1,
        )
    )

    line = (tmp_path / "metrics" / "train.jsonl").read_text().strip()
    assert "NaN" not in line
    row = json.loads(line, parse_constant=_reject_json_constant)
    assert row["value"] is None


def test_jsonl_metric_sink_writes_stable_identity_rows(tmp_path: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    identity = build_run_identity(cfg, run_dir=tmp_path, source_path=None)
    sink = JsonlMetricSink(run_dir=tmp_path, identity=identity)

    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/accuracy",
            value=0.75,
            step=1,
            epoch=1,
        )
    )

    row = json.loads((tmp_path / "metrics" / "train.jsonl").read_text())
    assert row["metric"] == "train/accuracy"
    assert row["value"] == 0.75
    assert row["run_identity"]["config_hash"] == identity.config_hash
    assert row["experiment"] == cfg.name
    assert row["dataset"] == cfg.dataset.name
    assert row["trainer"] == cfg.trainer.key
    assert row["seed"] == cfg.seed


def test_composite_metric_sink_fans_out_and_closes(tmp_path: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    sink = CompositeMetricSink(
        sinks=[
            JsonlMetricSink(run_dir=tmp_path),
            NoopMetricSink(),
        ]
    )

    sink.log(
        MetricRecord.from_config(
            cfg,
            phase="summary",
            metric="run/complete",
            value=1.0,
        )
    )
    sink.close()

    assert (tmp_path / "metrics" / "summary.jsonl").exists()


def test_composite_metric_sink_closes_all_sinks_and_reraises_first_error(
    tmp_path: Path,
) -> None:
    closed = []

    class RaisingSink(NoopMetricSink):
        def close(self, exit_code: int | None = None) -> None:
            raise RuntimeError("sink boom")

    class RecordingSink(NoopMetricSink):
        def close(self, exit_code: int | None = None) -> None:
            closed.append(True)

    sink = CompositeMetricSink(sinks=[RaisingSink(), RecordingSink()])

    with pytest.raises(RuntimeError, match="sink boom"):
        sink.close()

    # The second sink must be closed even though the first raised.
    assert closed == [True]
