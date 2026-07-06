from __future__ import annotations

import json
from pathlib import Path

import pytest

from drift_happens.experiments.registry import preset
from drift_happens.runtime.local import worker_main


@pytest.mark.integration
def test_synthetic_cpu_smoke_writes_valid_run_dir(tmp_artifacts: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    num_epochs = cfg.trainer.training["num_epochs"]
    assert isinstance(num_epochs, int)

    _EVAL_STAGE_ITERATIONS = 1  # worker_main sums train epochs + one eval pass
    result = worker_main(cfg, allow_overwrite=True)
    run_dir = result.run_dir

    assert run_dir.is_relative_to(tmp_artifacts / "runs")
    assert result.exit_status == "ok"
    assert result.iterations == num_epochs + _EVAL_STAGE_ITERATIONS
    assert 0.0 <= result.metrics["train/accuracy"] <= 1.0
    assert 0.0 <= result.metrics["eval/accuracy"] <= 1.0

    snapshot = json.loads((run_dir / "snapshot.json").read_text())
    assert snapshot == json.loads(cfg.to_snapshot_json())

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["seed"] == cfg.seed
    assert metadata["exit_status"] == "ok"
    assert metadata["host"]["effective_device"] == "cpu"
    assert metadata["wall_seconds"] is not None
    assert metadata["run_identity"]["config_hash"]

    events_path = run_dir / "logs" / "events.jsonl"
    events = [
        json.loads(line)["event"] for line in events_path.read_text().splitlines()
    ]
    assert events.count("stage_started") == 2
    assert events.count("stage_finished") == 2

    metrics_path = run_dir / "metrics" / "train.jsonl"
    metric_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    accuracy_rows = [row for row in metric_rows if row["metric"] == "train/accuracy"]
    assert len(accuracy_rows) == num_epochs
    assert accuracy_rows[-1]["value"] == result.metrics["train/accuracy"]
    assert accuracy_rows[-1]["run_identity"]["config_hash"]

    history = json.loads(
        (run_dir / "stages" / "train" / "training_history.json").read_text()
    )
    assert history["final"]["train/accuracy"] == result.metrics["train/accuracy"]
    summary = json.loads((run_dir / "results" / "summary.json").read_text())
    assert summary["primary_metric"] == "eval/accuracy"
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "stages" / "train" / "completion.json").exists()
    assert (run_dir / "stages" / "eval" / "completion.json").exists()


@pytest.mark.integration
def test_synthetic_cpu_smoke_is_reproducible_across_runs(tmp_artifacts: Path) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()

    first = worker_main(cfg, runs_root=tmp_artifacts / "runs_a")
    second = worker_main(cfg, runs_root=tmp_artifacts / "runs_b")

    assert first.metrics
    assert first.metrics == second.metrics
