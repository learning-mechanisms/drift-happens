from __future__ import annotations

from pathlib import Path

import pytest
import torch

from drift_happens.experiments.registry import preset
from drift_happens.runtime.adapters import (
    DatasetPipelineRuntimeAdapter,
    EvalUnit,
    TrainUnit,
)
from drift_happens.runtime.base import TaskResult
from drift_happens.runtime.dataset_pipeline import _prepare_dataset_pipeline
from drift_happens.runtime.local import worker_main


def test_worker_main_dispatches_supported_dataset_pipeline(
    tmp_artifacts: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_stage(phase: str, iterations: int):
        def _stage(cfg, *, run_dir, metric_sink, resume, identity, device):
            calls.append((phase, cfg.dataset.name, cfg.trainer.key, run_dir, device))
            return TaskResult(iterations=iterations, metrics={})

        return _stage

    monkeypatch.setattr(
        "drift_happens.runtime.adapters.run_dataset_pipeline_train_stage",
        fake_stage("train", 2),
    )
    monkeypatch.setattr(
        "drift_happens.runtime.adapters.run_dataset_pipeline_eval_stage",
        fake_stage("eval", 1),
    )
    cfg = preset("yearbook", "smoke-mlp-s").build()
    cfg = cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"device": "cpu"})}
    )

    result = worker_main(
        cfg,
        runs_root=tmp_artifacts / "runs",
        source_path=Path("configs/snapshots/presets/yearbook/smoke-mlp-s.json"),
    )

    assert result.exit_status == "ok"
    assert result.iterations == 3
    assert len(calls) == 2
    assert calls[0][:3] == ("train", "yearbook", "mlp_s")
    assert calls[1][:3] == ("eval", "yearbook", "mlp_s")
    assert calls[0][3] == calls[1][3]
    # The resolved runtime device must reach the dataset pipeline stages.
    assert calls[0][4] == calls[1][4] == torch.device("cpu")


def test_dataset_pipeline_adapter_reports_expected_units(monkeypatch) -> None:
    cfg = preset("yearbook", "smoke-mlp-s").build()
    adapter = DatasetPipelineRuntimeAdapter()

    monkeypatch.setattr(
        "drift_happens.runtime.adapters.expected_dataset_pipeline_slices",
        lambda cfg: (("1950", "1960"), ("1970",)),
    )

    assert adapter.expected_train_units(cfg) == (
        TrainUnit("mlp_s", "1950"),
        TrainUnit("mlp_s", "1960"),
    )
    assert adapter.expected_eval_units(cfg) == (
        EvalUnit("mlp_s", "1950", "1970"),
        EvalUnit("mlp_s", "1960", "1970"),
    )


@pytest.mark.parametrize(
    ("group", "name"),
    [("arxiv", "smoke-minilm-l6-frozen"), ("yearbook", "smoke-mlp-s")],
)
def test_dataset_pipelines_reject_non_conference_configs(group, name) -> None:
    cfg = preset(group, name).build()
    cfg = cfg.model_copy(update={"tags": ("preset", group, "smoke")})

    with pytest.raises(ValueError, match="only the conference lineup"):
        _prepare_dataset_pipeline(cfg)
