"""
The runtime-resolved device must reach the trainers it claims to describe.

The staged runtime records ``effective_device`` from ``cfg.runtime.device``;
these tests pin the chain that makes that record honest: adapter-passed device
-> dataset pipeline stage -> module ``train_single_model``/``eval_single_model``
-> trainer builders -> ``PytorchTrainerConfig.device``. The direct module CLI
passes no device and keeps the auto-detection helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from drift_happens.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    RunIdentity,
    TrainerConfig,
)
from drift_happens.pipeline.arxiv import run as arxiv_run
from drift_happens.pipeline.arxiv.trainers import (
    arxiv_conference_trainer_configs,
    build_trainers_from_configs,
)
from drift_happens.pipeline.image.trainers import (
    build_image_trainers_from_configs,
    conference_image_trainer_configs,
)
from drift_happens.pipeline.yearbook import run as yearbook_run
from drift_happens.runtime import dataset_pipeline
from drift_happens.runtime.dataset_pipeline import (
    run_dataset_pipeline_eval_stage,
    run_dataset_pipeline_train_stage,
)
from drift_happens.runtime.metrics import MetricRecord
from drift_happens.utils.pytorch import device_manual_mps_or_cuda_if_available


def _image_configs() -> dict:
    return {"mlp_s": conference_image_trainer_configs()["mlp_s"]}


def _arxiv_configs() -> dict:
    configs = arxiv_conference_trainer_configs(
        category_to_idx={"cs": 0, "math": 1},
        pos_weight=torch.ones(2),
    )
    return {"ffn_s": configs["ffn_s"]}


def test_image_builder_uses_the_passed_device() -> None:
    trainer = build_image_trainers_from_configs(_image_configs(), device="cpu")["mlp_s"]
    assert trainer._config.device == "cpu"


def test_image_builder_defaults_to_auto_detection() -> None:
    trainer = build_image_trainers_from_configs(_image_configs())["mlp_s"]
    assert trainer._config.device == device_manual_mps_or_cuda_if_available()


def test_text_builder_uses_the_passed_torch_device() -> None:
    trainer = build_trainers_from_configs(_arxiv_configs(), device=torch.device("cpu"))[
        "ffn_s"
    ]
    assert trainer._config.device == "cpu"


def test_text_builder_defaults_to_auto_detection() -> None:
    trainer = build_trainers_from_configs(_arxiv_configs())["ffn_s"]
    assert trainer._config.device == device_manual_mps_or_cuda_if_available()


def _stub_text_context() -> SimpleNamespace:
    return SimpleNamespace(
        trainer_configs={"ffn_s": object()},
        tensor_dataset=object(),
        dataset_splits=object(),
        train_time_slices={},
        artifacts_dir=Path("unused"),
    )


def test_arxiv_train_single_model_passes_device_to_the_builder(monkeypatch) -> None:
    recorded: dict = {}

    def fake_builder(configs, *, device=None):
        recorded["device"] = device
        return {"ffn_s": object()}

    monkeypatch.setattr(arxiv_run, "build_trainers_from_configs", fake_builder)
    monkeypatch.setattr(arxiv_run, "train_models_on_time_slices", lambda **kwargs: None)

    arxiv_run.train_single_model(
        _stub_text_context(), "ffn_s", device=torch.device("cpu")
    )

    assert recorded["device"] == torch.device("cpu")


def test_yearbook_eval_single_model_passes_device_to_the_builder(monkeypatch) -> None:
    recorded: dict = {}

    def fake_builder(configs, *, device=None):
        recorded["device"] = device
        return {"mlp_s": object()}

    monkeypatch.setattr(yearbook_run, "build_trainers_from_configs", fake_builder)
    monkeypatch.setattr(
        yearbook_run, "embed_dataset_if_needed", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        yearbook_run, "eval_models_on_time_slices", lambda **kwargs: None
    )
    ctx = SimpleNamespace(
        trainer_configs={"mlp_s": object()},
        dataset_splits=object(),
        train_time_slices={},
        artifacts_dir=Path("unused"),
    )

    yearbook_run.eval_single_model(
        ctx, "mlp_s", eval_time_slices={}, device=torch.device("cpu")
    )

    assert recorded["device"] == torch.device("cpu")


@dataclass
class _FakePipelineContext:
    artifacts_dir: Path
    train_time_slices: dict = field(default_factory=dict)


class _RecordingModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def train_single_model(self, ctx, key, **kwargs) -> None:
        self.calls.append({"key": key, **kwargs})

    def eval_single_model(self, ctx, key, eval_time_slices, **kwargs) -> None:
        self.calls.append({"key": key, **kwargs})


class _ListSink:
    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)

    def close(self, exit_code: int | None = None) -> None:
        return None


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        seed=7,
        dataset=DatasetConfig(name="synthetic"),
        trainer=TrainerConfig(key="fake"),
        evaluation=EvaluationConfig(metric="accuracy"),
    )


def _identity() -> RunIdentity:
    return RunIdentity(
        source_identity="src",
        config_hash="cfg",
        snapshot_sha256="snap",
        wandb_group="group",
        wandb_run_name="run",
    )


def test_pipeline_stages_forward_device_to_the_module(tmp_path, monkeypatch) -> None:
    module = _RecordingModule()
    monkeypatch.setattr(
        dataset_pipeline,
        "_prepare_dataset_pipeline",
        lambda cfg: (module, _FakePipelineContext(artifacts_dir=tmp_path), "fake", {}),
    )

    run_dataset_pipeline_train_stage(
        _config(),
        run_dir=tmp_path,
        metric_sink=_ListSink(),
        resume=True,
        identity=_identity(),
        device=torch.device("cpu"),
    )
    run_dataset_pipeline_eval_stage(
        _config(),
        run_dir=tmp_path,
        metric_sink=_ListSink(),
        resume=True,
        identity=_identity(),
        device=torch.device("cpu"),
    )

    assert [call["device"] for call in module.calls] == [torch.device("cpu")] * 2
