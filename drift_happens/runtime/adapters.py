"""Dataset runtime adapter registry for staged local execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.runtime.base import TaskResult
from drift_happens.runtime.dataset_pipeline import (
    expected_dataset_pipeline_slices,
    run_dataset_pipeline_eval_stage,
    run_dataset_pipeline_train_stage,
)
from drift_happens.runtime.metrics import MetricSink
from drift_happens.runtime.stages import RunStage
from drift_happens.runtime.synthetic import run_synthetic_eval, run_synthetic_train


@dataclass(frozen=True, slots=True)
class TrainUnit:
    """Expected resumable training work unit."""

    trainer_key: str
    train_slice: str


@dataclass(frozen=True, slots=True)
class EvalUnit:
    """Expected resumable evaluation work unit."""

    trainer_key: str
    train_slice: str
    eval_slice: str


@dataclass(frozen=True, slots=True)
class StageContext:
    """Inputs shared by concrete dataset runtime adapters."""

    cfg: ExperimentConfig
    run_dir: Path
    device: torch.device
    metric_sink: MetricSink
    resume: bool
    identity: RunIdentity


class DatasetRuntimeAdapter(Protocol):
    """Staged train/eval bridge for one dataset runtime family."""

    name: str

    def supports(self, cfg: ExperimentConfig) -> bool: ...

    def expected_train_units(self, cfg: ExperimentConfig) -> tuple[TrainUnit, ...]: ...

    def expected_eval_units(self, cfg: ExperimentConfig) -> tuple[EvalUnit, ...]: ...

    def train(self, ctx: StageContext) -> TaskResult: ...

    def eval(self, ctx: StageContext) -> TaskResult: ...


class SyntheticRuntimeAdapter:
    """Runtime adapter for the CPU synthetic smoke task."""

    name = "synthetic"

    def supports(self, cfg: ExperimentConfig) -> bool:
        return (
            cfg.dataset.name == "synthetic" and cfg.trainer.key == "linear-classifier"
        )

    def expected_train_units(self, cfg: ExperimentConfig) -> tuple[TrainUnit, ...]:
        return (TrainUnit(trainer_key=cfg.trainer.key, train_slice="synthetic"),)

    def expected_eval_units(self, cfg: ExperimentConfig) -> tuple[EvalUnit, ...]:
        return (
            EvalUnit(
                trainer_key=cfg.trainer.key,
                train_slice="synthetic",
                eval_slice="synthetic",
            ),
        )

    def train(self, ctx: StageContext) -> TaskResult:
        return run_synthetic_train(
            ctx.cfg,
            run_dir=ctx.run_dir,
            device=ctx.device,
            metric_sink=ctx.metric_sink,
            resume=ctx.resume,
        )

    def eval(self, ctx: StageContext) -> TaskResult:
        return run_synthetic_eval(
            ctx.cfg,
            run_dir=ctx.run_dir,
            device=ctx.device,
            metric_sink=ctx.metric_sink,
            resume=ctx.resume,
        )


class DatasetPipelineRuntimeAdapter:
    """Adapter for Yearbook, text, and IMDB staged pipeline modules."""

    name = "dataset_pipeline"
    _datasets = frozenset(
        {
            "amazon_reviews_23",
            "arxiv",
            "imdb_faces",
            "yearbook",
        }
    )

    def supports(self, cfg: ExperimentConfig) -> bool:
        return cfg.dataset.name in self._datasets

    def expected_train_units(self, cfg: ExperimentConfig) -> tuple[TrainUnit, ...]:
        train_slices, _ = expected_dataset_pipeline_slices(cfg)
        return tuple(
            TrainUnit(trainer_key=cfg.trainer.key, train_slice=train_slice)
            for train_slice in train_slices
        )

    def expected_eval_units(self, cfg: ExperimentConfig) -> tuple[EvalUnit, ...]:
        train_slices, eval_slices = expected_dataset_pipeline_slices(cfg)
        return tuple(
            EvalUnit(
                trainer_key=cfg.trainer.key,
                train_slice=train_slice,
                eval_slice=eval_slice,
            )
            for train_slice in train_slices
            for eval_slice in eval_slices
        )

    def train(self, ctx: StageContext) -> TaskResult:
        return run_dataset_pipeline_train_stage(
            ctx.cfg,
            run_dir=ctx.run_dir,
            metric_sink=ctx.metric_sink,
            resume=ctx.resume,
            identity=ctx.identity,
            device=ctx.device,
        )

    def eval(self, ctx: StageContext) -> TaskResult:
        return run_dataset_pipeline_eval_stage(
            ctx.cfg,
            run_dir=ctx.run_dir,
            metric_sink=ctx.metric_sink,
            resume=ctx.resume,
            identity=ctx.identity,
            device=ctx.device,
        )


def adapter_for_config(cfg: ExperimentConfig) -> DatasetRuntimeAdapter:
    """Return the first registered adapter that supports ``cfg``."""
    for adapter in registered_adapters():
        if adapter.supports(cfg):
            return adapter
    raise NotImplementedError(
        f"local run_stage does not support dataset={cfg.dataset.name!r} "
        f"trainer={cfg.trainer.key!r}"
    )


def run_adapter_stage(
    cfg: ExperimentConfig,
    *,
    stage: RunStage,
    run_dir: Path,
    device: torch.device,
    metric_sink: MetricSink,
    resume: bool,
    identity: RunIdentity,
) -> TaskResult:
    """Execute a stage through the registered dataset adapter."""
    adapter = adapter_for_config(cfg)
    ctx = StageContext(
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        metric_sink=metric_sink,
        resume=resume,
        identity=identity,
    )
    if stage == "train":
        return adapter.train(ctx)
    return adapter.eval(ctx)


def registered_adapters() -> tuple[DatasetRuntimeAdapter, ...]:
    """Return adapters in dispatch order."""
    return (SyntheticRuntimeAdapter(), DatasetPipelineRuntimeAdapter())
