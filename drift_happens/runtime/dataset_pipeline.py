"""Staged runtime bridge for dataset pipeline implementations."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.runtime.base import TaskResult
from drift_happens.runtime.metrics import MetricRecord, MetricSink


def run_dataset_pipeline_train_stage(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    metric_sink: MetricSink,
    resume: bool,
    identity: RunIdentity,
    device: torch.device | None = None,
) -> TaskResult:
    """Run the train half of a dataset pipeline under ``stages/train``."""
    module, ctx, key, _ = _prepare_dataset_pipeline(cfg)
    train_root = run_dir / "stages" / "train"
    ctx = replace(ctx, artifacts_dir=train_root)
    module.train_single_model(
        ctx,
        key,
        resume=resume,
        run_identity=identity,
        experiment_config=cfg,
        metric_sink=metric_sink,
        device=device,
    )
    completed = float(len(ctx.train_time_slices))
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/slices_completed",
            value=completed,
        )
    )
    return TaskResult(iterations=len(ctx.train_time_slices), metrics={})


def run_dataset_pipeline_eval_stage(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    metric_sink: MetricSink,
    resume: bool,
    identity: RunIdentity,
    device: torch.device | None = None,
) -> TaskResult:
    """Run the eval half of a dataset pipeline under ``stages/eval``."""
    module, ctx, key, eval_time_slices = _prepare_dataset_pipeline(cfg)
    eval_root = run_dir / "stages" / "eval"
    ctx = replace(ctx, artifacts_dir=eval_root)
    module.eval_single_model(
        ctx,
        key,
        eval_time_slices,
        resume=resume,
        run_identity=identity,
        experiment_config=cfg,
        metric_sink=metric_sink,
        model_artifacts_dir=run_dir / "stages" / "train",
        device=device,
    )
    summary_metrics = _write_pipeline_summary(run_dir, cfg)
    _log_pipeline_summary_metrics(metric_sink, cfg, summary_metrics)
    cells = float(len(ctx.train_time_slices) * len(eval_time_slices))
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="eval",
            metric="eval/cells_completed",
            value=cells,
        )
    )
    return TaskResult(iterations=int(cells), metrics=summary_metrics)


def expected_dataset_pipeline_slices(
    cfg: ExperimentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return expected train and eval slice keys for a dataset pipeline config."""
    _, ctx, _, eval_time_slices = _prepare_dataset_pipeline(cfg)
    return (
        tuple(str(key) for key in ctx.train_time_slices),
        tuple(str(key) for key in eval_time_slices),
    )


def _prepare_dataset_pipeline(
    cfg: ExperimentConfig,
) -> tuple[Any, Any, str, dict[Any, Any]]:
    if not any(tag.endswith("-conference") for tag in cfg.tags):
        raise ValueError(
            f"the {cfg.dataset.name} pipeline serves only the conference lineup; "
            "tag the config with its -conference group"
        )
    if cfg.dataset.name == "yearbook":
        from drift_happens.pipeline.yearbook import run as yearbook_module

        yearbook_ctx = yearbook_module.setup(trainer_keys=[cfg.trainer.key])
        eval_time_slices = yearbook_module.create_simple_time_slices(
            yearbook_ctx.df,
            time_col="year",
        )
        return yearbook_module, yearbook_ctx, cfg.trainer.key, eval_time_slices
    if cfg.dataset.name == "imdb_faces":
        from drift_happens.pipeline.imdb_faces import run as imdb_faces_module

        imdb_faces_ctx = imdb_faces_module.setup(trainer_keys=[cfg.trainer.key])
        eval_time_slices = imdb_faces_module.create_simple_time_slices(
            imdb_faces_ctx.df,
            time_col="photo_taken",
        )
        return imdb_faces_module, imdb_faces_ctx, cfg.trainer.key, eval_time_slices
    if cfg.dataset.name == "arxiv":
        from drift_happens.pipeline.arxiv import run as arxiv_module

        arxiv_ctx = arxiv_module.setup(trainer_keys=[cfg.trainer.key])
        eval_time_slices = arxiv_module.create_simple_time_slices(
            df=arxiv_ctx.df,
            time_col="year",
            min_time=2000,
        )
        return arxiv_module, arxiv_ctx, cfg.trainer.key, eval_time_slices
    if cfg.dataset.name == "amazon_reviews_23":
        from drift_happens.pipeline.amazon_reviews_23 import (
            run as amazon_reviews_module,
        )

        amazon_reviews_ctx = amazon_reviews_module.setup(trainer_keys=[cfg.trainer.key])
        eval_time_slices = amazon_reviews_module.create_simple_time_slices(
            df=amazon_reviews_ctx.df,
            time_col="half_year",
            min_time=(2014 - 2000) * 2,
        )
        return (
            amazon_reviews_module,
            amazon_reviews_ctx,
            cfg.trainer.key,
            eval_time_slices,
        )
    raise NotImplementedError(f"unsupported dataset runtime: {cfg.dataset.name}")


def _write_pipeline_summary(run_dir: Path, cfg: ExperimentConfig) -> dict[str, float]:
    matrix = _read_eval_matrix(run_dir, cfg)
    primary_metric = cfg.evaluation.metric
    primary_values = [
        metrics[primary_metric]
        for row in matrix.values()
        for metrics in row.values()
        if primary_metric is not None and primary_metric in metrics
    ]
    primary_value = (
        sum(primary_values) / len(primary_values) if primary_values else None
    )
    matrix_path = run_dir / "results" / "drift_matrix.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    _write_matrix_csv(run_dir / "results" / "drift_matrix.csv", matrix)

    # Seed summaries aggregate the summary's metrics dict, so the primary
    # metric must land there and not only in primary_value.
    summary_metrics: dict[str, float] = (
        {primary_metric: primary_value}
        if primary_metric is not None and primary_value is not None
        else {}
    )
    path = run_dir / "results" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": summary_metrics,
                "primary_metric": primary_metric,
                "primary_value": primary_value,
                "seed": cfg.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary_metrics


def _log_pipeline_summary_metrics(
    metric_sink: MetricSink,
    cfg: ExperimentConfig,
    summary_metrics: dict[str, float],
) -> None:
    """Log final eval summary metrics after the drift matrix has been written."""
    for name, value in sorted(summary_metrics.items()):
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="summary",
                metric=f"summary/{name}",
                value=value,
            )
        )
    primary_metric = cfg.evaluation.metric
    if primary_metric is not None and primary_metric in summary_metrics:
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="summary",
                metric="summary/primary_metric",
                value=summary_metrics[primary_metric],
                context={"summary/primary_metric_name": primary_metric},
            )
        )


def _read_eval_matrix(
    run_dir: Path,
    cfg: ExperimentConfig,
) -> dict[str, dict[str, dict[str, float]]]:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    eval_root = run_dir / "stages" / "eval" / cfg.trainer.key
    for completion_path in sorted(
        eval_root.glob("train_slice_*/eval_slice=*.completion.json")
    ):
        try:
            payload = json.loads(completion_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        train_slice = completion_path.parent.name.removeprefix("train_slice_")
        eval_slice = completion_path.name.removeprefix("eval_slice=").removesuffix(
            ".completion.json"
        )
        matrix.setdefault(train_slice, {})[eval_slice] = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    return matrix


def _write_matrix_csv(
    path: Path,
    matrix: dict[str, dict[str, dict[str, float]]],
) -> None:
    metric_names = sorted(
        {
            metric
            for evals in matrix.values()
            for metrics in evals.values()
            for metric in metrics
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["train_slice", "eval_slice", *metric_names],
        )
        writer.writeheader()
        for train_slice, evals in sorted(matrix.items()):
            for eval_slice, metrics in sorted(evals.items()):
                writer.writerow(
                    {
                        "train_slice": train_slice,
                        "eval_slice": eval_slice,
                        **metrics,
                    }
                )
