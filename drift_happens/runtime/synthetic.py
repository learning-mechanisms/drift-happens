"""Tiny synthetic tasks used to smoke-test the experiment runtime contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from drift_happens.configs import ExperimentConfig
from drift_happens.runtime.base import TaskResult
from drift_happens.runtime.metrics import MetricRecord, MetricSink


def run_synthetic_train(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    device: torch.device,
    metric_sink: MetricSink,
    resume: bool = True,
) -> TaskResult:
    """Train a small linear classifier on deterministic synthetic data."""
    params = cfg.trainer.training
    num_epochs = _int_param(params, "num_epochs", default=2, minimum=1)
    batch_size = _int_param(params, "batch_size", default=16, minimum=1)
    n_samples = _int_param(params, "n_samples", default=64, minimum=4)
    n_features = _int_param(params, "n_features", default=8, minimum=1)
    learning_rate = _float_param(
        params, "learning_rate", default=0.1, exclusive_minimum=0.0
    )

    stage_dir = run_dir / "stages" / "train"
    history_path = stage_dir / "training_history.json"
    checkpoint_path = stage_dir / "checkpoints" / "final.pt"
    if resume and history_path.exists() and checkpoint_path.exists():
        history_payload = json.loads(history_path.read_text())
        metrics = {
            str(k): float(v) for k, v in history_payload.get("final", {}).items()
        }
        return TaskResult(iterations=num_epochs, metrics=metrics)

    dataset = _make_dataset(
        seed=cfg.seed,
        n_samples=n_samples,
        n_features=n_features,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = nn.Linear(n_features, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history: list[dict[str, float | int]] = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        seen = 0
        model.train()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(labels.numel())
            running_loss += float(loss.item()) * batch_size_actual
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += batch_size_actual

        row = {
            "epoch": epoch + 1,
            "loss": running_loss / seen,
            "accuracy": correct / seen,
        }
        history.append(row)
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="train",
                metric="train/loss",
                value=float(row["loss"]),
                step=epoch + 1,
                epoch=epoch + 1,
            )
        )
        metric_sink.log(
            MetricRecord.from_config(
                cfg,
                phase="train",
                metric="train/accuracy",
                value=float(row["accuracy"]),
                step=epoch + 1,
                epoch=epoch + 1,
            )
        )

    final_metrics = {
        "train/loss": float(history[-1]["loss"]),
        "train/accuracy": float(history[-1]["accuracy"]),
    }
    _write_history(history_path, history, final_metrics)
    _write_history(run_dir / "training_history.json", history, final_metrics)
    _write_checkpoint(checkpoint_path, cfg, model, final_metrics)
    _write_checkpoint(run_dir / "checkpoints" / "final.pt", cfg, model, final_metrics)
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="summary",
            metric="summary/primary_metric",
            value=final_metrics["train/accuracy"],
            context={"summary/primary_metric_name": "train/accuracy"},
        )
    )
    return TaskResult(iterations=num_epochs, metrics=final_metrics)


def run_synthetic_eval(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    device: torch.device,
    metric_sink: MetricSink,
    resume: bool = True,
) -> TaskResult:
    """Evaluate the trained synthetic classifier in a separate stage."""
    params = cfg.trainer.training
    n_samples = _int_param(params, "n_samples", default=64, minimum=4)
    n_features = _int_param(params, "n_features", default=8, minimum=1)
    checkpoint_path = run_dir / "stages" / "train" / "checkpoints" / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"synthetic eval requires train checkpoint: {checkpoint_path}"
        )

    eval_path = run_dir / "stages" / "eval" / "eval.json"
    if resume and eval_path.exists():
        payload = json.loads(eval_path.read_text())
        metrics = {str(k): float(v) for k, v in payload.get("metrics", {}).items()}
        return TaskResult(iterations=1, metrics=metrics)

    dataset = _make_dataset(seed=cfg.seed, n_samples=n_samples, n_features=n_features)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    model = nn.Linear(n_features, 2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            count = int(labels.numel())
            total_loss += float(loss.item()) * count
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += count

    metrics = {
        "eval/loss": total_loss / seen,
        "eval/accuracy": correct / seen,
    }
    _write_eval_payload(eval_path, cfg, metrics)
    summary_metrics = {**_load_train_metrics(run_dir), **metrics}
    _write_summary(run_dir / "results" / "summary.json", cfg, summary_metrics)
    _write_drift_matrix(run_dir / "results" / "drift_matrix.json", metrics)
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="eval",
            metric="eval/loss",
            value=metrics["eval/loss"],
            step=1,
        )
    )
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="eval",
            metric="eval/accuracy",
            value=metrics["eval/accuracy"],
            step=1,
        )
    )
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="summary",
            metric="summary/primary_metric",
            value=metrics["eval/accuracy"],
            context={"summary/primary_metric_name": "eval/accuracy"},
        )
    )
    return TaskResult(iterations=1, metrics=metrics)


def _make_dataset(*, seed: int, n_samples: int, n_features: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, n_features, generator=generator)
    weights = torch.linspace(-1.0, 1.0, n_features)
    bias = torch.tensor(0.15)
    logits = inputs @ weights + bias
    labels = (logits > 0).long()
    return TensorDataset(inputs, labels)


def _write_history(
    path: Path,
    history: list[dict[str, float | int]],
    final_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "epochs": history,
                "final": final_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_eval_payload(
    path: Path,
    cfg: ExperimentConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metrics": metrics, "seed": cfg.seed}, indent=2, sort_keys=True)
        + "\n"
    )


def _write_summary(
    path: Path,
    cfg: ExperimentConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary_metric = _primary_metric_name(cfg, metrics)
    path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "primary_metric": primary_metric,
                "primary_value": metrics[primary_metric],
                "seed": cfg.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_train_metrics(run_dir: Path) -> dict[str, float]:
    history_path = run_dir / "stages" / "train" / "training_history.json"
    try:
        payload = json.loads(history_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    final = payload.get("final")
    if not isinstance(final, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in final.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metrics[str(key)] = float(value)
    return metrics


def _primary_metric_name(
    cfg: ExperimentConfig,
    metrics: dict[str, float],
) -> str:
    if cfg.evaluation.metric:
        eval_candidate = f"eval/{cfg.evaluation.metric}"
        if eval_candidate in metrics:
            return eval_candidate
        train_candidate = f"train/{cfg.evaluation.metric}"
        if train_candidate in metrics:
            return train_candidate
    if "eval/accuracy" in metrics:
        return "eval/accuracy"
    return sorted(metrics)[0]


def _write_drift_matrix(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"synthetic": {"synthetic": metrics}},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_checkpoint(
    path: Path,
    cfg: ExperimentConfig,
    model: nn.Module,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.model_dump(mode="json"),
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _int_param(
    params: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = int(params.get(key, default))
    if value < minimum:
        raise ValueError(f"trainer.training.{key} must be >= {minimum}")
    return value


def _float_param(
    params: dict[str, Any],
    key: str,
    *,
    default: float,
    exclusive_minimum: float,
) -> float:
    value = float(params.get(key, default))
    if value <= exclusive_minimum:
        raise ValueError(f"trainer.training.{key} must be > {exclusive_minimum}")
    return value
