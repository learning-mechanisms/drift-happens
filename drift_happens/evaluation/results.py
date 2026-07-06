"""Discover and shape runtime drift-matrix results for downstream evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from drift_happens.utils.paths import RUNS_DIR

MatrixPayload = Mapping[str, Mapping[str, Mapping[str, float]]]


@dataclass(frozen=True, slots=True)
class ResultRun:
    """One completed run with a runtime drift-matrix result."""

    run_dir: Path
    dataset: str
    trainer_key: str
    experiment: str
    seed: int | None
    source_identity: str
    tags: tuple[str, ...]
    primary_metric: str | None
    matrix: MatrixPayload


def discover_result_runs(
    input_paths: Sequence[Path] | None = None,
    *,
    runs_root: Path = RUNS_DIR,
) -> list[ResultRun]:
    """Discover runtime result matrices under run directories or roots."""
    roots = tuple(input_paths or (runs_root,))
    seen: set[Path] = set()
    runs: list[ResultRun] = []
    for root in roots:
        for run_dir in _iter_result_run_dirs(root):
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = _load_result_run(run_dir)
            if run is not None:
                runs.append(run)
    return sorted(
        runs,
        key=lambda run: (
            run.dataset,
            run.trainer_key,
            run.experiment,
            run.source_identity,
            run.seed if run.seed is not None else -1,
            str(run.run_dir),
        ),
    )


def resolve_metric(run: ResultRun, *, requested: str | None = None) -> str | None:
    """Resolve a requested or default metric name against a run matrix."""
    available = _available_metrics(run.matrix)
    if requested is not None:
        return _first_available_metric(requested, available)

    if run.primary_metric is not None:
        primary = _first_available_metric(run.primary_metric, available)
        if primary is not None:
            return primary

    for preferred in (
        "eval/accuracy",
        "accuracy",
        "eval/auc_macro",
        "auc_macro",
        "balanced_accuracy",
        "balanced_mse",
        "eval/loss",
        "loss",
    ):
        resolved = _first_available_metric(preferred, available)
        if resolved is not None:
            return resolved
    return sorted(available)[0] if available else None


def build_metric_dataframe(matrix: MatrixPayload, metric: str) -> pd.DataFrame:
    """Build a train-slice by eval-slice DataFrame for one scalar metric."""
    train_keys = sorted(matrix.keys(), key=_slice_sort_key)
    eval_keys = sorted(
        {eval_key for row in matrix.values() for eval_key in row.keys()},
        key=_slice_sort_key,
    )
    data: list[list[float | None]] = []
    for train_key in train_keys:
        row = matrix.get(train_key, {})
        data.append(
            [_metric_value(row.get(eval_key, {}), metric) for eval_key in eval_keys]
        )
    return pd.DataFrame(data, index=train_keys, columns=eval_keys, dtype=float)


def _iter_result_run_dirs(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.name == "drift_matrix.json":
        return [root.parent.parent]
    if (root / "results" / "drift_matrix.json").is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(
        path.parent.parent for path in root.rglob("results/drift_matrix.json")
    )


def _load_result_run(run_dir: Path) -> ResultRun | None:
    matrix = _read_json(run_dir / "results" / "drift_matrix.json")
    if not isinstance(matrix, dict) or not matrix:
        return None
    snapshot = _read_json(run_dir / "snapshot.json")
    summary = _read_json(run_dir / "results" / "summary.json")
    metadata = _read_json(run_dir / "metadata.json")
    manifest = _read_json(run_dir / "run_manifest.json")

    dataset = _nested_str(snapshot, "dataset", "name") or _str_value(
        manifest.get("dataset")
    )
    trainer_key = _nested_str(snapshot, "trainer", "key") or _str_value(
        manifest.get("trainer")
    )
    experiment = _str_value(snapshot.get("name")) or _str_value(
        manifest.get("experiment")
    )
    seed = _first_int(snapshot.get("seed"), summary.get("seed"))
    identity = _nested_dict(metadata, "run_identity") or _nested_dict(
        manifest,
        "identity",
    )
    source_identity = (
        _str_value(identity.get("source_identity"))
        or _str_value(identity.get("wandb_group"))
        or run_dir.name
    )
    tags = tuple(str(tag) for tag in snapshot.get("tags", ()) if isinstance(tag, str))
    primary_metric = _str_value(summary.get("primary_metric")) or _nested_str(
        snapshot,
        "evaluation",
        "metric",
    )
    numeric_matrix = _numeric_matrix(matrix)
    if (
        dataset is None
        or trainer_key is None
        or experiment is None
        or not numeric_matrix
    ):
        return None
    return ResultRun(
        run_dir=run_dir,
        dataset=dataset,
        trainer_key=trainer_key,
        experiment=experiment,
        seed=seed,
        source_identity=source_identity,
        tags=tags,
        primary_metric=primary_metric,
        matrix=numeric_matrix,
    )


def _numeric_matrix(data: Mapping[str, Any]) -> MatrixPayload:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for train_key, evals in data.items():
        if not isinstance(evals, dict):
            continue
        for eval_key, metrics in evals.items():
            if not isinstance(metrics, dict):
                continue
            numeric = {
                str(metric): float(value)
                for metric, value in metrics.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            if numeric:
                matrix.setdefault(str(train_key), {})[str(eval_key)] = numeric
    return matrix


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _available_metrics(matrix: MatrixPayload) -> set[str]:
    return {
        metric
        for evals in matrix.values()
        for metrics in evals.values()
        for metric in metrics
    }


def _first_available_metric(metric: str, available: set[str]) -> str | None:
    for alias in _metric_aliases(metric):
        if alias in available:
            return alias
    return None


def _metric_aliases(metric: str) -> tuple[str, ...]:
    aliases = [metric]
    if metric.startswith("eval/") or metric.startswith("train/"):
        aliases.append(metric.split("/", 1)[1])
    else:
        aliases.extend([f"eval/{metric}", f"train/{metric}"])
    return tuple(dict.fromkeys(aliases))


def _metric_value(metrics: Mapping[str, float], metric: str) -> float | None:
    for alias in _metric_aliases(metric):
        value = metrics.get(alias)
        if value is not None:
            return float(value)
    return None


def _slice_sort_key(value: object) -> tuple[int, float | str]:
    raw = str(value)
    try:
        return (0, float(raw))
    except ValueError:
        return (1, raw)


def _nested_str(data: Mapping[str, Any], *keys: str) -> str | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _str_value(value)


def _nested_dict(data: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _str_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_int(*values: object) -> int | None:
    for value in values:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None
