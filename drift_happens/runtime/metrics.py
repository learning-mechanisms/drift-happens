"""Metric record protocol and local sink implementations."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from drift_happens.configs import ExperimentConfig, RunIdentity


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One scalar metric observation with run and drift-matrix context."""

    experiment: str
    dataset: str
    dataset_variant: str | None
    trainer: str
    trainer_family: str | None
    seed: int
    phase: str
    metric: str
    value: float
    step: int | None = None
    epoch: int | None = None
    train_slice: str | None = None
    eval_slice: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig,
        *,
        phase: str,
        metric: str,
        value: float,
        step: int | None = None,
        epoch: int | None = None,
        train_slice: str | None = None,
        eval_slice: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> MetricRecord:
        """Create a record populated with the config's stable identity fields."""
        return cls(
            experiment=cfg.name,
            dataset=cfg.dataset.name,
            dataset_variant=cfg.dataset.variant,
            trainer=cfg.trainer.key,
            trainer_family=cfg.trainer.family,
            seed=cfg.seed,
            phase=phase,
            metric=metric,
            value=float(value),
            step=step,
            epoch=epoch,
            train_slice=train_slice,
            eval_slice=eval_slice,
            context=context or {},
        )

    def to_json_dict(self, identity: RunIdentity | None = None) -> dict[str, Any]:
        """Return a stable JSON-serializable row."""
        payload: dict[str, Any] = {
            "context": _jsonable(self.context),
            "dataset": self.dataset,
            "dataset_variant": self.dataset_variant,
            "epoch": self.epoch,
            "eval_slice": self.eval_slice,
            "experiment": self.experiment,
            "metric": self.metric,
            "phase": self.phase,
            "seed": self.seed,
            "step": self.step,
            "timestamp": self.timestamp,
            "train_slice": self.train_slice,
            "trainer": self.trainer,
            "trainer_family": self.trainer_family,
            "value": self.value if math.isfinite(self.value) else None,
        }
        if identity is not None:
            payload["run_identity"] = identity.model_dump(mode="json")
        return payload


class MetricSink(Protocol):
    """Receives scalar metric records."""

    def log(self, record: MetricRecord) -> None: ...

    def close(self, exit_code: int | None = None) -> None: ...


@dataclass(slots=True)
class NoopMetricSink:
    """Metric sink used when all outputs are disabled."""

    def log(self, record: MetricRecord) -> None:
        return None

    def close(self, exit_code: int | None = None) -> None:
        return None


@dataclass(slots=True)
class JsonlMetricSink:
    """Append metric records to ``metrics/<phase>.jsonl``."""

    run_dir: Path
    identity: RunIdentity | None = None

    def log(self, record: MetricRecord) -> None:
        path = self.run_dir / "metrics" / f"{record.phase}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(
                json.dumps(
                    record.to_json_dict(identity=self.identity),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def close(self, exit_code: int | None = None) -> None:
        return None


@dataclass(slots=True)
class CompositeMetricSink:
    """Fan out metric records to multiple sinks."""

    sinks: Iterable[MetricSink]

    def log(self, record: MetricRecord) -> None:
        for sink in self.sinks:
            sink.log(record)

    def close(self, exit_code: int | None = None) -> None:
        first_error: BaseException | None = None
        for sink in self.sinks:
            try:
                sink.close(exit_code=exit_code)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)
    return value
