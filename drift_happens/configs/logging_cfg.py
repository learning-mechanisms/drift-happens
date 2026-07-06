"""Logging and metric-sink configuration."""

from __future__ import annotations

from typing import Literal

from drift_happens.configs.base import BaseConfig

LogLevel = Literal["debug", "info", "warning", "error", "critical"]
WandbMode = Literal["online", "offline", "disabled"]


class WandbConfig(BaseConfig):
    """Weights & Biases run identity and artifact policy."""

    project: str
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: WandbMode = "online"
    run_name: str | None = None
    job_type: str = "train_eval"
    upload_artifacts: bool = True
    upload_checkpoints: bool = False
    artifact_aliases: tuple[str, ...] = ("latest",)
    resume: Literal["allow", "never", "must"] = "never"


class LoggingConfig(BaseConfig):
    """Local and remote logging sinks for an experiment run."""

    level: LogLevel = "info"
    stdout: bool = True
    plain_log_file: bool = True
    json_log_file: bool = True
    metrics_jsonl: bool = True
    wandb: WandbConfig | None = None
