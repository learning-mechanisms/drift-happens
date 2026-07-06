"""Shared runtime result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RunExitStatus = Literal["ok", "error"]
RunStageName = Literal["train", "eval"]


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Result returned by one concrete experiment task implementation."""

    iterations: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result returned by a local experiment worker process."""

    run_dir: Path
    exit_status: RunExitStatus
    iterations: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Result returned by one train/eval stage."""

    run_dir: Path
    stage: RunStageName
    exit_status: RunExitStatus
    iterations: int
    metrics: dict[str, float] = field(default_factory=dict)
