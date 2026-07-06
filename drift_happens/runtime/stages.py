"""Stage-level artifact contracts for local experiment runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from drift_happens.configs import BaseConfig, RunIdentity

RunStage = Literal["train", "eval"]
StageExitStatus = Literal["ok", "error", "running", "skipped"]
StageStatus = Literal["missing", "running", "partial", "ok", "failed", "blocked"]
WorkUnitKind = Literal["train_slice", "eval_cell"]
STAGE_CONTENTION_EXIT_CODE = 75


class StageCompletion(BaseConfig):
    """Durable completion marker for one run stage."""

    stage: RunStage
    exit_status: StageExitStatus
    seed: int
    source_identity: str
    config_hash: str
    completion_hash: str | None = None
    snapshot_sha256: str
    trainer_key: str
    dataset_name: str
    run_dir: str
    attempt_dir: str | None = None
    started_at: datetime | None = None
    ended_at: datetime
    iterations: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error_message: str | None = None
    lockfile_sha256: str | None = None
    git_commit: str | None = None


class WorkUnitCompletion(BaseConfig):
    """Durable completion marker for a resumable train slice or eval cell."""

    kind: WorkUnitKind
    stage: RunStage
    exit_status: StageExitStatus
    seed: int | None = None
    source_identity: str | None = None
    config_hash: str | None = None
    snapshot_sha256: str | None = None
    trainer_key: str
    train_slice: str
    eval_slice: str | None = None
    ended_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    error_message: str | None = None


def stage_dir(run_dir: Path, stage: RunStage) -> Path:
    """Return the root directory for one stage inside a canonical run dir."""
    return run_dir / "stages" / stage


def stage_completion_path(run_dir: Path, stage: RunStage) -> Path:
    """Return the stage-level completion marker path."""
    return stage_dir(run_dir, stage) / "completion.json"


def stage_metadata_path(run_dir: Path, stage: RunStage) -> Path:
    """Return the stage-level metadata path."""
    return stage_dir(run_dir, stage) / "metadata.json"


def work_unit_completion_matches(
    payload: dict[str, Any],
    *,
    identity: RunIdentity | None,
    trainer_key: str,
    train_slice: object,
    eval_slice: object | None = None,
) -> bool:
    """Return True only when exit_status is 'ok' and identity fields match."""
    if payload.get("exit_status") != "ok":
        return False
    if payload.get("trainer_key") != trainer_key:
        return False
    if payload.get("train_slice") != str(train_slice):
        return False
    if eval_slice is not None and payload.get("eval_slice") != str(eval_slice):
        return False
    if identity is None:
        return True
    if (
        identity.completion_hash is not None
        and payload.get("source_identity") == identity.source_identity
        and payload.get("completion_hash") == identity.completion_hash
    ):
        return True
    return (
        payload.get("source_identity") == identity.source_identity
        and payload.get("config_hash") == identity.config_hash
        and payload.get("snapshot_sha256") == identity.snapshot_sha256
    )


def stage_completion_matches(
    payload: dict[str, Any],
    *,
    identity: RunIdentity,
    seed: int,
    stage: RunStage,
) -> bool:
    """Return True when identity fields match; does NOT check exit_status."""
    if (
        payload.get("stage") == stage
        and payload.get("seed") == seed
        and payload.get("source_identity") == identity.source_identity
        and identity.completion_hash is not None
        and payload.get("completion_hash") == identity.completion_hash
    ):
        return True
    return (
        payload.get("stage") == stage
        and payload.get("seed") == seed
        and payload.get("source_identity") == identity.source_identity
        and payload.get("config_hash") == identity.config_hash
        and payload.get("snapshot_sha256") == identity.snapshot_sha256
    )


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object on missing/invalid files."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON durably enough for completion markers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
