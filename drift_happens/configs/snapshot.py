"""Run snapshot metadata models."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from drift_happens.configs.base import BaseConfig


class GitInfo(BaseConfig):
    """Git state captured at run start."""

    commit: str
    branch: str | None
    dirty: bool
    diff_truncated: str | None = Field(
        default=None,
        description="Full git diff text, truncated to a byte limit when large; not a boolean flag.",
    )


class HostInfo(BaseConfig):
    """Host and accelerator state captured at run start."""

    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cpu_count: int | None = None
    cpu_model: str | None = None
    memory_total_bytes: int | None = None
    cuda_available: bool
    cuda_version: str | None
    cuda_device_count: int = 0
    device_name: str | None
    gpu_memory_bytes: int | None = None
    mps_available: bool = False
    effective_device: str | None = None


class ExecutionInfo(BaseConfig):
    """Execution backend metadata for one run."""

    backend: str = "local"
    image_reference: str | None = Field(
        default=None,
        description="Container image reference used for remote execution.",
    )
    job_id: str | None = None
    worker_id: str | None = None
    slot_label: str | None = None
    device_request: str | None = None


class RunIdentity(BaseConfig):
    """Locally computed run identity persisted in metadata.json, including W&B group and
    run name."""

    source_identity: str
    config_hash: str
    completion_hash: str | None = None
    snapshot_sha256: str
    wandb_group: str
    wandb_run_name: str


class LockfileInfo(BaseConfig):
    """Dependency lockfile metadata captured at run start."""

    pixi_lock_sha256: str | None


class SnapshotMetadata(BaseConfig):
    """
    Volatile run-level metadata persisted to ``metadata.json``.

    This complements deterministic ``snapshot.json`` files, which contain only the
    resolved ``ExperimentConfig``.
    """

    seed: int
    started_at: datetime
    ended_at: datetime | None = None
    wall_seconds: float | None = None
    exit_status: str = "running"
    error_message: str | None = None
    last_completed_iteration: int | None = None
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    git: GitInfo
    host: HostInfo
    lockfile: LockfileInfo
    run_identity: RunIdentity | None = None
