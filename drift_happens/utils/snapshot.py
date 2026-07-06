"""Writers and builders for run metadata and frozen experiment snapshots."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

from drift_happens.configs.experiment import ExperimentConfig
from drift_happens.configs.snapshot import (
    ExecutionInfo,
    GitInfo,
    HostInfo,
    LockfileInfo,
    RunIdentity,
    SnapshotMetadata,
)
from drift_happens.utils.git import GitState, read_git_state
from drift_happens.utils.lockfile import pixi_lock_sha256

SNAPSHOT_KIND = "experiment_snapshot/v1"


def build_metadata(
    *,
    seed: int,
    started_at: datetime,
    execution: ExecutionInfo | None = None,
    source_git: GitInfo | GitState | None = None,
    effective_device: str | None = None,
    run_identity: RunIdentity | None = None,
) -> SnapshotMetadata:
    """Capture reproducibility metadata for a starting run."""
    return SnapshotMetadata(
        seed=seed,
        started_at=started_at,
        execution=execution or ExecutionInfo(),
        git=_git_info(source_git),
        host=_host_info(effective_device=effective_device),
        lockfile=LockfileInfo(pixi_lock_sha256=pixi_lock_sha256()),
        run_identity=run_identity,
    )


def finalise_metadata(
    meta: SnapshotMetadata,
    *,
    exit_status: str,
    error_message: str | None,
    last_completed_iteration: int | None,
    ended_at: datetime,
) -> SnapshotMetadata:
    """Return a copy with end-of-run fields filled in."""
    wall_seconds = (ended_at - meta.started_at).total_seconds()
    return meta.model_copy(
        update={
            "exit_status": exit_status,
            "error_message": error_message,
            "last_completed_iteration": last_completed_iteration,
            "ended_at": ended_at,
            "wall_seconds": wall_seconds,
        }
    )


def write_metadata(path: Path, meta: SnapshotMetadata) -> None:
    """Write ``metadata.json`` atomically in stable, pretty JSON form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(_pretty_json(meta.model_dump(mode="json")))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_snapshot(path: Path, cfg: ExperimentConfig) -> None:
    """Write a deterministic ``snapshot.json`` for one resolved run config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_snapshot_json())


def build_preset_snapshot(
    *,
    cfg: ExperimentConfig,
    group: str,
    name: str,
    seeds: tuple[int, ...],
    description: str = "",
    tags: tuple[str, ...] = (),
    comparison_group: str | None = None,
    comparison_role: str = "headline",
    variant_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    """
    Build a stable preset snapshot envelope around a deterministic config.

    Preset snapshots intentionally exclude wall-clock time, git state, host info, and
    lockfile hashes. Those belong to run-level ``metadata.json`` files.
    """
    return {
        "kind": SNAPSHOT_KIND,
        "name": name,
        "group": group,
        "description": description,
        "comparison_group": comparison_group,
        "comparison_role": comparison_role,
        "seeds": list(seeds),
        "tags": list(tags),
        "variant_fields": list(variant_fields),
        "config": cfg.model_dump(mode="json"),
    }


def write_preset_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    """Write a materialized preset snapshot in stable, pretty JSON form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty_json(snapshot))


def load_preset_snapshot(path: Path) -> dict[str, object]:
    """Load a materialized preset snapshot envelope."""
    return json.loads(path.read_text())


def experiment_from_preset_snapshot(snapshot: dict[str, object]) -> ExperimentConfig:
    """Extract and validate the inner ``ExperimentConfig`` from a preset snapshot."""
    config = snapshot.get("config")
    if not isinstance(config, dict):
        raise ValueError("snapshot is missing a 'config' object")
    return ExperimentConfig.model_validate(config)


def _git_info(source_git: GitInfo | GitState | None) -> GitInfo:
    if isinstance(source_git, GitInfo):
        return source_git
    git = source_git or read_git_state()
    return GitInfo(
        commit=git.commit,
        branch=git.branch,
        dirty=git.dirty,
        diff_truncated=git.diff_truncated,
    )


def _host_info(*, effective_device: str | None = None) -> HostInfo:
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_memory_bytes: int | None = None
    device_name: str | None = None
    if cuda_available and cuda_device_count:
        properties = torch.cuda.get_device_properties(0)
        gpu_memory_bytes = int(properties.total_memory)
        device_name = torch.cuda.get_device_name(0)

    mps_available = bool(torch.backends.mps.is_available())

    return HostInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cpu_count=os.cpu_count(),
        cpu_model=_cpu_model(),
        memory_total_bytes=_memory_total_bytes(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cuda_device_count=cuda_device_count,
        device_name=device_name,
        gpu_memory_bytes=gpu_memory_bytes,
        mps_available=mps_available,
        effective_device=effective_device,
    )


def _cpu_model() -> str | None:
    processor = platform.processor()
    if processor:
        return processor

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip() or None
    return None


def _memory_total_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int):
        return int(pages * page_size)
    return None


def _pretty_json(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
