"""
Classify and repair stale stage lock files.

The normal lock path only reclaims locks owned by dead local processes. Foreign-host
locks on a shared filesystem need outside evidence, so repair is explicit and dry-run by
default.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from drift_happens.runtime.locks import (
    LockOwner,
    heartbeat_age_seconds,
    lock_owner_is_dead,
    lock_owner_is_local,
    read_lock_owner,
    remove_lock,
)
from drift_happens.runtime.stages import RunStage

LockRepairAction = Literal["remove", "keep", "skip"]
WandbLockState = Literal[
    "missing",
    "retry",
    "retry_exhausted",
    "running",
    "complete",
    "unavailable",
]

_RECLAIMABLE_WANDB_STATES: set[WandbLockState] = {
    "missing",
    "retry",
    "retry_exhausted",
    "complete",
}


@dataclass(frozen=True, slots=True)
class LockRepairDecision:
    """One repair decision for a stage lock."""

    action: LockRepairAction
    reason: str
    lock_path: Path
    run_dir: Path
    stage: RunStage
    owner: LockOwner | None = None
    wandb_state: WandbLockState = "unavailable"
    slurm_running: bool | None = None
    heartbeat_age_seconds: float | None = None
    removed: bool = False

    @property
    def owner_label(self) -> str:
        """Human-readable owner summary."""
        if self.owner is None:
            return "unknown"
        host = self.owner.host or "?"
        return f"pid={self.owner.pid} host={host}"


def classify_stage_lock(
    *,
    lock_path: Path,
    run_dir: Path,
    stage: RunStage,
    wandb_state: WandbLockState = "unavailable",
    slurm_running: bool | None = None,
    stale_after_seconds: float = 3600.0,
    allow_legacy_foreign: bool = False,
) -> LockRepairDecision:
    """Classify whether a stage lock can be safely removed."""
    owner = read_lock_owner(lock_path)
    if owner is None:
        return LockRepairDecision(
            action="skip",
            reason="lock missing or unreadable",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=None,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
        )

    heartbeat_age = heartbeat_age_seconds(owner)
    if lock_owner_is_local(owner):
        if lock_owner_is_dead(owner):
            return LockRepairDecision(
                action="remove",
                reason="owner is a dead local process",
                lock_path=lock_path,
                run_dir=run_dir,
                stage=stage,
                owner=owner,
                wandb_state=wandb_state,
                slurm_running=slurm_running,
                heartbeat_age_seconds=heartbeat_age,
            )
        return LockRepairDecision(
            action="keep",
            reason="owner is local and still alive",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age_seconds=heartbeat_age,
        )

    if wandb_state == "running":
        return _keep_foreign(
            "matching W&B stage is running",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )
    if slurm_running is True:
        return _keep_foreign(
            "recorded Slurm job is running",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )
    if wandb_state not in _RECLAIMABLE_WANDB_STATES:
        return _keep_foreign(
            "W&B does not prove the stage is terminal",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )
    if owner.slurm_job_id and slurm_running is None:
        return _keep_foreign(
            "recorded Slurm job could not be checked",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )
    if heartbeat_age is None and not allow_legacy_foreign:
        return _keep_foreign(
            "foreign lock has no heartbeat metadata",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )
    if heartbeat_age is not None and heartbeat_age < stale_after_seconds:
        return _keep_foreign(
            "foreign lock heartbeat is still fresh",
            lock_path=lock_path,
            run_dir=run_dir,
            stage=stage,
            owner=owner,
            wandb_state=wandb_state,
            slurm_running=slurm_running,
            heartbeat_age=heartbeat_age,
        )

    reason = "foreign lock is stale and W&B stage is terminal"
    if heartbeat_age is None:
        reason = "legacy foreign lock and W&B stage is terminal"
    return LockRepairDecision(
        action="remove",
        reason=reason,
        lock_path=lock_path,
        run_dir=run_dir,
        stage=stage,
        owner=owner,
        wandb_state=wandb_state,
        slurm_running=slurm_running,
        heartbeat_age_seconds=heartbeat_age,
    )


def apply_lock_repair(decision: LockRepairDecision) -> LockRepairDecision:
    """Apply a remove decision and return the resulting decision."""
    if decision.action != "remove":
        return decision
    return LockRepairDecision(
        action=decision.action,
        reason=decision.reason,
        lock_path=decision.lock_path,
        run_dir=decision.run_dir,
        stage=decision.stage,
        owner=decision.owner,
        wandb_state=decision.wandb_state,
        slurm_running=decision.slurm_running,
        heartbeat_age_seconds=decision.heartbeat_age_seconds,
        removed=remove_lock(decision.lock_path),
    )


def slurm_job_is_running(job_id: str) -> bool | None:
    """Return whether Slurm currently reports ``job_id`` as active."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        if "Invalid job id specified" in result.stderr:
            return False
        return None
    return bool(result.stdout.strip())


def _keep_foreign(
    reason: str,
    *,
    lock_path: Path,
    run_dir: Path,
    stage: RunStage,
    owner: LockOwner,
    wandb_state: WandbLockState,
    slurm_running: bool | None,
    heartbeat_age: float | None,
) -> LockRepairDecision:
    return LockRepairDecision(
        action="keep",
        reason=reason,
        lock_path=lock_path,
        run_dir=run_dir,
        stage=stage,
        owner=owner,
        wandb_state=wandb_state,
        slurm_running=slurm_running,
        heartbeat_age_seconds=heartbeat_age,
    )
