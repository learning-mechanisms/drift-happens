from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from drift_happens.runtime.lock_repair import (
    apply_lock_repair,
    classify_stage_lock,
    slurm_job_is_running,
)


def _write_lock(
    path: Path,
    *,
    pid: int,
    host: str,
    heartbeat_at: str | None = None,
    slurm_job_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "pid": str(pid),
        "host": host,
        "heartbeat_at": heartbeat_at,
        "slurm_job_id": slurm_job_id,
    }
    path.write_text(
        "".join(
            f"{key}={value}\n" for key, value in fields.items() if value is not None
        )
    )


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_classify_removes_dead_local_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    _write_lock(lock, pid=_dead_pid(), host=socket.gethostname())

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="unavailable",
    )

    assert decision.action == "remove"
    assert decision.reason == "owner is a dead local process"


def test_classify_keeps_live_local_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    _write_lock(lock, pid=os.getpid(), host=socket.gethostname())

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="retry",
    )

    assert decision.action == "keep"
    assert decision.reason == "owner is local and still alive"


def test_classify_keeps_foreign_lock_when_wandb_running(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    old = datetime.now(UTC) - timedelta(hours=2)
    _write_lock(
        lock,
        pid=123,
        host="other-node",
        heartbeat_at=old.isoformat().replace("+00:00", "Z"),
    )

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="running",
        stale_after_seconds=60,
    )

    assert decision.action == "keep"
    assert decision.reason == "matching W&B stage is running"


def test_classify_removes_stale_foreign_lock_with_terminal_wandb(
    tmp_path: Path,
) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    old = datetime.now(UTC) - timedelta(hours=2)
    _write_lock(
        lock,
        pid=123,
        host="other-node",
        heartbeat_at=old.isoformat().replace("+00:00", "Z"),
        slurm_job_id="42",
    )

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="retry",
        slurm_running=False,
        stale_after_seconds=60,
    )
    applied = apply_lock_repair(decision)

    assert decision.action == "remove"
    assert applied.removed
    assert not lock.exists()


def test_classify_keeps_legacy_foreign_lock_without_opt_in(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    _write_lock(lock, pid=123, host="other-node")

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="retry",
    )

    assert decision.action == "keep"
    assert decision.reason == "foreign lock has no heartbeat metadata"


def test_classify_removes_legacy_foreign_lock_with_opt_in(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    _write_lock(lock, pid=123, host="other-node")

    decision = classify_stage_lock(
        lock_path=lock,
        run_dir=tmp_path,
        stage="train",
        wandb_state="retry",
        allow_legacy_foreign=True,
    )

    assert decision.action == "remove"
    assert decision.reason == "legacy foreign lock and W&B stage is terminal"


def test_slurm_invalid_job_id_means_not_running(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="slurm_load_jobs error: Invalid job id specified\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert slurm_job_is_running("42") is False


def test_slurm_unexpected_error_is_unknown(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="slurm temporarily unavailable\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert slurm_job_is_running("42") is None
