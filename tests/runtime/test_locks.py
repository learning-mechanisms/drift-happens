from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from drift_happens.runtime.locks import FileLock, read_lock_owner


def test_file_lock_rejects_contention_without_timeout(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"

    with FileLock(lock_path):
        with pytest.raises(RuntimeError, match="lock already held"):
            with FileLock(lock_path):
                pass


def test_file_lock_removes_lock_file_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"

    with FileLock(lock_path):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_file_lock_writes_scheduler_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    lock_path = tmp_path / "stage.lock"

    with FileLock(
        lock_path,
        metadata={
            "stage": "train",
            "run_dir": tmp_path,
            "seed": 7,
            "config_hash": "cfg",
        },
        heartbeat_seconds=0,
    ):
        owner = read_lock_owner(lock_path)

    assert owner is not None
    assert owner.host == socket.gethostname()
    assert owner.token is not None
    assert owner.created_at is not None
    assert owner.heartbeat_at is not None
    assert owner.slurm_job_id == "123"
    assert owner.slurm_step_id == "4"
    assert owner.stage == "train"
    assert owner.seed == "7"
    assert owner.config_hash == "cfg"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_file_lock_reclaims_stale_lock_from_dead_local_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"
    lock_path.write_text(f"pid={_dead_pid()}\nhost={socket.gethostname()}\n")

    with FileLock(lock_path):
        assert f"pid={os.getpid()}" in lock_path.read_text()

    assert not lock_path.exists()


def test_file_lock_rejects_lock_held_by_live_local_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"
    lock_path.write_text(f"pid={os.getpid()}\nhost={socket.gethostname()}\n")

    with pytest.raises(RuntimeError, match="lock already held"):
        with FileLock(lock_path):
            pass


def test_file_lock_never_reclaims_dead_pid_from_another_host(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"
    lock_path.write_text(f"pid={_dead_pid()}\nhost=some-other-node\n")

    with pytest.raises(RuntimeError, match="lock already held"):
        with FileLock(lock_path):
            pass


def test_file_lock_treats_hostless_legacy_payload_as_live(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"
    lock_path.write_text(f"pid={_dead_pid()}\n")

    with pytest.raises(RuntimeError, match="lock already held"):
        with FileLock(lock_path):
            pass


def test_file_lock_payload_write_failure_leaves_no_ownerless_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "stage.lock"

    def fail_write(fd: int, data: bytes) -> int:
        raise OSError("no space left on device")

    monkeypatch.setattr("drift_happens.runtime.locks.os.write", fail_write)
    with pytest.raises(OSError, match="no space left"):
        with FileLock(lock_path):
            pass
    monkeypatch.undo()

    # An ownerless lock file would block every later writer; the failed
    # acquisition must clean up after itself so the lock stays acquirable.
    assert not lock_path.exists()
    with FileLock(lock_path):
        assert lock_path.exists()


def test_file_lock_retries_reclaim_when_lock_owner_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "stage.lock"
    # The first observed owner is foreign, so the reclaim attempt fails.
    lock_path.write_text(f"pid={_dead_pid()}\nhost=some-other-node\n")
    local_dead = _dead_pid()

    def hand_lock_to_dead_local_owner(seconds: float) -> None:
        lock_path.write_text(f"pid={local_dead}\nhost={socket.gethostname()}\n")

    monkeypatch.setattr(
        "drift_happens.runtime.locks.time.sleep", hand_lock_to_dead_local_owner
    )

    # The owner changed to a dead local pid mid-wait; one reclaim attempt per
    # acquisition would miss it and time out instead.
    # timeout_seconds enables the retry loop; the patched sleep performs the handoff.
    with FileLock(lock_path, timeout_seconds=5.0):
        assert f"pid={os.getpid()}" in lock_path.read_text()


def test_file_lock_reclaims_holder_killed_mid_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "stage.lock"
    # The payload never changes: the holder is killed -9, not replaced.
    lock_path.write_text(f"pid={_dead_pid()}\nhost={socket.gethostname()}\n")
    alive = {"value": True}

    monkeypatch.setattr(
        "drift_happens.runtime.locks.lock_owner_is_dead",
        lambda owner: not alive["value"],
    )

    def kill_holder(seconds: float) -> None:
        alive["value"] = False

    monkeypatch.setattr("drift_happens.runtime.locks.time.sleep", kill_holder)

    # The owner dies between iterations without the payload changing; gating
    # reclaim on an owner change would never re-probe it and time out instead.
    # timeout_seconds enables the retry loop; the patched sleep performs the kill.
    with FileLock(lock_path, timeout_seconds=5.0):
        assert f"pid={os.getpid()}" in lock_path.read_text()


def test_file_lock_skips_reclaim_while_a_guard_exists(tmp_path: Path) -> None:
    lock_path = tmp_path / "stage.lock"
    lock_path.write_text(f"pid={_dead_pid()}\nhost={socket.gethostname()}\n")
    (tmp_path / "stage.lock.reclaim").touch()

    with pytest.raises(RuntimeError, match="lock already held"):
        with FileLock(lock_path):
            pass
