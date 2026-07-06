"""Small file-lock helper for stage-level local writers."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from drift_happens.utils.log import get_logger

logger = get_logger()

_DEFAULT_HEARTBEAT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class LockOwner:
    """Owner identity recorded inside a lock file."""

    pid: int
    host: str | None
    token: str | None = None
    created_at: str | None = None
    heartbeat_at: str | None = None
    slurm_job_id: str | None = None
    slurm_step_id: str | None = None
    stage: str | None = None
    run_dir: str | None = None
    seed: str | None = None
    source_identity: str | None = None
    config_hash: str | None = None
    snapshot_sha256: str | None = None
    completion_hash: str | None = None
    wandb_run_id: str | None = None


def read_lock_owner(path: Path) -> LockOwner | None:
    """Parse the owner payload of a lock file; ``None`` when unreadable."""
    fields = read_lock_payload(path)
    if not fields:
        return None
    try:
        pid = int(fields["pid"])
    except (KeyError, ValueError):
        return None
    return LockOwner(
        pid=pid,
        host=fields.get("host"),
        token=fields.get("token"),
        created_at=fields.get("created_at"),
        heartbeat_at=fields.get("heartbeat_at"),
        slurm_job_id=fields.get("slurm_job_id"),
        slurm_step_id=fields.get("slurm_step_id"),
        stage=fields.get("stage"),
        run_dir=fields.get("run_dir"),
        seed=fields.get("seed"),
        source_identity=fields.get("source_identity"),
        config_hash=fields.get("config_hash"),
        snapshot_sha256=fields.get("snapshot_sha256"),
        completion_hash=fields.get("completion_hash"),
        wandb_run_id=fields.get("wandb_run_id"),
    )


def read_lock_payload(path: Path) -> dict[str, str]:
    """Read a lock payload as key/value fields."""
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return {}
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def lock_owner_is_dead(owner: LockOwner) -> bool:
    """
    Whether the owner is provably a dead process on this host.

    Locks from other hosts (or without a recorded host) are never reclaimable: on a
    shared filesystem a foreign pid cannot be probed.
    """
    if owner.host is None or owner.host != socket.gethostname():
        return False
    try:
        os.kill(owner.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def lock_owner_is_local(owner: LockOwner) -> bool:
    """Return whether a lock owner belongs to this host."""
    return owner.host == socket.gethostname()


def heartbeat_age_seconds(owner: LockOwner) -> float | None:
    """Return age of the latest heartbeat, or ``None`` when absent/unparseable."""
    raw = owner.heartbeat_at or owner.created_at
    if raw is None:
        return None
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(UTC) - timestamp).total_seconds()


def _reclaim_stale_lock(path: Path) -> bool:
    """
    Remove ``path`` iff its recorded owner is a dead local process.

    Reclaim is serialized through an O_EXCL guard file so two live contenders can never
    both unlink and re-acquire the same lock. A guard orphaned by a crash mid-reclaim
    disables reclaim for that lock until removed by hand.
    """
    owner = read_lock_owner(path)
    if owner is None or not lock_owner_is_dead(owner):
        return False
    guard = path.with_name(path.name + ".reclaim")
    try:
        guard_fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        current = read_lock_owner(path)
        if current is None or not _same_lock_owner(current, owner):
            return False
        if not lock_owner_is_dead(current):
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        logger.warning(
            f"Reclaimed stale lock {path} held by dead pid {owner.pid} on {owner.host}"
        )
        return True
    finally:
        os.close(guard_fd)
        try:
            guard.unlink()
        except FileNotFoundError:
            pass


@dataclass(slots=True)
class FileLock:
    """Exclusive lock based on atomic file creation."""

    path: Path
    timeout_seconds: float = 0.0
    poll_seconds: float = 0.2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS
    _fd: int | None = None
    _owner: LockOwner | None = None
    _stop_heartbeat: threading.Event | None = None
    _heartbeat_thread: threading.Thread | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        owner_payload = self._owner_payload()
        while True:
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                owner = read_lock_owner(self.path)
                # Attempt reclaim on every iteration: a holder killed mid-wait
                # leaves its payload unchanged, so reclaim cannot be gated on
                # observing a new owner. The pid probe is cheap and the O_EXCL
                # guard file already serializes concurrent reclaimers.
                if _reclaim_stale_lock(self.path):
                    continue
                if self.timeout_seconds <= 0 or time.monotonic() >= deadline:
                    detail = (
                        f" (held by pid={owner.pid} host={owner.host or '?'})"
                        if owner
                        else ""
                    )
                    raise RuntimeError(
                        f"lock already held: {self.path}{detail}"
                    ) from None
                time.sleep(self.poll_seconds)
                continue
            try:
                os.write(fd, _format_lock_payload(owner_payload).encode())
            except OSError:
                # Never leave an ownerless lock behind: it could not be
                # attributed to a pid and so could never be reclaimed.
                os.close(fd)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                raise
            self._fd = fd
            self._owner = read_lock_owner(self.path)
            self._start_heartbeat()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop_heartbeat_thread()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._owner = None

    def _owner_payload(self) -> dict[str, str]:
        now = _utc_now()
        payload: dict[str, Any] = {
            "pid": str(os.getpid()),
            "host": socket.gethostname(),
            "token": uuid.uuid4().hex,
            "created_at": now,
            "heartbeat_at": now,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID")
            or os.environ.get("SLURM_STEPID"),
        }
        payload.update(self.metadata)
        return _string_metadata(payload)

    def _start_heartbeat(self) -> None:
        if self.heartbeat_seconds <= 0:
            return
        stop = threading.Event()
        self._stop_heartbeat = stop
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(stop,),
            daemon=True,
            name=f"drift-lock-heartbeat:{self.path.name}",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            self._touch_heartbeat()

    def _touch_heartbeat(self) -> None:
        owner = self._owner
        if owner is None:
            return
        current = read_lock_owner(self.path)
        if current is None or not _same_lock_owner(current, owner):
            return
        payload = read_lock_payload(self.path)
        if not payload:
            return
        payload["heartbeat_at"] = _utc_now()
        _write_lock_payload_atomic(self.path, payload)

    def _stop_heartbeat_thread(self) -> None:
        if self._stop_heartbeat is not None:
            self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        self._stop_heartbeat = None
        self._heartbeat_thread = None


def stage_lock(
    run_dir: Path,
    stage: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> FileLock:
    """Return the lock for a stage inside a run directory."""
    payload = {"run_dir": str(run_dir), "stage": stage}
    if metadata is not None:
        payload.update(metadata)
    return FileLock(run_dir / ".locks" / f"{stage}.lock", metadata=payload)


def remove_lock(path: Path) -> bool:
    """Remove one lock file if it still exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _same_lock_owner(current: LockOwner, expected: LockOwner) -> bool:
    if current.pid != expected.pid or current.host != expected.host:
        return False
    if current.token is None or expected.token is None:
        return True
    return current.token == expected.token


def _format_lock_payload(payload: Mapping[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in sorted(payload.items()))


def _write_lock_payload_atomic(path: Path, payload: Mapping[str, str]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(_format_lock_payload(payload))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _string_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in metadata.items()
        if value is not None and str(value) != ""
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
