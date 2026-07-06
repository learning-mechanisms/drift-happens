"""Git introspection used to stamp runs with reproducibility metadata."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from drift_happens.utils.paths import PROJECT_ROOT

_DIFF_TRUNCATE_CHARS = 64 * 1024


@dataclass(frozen=True)
class GitState:
    """Small, serializable view of repository state at run start."""

    commit: str
    short_commit: str
    branch: str | None
    dirty: bool
    diff_truncated: str | None


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",  # tolerate non-UTF-8 bytes in a diff
    ).stdout.strip()


def _try_run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return _run(cmd, cwd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def read_git_state(cwd: Path = PROJECT_ROOT) -> GitState:
    """
    Return commit, branch, and dirty status for ``cwd``.

    If git is unavailable or ``cwd`` is not inside a git repository, the function
    returns an explicit sentinel state. Treat that state as non-reproducible.
    """
    commit = _try_run(["git", "rev-parse", "HEAD"], cwd)
    if commit is None:
        return GitState(
            commit="unknown",
            short_commit="unknown",
            branch=None,
            dirty=True,
            diff_truncated=None,
        )

    branch_raw = _try_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    branch = branch_raw if branch_raw and branch_raw != "HEAD" else None

    status = _try_run(["git", "status", "--porcelain"], cwd)
    dirty = status is None or bool(status.strip())  # a failed status check is dirty

    diff_truncated: str | None = None
    if dirty:
        diff = _try_run(["git", "diff", "--no-color", "HEAD"], cwd) or ""
        if len(diff) > _DIFF_TRUNCATE_CHARS:
            diff_truncated = (
                diff[:_DIFF_TRUNCATE_CHARS]
                + f"\n... <truncated, {len(diff)} chars total>"
            )
        else:
            diff_truncated = diff or None

    return GitState(
        commit=commit,
        short_commit=commit[:7],
        branch=branch,
        dirty=dirty,
        diff_truncated=diff_truncated,
    )
