from __future__ import annotations

import subprocess
from pathlib import Path

from drift_happens.utils import git as git_module
from drift_happens.utils.git import _DIFF_TRUNCATE_CHARS, read_git_state


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_read_git_state_returns_sentinel_outside_git_repo(tmp_path: Path) -> None:
    state = read_git_state(tmp_path)

    assert state.commit == "unknown"
    assert state.short_commit == "unknown"
    assert state.branch is None
    assert state.dirty is True
    assert state.diff_truncated is None


def test_read_git_state_captures_clean_and_dirty_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")

    clean = read_git_state(tmp_path)
    expected_branch = _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD")

    assert clean.commit != "unknown"
    assert clean.short_commit == clean.commit[:7]
    assert clean.branch == expected_branch
    assert clean.dirty is False
    assert clean.diff_truncated is None

    tracked.write_text("changed\n")
    dirty = read_git_state(tmp_path)

    assert dirty.commit == clean.commit
    assert dirty.branch == expected_branch
    assert dirty.dirty is True
    assert dirty.diff_truncated is not None
    assert "changed" in dirty.diff_truncated


def test_read_git_state_truncates_large_diffs(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    big = tmp_path / "big.txt"
    big.write_text("x\n")
    _git(tmp_path, "add", "big.txt")
    _git(tmp_path, "commit", "-m", "init")
    # write more than the truncation threshold
    big.write_text("y" * (_DIFF_TRUNCATE_CHARS + 1024) + "\n")

    state = read_git_state(tmp_path)

    assert state.dirty is True
    assert state.diff_truncated is not None
    assert len(state.diff_truncated) <= _DIFF_TRUNCATE_CHARS + 200  # cap + marker
    assert "truncated" in state.diff_truncated
    assert "chars total" in state.diff_truncated


def test_failed_status_check_is_treated_as_dirty(monkeypatch) -> None:
    def fake_try_run(cmd, cwd):
        if cmd[:2] == ["git", "rev-parse"]:
            return "deadbeef" * 5
        if cmd[:2] == ["git", "status"]:
            return None
        return ""

    monkeypatch.setattr(git_module, "_try_run", fake_try_run)

    assert read_git_state(Path("/irrelevant")).dirty is True


def test_read_git_state_survives_non_utf8_diff(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@t.com")
    _git(tmp_path, "config", "user.name", "t")
    target = tmp_path / "latin1.txt"
    target.write_bytes(b"cafe baseline\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")
    target.write_bytes(bytes([0x63, 0x61, 0x66, 0xE9]) + b" modified\n")

    state = read_git_state(tmp_path)

    assert state.dirty is True
    assert state.diff_truncated is not None
