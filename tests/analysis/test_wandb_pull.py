"""Tests for synchronizing W&B analysis artifacts."""

from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

from drift_happens.analysis.export.wandb_pull import _MATRIX, pull_drift_matrices


class FakeEntry:
    """Small stand-in for a W&B artifact entry."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.digest = _wandb_digest(payload)
        self.calls: list[tuple[str | None, bool | None]] = []

    def download(self, root: str | None = None, skip_cache: bool | None = None) -> str:
        """Write the entry payload into the requested root."""
        self.calls.append((root, skip_cache))
        if root is None:
            raise ValueError("root is required")
        path = Path(root) / _MATRIX
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.payload)
        return str(path)


class FakeArtifact:
    """Small stand-in for a W&B artifact."""

    def __init__(
        self,
        entry: FakeEntry,
        *,
        aliases: tuple[str, ...] = (),
        type: str = "run",
        version: str | None = None,
    ) -> None:
        self.entry = entry
        self.aliases = aliases
        self.type = type
        self.version = version

    def get_entry(self, name: str) -> FakeEntry:
        """Return the requested fake artifact entry."""
        if name != _MATRIX:
            raise KeyError(name)
        return self.entry


class FakeRun:
    """Small stand-in for a W&B run."""

    def __init__(self, artifacts: list[FakeArtifact]) -> None:
        self.name = "yearbook__cnn_s__seed=0__eval"
        self.group = "yearbook-conference"
        self.artifacts = artifacts

    def logged_artifacts(self) -> list[FakeArtifact]:
        """Return logged artifacts in the configured order."""
        return self.artifacts


def test_pull_refreshes_existing_matrix_from_latest_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Existing local matrices are overwritten with the latest W&B entry."""
    stale_entry = FakeEntry('{"source": "stale"}\n')
    latest_entry = FakeEntry('{"source": "latest"}\n')
    run = FakeRun(
        [
            FakeArtifact(stale_entry),
            FakeArtifact(latest_entry, aliases=("latest",)),
        ]
    )
    _install_fake_wandb(monkeypatch, [run])

    matrix_path = tmp_path / run.name / _MATRIX
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text('{"source": "local"}\n')

    pulled = pull_drift_matrices(project="project", runs_root=tmp_path)

    assert pulled == [run.name]
    assert matrix_path.read_text() == latest_entry.payload
    assert stale_entry.calls == []
    assert latest_entry.calls == [(str(tmp_path / run.name), True)]


def test_pull_skips_download_for_already_current_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Already-current local matrices are left untouched and not counted."""
    entry = FakeEntry('{"source": "latest"}\n')
    run = FakeRun([FakeArtifact(entry, aliases=("latest",))])
    _install_fake_wandb(monkeypatch, [run])

    matrix_path = tmp_path / run.name / _MATRIX
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(entry.payload)

    pulled = pull_drift_matrices(project="project", runs_root=tmp_path)

    assert pulled == []
    assert matrix_path.read_text() == entry.payload
    assert entry.calls == []


def test_pull_falls_back_to_highest_version_without_latest_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Without a 'latest' alias, the highest-version artifact is selected."""
    old_entry = FakeEntry('{"source": "v0"}\n')
    new_entry = FakeEntry('{"source": "v1"}\n')
    run = FakeRun(
        [
            FakeArtifact(new_entry, version="v1"),
            FakeArtifact(old_entry, version="v0"),
        ]
    )
    _install_fake_wandb(monkeypatch, [run])

    pulled = pull_drift_matrices(project="project", runs_root=tmp_path)

    assert pulled == [run.name]
    assert (tmp_path / run.name / _MATRIX).read_text() == new_entry.payload
    assert old_entry.calls == []
    assert new_entry.calls == [(str(tmp_path / run.name), True)]


def _install_fake_wandb(monkeypatch, runs: list[FakeRun]) -> None:
    """Install a fake wandb module returning the provided runs."""
    fake_api = SimpleNamespace(runs=lambda project, filters: runs)
    fake_wandb = SimpleNamespace(Api=lambda: fake_api)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)


def _wandb_digest(payload: str) -> str:
    """Return W&B's base64-encoded MD5 digest representation."""
    hasher = hashlib.md5(usedforsecurity=False)
    hasher.update(payload.encode())
    return base64.b64encode(hasher.digest()).decode("ascii")
