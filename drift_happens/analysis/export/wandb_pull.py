"""Pull drift matrix artifacts from W&B into the local runs directory."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from drift_happens.utils.paths import RUNS_DIR

WANDB_PROJECT = "drift-happens/drift-happens"
_MATRIX = "results/drift_matrix.json"


def pull_drift_matrices(
    project: str = WANDB_PROJECT, runs_root: Path = RUNS_DIR
) -> list[str]:
    """
    Fetch finished conference eval drift matrices into runs_root/<run>/.

    Existing local matrices are checked against the W&B artifact entry digest and
    refreshed when the remote representation differs. Returns the run names whose local
    matrix was missing or changed before synchronization.
    """
    import wandb

    api = wandb.Api()
    pulled: list[str] = []
    for run in api.runs(project, filters={"state": "finished"}):
        if "conference" not in (run.group or "") or not run.name.endswith("__eval"):
            continue
        artifact = _latest_run_artifact(run)
        if artifact is None:
            continue
        try:
            entry = _artifact_entry(artifact)
        except (KeyError, ValueError):
            continue
        run_root = runs_root / run.name
        matrix_path = run_root / _MATRIX
        if _local_matches_entry(matrix_path, entry):
            continue
        entry.download(root=str(run_root), skip_cache=True)
        pulled.append(run.name)
    return pulled


def _latest_run_artifact(run: Any) -> Any | None:
    """Return the run artifact representing W&B's latest logged version."""
    artifacts = [
        artifact
        for artifact in run.logged_artifacts()
        if getattr(artifact, "type", None) == "run"
    ]
    if not artifacts:
        return None
    aliased = next(
        (artifact for artifact in artifacts if "latest" in _artifact_aliases(artifact)),
        None,
    )
    if aliased is not None:
        return aliased
    return max(artifacts, key=_artifact_version)


def _artifact_version(artifact: Any) -> int:
    """Return the numeric W&B version index, or -1 when unavailable."""
    version = getattr(artifact, "version", None)
    if isinstance(version, str) and version.startswith("v") and version[1:].isdigit():
        return int(version[1:])
    return -1


def _artifact_aliases(artifact: Any) -> tuple[str, ...]:
    """Return W&B artifact aliases as plain strings."""
    aliases = getattr(artifact, "aliases", ())
    if aliases is None:
        return ()
    return tuple(str(alias) for alias in aliases)


def _artifact_entry(artifact: Any) -> Any:
    """Return the drift matrix entry using the non-deprecated W&B API."""
    get_entry = getattr(artifact, "get_entry", None)
    if get_entry is not None:
        return get_entry(_MATRIX)
    return artifact.get_path(_MATRIX)


def _local_matches_entry(path: Path, entry: Any) -> bool:
    """Return whether path already matches the W&B artifact entry digest."""
    digest = getattr(entry, "digest", None)
    if digest is None or not path.is_file():
        return False
    return _file_md5_b64(path) == digest


def _file_md5_b64(path: Path) -> str:
    """Return W&B's base64-encoded MD5 digest representation for a local file."""
    hasher = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return base64.b64encode(hasher.digest()).decode("ascii")
