"""Local artifact listing and safe cleanup helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from drift_happens.configs import RunIdentity
from drift_happens.runtime.stage_status import inspect_run_status
from drift_happens.runtime.stages import read_json_object
from drift_happens.utils import paths
from drift_happens.utils.wandb_completion import WandbApiFactory, WandbCompletionIndex

ArtifactKind = Literal["runs", "sweeps", "all"]


@dataclass(frozen=True, slots=True)
class ArtifactRow:
    """One inspectable local artifact row."""

    kind: str
    path: Path
    status: str
    dataset: str | None = None
    trainer: str | None = None
    experiment: str | None = None
    seed: int | None = None
    source_identity: str | None = None
    config_hash: str | None = None
    train: str | None = None
    eval: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class DeletionPlanItem:
    """A path selected for safe deletion."""

    path: Path
    reason: str


def list_artifacts(
    *,
    root: Path | None = None,
    kind: ArtifactKind = "all",
    status: str | None = None,
) -> tuple[ArtifactRow, ...]:
    """List canonical run and sweep artifacts, tolerating malformed directories."""
    artifact_root = root or paths.ARTIFACTS_DIR
    rows: list[ArtifactRow] = []
    if kind in {"all", "runs"}:
        rows.extend(_list_runs(artifact_root / "runs"))
    if kind in {"all", "sweeps"}:
        rows.extend(_list_sweeps(artifact_root / "sweeps"))
    if status is not None:
        rows = [row for row in rows if row.status == status]
    return tuple(sorted(rows, key=lambda row: str(row.path)))


def plan_gc(
    *,
    root: Path | None = None,
    keep_attempts: int = 3,
) -> tuple[DeletionPlanItem, ...]:
    """Plan deletion of old attempt directories without deleting canonical runs."""
    if keep_attempts < 0:
        raise ValueError("keep_attempts must be >= 0")
    artifact_root = root or paths.ARTIFACTS_DIR
    items: list[DeletionPlanItem] = []
    for attempts_root in sorted((artifact_root / "runs").rglob("attempts")):
        for stage_root in (attempts_root / "train", attempts_root / "eval"):
            if not stage_root.exists():
                continue
            attempts = sorted(
                [path for path in stage_root.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            )
            for old in attempts[keep_attempts:]:
                items.append(DeletionPlanItem(path=old, reason="old stage attempt"))
    sweeps_root = artifact_root / "sweeps"
    sweeps_by_name: dict[str, list[Path]] = {}
    sweep_attempts = sorted(sweeps_root.iterdir()) if sweeps_root.exists() else ()
    for sweep_root in sweep_attempts:
        if sweep_root.is_dir() and _is_attempt_dir(sweep_root):
            name = sweep_root.name.split("__", 1)[1]
            sweeps_by_name.setdefault(name, []).append(sweep_root)
    for grouped in sweeps_by_name.values():
        ordered = sorted(grouped, key=lambda path: path.name, reverse=True)
        for old in ordered[keep_attempts:]:
            items.append(DeletionPlanItem(path=old, reason="old sweep attempt"))
    return tuple(items)


def plan_wandb_deleted_run_gc(
    *,
    project: str,
    entity: str | None = None,
    root: Path | None = None,
    api_factory: WandbApiFactory | None = None,
) -> tuple[DeletionPlanItem, ...]:
    """
    Plan deletion of local run dirs whose local W&B runs are gone remotely.

    A local run is eligible only when it has at least one online W&B run directory under
    ``run_dir/wandb``. It is kept if W&B still has one of those exact run ids, or if W&B
    has another run matching the persisted local run identity.
    """
    artifact_root = root or paths.ARTIFACTS_DIR
    index = WandbCompletionIndex(
        project=project,
        entity=entity,
        **({"api_factory": api_factory} if api_factory is not None else {}),
    )
    items: list[DeletionPlanItem] = []
    for manifest_path in sorted((artifact_root / "runs").rglob("run_manifest.json")):
        manifest = read_json_object(manifest_path)
        identity_payload = manifest.get("identity")
        seed = _as_int(manifest.get("seed"))
        if not isinstance(identity_payload, dict) or seed is None:
            continue
        try:
            identity = RunIdentity.model_validate(identity_payload)
        except Exception:
            continue

        run_dir = manifest_path.parent
        local_run_ids = _local_wandb_run_ids(run_dir)
        if not local_run_ids:
            continue

        remote_run_ids = set(index.run_ids(group=identity.wandb_group, seed=seed))
        if remote_run_ids.intersection(local_run_ids):
            continue

        matching_ids = index.matching_run_ids(
            group=identity.wandb_group,
            seed=seed,
            config_hash=identity.config_hash,
            snapshot_sha256=identity.snapshot_sha256,
            completion_hash=identity.completion_hash,
        )
        if matching_ids:
            continue

        local_ids = ",".join(local_run_ids)
        items.append(
            DeletionPlanItem(
                path=run_dir,
                reason=f"deleted wandb run ids={local_ids}",
            )
        )
    return tuple(items)


def apply_gc(
    items: tuple[DeletionPlanItem, ...],
    *,
    root: Path | None = None,
) -> tuple[Path, ...]:
    """Delete planned paths after verifying they are inside the artifact root."""
    artifact_root = (root or paths.ARTIFACTS_DIR).resolve()
    deleted: list[Path] = []
    for item in items:
        resolved = item.path.resolve()
        if not resolved.is_relative_to(artifact_root):
            raise ValueError(f"refusing to delete outside artifact root: {resolved}")
        if item.path.is_symlink():
            item.path.unlink()  # the link itself, not its target
            deleted.append(item.path)
        elif resolved.is_dir():
            shutil.rmtree(resolved)
            deleted.append(item.path)
        elif resolved.exists():
            resolved.unlink()
            deleted.append(item.path)
    return tuple(deleted)


def _list_runs(runs_root: Path) -> list[ArtifactRow]:
    if not runs_root.exists():
        return []
    rows: list[ArtifactRow] = []
    for manifest_path in runs_root.rglob("run_manifest.json"):
        manifest = read_json_object(manifest_path)
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            rows.append(
                ArtifactRow(kind="runs", path=manifest_path.parent, status="partial")
            )
            continue
        seed = _as_int(manifest.get("seed"))
        try:
            status = inspect_run_status(
                run_dir=manifest_path.parent,
                identity=RunIdentity.model_validate(identity),
                seed=seed if seed is not None else -1,
            )
            row_status = status.run
            train = status.train
            eval_status = status.eval
        except Exception:
            row_status = "partial"
            train = None
            eval_status = None
        rows.append(
            ArtifactRow(
                kind="runs",
                path=manifest_path.parent,
                status=row_status,
                dataset=_as_str(manifest.get("dataset")),
                trainer=_as_str(manifest.get("trainer")),
                experiment=_as_str(manifest.get("experiment")),
                seed=seed,
                source_identity=_as_str(identity.get("source_identity")),
                config_hash=_as_str(identity.get("config_hash")),
                train=train,
                eval=eval_status,
                updated_at=_updated_at(manifest_path.parent),
            )
        )
    return rows


def _local_wandb_run_ids(run_dir: Path) -> tuple[str, ...]:
    wandb_root = run_dir / "wandb"
    if not wandb_root.exists():
        return ()
    run_ids: set[str] = set()
    for child in wandb_root.iterdir():
        if not child.is_dir() or child.name.startswith("offline-run-"):
            continue
        child_ids: set[str] = set()
        for wandb_file in child.glob("run-*.wandb"):
            run_id = wandb_file.stem.removeprefix("run-")
            if run_id:
                child_ids.add(run_id)
        if not child_ids and child.name.startswith("run-"):
            run_id = child.name.rsplit("-", 1)[-1]
            if run_id:
                child_ids.add(run_id)
        run_ids.update(child_ids)
    return tuple(sorted(run_ids))


def _list_sweeps(sweeps_root: Path) -> list[ArtifactRow]:
    if not sweeps_root.exists():
        return []
    rows: list[ArtifactRow] = []
    for manifest_path in sweeps_root.rglob("manifest.json"):
        results = _read_json(manifest_path.parent / "results.json")
        status = "partial"
        if isinstance(results, list):
            statuses = {
                item.get("status")
                for item in results
                if isinstance(item, dict) and item.get("status")
            }
            status = "failed" if "failed" in statuses else "ok"
        rows.append(
            ArtifactRow(
                kind="sweeps",
                path=manifest_path.parent,
                status=status,
                experiment=_as_str(read_json_object(manifest_path).get("name")),
                updated_at=_updated_at(manifest_path.parent),
            )
        )
    return rows


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _updated_at(path: Path) -> str | None:
    try:
        return str(int(path.stat().st_mtime))
    except OSError:
        return None


def _is_attempt_dir(path: Path) -> bool:
    return "__" in path.name


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
