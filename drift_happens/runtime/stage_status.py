"""Local status inspection for canonical staged run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drift_happens.configs import RunIdentity
from drift_happens.runtime.locks import lock_owner_is_dead, read_lock_owner
from drift_happens.runtime.stages import (
    RunStage,
    StageStatus,
    read_json_object,
    stage_completion_matches,
    stage_completion_path,
    stage_dir,
)
from drift_happens.utils import paths


@dataclass(frozen=True, slots=True)
class StageStatusRow:
    """Status for one stage inside a local run directory."""

    stage: RunStage
    status: StageStatus
    run_dir: Path
    exit_status: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunStatusRow:
    """Combined train/eval status for one seed."""

    seed: int
    train: StageStatus
    eval: StageStatus
    run: StageStatus
    run_dir: Path | None = None
    exit_status: str | None = None


def local_run_statuses_by_identity(
    identities: dict[int, RunIdentity],
    *,
    runs_root: Path | None = None,
) -> tuple[RunStatusRow, ...]:
    """Scan local staged runs using a per-seed identity map."""
    root = runs_root or paths.RUNS_DIR
    rows: list[RunStatusRow] = []
    for seed, identity in identities.items():
        run_dirs = matching_run_dirs(identity, seed=seed, runs_root=root)
        rows.append(_best_status(seed, identity, run_dirs))
    return tuple(rows)


# Canonical run dirs live at ``root/dataset/trainer/name/seed=N/run_leaf``; the
# run-level manifest and metadata sit at that depth, while stage and attempt
# metadata live strictly deeper (``.../stages/<stage>`` and
# ``.../attempts/<stage>/<id>``). Globbing the run-dir depth for the requested
# seed reaches only the canonical run dirs, so a per-seed scan touches only those
# dirs rather than the whole tree.
_RUN_DIR_GLOB = "*/*/*/seed={seed}/*"


def matching_run_dirs(
    identity: RunIdentity,
    *,
    seed: int,
    runs_root: Path,
) -> tuple[Path, ...]:
    """Return run dirs whose manifest or metadata matches identity."""
    if not runs_root.exists():
        return ()
    matches: list[Path] = []
    known: set[Path] = set()
    for run_dir in runs_root.glob(_RUN_DIR_GLOB.format(seed=seed)):
        if not run_dir.is_dir() or run_dir in known:
            continue
        if _run_dir_matches(run_dir, identity, seed):
            matches.append(run_dir)
            known.add(run_dir)
    return tuple(matches)


def _run_dir_matches(run_dir: Path, identity: RunIdentity, seed: int) -> bool:
    """
    Whether a candidate run dir's manifest or metadata matches identity/seed.

    The run manifest is authoritative; metadata is consulted only as a fallback for runs
    written before the manifest.
    """
    manifest = read_json_object(run_dir / "run_manifest.json")
    if manifest:
        raw_identity = manifest.get("identity")
        if (
            isinstance(raw_identity, dict)
            and _as_int(manifest.get("seed")) == seed
            and _identity_matches(raw_identity, identity)
        ):
            return True
    metadata = read_json_object(run_dir / "metadata.json")
    if metadata:
        raw_identity = metadata.get("run_identity")
        if (
            isinstance(raw_identity, dict)
            and _as_int(metadata.get("seed")) == seed
            and _identity_matches(raw_identity, identity)
        ):
            return True
    return False


def inspect_run_status(
    *,
    run_dir: Path,
    identity: RunIdentity,
    seed: int,
) -> RunStatusRow:
    """Inspect train/eval/run state for one known run dir."""
    train = inspect_stage_status(
        run_dir=run_dir,
        identity=identity,
        seed=seed,
        stage="train",
    )
    eval_status = inspect_stage_status(
        run_dir=run_dir,
        identity=identity,
        seed=seed,
        stage="eval",
        blocked=train.status != "ok",
    )
    run_status = _combine_run_status(run_dir, train.status, eval_status.status)
    metadata = read_json_object(run_dir / "metadata.json")
    return RunStatusRow(
        seed=seed,
        train=train.status,
        eval=eval_status.status,
        run=run_status,
        run_dir=run_dir,
        exit_status=str(metadata.get("exit_status")) if metadata else None,
    )


def inspect_stage_status(
    *,
    run_dir: Path,
    identity: RunIdentity,
    seed: int,
    stage: RunStage,
    blocked: bool = False,
) -> StageStatusRow:
    """Inspect one stage completion marker and artifact directory."""
    if blocked:
        return StageStatusRow(stage=stage, status="blocked", run_dir=run_dir)
    lock_path = run_dir / ".locks" / f"{stage}.lock"
    if lock_path.exists():
        owner = read_lock_owner(lock_path)
        if owner is None or not lock_owner_is_dead(owner):
            return StageStatusRow(stage=stage, status="running", run_dir=run_dir)
        # A dead local owner means a crashed stage, not a running one; fall
        # through to the completion markers (run_stage reclaims the lock).
    completion_path = stage_completion_path(run_dir, stage)
    completion = read_json_object(completion_path)
    if completion:
        if not stage_completion_matches(
            completion,
            identity=identity,
            seed=seed,
            stage=stage,
        ):
            return StageStatusRow(
                stage=stage,
                status="failed",
                run_dir=run_dir,
                exit_status=str(completion.get("exit_status", "")),
                reason="completion identity mismatch",
            )
        exit_status = str(completion.get("exit_status"))
        if exit_status == "ok":
            return StageStatusRow(
                stage=stage,
                status="ok",
                run_dir=run_dir,
                exit_status=exit_status,
            )
        return StageStatusRow(
            stage=stage,
            status="failed",
            run_dir=run_dir,
            exit_status=exit_status,
        )
    if stage_dir(run_dir, stage).exists():
        return StageStatusRow(stage=stage, status="partial", run_dir=run_dir)
    return StageStatusRow(stage=stage, status="missing", run_dir=run_dir)


def local_run_complete(
    identity: RunIdentity,
    *,
    seed: int,
    runs_root: Path | None = None,
) -> bool:
    """Return whether local canonical state says the seed run is complete."""
    row = local_run_statuses_by_identity(
        {seed: identity},
        runs_root=runs_root,
    )[0]
    return row.run == "ok"


def _best_status(
    seed: int,
    identity: RunIdentity,
    run_dirs: tuple[Path, ...],
) -> RunStatusRow:
    if not run_dirs:
        return RunStatusRow(seed=seed, train="missing", eval="missing", run="missing")
    rows = [
        inspect_run_status(run_dir=run_dir, identity=identity, seed=seed)
        for run_dir in run_dirs
    ]
    for status in ("ok", "running", "partial", "failed", "blocked", "missing"):
        for row in rows:
            if row.run == status:
                return row
    return rows[0]


def _combine_run_status(
    run_dir: Path,
    train: StageStatus,
    eval_status: StageStatus,
) -> StageStatus:
    if train == "ok" and eval_status == "ok":
        metadata = read_json_object(run_dir / "metadata.json")
        return "ok" if metadata.get("exit_status") == "ok" else "partial"
    if train == "failed" or eval_status == "failed":
        return "failed"
    if train == "running" or eval_status == "running":
        return "running"
    if train == "ok" and eval_status in {"missing", "partial", "blocked"}:
        return "partial"
    if train == "missing" and eval_status in {"missing", "blocked"}:
        return "missing"
    return "partial"


def _identity_matches(raw_identity: dict, identity: RunIdentity) -> bool:
    if (
        identity.completion_hash is not None
        and raw_identity.get("source_identity") == identity.source_identity
        and raw_identity.get("completion_hash") == identity.completion_hash
    ):
        return True
    return (
        raw_identity.get("source_identity") == identity.source_identity
        and raw_identity.get("config_hash") == identity.config_hash
        and raw_identity.get("snapshot_sha256") == identity.snapshot_sha256
    )


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
