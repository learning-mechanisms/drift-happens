from __future__ import annotations

import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from drift_happens.configs import RunIdentity
from drift_happens.runtime.stage_status import (
    inspect_run_status,
    inspect_stage_status,
    matching_run_dirs,
)
from drift_happens.runtime.stages import RunStage, StageCompletion, write_json_atomic


def _identity(suffix: str = "") -> RunIdentity:
    return RunIdentity(
        source_identity=f"source{suffix}",
        config_hash=f"config{suffix}",
        snapshot_sha256=f"snapshot{suffix}",
        wandb_group="group",
        wandb_run_name="run",
    )


def _completion(run_dir: Path, stage: RunStage, identity: RunIdentity) -> None:
    write_json_atomic(
        run_dir / "stages" / stage / "completion.json",
        StageCompletion(
            stage=stage,
            exit_status="ok",
            seed=0,
            source_identity=identity.source_identity,
            config_hash=identity.config_hash,
            completion_hash=identity.completion_hash,
            snapshot_sha256=identity.snapshot_sha256,
            trainer_key="trainer",
            dataset_name="dataset",
            run_dir=str(run_dir),
            ended_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )


def test_inspect_stage_status_reports_running_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    lock.parent.mkdir()
    # A lock without a host line is never reclaimable regardless of pid.
    lock.write_text("pid=1")

    row = inspect_stage_status(
        run_dir=tmp_path, identity=_identity(), seed=0, stage="train"
    )

    assert row.status == "running"


def test_inspect_stage_status_reports_partial_stage_dir(tmp_path: Path) -> None:
    (tmp_path / "stages" / "train").mkdir(parents=True)

    row = inspect_stage_status(
        run_dir=tmp_path, identity=_identity(), seed=0, stage="train"
    )

    assert row.status == "partial"


def test_inspect_stage_status_reports_identity_mismatch_as_failed(
    tmp_path: Path,
) -> None:
    _completion(tmp_path, "train", _identity("other"))

    row = inspect_stage_status(
        run_dir=tmp_path, identity=_identity(), seed=0, stage="train"
    )

    assert row.status == "failed"
    assert row.reason == "completion identity mismatch"


def test_inspect_stage_status_accepts_completion_hash_match(tmp_path: Path) -> None:
    stored = _identity().model_copy(update={"completion_hash": "semantic"})
    current = RunIdentity(
        source_identity=stored.source_identity,
        config_hash="current-config",
        completion_hash=stored.completion_hash,
        snapshot_sha256="current-snapshot",
        wandb_group=stored.wandb_group,
        wandb_run_name=stored.wandb_run_name,
    )
    _completion(tmp_path, "train", stored)

    row = inspect_stage_status(
        run_dir=tmp_path, identity=current, seed=0, stage="train"
    )

    assert row.status == "ok"


def test_inspect_run_status_blocks_eval_when_train_not_ok(tmp_path: Path) -> None:
    _completion(tmp_path, "eval", _identity())

    row = inspect_run_status(run_dir=tmp_path, identity=_identity(), seed=0)

    assert row.train == "missing"
    assert row.eval == "blocked"
    assert row.run == "missing"


def _canonical_run_dir(runs_root: Path, *, seed: int, leaf: str = "leaf") -> Path:
    """A run dir at the canonical ``dataset/trainer/name/seed=N/leaf`` depth."""
    return runs_root / "dataset" / "trainer" / "name" / f"seed={seed}" / leaf


def test_matching_run_dirs_finds_manifest_and_metadata(tmp_path: Path) -> None:
    identity = _identity()
    manifest_dir = _canonical_run_dir(tmp_path, seed=0, leaf="from_manifest")
    manifest_dir.mkdir(parents=True)
    write_json_atomic(
        manifest_dir / "run_manifest.json",
        {"seed": 0, "identity": identity.model_dump(mode="json")},
    )
    metadata_dir = _canonical_run_dir(tmp_path, seed=0, leaf="from_metadata")
    metadata_dir.mkdir(parents=True)
    write_json_atomic(
        metadata_dir / "metadata.json",
        {"seed": 0, "run_identity": identity.model_dump(mode="json")},
    )

    assert set(matching_run_dirs(identity, seed=0, runs_root=tmp_path)) == {
        manifest_dir,
        metadata_dir,
    }


def test_matching_run_dirs_accepts_completion_hash_match(tmp_path: Path) -> None:
    stored = _identity().model_copy(update={"completion_hash": "semantic"})
    current = RunIdentity(
        source_identity=stored.source_identity,
        config_hash="current-config",
        completion_hash=stored.completion_hash,
        snapshot_sha256="current-snapshot",
        wandb_group=stored.wandb_group,
        wandb_run_name=stored.wandb_run_name,
    )
    run_dir = _canonical_run_dir(tmp_path, seed=0)
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "run_manifest.json",
        {"seed": 0, "identity": stored.model_dump(mode="json")},
    )

    assert matching_run_dirs(current, seed=0, runs_root=tmp_path) == (run_dir,)


def test_matching_run_dirs_skips_stage_dirs_inside_a_run(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _canonical_run_dir(tmp_path, seed=0)
    run_dir.mkdir(parents=True)
    payload = {"seed": 0, "run_identity": identity.model_dump(mode="json")}
    write_json_atomic(run_dir / "metadata.json", payload)
    # Stage-level and attempt-level metadata carry the same identity but mark
    # internal directories, not run dirs; they live deeper than the run-dir glob.
    write_json_atomic(run_dir / "stages" / "train" / "metadata.json", payload)
    write_json_atomic(run_dir / "stages" / "eval" / "metadata.json", payload)
    write_json_atomic(run_dir / "attempts" / "train" / "a1" / "metadata.json", payload)

    assert matching_run_dirs(identity, seed=0, runs_root=tmp_path) == (run_dir,)


def test_matching_run_dirs_filters_by_seed(tmp_path: Path) -> None:
    identity = _identity()
    wanted = _canonical_run_dir(tmp_path, seed=1)
    wanted.mkdir(parents=True)
    write_json_atomic(
        wanted / "run_manifest.json",
        {"seed": 1, "identity": identity.model_dump(mode="json")},
    )
    other_seed = _canonical_run_dir(tmp_path, seed=2)
    other_seed.mkdir(parents=True)
    write_json_atomic(
        other_seed / "run_manifest.json",
        {"seed": 2, "identity": identity.model_dump(mode="json")},
    )

    assert matching_run_dirs(identity, seed=1, runs_root=tmp_path) == (wanted,)


def test_inspect_stage_status_ignores_lock_of_dead_local_process(
    tmp_path: Path,
) -> None:
    lock = tmp_path / ".locks" / "train.lock"
    lock.parent.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    lock.write_text(f"pid={proc.pid}\nhost={socket.gethostname()}\n")
    (tmp_path / "stages" / "train").mkdir(parents=True)

    row = inspect_stage_status(
        run_dir=tmp_path, identity=_identity(), seed=0, stage="train"
    )

    assert row.status == "partial"


def test_write_json_atomic_cleans_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", boom)
    target = tmp_path / "completion.json"

    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic(target, {"stage": "train"})

    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_matching_run_dirs_coerces_string_seed_in_metadata(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _canonical_run_dir(tmp_path, seed=0, leaf="string_seed")
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "metadata.json",
        {"seed": "0", "run_identity": identity.model_dump(mode="json")},
    )

    assert matching_run_dirs(identity, seed=0, runs_root=tmp_path) == (run_dir,)
