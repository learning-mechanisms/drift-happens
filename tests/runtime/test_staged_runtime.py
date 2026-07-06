from __future__ import annotations

import json
import os
import socket
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from drift_happens.configs import WandbConfig
from drift_happens.experiments.registry import preset
from drift_happens.runtime.base import TaskResult
from drift_happens.runtime.local import _warn_environment_drift, run_stage
from drift_happens.runtime.locks import stage_lock
from drift_happens.runtime.run_store import resolve_run_store
from drift_happens.runtime.stage_status import inspect_run_status
from drift_happens.utils.git import read_git_state
from drift_happens.utils.lockfile import pixi_lock_sha256


def test_train_and_eval_stages_share_stable_run_dir(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"

    train = run_stage(cfg, stage="train", runs_root=runs_root)
    eval_result = run_stage(cfg, stage="eval", runs_root=runs_root)

    assert train.run_dir == eval_result.run_dir
    assert (train.run_dir / "stages" / "train" / "completion.json").exists()
    assert (train.run_dir / "stages" / "eval" / "completion.json").exists()
    status = inspect_run_status(
        run_dir=train.run_dir,
        identity=resolve_run_store(cfg, runs_root=runs_root).identity,
        seed=cfg.seed,
    )
    assert (status.train, status.eval, status.run) == ("ok", "ok", "ok")


def test_eval_stage_is_blocked_until_train_completes(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()

    with pytest.raises(RuntimeError, match="requires a completed train stage"):
        run_stage(cfg, stage="eval", runs_root=tmp_artifacts / "runs")


def test_train_stage_resume_skips_completed_stage(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"

    first = run_stage(cfg, stage="train", runs_root=runs_root)
    completion_path = first.run_dir / "stages" / "train" / "completion.json"
    first_completion = json.loads(completion_path.read_text())
    second = run_stage(cfg, stage="train", runs_root=runs_root, resume=True)
    second_completion = json.loads(completion_path.read_text())

    assert second.run_dir == first.run_dir
    assert second.iterations == cfg.trainer.training["num_epochs"]
    assert second_completion == first_completion


def test_stage_completion_records_environment(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"

    result = run_stage(cfg, stage="train", runs_root=runs_root)
    completion = json.loads(
        (result.run_dir / "stages" / "train" / "completion.json").read_text()
    )

    expected_sha = pixi_lock_sha256()
    expected_commit = read_git_state().commit
    assert expected_sha is not None, (
        "pixi.lock missing; cannot verify lockfile recording"
    )
    assert expected_commit != "unknown", (
        "git unavailable; cannot verify commit recording"
    )
    assert completion["lockfile_sha256"] == expected_sha
    assert completion["git_commit"] == expected_commit


def test_resume_under_changed_lockfile_warns_and_still_skips(
    tmp_artifacts, monkeypatch
) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    first = run_stage(cfg, stage="train", runs_root=runs_root)

    monkeypatch.setattr(
        "drift_happens.runtime.local.pixi_lock_sha256", lambda *a, **k: "changed" * 8
    )
    # No lock is held, so the resume takes the pre-lock fast path.
    with capture_logs() as logs:
        second = run_stage(cfg, stage="train", runs_root=runs_root, resume=True)

    # Drift must be surfaced whether or not a lock is held.
    assert any(e["event"] == "resume_environment_drift" for e in logs)
    # A changed lockfile must not force re-running completed work.
    assert second.run_dir == first.run_dir
    assert second.iterations == first.iterations


def test_attempt_metadata_preserves_each_attempt_environment(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    first = run_stage(cfg, stage="train", runs_root=runs_root)

    attempt_metas = list((first.run_dir / "attempts" / "train").glob("*/metadata.json"))
    assert len(attempt_metas) == 1
    payload = json.loads(attempt_metas[0].read_text())
    expected_sha = pixi_lock_sha256()
    expected_commit = read_git_state().commit
    assert expected_sha is not None, (
        "pixi.lock missing; cannot verify lockfile recording"
    )
    assert expected_commit != "unknown", (
        "git unavailable; cannot verify commit recording"
    )
    assert payload["lockfile"]["pixi_lock_sha256"] == expected_sha
    assert payload["git"]["commit"] == expected_commit
    assert payload["exit_status"] == "ok"
    snapshot = attempt_metas[0].read_text()

    # Resume skips (only one stage complete), so the attempt directory is untouched.
    run_stage(cfg, stage="train", runs_root=runs_root, resume=True)
    assert (
        list((first.run_dir / "attempts" / "train").glob("*/metadata.json"))
        == attempt_metas
    )
    assert attempt_metas[0].read_text() == snapshot


def test_warn_environment_drift_only_fires_on_real_change(monkeypatch) -> None:
    git = SimpleNamespace(commit="c" * 40)
    monkeypatch.setattr(
        "drift_happens.runtime.local.pixi_lock_sha256", lambda *a, **k: "current"
    )

    with capture_logs() as drift_logs:
        _warn_environment_drift(
            {"lockfile_sha256": "stale", "git_commit": "c" * 40}, git, "train"
        )
    assert any(e["event"] == "resume_environment_drift" for e in drift_logs)

    with capture_logs() as same_logs:
        _warn_environment_drift(
            {"lockfile_sha256": "current", "git_commit": "c" * 40}, git, "train"
        )
    assert not any(e["event"] == "resume_environment_drift" for e in same_logs)

    with capture_logs() as old_marker_logs:
        _warn_environment_drift({}, git, "train")  # pre-fix marker, no env recorded
    assert not any(e["event"] == "resume_environment_drift" for e in old_marker_logs)

    monkeypatch.setattr(
        "drift_happens.runtime.local.pixi_lock_sha256", lambda *a, **k: None
    )
    with capture_logs() as deleted_lock_logs:
        _warn_environment_drift(
            {"lockfile_sha256": "was-present", "git_commit": "c" * 40}, git, "train"
        )
    assert any(e["event"] == "resume_environment_drift" for e in deleted_lock_logs)


def test_run_stage_failure_writes_error_completion_and_metadata(
    tmp_artifacts, monkeypatch
) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    store = resolve_run_store(cfg, runs_root=runs_root)

    def fail_task(*args, **kwargs):
        raise RuntimeError("stage exploded")

    monkeypatch.setattr("drift_happens.runtime.local._run_stage_task", fail_task)

    with pytest.raises(RuntimeError, match="stage exploded"):
        run_stage(cfg, stage="train", runs_root=runs_root)

    completion = json.loads(
        (store.run_dir / "stages" / "train" / "completion.json").read_text()
    )
    metadata = json.loads(
        (store.run_dir / "stages" / "train" / "metadata.json").read_text()
    )
    assert completion["exit_status"] == "error"
    assert metadata["exit_status"] == "error"
    assert not (store.run_dir / ".locks" / "train.lock").exists()


def test_run_stage_failure_finishes_wandb_with_error_exit_code(
    tmp_artifacts,
    monkeypatch,
) -> None:
    cfg_without_wandb = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    run_stage(cfg_without_wandb, stage="train", runs_root=runs_root)
    cfg = cfg_without_wandb.model_copy(
        update={
            "logging": cfg_without_wandb.logging.model_copy(
                update={
                    "wandb": WandbConfig(
                        project="proj",
                        mode="offline",
                        upload_artifacts=False,
                    )
                }
            )
        }
    )
    logs: list[dict] = []
    exit_codes: list[int | None] = []

    class FakeRun:
        def log(self, payload):
            logs.append(payload)

        def finish(self, exit_code=None, quiet=None):
            exit_codes.append(exit_code)

    fake_run = FakeRun()
    fake_wandb = SimpleNamespace(
        define_metric=lambda *args, **kwargs: None,
        init=lambda **kwargs: fake_run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def fail_task(*args, **kwargs):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr("drift_happens.runtime.local._run_stage_task", fail_task)

    with pytest.raises(RuntimeError, match="eval exploded"):
        run_stage(cfg, stage="eval", runs_root=runs_root)

    assert exit_codes == [1]
    assert any(
        payload.get("stage/complete") == 0.0
        and payload.get("stage/exit_status") == "error"
        and payload.get("run/exit_status") == "error"
        and payload.get("run/stage") == "eval"
        for payload in logs
    )


def test_run_stage_resume_false_clears_previous_stage_outputs(
    tmp_artifacts, monkeypatch
) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    store = resolve_run_store(cfg, runs_root=runs_root)
    old_stage_file = store.run_dir / "stages" / "train" / "old.txt"
    old_metric_file = store.run_dir / "metrics" / "train.jsonl"
    old_stage_file.parent.mkdir(parents=True)
    old_metric_file.parent.mkdir(parents=True)
    old_stage_file.write_text("old")
    old_metric_file.write_text("old")
    monkeypatch.setattr(
        "drift_happens.runtime.local._run_stage_task",
        lambda *args, **kwargs: TaskResult(iterations=2, metrics={"loss": 1.0}),
    )

    result = run_stage(cfg, stage="train", runs_root=runs_root, resume=False)

    assert result.iterations == 2
    assert not old_stage_file.exists()
    assert not old_metric_file.exists()
    assert (store.run_dir / "stages" / "train" / "completion.json").exists()


def test_eval_allow_overwrite_clears_only_eval_outputs(
    tmp_artifacts, monkeypatch
) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    monkeypatch.setattr(
        "drift_happens.runtime.local._run_stage_task",
        lambda *args, **kwargs: TaskResult(iterations=1, metrics={}),
    )

    train = run_stage(cfg, stage="train", runs_root=runs_root)
    run_stage(cfg, stage="eval", runs_root=runs_root)
    train_artifact = train.run_dir / "stages" / "train" / "artifact.txt"
    eval_stale = train.run_dir / "stages" / "eval" / "stale.txt"
    train_artifact.write_text("keep")
    eval_stale.write_text("drop")

    run_stage(cfg, stage="eval", runs_root=runs_root, allow_overwrite=True)

    assert train_artifact.read_text() == "keep"
    assert not eval_stale.exists()
    assert (train.run_dir / "stages" / "eval" / "completion.json").exists()


def test_ensure_base_overwrite_preserves_locks_dir_and_held_lock(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    store = resolve_run_store(cfg, runs_root=tmp_artifacts / "runs")
    store.ensure_base()
    stale = store.run_dir / "stages" / "train" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")

    with stage_lock(store.run_dir, "train"):
        store.ensure_base(allow_overwrite=True)
        lock_path = store.run_dir / ".locks" / "train.lock"
        # Overwriting must never remove the directory holding the live lock.
        assert f"pid={os.getpid()}" in lock_path.read_text()

    assert not stale.exists()


def test_contended_stage_lock_blocks_destructive_clearing(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    store = resolve_run_store(cfg, runs_root=runs_root)
    store.ensure_base()
    old_stage_file = store.run_dir / "stages" / "train" / "old.txt"
    old_stage_file.parent.mkdir(parents=True)
    old_stage_file.write_text("old")
    lock_path = store.run_dir / ".locks" / "train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A live local owner is never reclaimed, so acquisition fails immediately.
    lock_path.write_text(f"pid={os.getpid()}\nhost={socket.gethostname()}\n")

    with pytest.raises(RuntimeError, match="lock already held"):
        run_stage(cfg, stage="train", runs_root=runs_root, resume=False)
    with pytest.raises(RuntimeError, match="lock already held"):
        run_stage(cfg, stage="train", runs_root=runs_root, allow_overwrite=True)

    # Neither --no-resume clearing nor the overwrite ran before acquisition.
    assert old_stage_file.read_text() == "old"


def test_train_clearing_blocks_on_a_live_eval_lock(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"
    store = resolve_run_store(cfg, runs_root=runs_root)
    store.ensure_base()
    eval_artifact = store.run_dir / "stages" / "eval" / "predictions.txt"
    eval_artifact.parent.mkdir(parents=True)
    eval_artifact.write_text("live")
    eval_lock = store.run_dir / ".locks" / "eval.lock"
    eval_lock.parent.mkdir(parents=True, exist_ok=True)
    # A live local owner is never reclaimed, so the nested eval acquisition
    # fails even though the train lock itself is free.
    eval_lock.write_text(f"pid={os.getpid()}\nhost={socket.gethostname()}\n")

    with pytest.raises(RuntimeError, match="lock already held"):
        run_stage(cfg, stage="train", runs_root=runs_root, resume=False)
    with pytest.raises(RuntimeError, match="lock already held"):
        run_stage(cfg, stage="train", runs_root=runs_root, allow_overwrite=True)

    # Train cleanup wipes eval outputs too, so it must not start while a live
    # eval runner holds its stage lock.
    assert eval_artifact.read_text() == "live"
    assert not (store.run_dir / ".locks" / "train.lock").exists()


def test_resume_with_missing_train_artifact_fails_downstream(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    runs_root = tmp_artifacts / "runs"

    train = run_stage(cfg, stage="train", runs_root=runs_root)
    for checkpoint in train.run_dir.rglob("final.pt"):
        checkpoint.unlink()

    # The completion marker still claims success, so resume skips the rerun and
    # the checkpoint stays missing.
    resumed = run_stage(cfg, stage="train", runs_root=runs_root, resume=True)
    assert resumed.run_dir == train.run_dir
    assert not list(train.run_dir.rglob("final.pt"))

    # The downstream eval stage then refuses to proceed rather than evaluating a
    # model that no longer exists.
    with pytest.raises(FileNotFoundError, match="requires train checkpoint"):
        run_stage(cfg, stage="eval", runs_root=runs_root, resume=True)


def test_stage_run_identity_honors_user_run_name_override(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    cfg = cfg.model_copy(
        update={
            "logging": cfg.logging.model_copy(
                update={"wandb": WandbConfig(project="p", run_name="my-run")}
            )
        }
    )
    store = resolve_run_store(cfg, runs_root=tmp_artifacts / "runs")

    name = store.stage_run_identity("train").wandb_run_name

    assert name == f"my-run__seed={cfg.seed}__train"


def test_stage_run_identity_without_override_uses_group(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    store = resolve_run_store(cfg, runs_root=tmp_artifacts / "runs")

    name = store.stage_run_identity("eval").wandb_run_name

    assert name == f"{store.identity.wandb_group}__seed={cfg.seed}__eval"


def test_attempt_dir_is_unique_within_the_same_second(tmp_artifacts) -> None:
    cfg = preset("smoke", "synthetic-classification-cpu").build()
    store = resolve_run_store(cfg, runs_root=tmp_artifacts / "runs")
    git = read_git_state()
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    first = store.attempt_dir(stage="train", started_at=started_at, git=git)
    second = store.attempt_dir(stage="train", started_at=started_at, git=git)

    assert first != second
    assert first.exists() and second.exists()
    # The bare suffix sorts before the disambiguated one, so the GC's
    # lexicographic newest-first ordering stays chronological.
    assert sorted([first.name, second.name]) == [first.name, second.name]
