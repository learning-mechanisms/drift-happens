from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from drift_happens.configs import ExperimentConfig
from drift_happens.utils.git import GitState
from drift_happens.utils.snapshot import (
    SNAPSHOT_KIND,
    _cpu_model,
    _memory_total_bytes,
    build_metadata,
    build_preset_snapshot,
    experiment_from_preset_snapshot,
    finalise_metadata,
    load_preset_snapshot,
    write_metadata,
    write_preset_snapshot,
    write_snapshot,
)


def _minimal_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "yearbook-smoke",
            "seed": 7,
            "dataset": {"name": "yearbook"},
            "trainer": {
                "key": "unit-trainer",
                "training": {"batch_size": 64, "learning_rate": 0.001},
            },
        }
    )


def test_write_snapshot_persists_deterministic_experiment_config(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config()
    path = tmp_path / "snapshot.json"

    write_snapshot(path, cfg)

    assert json.loads(path.read_text()) == json.loads(cfg.to_snapshot_json())


def test_metadata_lifecycle_serializes_run_reproducibility_state(
    tmp_path: Path,
) -> None:
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    source_git = GitState(
        commit="abcdef0123456789",
        short_commit="abcdef0",
        branch="main",
        dirty=True,
        diff_truncated="diff --git a/file b/file",
    )

    metadata = build_metadata(
        seed=7,
        started_at=started_at,
        source_git=source_git,
        effective_device="cpu",
    )
    finished = finalise_metadata(
        metadata,
        exit_status="success",
        error_message=None,
        last_completed_iteration=12,
        ended_at=started_at + timedelta(seconds=2.5),
    )
    path = tmp_path / "metadata.json"

    write_metadata(path, finished)
    payload = json.loads(path.read_text())

    assert payload["seed"] == 7
    assert payload["exit_status"] == "success"
    assert payload["wall_seconds"] == 2.5
    assert payload["last_completed_iteration"] == 12
    assert payload["git"]["commit"] == "abcdef0123456789"
    assert payload["git"]["dirty"] is True
    assert payload["host"]["effective_device"] == "cpu"
    assert "pixi_lock_sha256" in payload["lockfile"]


def test_preset_snapshot_round_trips_config_without_volatile_metadata(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config()
    snapshot = build_preset_snapshot(
        cfg=cfg,
        group="yearbook",
        name="smoke",
        seeds=(0, 1),
    )
    path = tmp_path / "preset.json"

    write_preset_snapshot(path, snapshot)
    loaded = load_preset_snapshot(path)
    restored = experiment_from_preset_snapshot(loaded)

    assert loaded["kind"] == SNAPSHOT_KIND
    assert loaded["seeds"] == [0, 1]
    assert "git" not in loaded
    assert "host" not in loaded
    assert restored == cfg


def test_host_info_fallbacks_when_platform_and_sysconf_are_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr("drift_happens.utils.snapshot.platform.processor", lambda: "")
    monkeypatch.setattr("drift_happens.utils.snapshot.Path.exists", lambda self: False)
    monkeypatch.setattr(
        "drift_happens.utils.snapshot.os.sysconf",
        lambda name: (_ for _ in ()).throw(ValueError(name)),
    )

    assert _cpu_model() is None
    assert _memory_total_bytes() is None


def test_write_metadata_failed_rename_keeps_prior_file(tmp_path, monkeypatch) -> None:
    meta = build_metadata(seed=1, started_at=datetime(2026, 1, 1, tzinfo=UTC))
    path = tmp_path / "metadata.json"
    write_metadata(path, meta)
    prior = path.read_bytes()

    def boom(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        write_metadata(path, meta)

    assert path.read_bytes() == prior
    assert not list(tmp_path.glob(".metadata.json.*.tmp"))
