from __future__ import annotations

import json
from pathlib import Path

from drift_happens.utils.artifact_remote import (
    build_filter_rules,
    build_rclone_transfer_command,
    make_remote_profile,
    read_remote_profile,
    remote_exists,
    remote_target,
    write_remote_profile,
)


def test_remote_profile_round_trip_normalizes_values(tmp_path: Path) -> None:
    profile = make_remote_profile(
        remote="pcloud:",
        path="drift-happens/artifacts/",
    )
    path = write_remote_profile(profile, tmp_path / "profile.json")

    loaded = read_remote_profile(path)

    assert loaded.remote == "pcloud"
    assert loaded.path == "/drift-happens/artifacts"
    assert loaded.roots == ("runs", "sweeps", "bundles")
    assert remote_target(loaded) == "pcloud:/drift-happens/artifacts"


def test_read_remote_profile_migrates_legacy_roots(tmp_path: Path) -> None:
    legacy = tmp_path / "profile.json"
    legacy.write_text(
        json.dumps(
            {
                "remote": "pcloud",
                "path": "/drift-happens/artifacts",
                "backend": "pcloud",
                "roots": ["runs", "sweeps"],
                "version": 1,
            }
        )
    )

    loaded = read_remote_profile(legacy)

    assert loaded.roots == ("runs", "sweeps", "bundles")
    assert loaded.version == 2


def test_build_rclone_push_command_defaults_to_safe_copy(tmp_path: Path) -> None:
    profile = make_remote_profile(remote="pcloud", path="/drift/artifacts")

    command = build_rclone_transfer_command(
        direction="push",
        profile=profile,
        artifacts_root=tmp_path / "artifacts",
        dry_run=True,
        progress=False,
        rclone_bin="rclone",
    )

    assert command[:4] == [
        "rclone",
        "copy",
        str(tmp_path / "artifacts"),
        "pcloud:/drift/artifacts",
    ]
    assert "--dry-run" in command
    assert "--progress" not in command
    assert "--filter" in command
    assert "+ /runs/**/results/**" in command
    assert "+ /sweeps/**/stages/train/**/trained_model.pt" in command
    assert "+ /bundles/**/*.tar.gz" in command
    assert "- /bundles/**/staged/**" in command
    assert "- /runs/**/attempts/**" in command


def test_build_rclone_pull_mirror_uses_sync(tmp_path: Path) -> None:
    profile = make_remote_profile(remote="pcloud", path="/drift/artifacts")

    command = build_rclone_transfer_command(
        direction="pull",
        profile=profile,
        artifacts_root=tmp_path / "artifacts",
        mirror=True,
        progress=False,
    )

    assert command[:4] == [
        "rclone",
        "sync",
        "pcloud:/drift/artifacts",
        str(tmp_path / "artifacts"),
    ]


def test_filter_rules_can_include_attempts_and_all_checkpoints() -> None:
    profile = make_remote_profile(remote="pcloud", path="/drift/artifacts")

    rules = build_filter_rules(
        profile,
        with_attempts=True,
        with_all_checkpoints=True,
    )

    assert "- /runs/**/attempts/**" not in rules
    assert "+ /runs/**/checkpoints/**" in rules
    assert rules[-1] == "- **"


def test_remote_exists_matches_rclone_listremotes_output() -> None:
    assert remote_exists("pcloud", "other:\npcloud:\n")
    assert remote_exists("pcloud:", "pcloud:\n")
    assert not remote_exists("pcloud", "drift:\n")
