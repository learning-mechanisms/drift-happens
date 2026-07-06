from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

from drift_happens.utils.paths import (
    PROJECT_ROOT,
    env_path,
    find_project_root,
    relative_to_project,
    slugify,
    utc_timestamp,
)


def test_find_project_root_from_nested_path() -> None:
    nested = PROJECT_ROOT / "drift_happens" / "configs" / "base.py"
    assert nested.is_file(), f"anchor file moved or renamed: {nested}"

    assert find_project_root(nested) == PROJECT_ROOT


def test_slugify_and_timestamp_are_filesystem_safe() -> None:
    assert slugify("Yearbook Smoke / MLP") == "Yearbook-Smoke-MLP"
    assert utc_timestamp(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == (
        "2026-01-02T03-04-05Z"
    )


def test_relative_to_project_keeps_repo_paths_short() -> None:
    assert relative_to_project(PROJECT_ROOT / "pixi.toml") == Path("pixi.toml")


def test_const_import_does_not_create_directories(tmp_path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    # const.py must not mkdir PLOTS_DIR at import.
    plot_root = tmp_path / "plots"
    monkeypatch.setattr("drift_happens.utils.paths.ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr("drift_happens.utils.paths.PLOTS_DIR", plot_root)

    import drift_happens.const as const

    importlib.reload(const)

    try:
        assert not artifact_root.exists()
        assert not plot_root.exists()
    finally:
        # Reload const with the unpatched paths so later tests see real constants.
        monkeypatch.undo()
        importlib.reload(const)


def test_env_path_empty_value_uses_default(monkeypatch) -> None:
    default = Path("/srv/drift/data")
    monkeypatch.setenv("DRIFT_DATA_DIR", "")

    assert env_path("DRIFT_DATA_DIR", default) == default.expanduser().resolve()


def test_env_path_set_value_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("DRIFT_DATA_DIR", "/custom/data")

    assert env_path("DRIFT_DATA_DIR", Path("/srv/d")) == Path("/custom/data").resolve()
