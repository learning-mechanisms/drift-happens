from __future__ import annotations

import json
from pathlib import Path

from drift_happens.configs import ExperimentConfig
from drift_happens.experiments.materialize import (
    PRESET_SNAPSHOT_INDEX_KIND,
    MaterializationDiff,
    build_index_payload,
    check_materialized_snapshots,
    expected_materialized_files,
    write_materialized_snapshots,
)
from drift_happens.experiments.registry import iter_presets
from drift_happens.utils.snapshot import experiment_from_preset_snapshot


def test_materialized_preset_snapshots_are_current() -> None:
    diff = check_materialized_snapshots()

    assert diff.ok, f"{diff.format()}; run `pixi run materialize`"


def test_materialization_round_trips_configs(tmp_path: Path) -> None:
    write_materialized_snapshots(tmp_path)

    for entry in iter_presets():
        path = tmp_path / entry.group / f"{entry.name}.json"
        snapshot = json.loads(path.read_text())
        restored = experiment_from_preset_snapshot(snapshot)

        assert restored == entry.build()


def test_materialization_check_detects_orphans(tmp_path: Path) -> None:
    write_materialized_snapshots(tmp_path)
    orphan = tmp_path / "yearbook" / "orphan.json"
    orphan.write_text("{}\n")

    diff = check_materialized_snapshots(tmp_path)

    assert not diff.ok
    assert orphan in diff.orphaned


def test_materialization_write_removes_orphans(tmp_path: Path) -> None:
    # A re-materialize must clear orphaned snapshots so `pixi run materialize`
    # can fix a check that fails on a removed/renamed preset.
    write_materialized_snapshots(tmp_path)
    orphan = tmp_path / "yearbook" / "orphan.json"
    orphan.write_text("{}\n")

    write_materialized_snapshots(tmp_path)

    assert not orphan.exists()
    assert check_materialized_snapshots(tmp_path).ok


def test_expected_materialized_files_include_index() -> None:
    expected = expected_materialized_files()
    index_path = next(path for path in expected if path.name == "index.json")
    index = json.loads(expected[index_path])

    assert index["kind"] == PRESET_SNAPSHOT_INDEX_KIND
    assert len(index["presets"]) == len(list(iter_presets()))


def test_index_cache_ids_match_configs() -> None:
    entries = tuple(iter_presets())
    index = build_index_payload(entries)
    configs: dict[tuple[str, str], ExperimentConfig] = {
        entry.key: entry.build() for entry in entries
    }

    for row in index["presets"]:
        key = (row["group"], row["name"])
        cache = configs[key].preprocessing.cache
        assert row["cache_id"] == (cache.cache_id if cache is not None else None)
        assert row["comparison_role"] in {"smoke", "headline"}
        assert isinstance(row["variant_fields"], list)


def test_materialization_check_detects_missing(tmp_path: Path) -> None:
    written = write_materialized_snapshots(tmp_path)
    victim = next(p for p in written if p.name != "index.json")
    victim.unlink()

    diff = check_materialized_snapshots(tmp_path)

    assert not diff.ok
    assert victim in diff.missing


def test_materialization_check_detects_stale(tmp_path: Path) -> None:
    written = write_materialized_snapshots(tmp_path)
    victim = next(p for p in written if p.name != "index.json")
    victim.write_text(victim.read_text() + " ")

    diff = check_materialized_snapshots(tmp_path)

    assert not diff.ok
    assert victim in diff.stale


def test_materialization_diff_format_renders_each_section(tmp_path: Path) -> None:
    written = write_materialized_snapshots(tmp_path)
    files = [p for p in written if p.name != "index.json"]
    files[0].unlink()
    files[1].write_text(files[1].read_text() + " ")
    (tmp_path / files[2].parent.name / "orphan.json").write_text("{}\n")

    rendered = check_materialized_snapshots(tmp_path).format()

    assert "missing:" in rendered
    assert "stale:" in rendered
    assert "orphaned:" in rendered
    assert MaterializationDiff((), (), ()).format() == (
        "materialized preset snapshots are current"
    )
