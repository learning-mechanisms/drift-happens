"""Materialize Python-defined presets to deterministic JSON snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drift_happens.experiments.registry import iter_presets
from drift_happens.experiments.types import PresetEntry
from drift_happens.utils.paths import SNAPSHOTS_DIR, relative_to_project
from drift_happens.utils.snapshot import build_preset_snapshot, write_preset_snapshot

PRESET_SNAPSHOT_INDEX_KIND = "preset_snapshot_index/v1"
PRESETS_ROOT = SNAPSHOTS_DIR / "presets"


@dataclass(frozen=True)
class MaterializationDiff:
    """Differences between materialized snapshots and the Python registry."""

    missing: tuple[Path, ...]
    stale: tuple[Path, ...]
    orphaned: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.stale and not self.orphaned

    def format(self) -> str:
        if self.ok:
            return "materialized preset snapshots are current"

        sections: list[str] = []
        for label, paths in (
            ("missing", self.missing),
            ("stale", self.stale),
            ("orphaned", self.orphaned),
        ):
            if not paths:
                continue
            rel_paths = ", ".join(str(relative_to_project(path)) for path in paths)
            sections.append(f"{label}: {rel_paths}")
        return "; ".join(sections)


def selected_presets(group: str | None = None) -> tuple[PresetEntry, ...]:
    """Return registry entries, optionally filtered by group."""
    entries = tuple(
        entry for entry in iter_presets() if group is None or entry.group == group
    )
    return entries


def preset_snapshot_path(entry: PresetEntry, root: Path = PRESETS_ROOT) -> Path:
    """Return the materialized JSON path for one preset."""
    return root / entry.group / f"{entry.name}.json"


def build_snapshot_payload(entry: PresetEntry) -> dict[str, object]:
    """Build the deterministic preset snapshot envelope for one entry."""
    return build_preset_snapshot(
        cfg=entry.build(),
        group=entry.group,
        name=entry.name,
        seeds=entry.seeds,
        description=entry.description,
        tags=entry.tags,
        comparison_group=entry.comparison_group,
        comparison_role=entry.comparison_role,
        variant_fields=entry.variant_fields,
    )


def build_index_payload(
    entries: Iterable[PresetEntry], root: Path = PRESETS_ROOT
) -> dict:
    """Build a deterministic index of materialized preset snapshots."""
    rows: list[dict[str, object]] = []
    for entry in sorted(entries, key=lambda item: item.key):
        cfg = entry.build()
        cache = cfg.preprocessing.cache
        rows.append(
            {
                "cache_id": cache.cache_id if cache is not None else None,
                "comparison_group": entry.comparison_group,
                "comparison_role": entry.comparison_role,
                "description": entry.description,
                "group": entry.group,
                "name": entry.name,
                "path": str(preset_snapshot_path(entry, root).relative_to(root)),
                "seeds": list(entry.seeds),
                "tags": list(entry.tags),
                "variant_fields": list(entry.variant_fields),
            }
        )
    return {
        "kind": PRESET_SNAPSHOT_INDEX_KIND,
        "presets": rows,
    }


def expected_materialized_files(root: Path = PRESETS_ROOT) -> dict[Path, str]:
    """Return expected snapshot and index files as stable JSON text."""
    entries = selected_presets()
    validate_comparison_groups(entries)
    expected = {
        preset_snapshot_path(entry, root): _stable_json(build_snapshot_payload(entry))
        for entry in entries
    }
    expected[root / "index.json"] = _stable_json(build_index_payload(entries, root))
    return expected


def write_materialized_snapshots(root: Path = PRESETS_ROOT) -> tuple[Path, ...]:
    """Write all registry snapshots and the index to ``root``."""
    entries = selected_presets()
    validate_comparison_groups(entries)
    written: list[Path] = []
    for entry in entries:
        path = preset_snapshot_path(entry, root)
        write_preset_snapshot(path, build_snapshot_payload(entry))
        written.append(path)

    index_path = root / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(_stable_json(build_index_payload(entries, root)))
    written.append(index_path)

    # Delete snapshots for presets no longer in the registry.
    for orphan in set(root.rglob("*.json")) - set(written):
        orphan.unlink()

    return tuple(written)


def check_materialized_snapshots(root: Path = PRESETS_ROOT) -> MaterializationDiff:
    """Compare ``root`` against the deterministic output of the registry."""
    expected = expected_materialized_files(root)
    missing: list[Path] = []
    stale: list[Path] = []

    for path, expected_text in expected.items():
        if not path.exists():
            missing.append(path)
            continue
        if path.read_text() != expected_text:
            stale.append(path)

    actual = set(root.rglob("*.json")) if root.exists() else set()
    orphaned = sorted(actual - set(expected))
    return MaterializationDiff(
        missing=tuple(sorted(missing)),
        stale=tuple(sorted(stale)),
        orphaned=tuple(orphaned),
    )


def validate_comparison_groups(entries: Iterable[PresetEntry]) -> None:
    """
    Validate invariant config fields for presets sharing a comparison group.

    The union of declared variant fields is removed before comparing snapshots. Seeds
    stay outside the root config and are checked separately.
    """
    groups: dict[str, list[PresetEntry]] = {}
    for entry in entries:
        if entry.comparison_group is None:
            continue
        groups.setdefault(entry.comparison_group, []).append(entry)

    for group, grouped in groups.items():
        expected_seeds = grouped[0].seeds
        variant_fields = sorted(
            {field for entry in grouped for field in entry.variant_fields}
        )
        baseline = _normalized_config(grouped[0], variant_fields)
        for entry in grouped[1:]:
            if entry.seeds != expected_seeds:
                raise ValueError(
                    f"{group}: {entry.group}/{entry.name} seeds {entry.seeds} "
                    f"do not match {grouped[0].group}/{grouped[0].name} "
                    f"seeds {expected_seeds}"
                )
            normalized = _normalized_config(entry, variant_fields)
            if normalized != baseline:
                raise ValueError(
                    f"{group}: {entry.group}/{entry.name} differs outside "
                    f"declared variant fields {variant_fields}"
                )


def _stable_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _normalized_config(
    entry: PresetEntry,
    variant_fields: list[str],
) -> dict[str, Any]:
    cfg = entry.build().model_dump(mode="json")
    for field in variant_fields:
        _drop_dotted(cfg, field)
    return cfg


def _drop_dotted(payload: dict[str, Any], dotted: str) -> None:
    cur: Any = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            return
        cur = cur.get(part)
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
