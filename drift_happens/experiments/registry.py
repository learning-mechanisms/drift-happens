"""Aggregate registry of Python-defined experiment presets."""

from __future__ import annotations

from collections.abc import Iterator

from drift_happens.experiments import (
    amazon_reviews_23,
    arxiv,
    imdb_faces,
    smoke,
    yearbook,
)
from drift_happens.experiments.types import PresetEntry


def _build_registry() -> tuple[PresetEntry, ...]:
    entries = (
        *yearbook.presets(),
        *arxiv.presets(),
        *amazon_reviews_23.presets(),
        *imdb_faces.presets(),
        *smoke.presets(),
    )
    _check_unique(entries)
    # sorted by (group, name) — callers rely on this invariant
    return tuple(sorted(entries, key=lambda entry: entry.key))


def _check_unique(entries: tuple[PresetEntry, ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if entry.key in seen:
            raise ValueError(f"duplicate preset (group, name): {entry.key}")
        seen.add(entry.key)


# Built at import time; submodule imports above are already eager
_REGISTRY: tuple[PresetEntry, ...] = _build_registry()


def iter_presets() -> Iterator[PresetEntry]:
    """Yield every registered preset in deterministic order."""
    yield from _REGISTRY


def preset(group: str, name: str) -> PresetEntry:
    """Return one registered preset by group/name."""
    for entry in _REGISTRY:
        if entry.group == group and entry.name == name:
            return entry
    raise KeyError(f"no preset registered with group={group!r}, name={name!r}")


def preset_groups() -> dict[str, list[str]]:
    """Return ``{group: [name, ...]}`` for help and listing commands."""
    # _REGISTRY is sorted by (group, name); insertion order gives sorted groups and names
    out: dict[str, list[str]] = {}
    for entry in _REGISTRY:
        out.setdefault(entry.group, []).append(entry.name)
    return out
