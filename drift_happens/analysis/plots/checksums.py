"""Output manifest: hash rendered files and check them against a committed list."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def manifest(paths: Iterable[Path], root: Path) -> dict[str, str]:
    """Map each output, keyed by its path relative to ``root``, to its sha256."""
    digests = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in set(paths)
        if path.is_file()
    }
    return {name: digests[name] for name in sorted(digests)}


def write(paths: Iterable[Path], root: Path, destination: Path) -> Path:
    # GNU sha256sum manifest format: digest, two spaces, then the path.
    lines = [f"{digest}  {name}" for name, digest in manifest(paths, root).items()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def verify(paths: Iterable[Path], root: Path, source: Path) -> list[str]:
    """Return the names whose hash differs from ``source`` (empty when all match)."""
    expected = _read(source)
    actual = manifest(paths, root)
    return sorted(
        name
        for name in set(expected) | set(actual)
        if expected.get(name) != actual.get(name)
    )


def _read(source: Path) -> dict[str, str]:
    entries = {}
    for line in source.read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            entries[name] = digest
    return entries
