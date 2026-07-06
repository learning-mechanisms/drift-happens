"""Pixi lockfile hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from drift_happens.utils.paths import PIXI_LOCK


def pixi_lock_sha256(path: Path = PIXI_LOCK) -> str | None:
    """Return the SHA-256 digest of ``pixi.lock``, or ``None`` if it is missing."""
    if not path.exists():
        return None

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
