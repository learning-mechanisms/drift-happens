"""Stable, filesystem-safe identifiers for artifacts, runs, and sweeps."""

from __future__ import annotations

import re
from datetime import UTC, datetime

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str) -> str:
    """Return a filesystem-safe identifier for names used in artifact paths."""
    slug = _SLUG_RE.sub("-", value).strip("-")
    if not slug or not slug.strip("."):
        return "unnamed"
    return slug


def utc_timestamp(now: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp: ``YYYY-MM-DDThh-mm-ssZ``."""
    if now is None:
        now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")
