"""Progress event side channel for subprocess orchestration."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

SWEEP_PROGRESS_FILE_ENV = "DRIFT_SWEEP_PROGRESS_FILE"


def sweep_progress_requested() -> bool:
    """Return true when a sweep parent owns child progress rendering."""
    return os.environ.get(SWEEP_PROGRESS_FILE_ENV) is not None


def write_sweep_progress_event(event: str, **payload: object) -> None:
    """Append one sweep progress event when a parent requested it."""
    raw_path = os.environ.get(SWEEP_PROGRESS_FILE_ENV)
    if raw_path is None:
        return
    record = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **payload,
    }
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
