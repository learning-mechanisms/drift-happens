"""
Repository path utilities.

This module is the non-mutating source of truth for project locations. It computes paths
but does not create directories at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from drift_happens.utils.ids import slugify as slugify
from drift_happens.utils.ids import utc_timestamp as utc_timestamp


def find_project_root(start: Path | None = None, marker: str = "pixi.toml") -> Path:
    """Return the nearest parent containing ``marker``."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"could not locate project root: no {marker} found above {here}")


def env_path(name: str, default: Path) -> Path:
    """Resolve a path from an environment variable or a default value."""
    value = os.environ.get(name)
    return (Path(value) if value and value.strip() else default).expanduser().resolve()


def relative_to_project(path: Path) -> Path:
    """Return ``path`` relative to the project root when possible."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved


PROJECT_ROOT: Path = find_project_root()
PIXI_LOCK: Path = PROJECT_ROOT / "pixi.lock"

DATA_DIR: Path = env_path("DRIFT_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACTS_DIR: Path = env_path("DRIFT_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
RUNS_DIR: Path = ARTIFACTS_DIR / "runs"
SWEEPS_DIR: Path = ARTIFACTS_DIR / "sweeps"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
EXPERIMENT_PLANS_DIR: Path = ARTIFACTS_DIR / "experiment_plans"
BUNDLES_DIR: Path = ARTIFACTS_DIR / "bundles"
PLOTS_DIR: Path = env_path("DRIFT_PLOTS_DIR", PROJECT_ROOT / "plots")
ARTIFACT_PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
SNAPSHOTS_DIR: Path = CONFIGS_DIR / "snapshots"
