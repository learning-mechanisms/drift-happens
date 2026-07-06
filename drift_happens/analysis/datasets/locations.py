"""Filesystem locations for frozen analysis artifacts and rendered paper assets."""

from __future__ import annotations

from pathlib import Path

from drift_happens.utils.paths import ARTIFACTS_DIR, PROJECT_ROOT

# Generated frozen data (written by `drift analysis export`; gitignored).
_FROZEN_DIR: Path = ARTIFACTS_DIR / "analysis"
RESULTS_PARQUET: Path = _FROZEN_DIR / "results.parquet"
DATASET_STATS_PARQUET: Path = _FROZEN_DIR / "dataset_stats.parquet"
PARAMS_PARQUET: Path = _FROZEN_DIR / "params.parquet"
RUNS_LOCK: Path = _FROZEN_DIR / "runs.lock.json"

# Rendered paper and website assets (committed) plus the verification manifest.
_PAPER_DIR: Path = PROJECT_ROOT / "paper"
DEFAULT_FIGURES_DIR: Path = _PAPER_DIR / "plots_experiments"
DEFAULT_TABLES_DIR: Path = _PAPER_DIR / "tables"
DEFAULT_PAGES_DIR: Path = _PAPER_DIR / "pages"
DEFAULT_VALUES_PATH: Path = _PAPER_DIR / "generated" / "values.tex"
DEFAULT_VALUES_JSON: Path = PROJECT_ROOT / "website" / "data" / "values.json"
FIGURES_MANIFEST: Path = _PAPER_DIR / "figures.sha256"
DEFAULT_SITE_DIR: Path = PROJECT_ROOT / "website"
