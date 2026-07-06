"""
Compute drift robustness scores for all trained models.

Outputs: S, R_abs, R_rel, LDRS, CVaR, H for each model. The source is the runtime run
layout under ``artifacts/runs`` (one ``results/drift_matrix.json`` per run), aggregated
across seeds per (dataset, trainer).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drift_happens.const import (
    DATASET_EXPERIMENT_DIRS,
    ROBUSTNESS_RESULTS_DIR,
)
from drift_happens.evaluation.results import (
    MatrixPayload,
    ResultRun,
    build_metric_dataframe,
    discover_result_runs,
    resolve_metric,
)
from drift_happens.evaluation.robustness.metrics import (
    DriftRobustnessResult,
    analyze_drift_robustness,
)
from drift_happens.evaluation.robustness.transforms import MetricType
from drift_happens.utils.paths import RUNS_DIR

ROBUSTNESS_FIELDS: dict[str, str] = {
    "S": "strength",
    "R_abs": "robustness_abs",
    "R_rel": "robustness_rel",
    "LDRS": "ldrs",
    "CVaR": "cvar",
    "H": "combined_harmonic",
}

_PRIMARY_METRIC_BY_DATASET: dict[str, str] = {
    name: config.metric for name, config in DATASET_EXPERIMENT_DIRS.items()
}
_LOWER_IS_BETTER_TOKENS = ("loss", "mse", "rmse", "mae", "error")


@dataclass(frozen=True)
class SkippedEntry:
    """A model or run left out of the report, with the reason."""

    subject: str
    reason: str


def _record_skip(
    skipped: list[SkippedEntry] | None,
    subject: str,
    reason: str,
) -> None:
    print(f"[SKIP] {subject}: {reason}")
    if skipped is not None:
        skipped.append(SkippedEntry(subject=subject, reason=reason))


def _report_skipped(skipped: list[SkippedEntry], *, strict: bool) -> None:
    if not skipped:
        return
    print(f"\nSkipped {len(skipped)} {'entry' if len(skipped) == 1 else 'entries'}:")
    for entry in skipped:
        print(f"  - {entry.subject}: {entry.reason}")
    if strict:
        raise RuntimeError(
            f"robustness computation skipped {len(skipped)} of the inputs (--strict)"
        )


# --------------------------------- RUNTIME READER ----------------------------------- #


def select_run_metric(run: ResultRun) -> str | None:
    """
    Pick the dataset's primary metric.

    Datasets without a configured primary metric fall back to the run's own.
    """
    primary = _PRIMARY_METRIC_BY_DATASET.get(run.dataset)
    if primary is not None:
        return primary
    return resolve_metric(run)


def compute_runtime_robustness_scores(
    runs_root: Path = RUNS_DIR,
    skipped: list[SkippedEntry] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate robustness scores across seeds per (dataset, trainer).

    Discovers runtime result runs under ``runs_root``, scores each seed with
    ``analyze_drift_robustness`` on the dataset's primary metric, and returns one
    mean/std table per dataset ranked by the headline metric H.
    """
    results_by_dataset: dict[str, dict[str, list[DriftRobustnessResult]]] = {}
    for run in discover_result_runs(runs_root=runs_root):
        metric = select_run_metric(run)
        if metric is None:
            _record_skip(skipped, str(run.run_dir), "no metric available")
            continue

        M = _square_matrix(run.matrix, metric)
        if M is None:
            _record_skip(
                skipped, str(run.run_dir), f"no {metric!r} values on shared slices"
            )
            continue

        result = analyze_drift_robustness(M, metric_type=_metric_type(metric))
        results_by_dataset.setdefault(run.dataset, {}).setdefault(
            run.trainer_key, []
        ).append(result)

    return {
        dataset: _aggregate_trainer_results(by_trainer)
        for dataset, by_trainer in sorted(results_by_dataset.items())
    }


def _metric_type(metric: str) -> MetricType:
    base = metric.split("/", 1)[-1].lower()
    if any(token in base for token in _LOWER_IS_BETTER_TOKENS):
        return "lower_is_better"
    return "higher_is_better"


def _square_matrix(matrix: MatrixPayload, metric: str) -> np.ndarray | None:
    """Restrict a runtime drift matrix to the shared train/eval slice keys."""
    dataframe = build_metric_dataframe(matrix, metric)
    columns = set(dataframe.columns)
    common = [key for key in dataframe.index if key in columns]
    if len(common) < 2:
        # Robustness needs at least two shared slices to form a drift pair.
        return None
    M = dataframe.loc[common, common].to_numpy(dtype=float)
    if np.isnan(np.diag(M)).any():
        # A NaN on the diagonal (missing in-distribution baseline) poisons the
        # scores, so treat the matrix as incomplete and skip it.
        return None
    return M


def _aggregate_trainer_results(
    by_trainer: dict[str, list[DriftRobustnessResult]],
) -> pd.DataFrame:
    rows = []
    for trainer_key, results in by_trainer.items():
        scores = pd.DataFrame(
            [
                {
                    column: getattr(result, field)
                    for column, field in ROBUSTNESS_FIELDS.items()
                }
                for result in results
            ]
        )
        mean = scores.mean()
        std = scores.std(ddof=1).fillna(0.0)
        row: dict[str, object] = {"model": trainer_key, "seeds": len(results)}
        for column in ROBUSTNESS_FIELDS:
            row[f"{column}_mean"] = float(mean[column])
            row[f"{column}_std"] = float(std[column])
        rows.append(row)

    return (
        pd.DataFrame(rows).sort_values("H_mean", ascending=False).reset_index(drop=True)
    )


# ------------------------------------- REPORTS -------------------------------------- #


def write_robustness_reports(
    *,
    runs_root: Path = RUNS_DIR,
    output_dir: Path = ROBUSTNESS_RESULTS_DIR,
    strict: bool = False,
) -> dict[str, Path]:
    """
    Write one robustness CSV per dataset and return the written paths.

    Skipped models and runs are summarized at the end; ``strict`` turns any skip into a
    ``RuntimeError``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    skipped: list[SkippedEntry] = []

    frames = compute_runtime_robustness_scores(runs_root=runs_root, skipped=skipped)
    if not frames:
        print(f"No result runs found under {runs_root}")
    for dataset, df in frames.items():
        metric = _PRIMARY_METRIC_BY_DATASET.get(dataset, "run primary metric")
        print(f"\n=== {dataset.upper()} ({metric}) ===")
        written[dataset] = _write_dataset_csv(output_dir, dataset, df)

    _report_skipped(skipped, strict=strict)
    return written


def _write_dataset_csv(output_dir: Path, name: str, df: pd.DataFrame) -> Path:
    print(df.to_string(index=False))
    output_file = output_dir / f"{name}_robustness.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    return output_file
