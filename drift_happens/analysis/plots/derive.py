"""Seed aggregation, drift-matrix assembly, and coverage over frozen results."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import polars as pl

from drift_happens.analysis.datasets import DATASETS

DEFAULT_CUTOFF_COUNT = 4

_LEADING_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?)(.*)$")


@dataclass(frozen=True)
class DriftMatrix:
    model: str
    slices: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class ForgettingCurve:
    model: str
    lag: np.ndarray
    mean: np.ndarray
    std: np.ndarray


FUTURE_PERFORMANCE = "future_performance"
DECAY = "decay"


@dataclass(frozen=True)
class Ranking:
    kind: str
    models: tuple[str, ...]
    score: np.ndarray
    higher_is_better: bool


@dataclass(frozen=True)
class Coverage:
    dataset: str
    deliverable: str
    present: tuple[str, ...]
    missing: tuple[str, ...]


def _slice_key(value: str) -> tuple[int, float, str]:
    match = _LEADING_NUMBER.match(value)
    if match:
        return (0, float(match.group(1)), match.group(2))
    return (1, float("inf"), value)


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _nan_reduce(stack: np.ndarray, reducer: Callable[..., np.ndarray]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reducer(stack, axis=0)


def _nan_sample_std(stack: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(np.nanstd(stack, axis=axis, ddof=1))


def _eval_rows(frame: pl.DataFrame, dataset: str) -> pl.DataFrame:
    spec = DATASETS[dataset]
    return frame.filter(
        (pl.col("dataset") == dataset)
        & (pl.col("phase") == "eval")
        # ledgers phase-prefix the metric name ("eval/accuracy")
        & pl.col("metric").is_in([spec.metric, f"eval/{spec.metric}"])
        & pl.col("train_slice").is_not_null()
        & pl.col("eval_slice").is_not_null()
        & pl.col("value").is_not_null()
    )


def _slices(rows: pl.DataFrame) -> tuple[str, ...]:
    labels = set(rows.get_column("train_slice").to_list())
    labels |= set(rows.get_column("eval_slice").to_list())
    return tuple(sorted(labels, key=_slice_key))


def _seed_split(
    rows: pl.DataFrame, expected_seeds: tuple[int, ...]
) -> tuple[list[str], list[str]]:
    grouped = rows.group_by("trainer").agg(pl.col("seed").unique().alias("seeds"))
    complete, incomplete = [], []
    for record in grouped.iter_rows(named=True):
        present = set(record["seeds"])
        if all(seed in present for seed in expected_seeds):
            complete.append(record["trainer"])
        else:
            incomplete.append(record["trainer"])
    return sorted(complete), sorted(incomplete)


def _benchmark_seeds() -> tuple[int, ...]:
    from drift_happens.experiments.common import BENCHMARK_SEEDS

    return BENCHMARK_SEEDS


def _yearbook_benchmark_seeds() -> tuple[int, ...]:
    from drift_happens.experiments.yearbook import YEARBOOK_BENCHMARK_SEEDS

    return YEARBOOK_BENCHMARK_SEEDS


EXPECTED_SEEDS_BY_DATASET: dict[str, Callable[[], tuple[int, ...]]] = {
    "amazon_reviews_23": _benchmark_seeds,
    "arxiv": _benchmark_seeds,
    "yearbook": _yearbook_benchmark_seeds,
}


def _expected_seeds(
    dataset: str, expected_seeds: tuple[int, ...] | None
) -> tuple[int, ...]:
    if expected_seeds is not None:
        return expected_seeds
    seed_factory = EXPECTED_SEEDS_BY_DATASET.get(dataset, _benchmark_seeds)
    return seed_factory()


def _matrix(
    rows: pl.DataFrame, model: str, slices: tuple[str, ...], scale: float
) -> DriftMatrix:
    index = {label: position for position, label in enumerate(slices)}
    size = len(slices)
    mean = np.full((size, size), np.nan)
    std = np.full((size, size), np.nan)
    cells = rows.group_by("train_slice", "eval_slice").agg(
        pl.col("value").mean().alias("mean"),
        pl.col("value").std(ddof=1).alias("std"),
    )
    for record in cells.iter_rows(named=True):
        row = index[record["train_slice"]]
        column = index[record["eval_slice"]]
        mean[row, column] = record["mean"]
        std[row, column] = np.nan if record["std"] is None else record["std"]
    return DriftMatrix(model, slices, mean * scale, std * scale)


def per_model_matrices(
    frame: pl.DataFrame, dataset: str, expected_seeds: tuple[int, ...] | None = None
) -> tuple[list[DriftMatrix], Coverage]:
    """Seed mean and std drift matrix for each model that has all seeds."""
    spec = DATASETS[dataset]
    rows = _eval_rows(frame, dataset)
    slices = _slices(rows)
    complete, incomplete = _seed_split(rows, _expected_seeds(dataset, expected_seeds))
    matrices = [
        _matrix(
            rows.filter(pl.col("trainer") == model), model, slices, spec.value_scale
        )
        for model in complete
    ]
    return matrices, Coverage(dataset, "matrices", tuple(complete), tuple(incomplete))


class Status(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True)
class LineupEntry:
    matrix: DriftMatrix
    status: Status


def lineup_matrices(
    frame: pl.DataFrame,
    dataset: str,
    expected: list[str] | None = None,
    expected_seeds: tuple[int, ...] | None = None,
) -> list[LineupEntry]:
    """
    Every expected trainer's matrix over available data, flagged by completeness.

    ``expected`` is the planned trainer list; when ``None`` the trainers present in the
    frame are used. An absent trainer yields a blank matrix flagged ``MISSING``.
    """
    spec = DATASETS[dataset]
    resolved_seeds = _expected_seeds(dataset, expected_seeds)
    rows = _eval_rows(frame, dataset)
    slices = _slices(rows)
    size = len(slices)
    if expected is None:
        expected = sorted(rows.get_column("trainer").unique().to_list())
    lineup = []
    for model in expected:
        model_rows = rows.filter(pl.col("trainer") == model)
        if size == 0 or model_rows.is_empty():
            blank = np.full((size, size), np.nan)
            entry = LineupEntry(
                DriftMatrix(model, slices, blank, blank), Status.MISSING
            )
        else:
            matrix = _matrix(model_rows, model, slices, spec.value_scale)
            present = set(model_rows.get_column("seed").to_list())
            complete = (
                all(seed in present for seed in resolved_seeds)
                and not np.isnan(matrix.mean).any()
            )
            entry = LineupEntry(matrix, Status.COMPLETE if complete else Status.PARTIAL)
        lineup.append(entry)
    return lineup


def mean_over_models(matrices: list[DriftMatrix]) -> DriftMatrix:
    """Average the per-model means, keeping only cells present for every model."""
    if not matrices:
        raise ValueError("no matrices to average")
    stack = np.stack([matrix.mean for matrix in matrices])
    thin = np.sum(np.isfinite(stack), axis=0) < len(matrices)
    mean = _nan_reduce(stack, np.nanmean)
    std = _nan_reduce(stack, _nan_sample_std)
    mean[thin] = np.nan
    std[thin] = np.nan
    return DriftMatrix("mean", matrices[0].slices, mean, std)


def deviations(
    matrices: list[DriftMatrix], reference: DriftMatrix
) -> list[DriftMatrix]:
    """Each model's mean matrix minus the reference grid."""
    return [
        DriftMatrix(
            matrix.model, matrix.slices, matrix.mean - reference.mean, matrix.std
        )
        for matrix in matrices
    ]


def deviation_extent(matrices: list[DriftMatrix], percentile: float = 99.0) -> float:
    """One symmetric colour extent shared by every deviation matrix."""
    if not matrices:
        return 1.0
    stacked = np.concatenate([matrix.mean.ravel() for matrix in matrices])
    finite = np.abs(stacked[np.isfinite(stacked)])
    return float(np.percentile(finite, percentile)) if finite.size else 1.0


def raw_extent(
    matrices: list[DriftMatrix], percentile: float = 95.0
) -> tuple[float, float]:
    """One (low, high) colour range shared by every raw drift-matrix heatmap."""
    if not matrices:
        return (0.0, 1.0)
    stacked = np.concatenate([matrix.mean.ravel() for matrix in matrices])
    finite = stacked[np.isfinite(stacked)]
    if not finite.size:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 100.0 - percentile))
    hi = float(np.percentile(finite, percentile))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def coarsen(
    matrix: DriftMatrix, k_max: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Block-average a drift matrix down to at most ``k_max`` periods.

    The slices are partitioned into ``k`` contiguous, near-equal buckets and each coarse
    cell is the finite mean over its sub-block of ``matrix.mean`` (an empty block stays
    NaN). Block-averaging lightly blurs the in-distribution diagonal, which is acceptable
    for the schematic replay overview. Returns the inclusive slice-index range of each
    bucket and the resulting ``k x k`` grid; a matrix already at or below ``k_max``
    periods yields one bucket per slice.

    Args:
        matrix: The drift matrix to coarsen.
        k_max: Maximum number of periods along each axis.

    Returns:
        The inclusive ``(start, end)`` slice-index range per bucket and the coarse grid.

    Raises:
        ValueError: If ``k_max`` is not positive.
    """
    if k_max < 1:
        raise ValueError("k_max must be positive")
    size = len(matrix.slices)
    if size == 0:
        return [], np.empty((0, 0))
    groups = np.array_split(np.arange(size), min(k_max, size))
    buckets = [(int(group[0]), int(group[-1])) for group in groups]
    coarse = np.full((len(groups), len(groups)), np.nan)
    for row, rows in enumerate(groups):
        for column, columns in enumerate(groups):
            coarse[row, column] = _safe_mean(matrix.mean[np.ix_(rows, columns)])
    return buckets, coarse


def forgetting_curves(
    frame: pl.DataFrame, dataset: str, expected_seeds: tuple[int, ...] | None = None
) -> tuple[list[ForgettingCurve], Coverage]:
    """Mean performance by slices elapsed since training, with seed std."""
    spec = DATASETS[dataset]
    rows = _eval_rows(frame, dataset)
    complete, incomplete = _seed_split(rows, _expected_seeds(dataset, expected_seeds))
    if not complete:
        return [], Coverage(dataset, "forgetting", (), tuple(incomplete))
    rank = {label: position for position, label in enumerate(_slices(rows))}
    lagged = rows.with_columns(
        (
            pl.col("eval_slice").replace_strict(rank)
            - pl.col("train_slice").replace_strict(rank)
        ).alias("lag")
    ).filter(pl.col("lag") >= 0)
    curves = []
    for model in complete:
        per_seed = (
            lagged.filter(pl.col("trainer") == model)
            .group_by("seed", "lag")
            .agg(pl.col("value").mean().alias("value"))
        )
        aggregated = (
            per_seed.group_by("lag")
            .agg(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std(ddof=1).alias("std"),
            )
            .sort("lag")
        )
        curves.append(
            ForgettingCurve(
                model,
                aggregated.get_column("lag").to_numpy(),
                aggregated.get_column("mean").to_numpy() * spec.value_scale,
                aggregated.get_column("std").to_numpy() * spec.value_scale,
            )
        )
    return curves, Coverage(dataset, "forgetting", tuple(complete), tuple(incomplete))


def forgetting_by_family(
    curves: list[ForgettingCurve], family_of: dict[str, str]
) -> list[ForgettingCurve]:
    """Average the per-model forgetting curves within each model family."""
    groups: dict[str, list[ForgettingCurve]] = {}
    for curve in curves:
        groups.setdefault(family_of.get(curve.model, "other"), []).append(curve)
    families = []
    for family in sorted(groups):
        by_lag: dict[int, list[float]] = {}
        for curve in groups[family]:
            for lag, value in zip(curve.lag.tolist(), curve.mean.tolist(), strict=True):
                by_lag.setdefault(int(lag), []).append(value)
        lags = sorted(by_lag)
        mean = np.array([float(np.mean(by_lag[lag])) for lag in lags])
        std = np.array(
            [
                float(np.std(by_lag[lag], ddof=1)) if len(by_lag[lag]) > 1 else 0.0
                for lag in lags
            ]
        )
        families.append(ForgettingCurve(family, np.array(lags), mean, std))
    return families


def _diagonal_mean(matrix: DriftMatrix) -> float:
    return _safe_mean(np.diagonal(matrix.mean))


def _future_mean(matrix: DriftMatrix) -> float:
    rows, columns = np.triu_indices(len(matrix.slices), k=1)
    return _safe_mean(matrix.mean[rows, columns])


def _ranking(
    kind: str, scored: list[tuple[str, float]], higher_is_better: bool
) -> Ranking:
    clean = [(model, score) for model, score in scored if not np.isnan(score)]
    clean.sort(key=lambda item: item[1], reverse=higher_is_better)
    return Ranking(
        kind,
        tuple(model for model, _ in clean),
        np.array([score for _, score in clean]),
        higher_is_better,
    )


def rankings(dataset: str, matrices: list[DriftMatrix]) -> list[Ranking]:
    """Future-performance and decay score per model."""
    spec = DATASETS[dataset]
    future, decay = [], []
    for matrix in matrices:
        in_distribution = _diagonal_mean(matrix)
        future_value = _future_mean(matrix)
        future.append((matrix.model, future_value))
        drop = (
            in_distribution - future_value
            if spec.higher_is_better
            else future_value - in_distribution
        )
        decay.append((matrix.model, drop))
    return [
        _ranking(FUTURE_PERFORMANCE, future, spec.higher_is_better),
        _ranking(DECAY, decay, False),
    ]


@dataclass(frozen=True)
class CutoffRow:
    model: str
    in_distribution: float
    future: float
    decay: float


@dataclass(frozen=True)
class FamilyRow:
    family: str
    cells: tuple[tuple[float, float], ...]


def select_cutoffs(
    slices: tuple[str, ...], count: int = DEFAULT_CUTOFF_COUNT
) -> list[str]:
    """Evenly spaced training cutoffs that still have future slices."""
    usable = list(slices[:-1]) if len(slices) > 1 else list(slices)
    if count < 2 or len(usable) <= count:
        return usable[:count]
    step = (len(usable) - 1) / (count - 1)
    return [usable[round(position * step)] for position in range(count)]


def _cutoff_stats(dataset: str, matrix: DriftMatrix, index: int) -> CutoffRow:
    spec = DATASETS[dataset]
    in_distribution = float(matrix.mean[index, index])
    future = _safe_mean(matrix.mean[index, index + 1 :])
    decay = (
        in_distribution - future if spec.higher_is_better else future - in_distribution
    )
    return CutoffRow(matrix.model, in_distribution, future, decay)


def cutoff_rows(
    dataset: str, matrices: list[DriftMatrix], cutoff: str, top_n: int | None = None
) -> list[CutoffRow]:
    """Models trained up to ``cutoff``, ordered by future performance (all by
    default)."""
    spec = DATASETS[dataset]
    index = matrices[0].slices.index(cutoff)
    rows = [_cutoff_stats(dataset, matrix, index) for matrix in matrices]
    rows = [
        row
        for row in rows
        if not np.isnan(row.in_distribution) and not np.isnan(row.future)
    ]
    rows.sort(key=lambda row: row.future, reverse=spec.higher_is_better)
    return rows[:top_n]


def families(frame: pl.DataFrame, dataset: str) -> dict[str, str]:
    """Map each trainer to its raw family key."""
    rows = _eval_rows(frame, dataset).select("trainer", "trainer_family").unique()
    return dict(rows.iter_rows())


def family_rows(
    frame: pl.DataFrame,
    dataset: str,
    matrices: list[DriftMatrix],
    cutoffs: list[str],
) -> list[FamilyRow]:
    """Mean future performance and decay per model family at each cutoff."""
    family_of = families(frame, dataset)
    grouped: dict[str, list[DriftMatrix]] = {}
    for matrix in matrices:
        grouped.setdefault(family_of.get(matrix.model, "other"), []).append(matrix)
    rows = []
    for family in sorted(grouped):
        cells = []
        for cutoff in cutoffs:
            index = matrices[0].slices.index(cutoff)
            stats = [
                _cutoff_stats(dataset, matrix, index) for matrix in grouped[family]
            ]
            future = _safe_mean(np.array([stat.future for stat in stats]))
            decay = _safe_mean(np.array([stat.decay for stat in stats]))
            cells.append((future, decay))
        rows.append(FamilyRow(family, tuple(cells)))
    return rows


def coverage_summary(coverages: list[Coverage]) -> str:
    """One line per (dataset, deliverable) with the incomplete models listed."""
    lines = []
    for coverage in coverages:
        total = len(coverage.present) + len(coverage.missing)
        lines.append(
            f"{coverage.dataset} [{coverage.deliverable}]: "
            f"{len(coverage.present)}/{total} complete"
        )
        if coverage.missing:
            lines.append(f"  incomplete: {', '.join(coverage.missing)}")
    return "\n".join(lines)
