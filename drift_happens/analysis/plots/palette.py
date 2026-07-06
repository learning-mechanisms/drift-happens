"""Colormaps and norms derived from the dataset registry."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize

from drift_happens.analysis.datasets import DatasetSpec

_MISSING_COLOR = "#f0f0f0"
_CONTINUOUS_CMAP = "turbo"

# qualitative colormaps paired with the number of distinct colors each carries
_QUALITATIVE_CMAPS = (("tab10", 10), ("tab20", 20))

BAR_COLOR = "#3b6fb0"

RAW_PERCENTILE = 95.0


def axis_label(spec: DatasetSpec) -> str:
    if spec.unit_suffix:
        return f"{spec.metric_label} ({spec.unit_suffix})"
    return spec.metric_label


def categorical(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Distinct colors per label, sampling a continuous map once the palettes run
    out."""
    for name, capacity in _QUALITATIVE_CMAPS:
        if len(models) <= capacity:
            cmap = colormaps[name]
            return {model: cmap(index) for index, model in enumerate(models)}
    cmap = colormaps[_CONTINUOUS_CMAP]
    span = max(len(models) - 1, 1)
    return {model: cmap(index / span) for index, model in enumerate(models)}


def sequential(
    spec: DatasetSpec, values: np.ndarray, extent: tuple[float, float] | None = None
) -> tuple[Colormap, Normalize]:
    """Colormap and norm for a metric heatmap; ``extent`` shares one scale across
    matrices."""
    cmap = colormaps[spec.sequential_cmap].with_extremes(bad=_MISSING_COLOR)
    bounds = (
        _percentile_range(values, RAW_PERCENTILE)
        if extent is None
        else _padded_range(*extent)
    )
    return cmap, Normalize(*bounds)


def _percentile_range(values: np.ndarray, pct: float) -> tuple[float, float]:
    finite = _finite(values)
    if not finite.size:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 100.0 - pct))
    hi = float(np.percentile(finite, pct))
    return _padded_range(lo, hi)


def diverging(
    spec: DatasetSpec, values: np.ndarray, extent: float | None = None
) -> tuple[Colormap, Normalize]:
    """Colormap and zero-centred norm for a deviation heatmap."""
    cmap = colormaps[spec.diverging_cmap].with_extremes(bad=_MISSING_COLOR)
    span = _abs_max(values) if extent is None else extent
    span = span if span > 0.0 else 1.0
    return cmap, Normalize(-span, span)


def _padded_range(low: float, high: float) -> tuple[float, float]:
    """A non-degenerate (low, high), widening by 1 when the span collapses."""
    return (low, high) if high > low else (low, low + 1.0)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _abs_max(values: np.ndarray) -> float:
    finite = _finite(values)
    if not finite.size:
        return 1.0
    span = float(np.abs(finite).max())
    return span if span > 0.0 else 1.0
