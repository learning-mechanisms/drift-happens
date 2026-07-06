"""Drift-matrix heatmaps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.image import AxesImage

from drift_happens.analysis.datasets import DatasetSpec
from drift_happens.analysis.plots import palette, style
from drift_happens.analysis.plots.derive import DriftMatrix
from drift_happens.analysis.plots.names import slice_label

_TARGET_TICKS = 12
_DIAGONAL_WIDTH = 0.8
_DIAGONAL_STROKE = 2 * _DIAGONAL_WIDTH  # black halo behind the white dashed line
_DIAGONAL_DASH = (0, (4, 3))

# Combined 3-panel cohort-mean figure (paper main body). Small heatmaps with
# larger, sparser labels so a full row of matrices stays legible at \textwidth.
# The y-tick labels are set diagonally (matching the x axis) so wide labels like
# "2020-H2" cost little horizontal room, leaving the heatmaps wider.
_COMBINED_SIZE = (6.6, 2.7)
_COMBINED_TICKS = 4
_COMBINED_TITLE_FS = 12
_COMBINED_LABEL_FS = 11
_COMBINED_TICK_FS = 9
_COMBINED_CBAR_FS = 9
_COMBINED_TICK_ROT = 45


def heatmap(
    matrix: DriftMatrix,
    spec: DatasetSpec,
    path: Path,
    *,
    diverging: bool = False,
    extent: float | None = None,
    raw_range: tuple[float, float] | None = None,
) -> Path:
    """Drift-matrix heatmap; ``diverging`` centres a signed deviation map on zero."""
    if diverging:
        cmap, norm = palette.diverging(spec, matrix.mean, extent)
        label = f"Δ {spec.metric_label}"
    else:
        cmap, norm = palette.sequential(spec, matrix.mean, raw_range)
        label = palette.axis_label(spec)
    return _draw(matrix.mean, matrix.slices, cmap, norm, label, spec, path)


def heatmap_axes(
    values: np.ndarray,
    labels: tuple[str, ...],
    spec: DatasetSpec,
    cmap: Colormap,
    norm: Normalize,
    colorbar_label: str,
) -> tuple[plt.Figure, plt.Axes, AxesImage]:
    """Drift-matrix canvas for the static heatmaps."""
    fig, ax = plt.subplots(figsize=style.HEATMAP_SIZE)
    image = ax.imshow(values, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    edge = len(labels) - 1
    ax.plot(
        [0, edge],
        [0, edge],
        color="white",
        linewidth=_DIAGONAL_WIDTH,
        linestyle=_DIAGONAL_DASH,
        path_effects=[pe.withStroke(linewidth=_DIAGONAL_STROKE, foreground="black")],
    )
    slice_ticks(ax, labels)
    ax.set_xlabel(f"Evaluation {spec.slice_noun}")
    ax.set_ylabel(f"Training {spec.slice_noun}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(colorbar_label)
    return fig, ax, image


def _draw(
    values: np.ndarray,
    slices: tuple[str, ...],
    cmap: Colormap,
    norm: Normalize,
    label: str,
    spec: DatasetSpec,
    path: Path,
) -> Path:
    labels = tuple(slice_label(value, spec) for value in slices)
    fig, _, _ = heatmap_axes(values, labels, spec, cmap, norm, label)
    return style.save(fig, path)


def slice_ticks(ax: plt.Axes, labels: tuple[str, ...]) -> None:
    """Thin, evenly spaced tick labels for a slice axis."""
    size = len(labels)
    step = max(1, math.ceil(size / _TARGET_TICKS))
    positions = list(range(0, size, step))
    if positions and positions[-1] != size - 1:
        positions.append(size - 1)
    shown = [labels[position] for position in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(shown, rotation=45, ha="right")
    ax.set_yticks(positions)
    ax.set_yticklabels(shown)
    ax.tick_params(length=0)


def combined_means(
    panels: Sequence[tuple[DriftMatrix, DatasetSpec, tuple[float, float] | None]],
    path: Path,
) -> Path:
    """
    One figure row of cohort-mean heatmaps, one panel per dataset.

    Replaces the separate per-dataset mean matrices in the paper body: each panel
    keeps its own colormap, norm, and colorbar (the metrics differ), titled by
    dataset. ``raw_range`` shares a panel's scale with its appendix lineup.
    """
    fig, axes = plt.subplots(
        1, len(panels), figsize=_COMBINED_SIZE, constrained_layout=True
    )
    for ax, (matrix, spec, raw_range) in zip(np.atleast_1d(axes), panels, strict=True):
        _combined_panel(ax, matrix, spec, raw_range)
    return style.save(fig, path)


def _combined_panel(
    ax: plt.Axes,
    matrix: DriftMatrix,
    spec: DatasetSpec,
    raw_range: tuple[float, float] | None,
) -> None:
    cmap, norm = palette.sequential(spec, matrix.mean, raw_range)
    image = ax.imshow(matrix.mean, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    edge = len(matrix.slices) - 1
    ax.plot(
        [0, edge],
        [0, edge],
        color="white",
        linewidth=_DIAGONAL_WIDTH,
        linestyle=_DIAGONAL_DASH,
        path_effects=[pe.withStroke(linewidth=_DIAGONAL_STROKE, foreground="black")],
    )
    labels = tuple(_combined_tick_label(value, spec) for value in matrix.slices)
    positions = _sparse_ticks(len(labels), _COMBINED_TICKS)
    shown = [labels[position] for position in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        shown,
        rotation=_COMBINED_TICK_ROT,
        ha="right",
        rotation_mode="anchor",
        fontsize=_COMBINED_TICK_FS,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        shown,
        rotation=_COMBINED_TICK_ROT,
        ha="right",
        va="center",
        rotation_mode="anchor",
        fontsize=_COMBINED_TICK_FS,
    )
    ax.tick_params(length=0)
    ax.set_xlabel(f"Evaluation {spec.slice_noun}", fontsize=_COMBINED_LABEL_FS)
    ax.set_ylabel(f"Training {spec.slice_noun}", fontsize=_COMBINED_LABEL_FS)
    ax.set_title(spec.title, fontsize=_COMBINED_TITLE_FS, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(palette.axis_label(spec), fontsize=_COMBINED_CBAR_FS)
    bar.ax.tick_params(labelsize=_COMBINED_CBAR_FS - 1)
    bar.ax.locator_params(nbins=4)


def _combined_tick_label(value: str, spec: DatasetSpec) -> str:
    """
    Tick label for the compact combined figure.

    Half-year slices drop their ``-H1``/``-H2`` suffix and show the year only: the axis
    is already titled "half-year" and the sub-year mark carries little signal at four
    sparse ticks, so the shorter label leaves more room for the heatmap.
    """
    label = slice_label(value, spec)
    if spec.slice_noun == "half-year":
        return label.split("-", 1)[0]
    return label


def _sparse_ticks(size: int, count: int) -> list[int]:
    """
    ``count`` roughly-even tick positions in ``[0, size)``, incl.

    first and last.
    """
    if size <= count:
        return list(range(size))
    return sorted({round(i * (size - 1) / (count - 1)) for i in range(count)})
