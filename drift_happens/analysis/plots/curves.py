"""Forgetting curves."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from drift_happens.analysis.datasets import DatasetSpec
from drift_happens.analysis.plots import palette, style
from drift_happens.analysis.plots.derive import ForgettingCurve
from drift_happens.analysis.plots.names import FAMILY_LABELS, get_display_name

# Distinct marker shapes cycled across lines in every forgetting layout, so
# curves stay tellable apart by shape (not colour alone) for colour-blind
# readers and in greyscale. Consecutive lines get different shapes, so
# neighbours that the continuous colour map renders in near-identical hues are
# still separable.
_LINE_MARKERS = (
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "*",
    "<",
    ">",
    "p",
    "h",
    "d",
    "8",
    "H",
)


def forgetting(curves: list[ForgettingCurve], spec: DatasetSpec, path: Path) -> Path:
    """One line per model with a seed std band."""
    return _forgetting(curves, spec, path, lambda curve: get_display_name(curve.model))


def forgetting_compact(
    curves: list[ForgettingCurve], spec: DatasetSpec, path: Path
) -> Path:
    """
    Space-efficient per-model variant for the single-column LNCS build.

    Same data as :func:`forgetting`, but with a wide, short aspect and the legend below
    the axes so the figure stays readable at column width.
    """
    return _forgetting(
        curves, spec, path, lambda curve: get_display_name(curve.model), compact=True
    )


def forgetting_families(
    curves: list[ForgettingCurve], spec: DatasetSpec, path: Path
) -> Path:
    """One line per model family, averaged over its members."""
    return _forgetting(
        curves, spec, path, lambda curve: FAMILY_LABELS.get(curve.model, curve.model)
    )


def _forgetting(
    curves: list[ForgettingCurve],
    spec: DatasetSpec,
    path: Path,
    label: Callable[[ForgettingCurve], str],
    *,
    compact: bool = False,
) -> Path:
    colors = palette.categorical([curve.model for curve in curves])
    figsize = style.FORGETTING_COMPACT_SIZE if compact else style.FIGURE_SIZE
    linewidth = 1.1 if compact else 1.5
    fig, ax = plt.subplots(figsize=figsize)
    markersize = 3.2 if compact else 4.0
    for index, curve in enumerate(curves):
        color = colors[curve.model]
        band = np.nan_to_num(curve.std)
        # A shape per line, sampled a handful of times along it and offset per
        # line so the marks spread out rather than stacking on one x. Applied in
        # every layout so curves are separable by shape, not colour alone.
        marker = _LINE_MARKERS[index % len(_LINE_MARKERS)]
        stride = max(len(curve.lag) // 6, 1)
        markevery = (index % stride, stride)
        ax.plot(
            curve.lag,
            curve.mean,
            color=color,
            linewidth=linewidth,
            label=label(curve),
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            markeredgecolor="0.15",
            markeredgewidth=0.3,
        )
        ax.fill_between(
            curve.lag,
            curve.mean - band,
            curve.mean + band,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    values = np.concatenate([curve.mean[np.isfinite(curve.mean)] for curve in curves])
    if values.size:
        clip = 97
        if spec.higher_is_better:
            low, high = float(np.percentile(values, 100 - clip)), float(values.max())
        else:
            low, high = float(values.min()), float(np.percentile(values, clip))
        margin = (high - low) * 0.05 or 1.0
        ax.set_ylim(low - margin, high + margin)
    label_size = 8 if compact else None
    ax.set_xlabel(
        f"{spec.slice_noun.capitalize()}s since training", fontsize=label_size
    )
    ax.set_ylabel(palette.axis_label(spec), fontsize=label_size)
    if compact:
        ax.tick_params(labelsize=7)
        # Legend below the axes in columns, so the plot spans the text width
        # instead of ceding a third of it to a right-hand legend.
        ncol = min(6, len(curves))
        ax.legend(
            fontsize=6,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=ncol,
            columnspacing=0.8,
            labelspacing=0.25,
            handlelength=1.4,
            handletextpad=0.3,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            fontsize="x-small",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
    return style.save(fig, path)
