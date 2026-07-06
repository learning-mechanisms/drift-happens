"""Dataset overview plots from frozen statistics."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from drift_happens.analysis.datasets import DATASETS
from drift_happens.analysis.plots import palette, style
from drift_happens.analysis.plots.names import half_year_label

_SLICE_LABELS = {"year": "Year", "half_year": "Half-year"}
_GROUP_TITLES = {"gender": "Gender", "rating": "Rating"}
_MIN_BAR_HEIGHT = 2.5
_BAR_HEIGHT = 0.3


def build_overview(stats: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Render a count plot per group kind for every dataset in ``stats``."""
    outputs = []
    for dataset in sorted(stats.get_column("dataset").unique().to_list()):
        if dataset not in DATASETS:
            continue
        spec = DATASETS[dataset]
        block = stats.filter(pl.col("dataset") == dataset)
        target = out_dir / spec.slug / "overview"
        for group_kind in sorted(block.get_column("group_kind").unique().to_list()):
            if group_kind == "subject_share":
                continue  # joint cross-tab consumed only by the site EDA renderer
            rows = block.filter(pl.col("group_kind") == group_kind)
            path = target / f"{group_kind}.pdf"
            outputs.append(
                _bars(rows, spec.count_noun, path)
                if group_kind == "subject"
                else _lines(rows, group_kind, spec.count_noun, path)
            )
    return outputs


def _lines(rows: pl.DataFrame, group_kind: str, count_noun: str, path: Path) -> Path:
    slice_kind = rows.get_column("slice_kind")[0]
    groups = sorted(rows.get_column("group").unique().to_list())
    colors = palette.categorical(groups)
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    for group in groups:
        series = rows.filter(pl.col("group") == group).sort("slice")
        ax.plot(
            series.get_column("slice").to_numpy(),
            series.get_column("count").to_numpy(),
            color=colors[group],
            linewidth=1.5,
            label=None if group_kind == "total" else group,
        )
    ax.set_xlabel(_SLICE_LABELS.get(slice_kind, slice_kind))
    ax.set_ylabel(count_noun)
    # Ticks land on real (integer) slice indices, never fractional positions.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if slice_kind == "half_year":
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: half_year_label(int(value)))
        )
    if group_kind != "total":
        ax.legend(
            title=_GROUP_TITLES.get(group_kind, group_kind),
            fontsize="x-small",
            frameon=False,
        )
    return style.save(fig, path)


def _bars(rows: pl.DataFrame, count_noun: str, path: Path) -> Path:
    ordered = rows.group_by("group").agg(pl.col("count").sum()).sort(["count", "group"])
    labels = ordered.get_column("group").to_list()
    height = max(_MIN_BAR_HEIGHT, _BAR_HEIGHT * len(labels))
    fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH, height))
    ax.barh(
        range(len(labels)),
        ordered.get_column("count").to_numpy(),
        color=palette.BAR_COLOR,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(count_noun)
    return style.save(fig, path)
