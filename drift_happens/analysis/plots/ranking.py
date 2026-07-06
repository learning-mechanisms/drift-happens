"""Robustness ranking bar charts."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from drift_happens.analysis.datasets import DatasetSpec
from drift_happens.analysis.plots import palette, style
from drift_happens.analysis.plots.derive import DECAY, FUTURE_PERFORMANCE, Ranking
from drift_happens.analysis.plots.names import get_display_name

_TITLES = {FUTURE_PERFORMANCE: "Future performance", DECAY: "Decay"}
_BAR_HEIGHT = 0.32
_MIN_HEIGHT = 2.5


def bars(ranking: Ranking, spec: DatasetSpec, path: Path) -> Path:
    """Horizontal bar chart of one ranking, best on top."""
    height = max(_MIN_HEIGHT, _BAR_HEIGHT * len(ranking.models))
    fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH, height))
    positions = range(len(ranking.models))
    drawn = ax.barh(list(positions), ranking.score, color=palette.BAR_COLOR)
    ax.bar_label(drawn, fmt=f"%{spec.value_fmt}", padding=3, fontsize="x-small")
    ax.set_yticks(list(positions))
    ax.set_yticklabels([get_display_name(model) for model in ranking.models])
    ax.invert_yaxis()
    ax.set_xlabel(_xlabel(ranking, spec))
    ax.set_title(_TITLES.get(ranking.kind, ranking.kind))
    return style.save(fig, path)


def _xlabel(ranking: Ranking, spec: DatasetSpec) -> str:
    if ranking.kind == DECAY:
        return f"Decay ({spec.unit_suffix})" if spec.unit_suffix else "Decay"
    return palette.axis_label(spec)
