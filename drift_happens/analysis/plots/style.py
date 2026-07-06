"""Matplotlib defaults and deterministic figure output."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

# pin the build epoch so matplotlib writes byte-reproducible PDF timestamps
os.environ["SOURCE_DATE_EPOCH"] = "0"
plt.switch_backend("Agg")

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
    }
)


FIGURE_SIZE = (6.0, 4.0)
FIGURE_WIDTH = 6.0
HEATMAP_SIZE = (6.0, 5.0)

# Wide, short forgetting-curve variant for the single-column LNCS build: the
# legend moves below the axes so the plot fills the text width and stays
# readable when included at \linewidth, instead of the roomy right-hand legend
# of FIGURE_SIZE that only reads well in the two-column / web layouts.
FORGETTING_COMPACT_SIZE = (5.6, 2.0)

_STRIP_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}


def save(fig: plt.Figure, path: Path, *, tight: bool = True) -> Path:
    """Write ``fig`` to ``path`` without machine- or time-varying metadata, then close
    it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight" if tight else None, metadata=_STRIP_METADATA)
    plt.close(fig)
    return path
