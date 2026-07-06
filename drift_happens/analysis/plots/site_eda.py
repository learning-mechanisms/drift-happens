"""
Site-style exploratory-analysis charts rendered as SVG from the frozen dataset
statistics.

Reads ``dataset_stats.parquet`` (per-slice counts) and writes the EDA figures shown on
the dataset pages, in the same palette as the rest of the site. Kept separate from the
paper's matplotlib overview so the website assets carry no external plotting dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets.locations import DEFAULT_SITE_DIR
from drift_happens.analysis.plots.names import half_year_year

_W, _L, _R, _T, _B, _H = 660, 58, 642, 20, 206, 250
_AXIS, _DIM, _ACC, _FILL = "#1f2230", "#5b626e", "#2f43c4", "#dde1f6"
_PALETTE = ["#2f43c4", "#e07a2f", "#2e9e6b", "#b5429a", "#c9a227"]
_GENDER = {
    "F": "female",
    "M": "male",
    "0": "female",
    "1": "male",
    "f": "female",
    "m": "male",
}


def build_site_eda(
    stats: pl.DataFrame, site_dir: Path = DEFAULT_SITE_DIR
) -> list[Path]:
    """Render every site EDA chart from the statistics frame; return the files
    written."""
    out = site_dir / "img" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    charts = (
        _yearbook_per_year(stats, out / "yearbook_per_year.svg"),
        _yearbook_gender(stats, out / "yearbook_gender_balance.svg"),
        _arxiv_per_year(stats, out / "arxiv_per_year.svg"),
        _arxiv_subject_mix(stats, out / "arxiv_subject_mix.svg"),
        _amazon_rating(stats, out / "amazon_rating.svg"),
        _amazon_mean(stats, out / "amazon_mean.svg"),
    )
    return [path for path in charts if path is not None]


def _sx(value: float, lo: float, hi: float) -> float:
    return _L if hi == lo else _L + (value - lo) / (hi - lo) * (_R - _L)


def _sy(value: float, vmin: float, vmax: float, top: int = _T) -> float:
    return _B if vmax == vmin else _B - (value - vmin) / (vmax - vmin) * (_B - top)


def _nice(value: float) -> float:
    if value <= 0:
        return 1
    base = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * base:
            return step * base
    return 10 * base


def _human(value: float) -> str:
    value = float(value)
    if value >= 1e6:
        return f"{value / 1e6:.0f}M" if value / 1e6 >= 10 else f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    if value <= 1:
        return f"{value:.0%}"
    return f"{value:.0f}"


def _year_ticks(lo: int, hi: int, step: int) -> list[tuple[float, str]]:
    start = math.ceil(lo / step) * step
    return [(_sx(year, lo, hi), str(year)) for year in range(start, hi + 1, step)]


def _axes(
    xticks: list[tuple[float, str]],
    ybot: tuple[float, str],
    ytop: tuple[float, str],
    xlabel: str,
    ylabel: str,
    top: int = _T,
) -> str:
    parts = [
        f'<line x1="{_L}" y1="{_B}" x2="{_R}" y2="{_B}" stroke="{_AXIS}" stroke-width="1.1"/>',
        f'<line x1="{_L}" y1="{_B}" x2="{_L}" y2="{top}" stroke="{_AXIS}" stroke-width="1.1"/>',
    ]
    for xp, label in xticks:
        parts.append(
            f'<line x1="{xp:.1f}" y1="{_B}" x2="{xp:.1f}" y2="{_B + 4}" stroke="{_AXIS}" stroke-width="1"/>'
            f'<text x="{xp:.1f}" y="{_B + 18}" font-size="11.5" fill="{_DIM}" text-anchor="middle">{label}</text>'
        )
    for yp, label in (ybot, ytop):
        parts.append(
            f'<text x="{_L - 8}" y="{yp + 4:.1f}" font-size="11" fill="{_DIM}" text-anchor="end">{label}</text>'
            f'<line x1="{_L - 3}" y1="{yp:.1f}" x2="{_L}" y2="{yp:.1f}" stroke="{_AXIS}" stroke-width="1"/>'
        )
    parts.append(
        f'<text x="350" y="{_H - 5}" font-size="12" fill="{_DIM}" text-anchor="middle" font-style="italic">{xlabel}</text>'
        f'<text x="15" y="113" font-size="12" fill="{_DIM}" text-anchor="middle" font-style="italic" transform="rotate(-90 15 113)">{ylabel}</text>'
    )
    return "".join(parts)


def _legend(items: list[tuple[str, str]]) -> str:
    parts, x = [], float(_L)
    for label, color in items:
        parts.append(
            f'<line x1="{x:.1f}" y1="11" x2="{x + 16:.1f}" y2="11" stroke="{color}" stroke-width="2.4"/>'
            f'<text x="{x + 21:.1f}" y="14.5" font-size="11" fill="{_DIM}">{label}</text>'
        )
        x += 26 + len(label) * 6.6
    return "".join(parts)


def _area_line(xy: list[tuple[float, float]], vmax: float) -> str:
    pts = [(_sx(x, xy[0][0], xy[-1][0]), _sy(y, 0, vmax)) for x, y in xy]
    area = (
        f"M {pts[0][0]:.1f} {_B} "
        + " ".join(f"L {x:.1f} {y:.1f}" for x, y in pts)
        + f" L {pts[-1][0]:.1f} {_B} Z"
    )
    line = " ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return (
        f'<path d="{area}" fill="{_FILL}"/>'
        f'<polyline points="{line}" fill="none" stroke="{_ACC}" stroke-width="2"/>'
    )


def _write(path: Path, body: str) -> Path:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'font-family="Inter, system-ui, sans-serif">{body}</svg>'
    )
    path.write_text(svg + "\n")
    return path


def _yearbook_per_year(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = (
        stats.filter(pl.col("dataset") == "yearbook")
        .group_by("slice")
        .agg(pl.col("count").sum())
        .sort("slice")
    )
    years = frame.get_column("slice").to_list()
    counts = frame.get_column("count").to_list()
    if not years:
        return None
    vmax = _nice(max(counts))
    body = _area_line(list(zip(years, counts, strict=True)), vmax)
    body += _axes(
        _year_ticks(years[0], years[-1], 20),
        (_B, "0"),
        (_T, _human(vmax)),
        "year",
        "portraits",
    )
    return _write(path, body)


def _yearbook_gender(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = stats.filter(
        (pl.col("dataset") == "yearbook") & (pl.col("group_kind") == "gender")
    )
    if frame.is_empty():
        return None
    wide = (
        frame.pivot(values="count", index="slice", on="group", aggregate_function="sum")
        .sort("slice")
        .fill_null(0)
    )
    years = wide.get_column("slice").to_list()
    groups = [c for c in wide.columns if c != "slice"]
    rows = wide.to_dicts()
    series, legend = [], []
    for index, group in enumerate(sorted(groups)):
        color = _PALETTE[index % len(_PALETTE)]
        xy = [(r["slice"], r[group] / max(sum(r[g] for g in groups), 1)) for r in rows]
        series.append((color, xy))
        legend.append((_GENDER.get(group, group), color))
    body = "".join(
        '<polyline points="'
        + " ".join(
            f"{_sx(x, years[0], years[-1]):.1f} {_sy(y, 0, 1, top=30):.1f}"
            for x, y in xy
        )
        + f'" fill="none" stroke="{color}" stroke-width="2"/>'
        for color, xy in series
    )
    body += _legend(legend)
    body += _axes(
        _year_ticks(years[0], years[-1], 20),
        (_B, "0%"),
        (30, "100%"),
        "year",
        "share of class",
        top=30,
    )
    return _write(path, body)


def _arxiv_per_year(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = stats.filter(
        (pl.col("dataset") == "arxiv") & (pl.col("group_kind") == "total")
    ).sort("slice")
    years = frame.get_column("slice").to_list()
    counts = frame.get_column("count").to_list()
    if not years:
        return None
    vmax = _nice(max(counts))
    body = _area_line(list(zip(years, counts, strict=True)), vmax)
    body += _axes(
        _year_ticks(years[0], years[-1], 5),
        (_B, "0"),
        (_T, _human(vmax)),
        "year",
        "papers",
    )
    return _write(path, body)


def _arxiv_subject_mix(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = stats.filter(
        (pl.col("dataset") == "arxiv") & (pl.col("group_kind") == "subject_share")
    )
    if frame.is_empty():
        return None
    totals = (
        frame.group_by("group")
        .agg(pl.col("count").sum())
        .sort(["count", "group"], descending=[True, False])
    )
    top = totals.get_column("group").to_list()[:5]
    years = sorted(frame.get_column("slice").unique().to_list())
    per_year = {
        r["slice"]: r["count"]
        for r in frame.group_by("slice")
        .agg(pl.col("count").sum())
        .iter_rows(named=True)
    }
    series, legend, peak = [], [], 0.0
    for index, subject in enumerate(top):
        color = _PALETTE[index % len(_PALETTE)]
        counts = {
            r["slice"]: r["count"]
            for r in frame.filter(pl.col("group") == subject).iter_rows(named=True)
        }
        xy = [(year, counts.get(year, 0) / per_year[year]) for year in years]
        peak = max(peak, max(share for _, share in xy))
        series.append((color, xy))
        legend.append((subject, color))
    vmax = min(1.0, math.ceil(peak * 10) / 10)
    body = "".join(
        '<polyline points="'
        + " ".join(
            f"{_sx(x, years[0], years[-1]):.1f} {_sy(y, 0, vmax, top=30):.1f}"
            for x, y in xy
        )
        + f'" fill="none" stroke="{color}" stroke-width="2"/>'
        for color, xy in series
    )
    body += _legend(legend)
    body += _axes(
        _year_ticks(years[0], years[-1], 5),
        (_B, "0%"),
        (30, _human(vmax)),
        "year",
        "share of papers",
        top=30,
    )
    return _write(path, body)


def _amazon_rating(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = stats.filter(
        (pl.col("dataset") == "amazon_reviews_23") & (pl.col("group_kind") == "rating")
    )
    if frame.is_empty():
        return None
    totals = {
        int(float(r["group"])): r["count"]
        for r in frame.group_by("group")
        .agg(pl.col("count").sum())
        .iter_rows(named=True)
    }
    cats = [1, 2, 3, 4, 5]
    values = [totals.get(c, 0) for c in cats]
    vmax = _nice(max(values))
    slot, width = (_R - _L) / len(cats), 44
    body = _axes([], (_B, "0"), (_T, _human(vmax)), "star rating", "reviews")
    for index, (cat, value) in enumerate(zip(cats, values, strict=True)):
        cx = _L + slot * (index + 0.5)
        height = value / vmax * (_B - _T)
        y = _B - height
        body += (
            f'<rect x="{cx - width / 2:.1f}" y="{y:.1f}" width="{width}" height="{height:.1f}" rx="3" fill="{_ACC}"/>'
            f'<text x="{cx:.1f}" y="{_B + 18}" font-size="12" fill="{_DIM}" text-anchor="middle">{cat}★</text>'
            f'<text x="{cx:.1f}" y="{y - 6:.1f}" font-size="10.5" fill="{_DIM}" text-anchor="middle">{_human(value)}</text>'
        )
    return _write(path, body)


def _amazon_mean(stats: pl.DataFrame, path: Path) -> Path | None:
    frame = stats.filter(
        (pl.col("dataset") == "amazon_reviews_23") & (pl.col("group_kind") == "rating")
    ).with_columns(
        (pl.col("group").cast(pl.Float64) * pl.col("count")).alias("weighted")
    )
    agg = (
        frame.group_by("slice")
        .agg(
            [
                pl.col("weighted").sum().alias("weighted"),
                pl.col("count").sum().alias("n"),
            ]
        )
        .sort("slice")
    )
    slices = agg.get_column("slice").to_list()
    if not slices:
        return None
    means = [
        w / n
        for w, n in zip(agg.get_column("weighted"), agg.get_column("n"), strict=True)
    ]
    vmin, vmax = math.floor(min(means) * 2) / 2, math.ceil(max(means) * 2) / 2
    lo, hi = slices[0], slices[-1]
    line = " ".join(
        f"{_sx(s, lo, hi):.1f} {_sy(m, vmin, vmax):.1f}"
        for s, m in zip(slices, means, strict=True)
    )
    step = max(2, ((hi - lo) // 5 // 2) * 2)
    xticks = [
        (_sx(s, lo, hi), str(half_year_year(s)))
        for s in range(lo + (lo % 2), hi + 1, step)
    ]
    body = f'<polyline points="{line}" fill="none" stroke="{_ACC}" stroke-width="2"/>'
    body += _axes(
        xticks, (_B, f"{vmin:.1f}"), (_T, f"{vmax:.1f}"), "year", "mean rating"
    )
    return _write(path, body)
