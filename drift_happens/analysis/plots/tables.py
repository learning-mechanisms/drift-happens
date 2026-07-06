"""LaTeX tables built from frozen results."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets import DATASETS, DatasetSpec
from drift_happens.analysis.plots.derive import (
    DECAY,
    FUTURE_PERFORMANCE,
    CutoffRow,
    FamilyRow,
    Ranking,
)
from drift_happens.analysis.plots.latex import tex, tex_name
from drift_happens.analysis.plots.names import FAMILY_LABELS, slice_label, slugify


def robustness_table(dataset: str, rankings: list[Ranking], path: Path) -> Path:
    """
    Per-model future performance and decay, ordered by decay.

    Rendered as two side-by-side panels under a single caption (the right panel
    continues the left), so the narrow table does not strand a full column of whitespace
    in the appendix.
    """
    spec = DATASETS[dataset]
    by_kind = {ranking.kind: ranking for ranking in rankings}
    future = dict(
        zip(
            by_kind[FUTURE_PERFORMANCE].models,
            by_kind[FUTURE_PERFORMANCE].score,
            strict=True,
        )
    )
    decay = by_kind[DECAY]
    unit = _unit_paren(spec)
    header = rf"Model & Future{unit} & Decay{unit} \\"
    body = [
        " & ".join(
            (
                tex_name(model),
                _format(future.get(model, math.nan), spec.value_fmt),
                _format(score, spec.value_fmt),
            )
        )
        + r" \\"
        for model, score in zip(decay.models, decay.score, strict=True)
    ]
    return _write_split_table(
        caption=rf"Temporal robustness on {tex(spec.title)}.",
        label=f"robustness_{spec.slug}",
        columns="l rr",
        header=header,
        body=body,
        path=path,
    )


def cutoff_table(dataset: str, cutoff: str, rows: list[CutoffRow], path: Path) -> Path:
    """
    Top models trained up to one cutoff, with future performance and decay.

    Emitted as a side-by-side ``minipage`` panel rather than a standalone float, so the
    per-cutoff tables pack two-up in the appendix (see ``_result_tables`` in
    ``appendix.py``). Each panel still carries its own numbered table caption via
    ``\\captionof``. The dense scale and reduced column padding keep two panels within
    the LNCS text width.
    """
    spec = DATASETS[dataset]
    unit = _unit_paren(spec)
    caption = (
        rf"{tex(spec.title)}: models trained up to "
        rf"{tex(slice_label(cutoff, spec))}, ordered by future performance."
    )
    lines = [
        r"\begin{minipage}[t]{0.49\linewidth}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\captionof{{table}}{{{caption}}}",
        rf"\label{{tab:{spec.slug}_cutoff_{slugify(cutoff)}}}",
        # Shrink to the panel width only if the table is wider (as in the narrow
        # LNCS column); never enlarge it in the roomy one-column preprint appendix.
        r"\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%",
        _open("c l rrr"),
        r"\toprule",
        rf"Rank & Model & {tex(spec.metric_label)}{unit} & Future{unit} & Decay{unit} \\",
        r"\midrule",
    ]
    for rank, row in enumerate(rows, start=1):
        cells = [
            str(rank),
            tex_name(row.model),
            _format(row.in_distribution, spec.value_fmt),
            _format(row.future, spec.value_fmt),
            _format(row.decay, spec.value_fmt),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{minipage}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def family_table(
    dataset: str, cutoffs: list[str], rows: list[FamilyRow], path: Path
) -> Path:
    """Mean future performance and decay per model family at each cutoff."""
    spec = DATASETS[dataset]
    columns = "l " + "rr" * len(cutoffs)
    spanned = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{tex(slice_label(cutoff, spec))}}}"
        for cutoff in cutoffs
    )
    subheader = " & ".join("Future & Decay" for _ in cutoffs)
    lines = [
        *_header(
            rf"{tex(spec.title)}: future performance and decay by model family.",
            f"{spec.slug}_by_family",
        ),
        _open(columns),
        r"\toprule",
        rf" & {spanned} \\",
        rf"Family & {subheader} \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [_family_label(row.family)]
        for future, decay in row.cells:
            cells += [_format(future, spec.value_fmt), _format(decay, spec.value_fmt)]
        lines.append(" & ".join(cells) + r" \\")
    return _write(lines, path)


_FROZEN_FAMILIES = (
    "image-transfer",
    "text-frozen-head",
    "text-frozen-head-regression",
)


def roster_table(dataset: str, rows: pl.DataFrame, path: Path) -> Path:
    """
    Parameter counts split into from-scratch architectures and frozen encoders.

    The two rosters are emitted as side-by-side panels (each independently numbered)
    rather than stacked full-width floats, which roughly halves their vertical footprint
    in the appendix.
    """
    spec = DATASETS[dataset]
    frozen = pl.col("trainer_family").is_in(_FROZEN_FAMILIES)
    panels = [
        panel
        for panel in (
            _roster_panel(
                spec,
                rows.filter(~frozen),
                "models trained from scratch",
                f"{spec.slug}_roster",
            ),
            _roster_panel(
                spec,
                rows.filter(frozen),
                "frozen pretrained encoders, with trainable head and total parameters",
                f"{spec.slug}_roster_frozen",
            ),
        )
        if panel
    ]
    lines = [r"\noindent"]
    for index, panel in enumerate(panels):
        if index:
            lines.append(r"\hfill")
        lines += panel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _roster_panel(
    spec: DatasetSpec, rows: pl.DataFrame, caption: str, label: str
) -> list[str]:
    if rows.is_empty():
        return []
    ordered = rows.sort(["total", "trainer"])
    head_only = (
        ordered.filter(pl.col("trainable") == pl.col("total")).height == ordered.height
    )
    if head_only:
        columns = "l l r"
        header = r"Model & Family & Parameters \\"
        body = [
            f"{tex_name(record['trainer'])} & "
            f"{_family_label(record['trainer_family'])} & "
            f"{_humanize(record['total'])} \\\\"
            for record in ordered.iter_rows(named=True)
        ]
    else:
        columns = "l l rr"
        header = r"Model & Family & Trainable & Total \\"
        body = [
            f"{tex_name(record['trainer'])} & "
            f"{_family_label(record['trainer_family'])} & "
            f"{_humanize(record['trainable'])} & {_humanize(record['total'])} \\\\"
            for record in ordered.iter_rows(named=True)
        ]
    return [
        r"\begin{minipage}[t]{0.49\linewidth}",
        r"\centering",
        r"\footnotesize",
        rf"\captionof{{table}}{{{tex(spec.title)}: {caption}.}}",
        rf"\label{{tab:{label}}}",
        *_panel_tabular(columns, header, body),
        r"\end{minipage}",
    ]


def _humanize(value: int) -> str:
    """Compact parameter count: 124_729_620 -> '124.7M', 83_988 -> '84k', 770 ->
    '770'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _header(caption: str, label: str) -> list[str]:
    return [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
    ]


def _open(columns: str) -> str:
    return rf"\begin{{tabular}}{{{columns}}}"


def _write(lines: list[str], path: Path) -> Path:
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _panel_tabular(columns: str, header: str, body: list[str]) -> list[str]:
    """
    A booktabs tabular that shrinks to the enclosing panel only if it is wider.

    The ``\\resizebox`` guard keeps the natural size in the roomy one-column preprint
    appendix and shrinks only in the narrower LNCS text column, matching the per-cutoff
    panels.
    """
    return [
        r"\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%",
        _open(columns),
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
    ]


def _write_split_table(
    *,
    caption: str,
    label: str,
    columns: str,
    header: str,
    body: list[str],
    path: Path,
) -> Path:
    """
    One captioned table float whose rows are split across two side-by-side panels.

    The right panel continues the left, halving the vertical footprint of a narrow but
    long table.
    """
    mid = (len(body) + 1) // 2
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        r"\begin{minipage}[t]{0.49\linewidth}",
        r"\centering",
        *_panel_tabular(columns, header, body[:mid]),
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\linewidth}",
        r"\centering",
        *_panel_tabular(columns, header, body[mid:]),
        r"\end{minipage}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, tex(family))


def _unit(suffix: str) -> str:
    return r"\%" if suffix == "%" else suffix


def _unit_paren(spec: DatasetSpec) -> str:
    return f" ({_unit(spec.unit_suffix)})" if spec.unit_suffix else ""


def _format(value: float, fmt: str) -> str:
    return "--" if math.isnan(value) else f"{value:{fmt}}"
