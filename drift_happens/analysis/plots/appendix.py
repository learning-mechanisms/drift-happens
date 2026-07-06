"""Per-dataset drift-matrix appendix pages, generated from the frozen results."""

from __future__ import annotations

from pathlib import Path

from drift_happens.analysis.datasets import DatasetSpec
from drift_happens.analysis.plots.derive import LineupEntry, Status
from drift_happens.analysis.plots.latex import tex, tex_name
from drift_happens.analysis.plots.names import (
    FAMILY_LABELS,
    figure_name,
    get_display_name,
    slugify,
)
from drift_happens.dataset.arxiv.scope import ARXIV_TARGET_LABELS

_APPENDIX_STEM = {
    "yearbook": "c_yearbook",
    "amazon_reviews_23": "d_amazon",
    "arxiv": "e_arxiv",
}
_FIGURES_PREFIX = "plots_experiments"
_TABLES_PREFIX = "tables"

# Families are grouped in this order; any family absent from the list follows, sorted.
_FAMILY_ORDER = (
    "image-mlp",
    "image-cnn",
    "image-resnet",
    "image-vit",
    "image-transfer",
    "text-ffn",
    "text-ffn-regression",
    "text-textcnn",
    "text-textcnn-regression",
    "text-rnn",
    "text-rnn-regression",
    "text-tx",
    "text-tx-regression",
    "text-frozen-head",
    "text-frozen-head-regression",
)
_SIZE_RANK = {"S": 0, "M": 1, "B": 2, "L": 3}


def page_path(dataset: str, pages_dir: Path) -> Path:
    return pages_dir / f"{_APPENDIX_STEM[dataset]}.tex"


def appendix_page(
    spec: DatasetSpec,
    lineup: list[LineupEntry],
    cutoffs: list[str],
    family_of: dict[str, str],
    has_roster: bool,
    path: Path,
) -> Path:
    """Write the drift-matrix appendix for one dataset: a per-family gallery, the
    forgetting and ranking summaries, and the result tables."""
    shown = [item for item in lineup if item.status is Status.COMPLETE]
    intro = (
        r"\noindent The cohort-mean and per-model deviation matrices shown here, and the "
        r"in-distribution, future, and decay quantities tabulated below, are defined in "
        r"Section~\ref{subsec:drift_summary}."
    )
    if spec.slug == "arxiv":
        labels = [rf"\texttt{{{c}}}" for c in ARXIV_TARGET_LABELS]
        cats = ", ".join(labels[:-1]) + ", and " + labels[-1]
        intro += rf" The label space comprises the leaf categories {cats}."
    lines = [
        r"\captionsetup[figure]{font=footnotesize,labelfont=footnotesize}",
        "",
        rf"\section{{{tex(spec.title)} -- Drift Matrices}}",
        rf"\label{{app:{spec.slug}}}",
        "",
        intro,
        "",
        *_mean_figure(spec),
    ]
    if has_roster:
        lines += [
            r"\subsection{Model Roster}",
            "",
            rf"\input{{{_TABLES_PREFIX}/{spec.slug}_roster.tex}}",
            "",
        ]
    for key in _ordered_families(shown, family_of):
        members = sorted(_members(shown, family_of, key), key=_model_order)
        lines.append(rf"\subsection{{{tex(FAMILY_LABELS.get(key, key))}}}")
        lines.append("")
        lines += _family_figure(spec, key, members)
    lines += _summary_figures(spec)
    lines += _result_tables(spec, cutoffs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _ordered_families(shown: list[LineupEntry], family_of: dict[str, str]) -> list[str]:
    present = {family_of.get(item.matrix.model, "other") for item in shown}
    head = [key for key in _FAMILY_ORDER if key in present]
    return head + sorted(present - set(head))


def _members(
    shown: list[LineupEntry], family_of: dict[str, str], key: str
) -> list[LineupEntry]:
    return [item for item in shown if family_of.get(item.matrix.model, "other") == key]


def _model_order(item: LineupEntry) -> tuple[str, int, str]:
    name = get_display_name(item.matrix.model)
    architecture, _, size = name.rpartition("-")
    return architecture.lower(), _SIZE_RANK.get(size, 99), name


def _mean_figure(spec: DatasetSpec) -> list[str]:
    source = f"{_FIGURES_PREFIX}/{spec.slug}/mean_{spec.matrix_stem}_matrix.pdf"
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=0.5\textwidth]{{{source}}}",
        rf"\caption{{Cohort-mean {tex(spec.metric_label)} matrix $\bar{{M}}$ over the "
        rf"{tex(spec.title)} models. Cell $(i,j)$ is the mean across "
        r"those models of the score from training through slice $i$ and evaluating on "
        r"slice $j$.}",
        rf"\label{{fig:{spec.slug}_mean_matrix}}",
        r"\end{figure}",
        "",
    ]


def _model_panel(spec: DatasetSpec, item: LineupEntry) -> str:
    name = figure_name(item.matrix.model)
    raw = f"{_FIGURES_PREFIX}/{spec.slug}/raw/{name}_drift_matrix.pdf"
    deviation = f"{_FIGURES_PREFIX}/{spec.slug}/deviation/{name}_deviation.pdf"
    return "\n".join(
        [
            r"\begin{subfigure}[t]{0.49\textwidth}\centering",
            rf"    \includegraphics[width=0.49\linewidth]{{{raw}}}\hfill",
            rf"    \includegraphics[width=0.49\linewidth]{{{deviation}}}",
            rf"    \caption{{{tex_name(item.matrix.model)}}}",
            rf"    \label{{fig:{spec.slug}_{name.replace('-', '_')}}}",
            r"\end{subfigure}",
        ]
    )


def _family_figure(
    spec: DatasetSpec, key: str, members: list[LineupEntry]
) -> list[str]:
    rows = [
        "\n\\hfill\n".join(
            _model_panel(spec, item) for item in members[start : start + 2]
        )
        for start in range(0, len(members), 2)
    ]
    family = tex(FAMILY_LABELS.get(key, key))
    return [
        r"\begin{figure}[H]",
        r"\centering",
        "\n\n".join(rows),
        rf"\caption{{{family} models: {tex(spec.metric_label)} drift matrix $M^{{(m)}}$ "
        r"and deviation from the cohort mean $\Delta^{(m)} = M^{(m)} - \bar{M}$ for each "
        r"model, shown on a sequential and a zero-centred diverging scale, respectively.}",
        rf"\label{{fig:{spec.slug}_family_{slugify(key)}}}",
        r"\end{figure}",
        "",
    ]


def _summary_figures(spec: DatasetSpec) -> list[str]:
    forgetting = f"{_FIGURES_PREFIX}/{spec.slug}/forgetting.pdf"
    family = f"{_FIGURES_PREFIX}/{spec.slug}/forgetting_by_family.pdf"
    future = f"{_FIGURES_PREFIX}/{spec.slug}/ranking_future_performance.pdf"
    decay = f"{_FIGURES_PREFIX}/{spec.slug}/ranking_decay.pdf"
    return [
        r"\subsection{Forgetting and Rankings}",
        rf"\label{{app:{spec.slug}_forgetting}}",
        "",
        r"\noindent To see how quickly each model forgets, we summarize its drift matrix as a "
        r"forgetting curve. The curve plots the "
        rf"{tex(spec.metric_label)} against the lag $\ell = j - i$, the number of slices between "
        r"the training cutoff $i$ and the evaluation slice $j$. At each lag we average over all "
        r"training cutoffs,",
        r"\[ F(\ell) = \operatorname{mean}_{i} M_{i,\,i+\ell}. \]",
        r"The result is the "
        rf"{tex(spec.metric_label)} at a fixed temporal distance, independent of which period a "
        r"model was trained on. This separates the effect of temporal distance from the difficulty "
        r"of any single slice.",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{subfigure}[t]{0.49\textwidth}\centering",
        rf"    \includegraphics[width=\linewidth]{{{forgetting}}}",
        r"    \caption{Per model}",
        rf"    \label{{fig:{spec.slug}_forgetting}}",
        r"\end{subfigure}\hfill",
        r"\begin{subfigure}[t]{0.49\textwidth}\centering",
        rf"    \includegraphics[width=\linewidth]{{{family}}}",
        r"    \caption{Per model family}",
        rf"    \label{{fig:{spec.slug}_forgetting_family}}",
        r"\end{subfigure}",
        r"\caption{Forgetting curves: each model (left) and averaged within each "
        r"family (right).}",
        rf"\label{{fig:{spec.slug}_forgetting_combined}}",
        r"\end{figure}",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{subfigure}[t]{0.49\textwidth}\centering",
        rf"    \includegraphics[width=\linewidth]{{{future}}}",
        r"    \caption{By future performance}",
        rf"    \label{{fig:{spec.slug}_ranking_future}}",
        r"\end{subfigure}\hfill",
        r"\begin{subfigure}[t]{0.49\textwidth}\centering",
        rf"    \includegraphics[width=\linewidth]{{{decay}}}",
        r"    \caption{By decay}",
        rf"    \label{{fig:{spec.slug}_ranking_decay}}",
        r"\end{subfigure}",
        r"\caption{Models ranked by mean future performance and by temporal decay.}",
        rf"\label{{fig:{spec.slug}_rankings}}",
        r"\end{figure}",
        "",
    ]


def _result_tables(spec: DatasetSpec, cutoffs: list[str]) -> list[str]:
    """
    Robustness table, then the per-cutoff tables packed two-up, then the per-family
    table.

    The cutoff tables are ``minipage`` panels (see ``tables.cutoff_table``), so two sit
    side by side per row.
    """
    lines = [
        r"\subsection{Result Tables}",
        "",
        rf"\input{{{_TABLES_PREFIX}/robustness_{spec.slug}.tex}}",
        "",
    ]
    stems = [f"{spec.slug}_cutoff_{slugify(cutoff)}" for cutoff in cutoffs]
    for start in range(0, len(stems), 2):
        row = "\\hfill\n".join(
            rf"\input{{{_TABLES_PREFIX}/{stem}.tex}}"
            for stem in stems[start : start + 2]
        )
        lines += [r"\noindent", row, ""]
        if start + 2 < len(stems):
            lines += [r"\vspace{1.5ex}", ""]
    lines += [rf"\input{{{_TABLES_PREFIX}/{spec.slug}_by_family.tex}}"]
    return lines
