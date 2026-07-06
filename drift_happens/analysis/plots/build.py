"""Render every paper figure and table from frozen results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets import DATASETS, DatasetSpec
from drift_happens.analysis.datasets.locations import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_PAGES_DIR,
    DEFAULT_TABLES_DIR,
    DEFAULT_VALUES_JSON,
    DEFAULT_VALUES_PATH,
)
from drift_happens.analysis.plots import (
    appendix,
    curves,
    derive,
    matrices,
    overview,
    ranking,
    sources,
    tables,
    values,
)
from drift_happens.analysis.plots.names import figure_name, slugify


@dataclass
class BuildReport:
    outputs: list[Path]
    coverage: list[derive.Coverage]


def build(
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
    pages_dir: Path | None = DEFAULT_PAGES_DIR,
    *,
    frame: pl.DataFrame | None = None,
    stats: pl.DataFrame | None = None,
    params: pl.DataFrame | None = None,
    expected: dict[str, list[str]] | None = None,
) -> BuildReport:
    """
    Render every figure, table, and appendix page from frozen results.

    ``expected`` maps each dataset to its planned trainers; models that are absent or
    only partially run are still rendered (and flagged) so the appendix shows the whole
    lineup. Aggregate figures and tables only use models with complete runs.
    """
    frame = sources.results() if frame is None else frame
    outputs: list[Path] = []
    coverage: list[derive.Coverage] = []
    combined_panels: list[
        tuple[derive.DriftMatrix, DatasetSpec, tuple[float, float]]
    ] = []

    if stats is not None:
        outputs.extend(overview.build_overview(stats, figures_dir))
    if params is not None:
        for dataset, spec in DATASETS.items():
            roster = params.filter(pl.col("dataset") == dataset)
            if not roster.is_empty():
                outputs.append(
                    tables.roster_table(
                        dataset, roster, tables_dir / f"{spec.slug}_roster.tex"
                    )
                )
        values.write_values(frame, params, DEFAULT_VALUES_PATH, DEFAULT_VALUES_JSON)

    for dataset, spec in DATASETS.items():
        models, model_coverage = derive.per_model_matrices(frame, dataset)
        coverage.append(model_coverage)
        if not models:
            continue
        target = figures_dir / spec.slug
        mean = derive.mean_over_models(models)
        forgetting, forgetting_coverage = derive.forgetting_curves(frame, dataset)
        coverage.append(forgetting_coverage)
        family_of = derive.families(frame, dataset)
        scores = derive.rankings(dataset, models)
        cutoffs = derive.select_cutoffs(models[0].slices)

        lineup = derive.lineup_matrices(
            frame, dataset, expected.get(dataset) if expected else None
        )
        shown = [entry for entry in lineup if entry.status is derive.Status.COMPLETE]
        raw_range = derive.raw_extent([mean, *(entry.matrix for entry in shown)])
        outputs.extend(_render_lineup(shown, mean, spec, target, raw_range))
        outputs.append(
            matrices.heatmap(
                mean,
                spec,
                target / f"mean_{spec.matrix_stem}_matrix.pdf",
                raw_range=raw_range,
            )
        )
        combined_panels.append((mean, spec, raw_range))
        outputs.append(curves.forgetting(forgetting, spec, target / "forgetting.pdf"))
        outputs.append(
            curves.forgetting_compact(forgetting, spec, target / "forgetting_lncs.pdf")
        )
        outputs.append(
            curves.forgetting_families(
                derive.forgetting_by_family(forgetting, family_of),
                spec,
                target / "forgetting_by_family.pdf",
            )
        )
        for score in scores:
            outputs.append(
                ranking.bars(score, spec, target / f"ranking_{score.kind}.pdf")
            )
        outputs.extend(
            _render_tables(dataset, spec, models, scores, cutoffs, frame, tables_dir)
        )
        if pages_dir is not None:
            has_roster = (
                params is not None
                and not params.filter(pl.col("dataset") == dataset).is_empty()
            )
            outputs.append(
                appendix.appendix_page(
                    spec,
                    lineup,
                    cutoffs,
                    family_of,
                    has_roster,
                    appendix.page_path(dataset, pages_dir),
                )
            )

    if combined_panels:
        outputs.append(
            matrices.combined_means(
                combined_panels, figures_dir / "combined" / "mean_matrices.pdf"
            )
        )
    return BuildReport(outputs, coverage)


def _render_lineup(
    lineup: list[derive.LineupEntry],
    mean: derive.DriftMatrix,
    spec: DatasetSpec,
    target: Path,
    raw_range: tuple[float, float],
) -> list[Path]:
    """Raw and shared-scale deviation heatmaps for each model in the lineup."""
    deviations = derive.deviations([entry.matrix for entry in lineup], mean)
    extent = derive.deviation_extent(deviations)
    paths: list[Path] = []
    for entry, deviation in zip(lineup, deviations, strict=True):
        name = figure_name(entry.matrix.model)
        paths.append(
            matrices.heatmap(
                entry.matrix,
                spec,
                target / "raw" / f"{name}_drift_matrix.pdf",
                raw_range=raw_range,
            )
        )
        paths.append(
            matrices.heatmap(
                deviation,
                spec,
                target / "deviation" / f"{name}_deviation.pdf",
                diverging=True,
                extent=extent,
            )
        )
    return paths


def _render_tables(
    dataset: str,
    spec: DatasetSpec,
    models: list[derive.DriftMatrix],
    scores: list[derive.Ranking],
    cutoffs: list[str],
    frame: pl.DataFrame,
    tables_dir: Path,
) -> list[Path]:
    """Robustness, per-cutoff, and per-family tables for one dataset."""
    paths = [
        tables.robustness_table(
            dataset, scores, tables_dir / f"robustness_{spec.slug}.tex"
        )
    ]
    for cutoff in cutoffs:
        rows = derive.cutoff_rows(dataset, models, cutoff)
        paths.append(
            tables.cutoff_table(
                dataset,
                cutoff,
                rows,
                tables_dir / f"{spec.slug}_cutoff_{slugify(cutoff)}.tex",
            )
        )
    family = derive.family_rows(frame, dataset, models, cutoffs)
    paths.append(
        tables.family_table(
            dataset, cutoffs, family, tables_dir / f"{spec.slug}_by_family.tex"
        )
    )
    return paths
