"""Analysis commands: build results, render figures and tables, export site data,
verify."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from drift_happens.analysis.datasets.locations import (
    DATASET_STATS_PARQUET,
    DEFAULT_FIGURES_DIR,
    DEFAULT_PAGES_DIR,
    DEFAULT_TABLES_DIR,
    FIGURES_MANIFEST,
    PARAMS_PARQUET,
)
from drift_happens.utils.paths import PROJECT_ROOT, RUNS_DIR, relative_to_project

if TYPE_CHECKING:
    import polars as pl

app = typer.Typer(
    help="Build results, render paper figures and tables, export site data, and verify.",
    no_args_is_help=True,
)


@app.command("export")
def export_runs(
    model_params_cache_only: Annotated[
        bool,
        typer.Option(
            "--model-params-cache-only",
            help="Use only locally cached Hugging Face files for parameter counts.",
        ),
    ] = False,
) -> None:
    """Freeze local runs and dataset statistics into the analysis parquets."""
    from drift_happens.analysis.export.__main__ import export
    from drift_happens.analysis.export.dataset_stats import freeze_dataset_stats
    from drift_happens.analysis.export.params import freeze_params

    lock = export()
    typer.echo(f"froze {len(lock['runs'])} runs, {len(lock['missing'])} missing")
    try:
        freeze_dataset_stats()
        typer.echo("froze dataset statistics")
    except Exception as error:  # raw data may be absent; statistics are optional
        typer.echo(f"dataset statistics skipped: {error}")
    try:
        freeze_params(local_files_only=model_params_cache_only)
        typer.echo("froze model parameters")
    except Exception as error:  # model construction may be unavailable
        typer.echo(f"model parameters skipped: {error}")


@app.command("pull")
def pull_runs() -> None:
    """Download finished conference eval drift matrices from W&B into the runs dir."""
    from drift_happens.analysis.export.wandb_pull import pull_drift_matrices

    pulled = pull_drift_matrices()
    typer.echo(f"pulled {len(pulled)} drift matrices")


@app.command("figures")
def render_figures() -> None:
    """Render every figure and table, then write the checksum manifest."""
    from drift_happens.analysis.export import runs
    from drift_happens.analysis.plots import build, checksums, derive

    stats, params = _frozen_inputs()
    try:
        report = build.build(
            stats=stats, params=params, expected=runs.expected_trainers()
        )
    except FileNotFoundError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(derive.coverage_summary(report.coverage))
    checksums.write(report.outputs, PROJECT_ROOT, FIGURES_MANIFEST)
    typer.echo(f"wrote {len(report.outputs)} files")


@app.command("saliency")
def render_saliency(
    cutoff: list[int] = typer.Option(
        ..., "--cutoff", help="Train cutoffs (repeatable)."
    ),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    trainer: list[str] = typer.Option(
        ["cnn_l", "resnet_s", "mlp_l"], "--trainer", help="Model presets (repeatable)."
    ),
    eval_year: list[int] = typer.Option([], "--eval-year", help="Evaluation years."),
    out: Path = typer.Option(
        DEFAULT_FIGURES_DIR / "yearbook" / "saliency.pdf", "--out"
    ),
    panels_dir: Path | None = typer.Option(
        None, "--panels-dir", help="Also save one image per model and cutoff."
    ),
    sample_per_year: int | None = typer.Option(
        None,
        "--sample-per-year",
        help="Sampling cap per evaluation year; uncapped years use all available portraits.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for deterministic sampling."),
    sample_seed: list[int] = typer.Option(
        [],
        "--sample-seed",
        help="Sample seed per evaluation year, in --eval-year order.",
    ),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Render the Yearbook gradient-saliency grid from trained checkpoints."""
    import torch

    from drift_happens.analysis.plots.saliency import (
        DEFAULT_EVAL_YEARS,
        build_yearbook_saliency,
        save_yearbook_saliency_panels,
    )

    try:
        selected_eval_years = eval_year or DEFAULT_EVAL_YEARS
        path = build_yearbook_saliency(
            runs_root,
            trainer,
            cutoff,
            out,
            eval_years=selected_eval_years,
            sample_per_year=sample_per_year,
            seed=seed,
            sample_seeds=sample_seed or None,
            device=torch.device(device),
        )
        if panels_dir is not None:
            save_yearbook_saliency_panels(
                runs_root,
                trainer,
                cutoff,
                panels_dir,
                eval_years=selected_eval_years,
                sample_per_year=sample_per_year,
                seed=seed,
                sample_seeds=sample_seed or None,
                device=torch.device(device),
            )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"wrote {relative_to_project(path)}")


@app.command("site")
def export_site() -> None:
    """Export drift matrices, roster manifests, and result tables as JSON for the
    website."""
    from drift_happens.analysis.export import runs
    from drift_happens.analysis.plots import site, site_eda, sources

    try:
        paths = site.build_site_data(
            sources.results(), expected=runs.expected_trainers()
        )
    except FileNotFoundError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    if DATASET_STATS_PARQUET.exists():
        paths += site_eda.build_site_eda(sources.dataset_stats())
    typer.echo(f"wrote {len(paths)} site data files")


@app.command("verify")
def verify_figures() -> None:
    """Rebuild the figures and check them against the committed manifest."""
    from drift_happens.analysis.plots import build, checksums

    if not FIGURES_MANIFEST.exists():
        typer.echo(
            f"{relative_to_project(FIGURES_MANIFEST)} not found; run figures first"
        )
        raise typer.Exit(code=1)
    stats, params = _frozen_inputs()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        try:
            report = build.build(
                root / DEFAULT_FIGURES_DIR.relative_to(PROJECT_ROOT),
                root / DEFAULT_TABLES_DIR.relative_to(PROJECT_ROOT),
                root / DEFAULT_PAGES_DIR.relative_to(PROJECT_ROOT),
                stats=stats,
                params=params,
            )
        except FileNotFoundError as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error
        mismatches = checksums.verify(report.outputs, root, FIGURES_MANIFEST)
    if mismatches:
        typer.echo("outputs differ from the manifest:")
        for name in mismatches:
            typer.echo(f"  {name}")
        raise typer.Exit(code=1)
    typer.echo("outputs match the manifest")


def _frozen_inputs() -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load the optional dataset-statistics and model-parameter tables if present."""
    from drift_happens.analysis.plots import sources

    stats = sources.dataset_stats() if DATASET_STATS_PARQUET.exists() else None
    params = sources.params() if PARAMS_PARQUET.exists() else None
    return stats, params
