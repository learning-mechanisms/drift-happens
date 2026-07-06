from __future__ import annotations

import typer

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def create_app(*, help_text: str = "Download and prepare datasets.") -> typer.Typer:
    app = typer.Typer(
        help=help_text,
        no_args_is_help=True,
        context_settings=CONTEXT_SETTINGS,
    )
    app.add_typer(_yearbook_app(), name="yearbook")
    app.add_typer(_arxiv_app(), name="arxiv")
    app.add_typer(_amazon_reviews_23_app(), name="amazon-reviews-23")
    app.add_typer(_imdb_faces_app(), name="imdb-faces")
    return app


def _yearbook_app() -> typer.Typer:
    app = typer.Typer(
        help="Set up the Yearbook dataset.",
        no_args_is_help=True,
        context_settings=CONTEXT_SETTINGS,
    )

    @app.command("download")
    def download() -> None:
        from drift_happens.dataset.yearbook.cli import download as run

        run()

    @app.command("unpack")
    def unpack() -> None:
        from drift_happens.dataset.yearbook.cli import unpack as run

        run()

    @app.command("prepare")
    def prepare() -> None:
        from drift_happens.dataset.yearbook.cli import prepare as run

        run()

    @app.command("full")
    def full(yes: bool = typer.Option(False, "--yes")) -> None:
        from drift_happens.dataset.yearbook.cli import full_pipeline as run

        run(yes=yes)

    return app


def _arxiv_app() -> typer.Typer:
    app = typer.Typer(
        help="Set up the arXiv dataset.",
        no_args_is_help=True,
        context_settings=CONTEXT_SETTINGS,
    )

    @app.command("download")
    def download() -> None:
        from drift_happens.dataset.arxiv.cli import download as run

        run()

    @app.command("prepare")
    def prepare() -> None:
        from drift_happens.dataset.arxiv.cli import prepare as run

        run()

    @app.command("full")
    def full(yes: bool = typer.Option(False, "--yes")) -> None:
        from drift_happens.dataset.arxiv.cli import full_pipeline as run

        run(yes=yes)

    return app


def _amazon_reviews_23_app() -> typer.Typer:
    app = typer.Typer(
        help="Set up the Amazon Reviews 2023 dataset.",
        no_args_is_help=True,
        context_settings=CONTEXT_SETTINGS,
    )

    @app.command("build-from-cache")
    def build_from_cache() -> None:
        from drift_happens.dataset.amazon_reviews_23.cli import build_from_cache as run

        run()

    @app.command("download-reviews")
    def download_reviews(skip_existing: bool = True) -> None:
        from drift_happens.dataset.amazon_reviews_23.cli import download_reviews as run

        run(skip_existing=skip_existing)

    @app.command("merge-review-categories")
    def merge_review_categories() -> None:
        from drift_happens.dataset.amazon_reviews_23.cli import (
            merge_review_categories as run,
        )

        run()

    @app.command("full")
    def full(from_cache: bool = True, yes: bool = typer.Option(False, "--yes")) -> None:
        from drift_happens.dataset.amazon_reviews_23.cli import full_pipeline as run

        run(from_cache=from_cache, yes=yes)

    return app


def _imdb_faces_app() -> typer.Typer:
    app = typer.Typer(
        help="Set up the IMDB Faces dataset.",
        no_args_is_help=True,
        context_settings=CONTEXT_SETTINGS,
    )

    @app.command("download")
    def download() -> None:
        from drift_happens.dataset.imdb_faces.cli import download as run

        run()

    @app.command("unpack")
    def unpack() -> None:
        from drift_happens.dataset.imdb_faces.cli import unpack as run

        run()

    @app.command("prepare")
    def prepare() -> None:
        from drift_happens.dataset.imdb_faces.cli import prepare as run

        run()

    @app.command("full")
    def full(yes: bool = typer.Option(False, "--yes")) -> None:
        from drift_happens.dataset.imdb_faces.cli import full_pipeline as run

        run(yes=yes)

    return app
