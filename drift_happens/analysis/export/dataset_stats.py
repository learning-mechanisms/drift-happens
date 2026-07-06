"""Freeze per-slice dataset counts into the statistics parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from drift_happens.analysis.datasets import schema
from drift_happens.analysis.datasets.locations import DATASET_STATS_PARQUET


def freeze_dataset_stats(output: Path = DATASET_STATS_PARQUET) -> Path:
    """Load each raw dataset once and write its slice counts to the parquet."""
    stats = pl.concat([_yearbook(), _amazon(), _arxiv()], how="vertical")
    output.parent.mkdir(parents=True, exist_ok=True)
    schema.check_stats(stats).write_parquet(output)
    return output


def _yearbook() -> pl.DataFrame:
    from drift_happens.dataset.yearbook.transform import load_downscaled_images_into_df

    frame = pl.from_pandas(load_downscaled_images_into_df()[["year", "gender"]])
    return (
        frame.group_by("year", "gender")
        .len()
        .select(
            dataset=pl.lit("yearbook"),
            slice_kind=pl.lit("year"),
            slice=pl.col("year").cast(pl.Int64),
            group_kind=pl.lit("gender"),
            group=pl.col("gender").cast(pl.Utf8),
            count=pl.col("len").cast(pl.Int64),
        )
    )


def _arxiv() -> pl.DataFrame:
    from collections import Counter

    from drift_happens.dataset.arxiv.load import ARXIV_PREPROCESSED_DF
    from drift_happens.dataset.arxiv.scope import (
        ARXIV_MAX_YEAR,
        ARXIV_MIN_YEAR,
        map_arxiv_categories_to_labels,
    )

    # Read only the two columns the scope needs; the title/abstract text is multi-GB.
    light = pl.read_parquet(ARXIV_PREPROCESSED_DF, columns=["created", "categories"])
    created = light["created"]
    if created.dtype not in (pl.Date, pl.Datetime):
        created = created.str.to_datetime(strict=False)
    year_total: Counter[int] = Counter()
    subject_total: Counter[str] = Counter()
    subject_year: Counter[tuple[int, str]] = Counter()
    for year, categories in zip(
        created.dt.year().to_list(), light["categories"].to_list(), strict=True
    ):
        if year is None or year < ARXIV_MIN_YEAR or year > ARXIV_MAX_YEAR:
            continue
        mapped = map_arxiv_categories_to_labels(categories)
        if not mapped:
            continue
        year_total[year] += 1
        for subject in mapped:
            subject_total[subject] += 1
            subject_year[(year, subject)] += 1
    years = pl.DataFrame(
        {"slice": list(year_total), "count": list(year_total.values())}
    ).select(
        dataset=pl.lit("arxiv"),
        slice_kind=pl.lit("year"),
        slice=pl.col("slice").cast(pl.Int64),
        group_kind=pl.lit("total"),
        group=pl.lit("all"),
        count=pl.col("count").cast(pl.Int64),
    )
    subjects = pl.DataFrame(
        {"group": list(subject_total), "count": list(subject_total.values())}
    ).select(
        dataset=pl.lit("arxiv"),
        slice_kind=pl.lit("subject"),
        slice=pl.lit(None, dtype=pl.Int64),
        group_kind=pl.lit("subject"),
        group=pl.col("group").cast(pl.Utf8),
        count=pl.col("count").cast(pl.Int64),
    )
    subject_year_df = pl.DataFrame(
        {
            "slice": [year for year, _ in subject_year],
            "group": [subject for _, subject in subject_year],
            "count": list(subject_year.values()),
        }
    ).select(
        dataset=pl.lit("arxiv"),
        slice_kind=pl.lit("year"),
        slice=pl.col("slice").cast(pl.Int64),
        group_kind=pl.lit("subject_share"),
        group=pl.col("group").cast(pl.Utf8),
        count=pl.col("count").cast(pl.Int64),
    )
    return pl.concat([years, subjects, subject_year_df], how="vertical")


def _amazon() -> pl.DataFrame:
    from drift_happens.dataset.amazon_reviews_23.const import (
        AR23_PREPROCESSED_REVIEWS_MERGED_PATH,
    )
    from drift_happens.dataset.amazon_reviews_23.scope import (
        build_amazon_reviews_23_scope,
    )

    # Read only timestamp and rating; the review text is multi-GB and unused by the scope.
    reviews = pl.read_parquet(
        AR23_PREPROCESSED_REVIEWS_MERGED_PATH, columns=["timestamp", "rating"]
    )
    counts = (
        build_amazon_reviews_23_scope(reviews).group_by("half_year", "rating").len()
    )
    return counts.select(
        dataset=pl.lit("amazon_reviews_23"),
        slice_kind=pl.lit("half_year"),
        slice=pl.col("half_year").cast(pl.Int64),
        group_kind=pl.lit("rating"),
        group=pl.col("rating").cast(pl.Utf8),
        count=pl.col("len").cast(pl.Int64),
    )
