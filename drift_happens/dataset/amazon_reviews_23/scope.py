"""Canonical Amazon Reviews 2023 scope for conference experiments."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

# Single source for the canonical window; all bounds below derive from these.
_HALF_YEAR_ORIGIN_YEAR = 2000
_SCOPE_MIN_YEAR = 2014
_SCOPE_MAX_YEAR = 2023

AMAZON_REVIEWS_23_SAMPLE_SIZE = 300_000
AMAZON_REVIEWS_23_SCOPE_VARIANT = "sample300k_2014h1_2023h2_text_cumulative_v1"
AMAZON_REVIEWS_23_MIN_HALF_YEAR = (_SCOPE_MIN_YEAR - _HALF_YEAR_ORIGIN_YEAR) * 2
AMAZON_REVIEWS_23_MAX_HALF_YEAR = (_SCOPE_MAX_YEAR - _HALF_YEAR_ORIGIN_YEAR) * 2 + 1


def add_half_year_column(
    df: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    output_col: str = "half_year",
) -> pl.DataFrame:
    """Add half-year integer time slices relative to year 2000."""
    return df.with_columns(
        (
            (pl.col(timestamp_col).dt.year() - _HALF_YEAR_ORIGIN_YEAR) * 2
            + ((pl.col(timestamp_col).dt.month() - 1) // 6)
        ).alias(output_col)
    )


def filter_canonical_time_range(
    df: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Filter to the canonical [min year, max year + 1) timestamp range."""
    return df.filter(
        (pl.col(timestamp_col) >= pl.datetime(_SCOPE_MIN_YEAR, 1, 1))
        & (pl.col(timestamp_col) < pl.datetime(_SCOPE_MAX_YEAR + 1, 1, 1))
    )


def filter_valid_ratings(
    df: pl.DataFrame,
    *,
    rating_col: str = "rating",
) -> pl.DataFrame:
    """
    Keep only reviews with a valid 1-5 star rating.

    The raw dumps contain a handful of rating-0 rows; the stratified sample guarantees
    one row per (half_year, rating) stratum, so without this filter they always end up
    in the canonical sample and break the rating loss.
    """
    return df.filter(pl.col(rating_col).is_between(1, 5))


def deterministic_stratified_sample(
    df: pl.DataFrame,
    *,
    n: int,
    strata: Sequence[str],
    seed: int,
    row_id_col: str = "row_id",
) -> pl.DataFrame:
    """
    Deterministically sample up to ``n`` rows while covering available strata.

    The sample preserves natural proportions through a seeded global sample for the
    remaining budget after taking one guaranteed row from each stratum when feasible.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or df.height == 0:
        return _ensure_row_id(df, row_id_col).head(0)
    if n >= df.height:
        return _ensure_row_id(df, row_id_col).sort(row_id_col)

    working = _ensure_row_id(df, row_id_col)
    if n < working.select(strata).unique().height:
        return working.sample(n=n, with_replacement=False, seed=seed).sort(row_id_col)

    guaranteed = (
        working.sort(row_id_col)
        .group_by(list(strata), maintain_order=True)
        .head(1)
        .select(row_id_col)
    )
    guaranteed_ids = guaranteed.get_column(row_id_col)
    remaining_budget = n - guaranteed.height

    guaranteed_rows = working.filter(pl.col(row_id_col).is_in(guaranteed_ids))
    if remaining_budget <= 0:
        return guaranteed_rows.sort(row_id_col)

    remaining_rows = working.filter(~pl.col(row_id_col).is_in(guaranteed_ids))
    extra_rows = remaining_rows.sample(
        n=remaining_budget,
        with_replacement=False,
        seed=seed,
    )
    return pl.concat([guaranteed_rows, extra_rows]).sort(row_id_col)


def build_amazon_reviews_23_scope(
    df: pl.DataFrame,
    *,
    sample_size: int = AMAZON_REVIEWS_23_SAMPLE_SIZE,
    sample_seed: int = 42,
) -> pl.DataFrame:
    """Apply the canonical date and rating filters, then the deterministic sample."""
    scoped = filter_canonical_time_range(df)
    scoped = filter_valid_ratings(scoped)
    scoped = add_half_year_column(scoped)
    return deterministic_stratified_sample(
        scoped,
        n=sample_size,
        strata=("half_year", "rating"),
        seed=sample_seed,
        row_id_col="row_id",
    )


def amazon_reviews_23_scope_report(df: pl.DataFrame) -> dict[str, object]:
    """Return lightweight distribution metadata for the scoped dataframe."""
    half_year_counts = dict(
        df.group_by("half_year").len().sort("half_year").iter_rows()
    )
    rating_counts = dict(df.group_by("rating").len().sort("rating").iter_rows())
    return {
        "dataset": "amazon_reviews_23",
        "variant": AMAZON_REVIEWS_23_SCOPE_VARIANT,
        "row_count": int(df.height),
        "half_year_counts": {int(k): int(v) for k, v in half_year_counts.items()},
        "rating_counts": {int(k): int(v) for k, v in rating_counts.items()},
    }


def _ensure_row_id(df: pl.DataFrame, row_id_col: str) -> pl.DataFrame:
    if row_id_col in df.columns:
        return df
    return df.with_row_index(row_id_col)
