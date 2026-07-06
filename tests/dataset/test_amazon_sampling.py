from __future__ import annotations

from datetime import datetime

import polars as pl

from drift_happens.dataset.amazon_reviews_23.scope import (
    add_half_year_column,
    amazon_reviews_23_scope_report,
    build_amazon_reviews_23_scope,
    deterministic_stratified_sample,
    filter_canonical_time_range,
    filter_valid_ratings,
)


def test_half_year_column_uses_2000_origin() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2014, 1, 1), datetime(2014, 7, 1)],
        }
    )

    out = add_half_year_column(df)

    assert out.get_column("half_year").to_list() == [28, 29]


def test_deterministic_stratified_sample_is_repeatable_and_covers_strata() -> None:
    df = pl.DataFrame(
        {
            "half_year": [28, 28, 29, 29, 29, 30],
            "rating": [1, 1, 5, 5, 4, 2],
            "text": ["a", "b", "c", "d", "e", "f"],
        }
    )

    first = deterministic_stratified_sample(
        df, n=4, strata=("half_year", "rating"), seed=42
    )
    second = deterministic_stratified_sample(
        df, n=4, strata=("half_year", "rating"), seed=42
    )

    assert first.to_dict(as_series=False) == second.to_dict(as_series=False)
    assert first.select("half_year", "rating").unique().height == 4


def test_deterministic_stratified_sample_seeded_remainder_is_deterministic() -> None:
    # 10 strata, n=14, 30 rows: guaranteed 10 + seeded remainder 4 = 14 total.
    half_years = [28, 28, 29, 29, 30, 30, 31, 31, 32, 32]
    ratings = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    rows_per_stratum = 3
    df = pl.DataFrame(
        {
            "half_year": half_years * rows_per_stratum,
            "rating": ratings * rows_per_stratum,
            "text": [str(i) for i in range(len(half_years) * rows_per_stratum)],
        }
    )

    first = deterministic_stratified_sample(
        df, n=14, strata=("half_year", "rating"), seed=42
    )
    second = deterministic_stratified_sample(
        df, n=14, strata=("half_year", "rating"), seed=42
    )

    assert first.height == 14
    # Same seed always produces identical output (seeded remainder path).
    assert first.to_dict(as_series=False) == second.to_dict(as_series=False)
    # All strata are represented in the result.
    assert first.select("half_year", "rating").unique().height == 10
    # No duplicate rows by row_id.
    assert first.get_column("row_id").n_unique() == 14


def test_filter_canonical_time_range_is_half_open_2014_to_2024() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2013, 12, 31),
                datetime(2014, 1, 1),
                datetime(2023, 12, 31),
                datetime(2024, 1, 1),
            ],
            "value": [0, 1, 2, 3],
        }
    )

    out = filter_canonical_time_range(df)

    assert out.get_column("value").to_list() == [1, 2]


def test_deterministic_sample_zero_and_full_dataset_paths() -> None:
    df = pl.DataFrame({"half_year": [28, 29], "rating": [1, 5]})

    empty = deterministic_stratified_sample(df, n=0, strata=("half_year",), seed=1)
    full = deterministic_stratified_sample(df, n=9, strata=("half_year",), seed=1)

    assert empty.height == 0
    # empty path shares the row_id-bearing schema of every other path
    assert "row_id" in empty.columns
    assert empty.columns == full.columns
    assert full.get_column("row_id").to_list() == [0, 1]


def test_deterministic_sample_empty_input_includes_row_id() -> None:
    df = pl.DataFrame(
        {"half_year": [], "rating": []},
        schema={"half_year": pl.Int64, "rating": pl.Int64},
    )

    out = deterministic_stratified_sample(df, n=5, strata=("half_year",), seed=1)

    assert out.height == 0
    assert "row_id" in out.columns


def test_deterministic_sample_when_budget_less_than_strata_count() -> None:
    df = pl.DataFrame({"half_year": [28, 29, 30, 31], "rating": [1, 2, 3, 4]})

    out = deterministic_stratified_sample(
        df, n=2, strata=("half_year", "rating"), seed=42
    )

    assert out.height == 2
    assert "row_id" in out.columns


def test_scope_report_returns_int_keyed_counts() -> None:
    report = amazon_reviews_23_scope_report(
        pl.DataFrame({"half_year": [28, 28, 29], "rating": [1, 5, 5]})
    )

    assert report["row_count"] == 3
    assert report["half_year_counts"] == {28: 2, 29: 1}
    assert report["rating_counts"] == {1: 1, 5: 2}


def test_filter_valid_ratings_drops_out_of_range_rows() -> None:
    df = pl.DataFrame({"rating": [0, 1, 3, 5, 6], "text": ["a", "b", "c", "d", "e"]})

    out = filter_valid_ratings(df)

    assert out.get_column("rating").to_list() == [1, 3, 5]


def test_canonical_scope_never_samples_invalid_ratings() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2015, 3, 1)] * 6,
            "rating": [0, 1, 2, 3, 4, 5],
            "text": ["zero", "a", "b", "c", "d", "e"],
        }
    )

    scoped = build_amazon_reviews_23_scope(df, sample_size=6, sample_seed=42)

    assert 0 not in scoped.get_column("rating").to_list()
    assert scoped.height == 5
