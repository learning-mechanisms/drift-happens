from __future__ import annotations

from datetime import UTC, datetime

from drift_happens.utils.ids import slugify, utc_timestamp


def test_slugify_and_timestamp_are_filesystem_safe() -> None:
    assert slugify("Yearbook Smoke / MLP") == "Yearbook-Smoke-MLP"
    assert slugify("   ") == "unnamed"
    assert utc_timestamp(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == (
        "2026-01-02T03-04-05Z"
    )


def test_slugify_neutralizes_pure_dot_path_traversal() -> None:
    for value in ("..", ".", "...", "  ..  "):
        assert slugify(value) == "unnamed"
    assert slugify("cnn-small") == "cnn-small"
    assert slugify("amazon_reviews_23") == "amazon_reviews_23"
