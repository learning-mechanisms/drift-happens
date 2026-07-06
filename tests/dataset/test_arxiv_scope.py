from __future__ import annotations

from typing import cast

import pandas as pd

from drift_happens.dataset.arxiv.scope import (
    add_arxiv_scope_columns,
    arxiv_scope_report,
    filter_arxiv_top_leaf_label_scope,
    label_schema_hash_input,
    map_arxiv_categories_to_labels,
    normalize_title_abstract,
)


def test_arxiv_category_mapping_keeps_target_leaf_labels_in_canonical_order() -> None:
    assert map_arxiv_categories_to_labels(
        ["quant-ph", "math.AG", "cs.AI", "cs.LG", "cs.AI"]
    ) == ["cs.LG", "cs.AI", "quant-ph"]


def test_arxiv_scope_adds_title_abstract_and_filters_to_strict_target_labels() -> None:
    df = pd.DataFrame(
        {
            "created": pd.to_datetime(
                ["2001-01-01", "1999-01-01", "2020-01-01", "2021-01-01"]
            ),
            "title": ["A  title", "Old", "Unknown", "Mixed"],
            "abstract": ["An\nabstract", "No", "No", "Outside"],
            "categories": [["cs.AI"], ["cs.CL"], ["unknown.X"], ["cs.AI", "math.AG"]],
        }
    )

    scoped = filter_arxiv_top_leaf_label_scope(df)

    assert len(scoped) == 1
    assert scoped.loc[0, "year"] == 2001
    assert scoped.loc[0, "top_subjects"] == ["cs.AI"]
    assert scoped.loc[0, "title_abstract"] == "A title\n\nAn abstract"


def test_normalize_title_abstract_handles_missing_values() -> None:
    assert normalize_title_abstract(None, float("nan")) == ""


def test_add_arxiv_scope_columns_adds_missing_abstract() -> None:
    df = pd.DataFrame(
        {
            "created": ["2020-01-01"],
            "title": ["Title"],
            "categories": [["cs.AI"]],
        }
    )

    out = add_arxiv_scope_columns(df)

    assert out.loc[0, "abstract"] == ""
    assert out.loc[0, "title_abstract"] == "Title"
    assert out.loc[0, "top_subjects"] == ["cs.AI"]


def test_filter_arxiv_scope_respects_year_bounds() -> None:
    df = pd.DataFrame(
        {
            "created": pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]),
            "title": ["old", "in", "new"],
            "abstract": ["", "", ""],
            "categories": [["cs.AI"], ["hep-th"], ["cs.CL"]],
        }
    )

    out = filter_arxiv_top_leaf_label_scope(df, min_year=2001, max_year=2001)

    assert out["title"].tolist() == ["in"]


def test_arxiv_scope_report_counts_label_membership() -> None:
    report = arxiv_scope_report(
        pd.DataFrame(
            {
                "year": [2020, 2020, 2021],
                "top_subjects": [["cs.LG", "cs.AI"], ["cs.LG"], ["hep-th"]],
            }
        )
    )

    assert report["row_count"] == 3
    assert report["year_counts"] == {2020: 2, 2021: 1}
    label_counts = cast(dict[str, int], report["label_counts"])
    assert label_counts["cs.LG"] == 2
    assert label_counts["cs.AI"] == 1
    assert label_counts["hep-th"] == 1


def test_label_schema_hash_input_is_stable() -> None:
    assert label_schema_hash_input(["b", "a"]) == "b\na"
    # Pin the exact default output so any reordering or edit to the target label
    # schema fails here (cache ids depend on this string).
    expected_default = "cs.LG\nhep-ph\ncs.CV\ncs.AI\nhep-th\nquant-ph\ngr-qc"
    assert label_schema_hash_input() == expected_default
