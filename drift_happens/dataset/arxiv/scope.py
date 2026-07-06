"""Canonical arXiv dataset scope for the conference benchmark."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, cast

import pandas as pd

ARXIV_TARGET_LABELS: tuple[str, ...] = (
    "cs.LG",
    "hep-ph",
    "cs.CV",
    "cs.AI",
    "hep-th",
    "quant-ph",
    "gr-qc",
)

ARXIV_SCOPE_VARIANT = "top7_leaf_title_abstract_2000_2025_cumulative_v1"
ARXIV_MIN_YEAR: int = 2000
ARXIV_MAX_YEAR: int = 2025

_WHITESPACE_RE = re.compile(r"\s+")


def map_arxiv_categories_to_labels(categories: Iterable[str]) -> list[str]:
    """Map raw arXiv categories to sorted, deduplicated target labels."""
    target_set = set(ARXIV_TARGET_LABELS)
    mapped = {
        str(category).strip()
        for category in categories
        if str(category).strip() in target_set
    }
    return sorted(mapped, key=ARXIV_TARGET_LABELS.index)


def normalize_title_abstract(title: object, abstract: object) -> str:
    """Create the canonical title + blank line + abstract text field."""
    clean_title = _WHITESPACE_RE.sub(
        " ", "" if pd.isna(cast(Any, title)) else str(title)
    ).strip()
    clean_abstract = _WHITESPACE_RE.sub(
        " ", "" if pd.isna(cast(Any, abstract)) else str(abstract)
    ).strip()
    return f"{clean_title}\n\n{clean_abstract}".strip()


def add_arxiv_scope_columns(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    abstract_col: str = "abstract",
    categories_col: str = "categories",
) -> pd.DataFrame:
    """Add canonical year, text, and target-label columns."""
    out = df.copy()
    out["created"] = pd.to_datetime(out["created"])
    out["year"] = out["created"].dt.year
    if abstract_col not in out.columns:
        out[abstract_col] = ""
    out["title_abstract"] = [
        normalize_title_abstract(title, abstract)
        for title, abstract in zip(out[title_col], out[abstract_col])
    ]
    out["top_subjects"] = [
        map_arxiv_categories_to_labels(categories) for categories in out[categories_col]
    ]
    return out


def filter_arxiv_top_leaf_label_scope(
    df: pd.DataFrame,
    *,
    min_year: int = ARXIV_MIN_YEAR,
    max_year: int = ARXIV_MAX_YEAR,
) -> pd.DataFrame:
    """Filter to the canonical strict top-leaf-label title+abstract scope."""
    scoped = add_arxiv_scope_columns(df)
    scoped = scoped[(scoped["year"] >= min_year) & (scoped["year"] <= max_year)]
    scoped = scoped[scoped["categories"].map(_has_only_target_labels)]
    return scoped.reset_index(drop=True)


def arxiv_scope_report(df: pd.DataFrame) -> dict[str, object]:
    """Return lightweight distribution metadata for the scoped dataframe."""
    year_counts = df.groupby("year").size().sort_index().astype(int).to_dict()
    label_counts = {
        label: int(df["top_subjects"].map(lambda labels: label in labels).sum())
        for label in ARXIV_TARGET_LABELS
    }
    return {
        "dataset": "arxiv",
        "variant": ARXIV_SCOPE_VARIANT,
        "row_count": int(len(df)),
        "year_counts": year_counts,
        "label_counts": label_counts,
    }


def label_schema_hash_input(labels: Sequence[str] = ARXIV_TARGET_LABELS) -> str:
    """Stable text used by cache-id builders for the arXiv label schema."""
    return "\n".join(labels)


def _has_only_target_labels(categories: Iterable[str]) -> bool:
    target_set = set(ARXIV_TARGET_LABELS)
    cleaned = [str(category).strip() for category in categories]
    return bool(cleaned) and all(category in target_set for category in cleaned)
