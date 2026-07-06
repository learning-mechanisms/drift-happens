"""Per-dataset headline metric and figure-presentation specs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    metric: str
    matrix_stem: str
    higher_is_better: bool
    metric_label: str
    value_scale: float
    unit_suffix: str
    value_fmt: str
    sequential_cmap: str
    diverging_cmap: str
    value_range: tuple[float, float] | None
    slice_noun: str
    count_noun: str


DATASETS: dict[str, DatasetSpec] = {
    "yearbook": DatasetSpec(
        slug="yearbook",
        slice_noun="year",
        count_noun="Photos",
        title="Yearbook",
        metric="accuracy",
        matrix_stem="accuracy",
        higher_is_better=True,
        metric_label="Accuracy",
        value_scale=100.0,
        unit_suffix="%",
        value_fmt=".1f",
        sequential_cmap="RdYlBu_r",
        diverging_cmap="RdBu_r",
        value_range=(0.0, 100.0),
    ),
    "amazon_reviews_23": DatasetSpec(
        slug="amazon_reviews",
        slice_noun="half-year",
        count_noun="Reviews",
        title="Amazon Reviews",
        metric="balanced_mse",
        matrix_stem="mse",
        higher_is_better=False,
        metric_label="Balanced MSE",
        value_scale=1.0,
        unit_suffix="",
        value_fmt=".3f",
        sequential_cmap="RdYlBu",
        diverging_cmap="RdBu",
        value_range=None,
    ),
    "arxiv": DatasetSpec(
        slug="arxiv",
        slice_noun="year",
        count_noun="Submissions",
        title="arXiv",
        metric="auc_macro",
        matrix_stem="auc",
        higher_is_better=True,
        metric_label="Macro AUC",
        value_scale=100.0,
        unit_suffix="%",
        value_fmt=".1f",
        sequential_cmap="RdYlBu_r",
        diverging_cmap="RdBu_r",
        value_range=(0.0, 100.0),
    ),
}
