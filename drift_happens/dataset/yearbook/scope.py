"""Canonical Yearbook dataset scope metadata."""

from __future__ import annotations

from drift_happens.configs import (
    DatasetScopeProtocol,
    EvaluationProtocol,
    SplitProtocol,
    TimeSliceProtocol,
)

YEARBOOK_SCOPE_VARIANT = "faces_32x32_cumulative_v2"
YEARBOOK_IMAGE_SOURCE = "faces_aligned_small_mirrored_co_aligned_cropped_cleaned"


def yearbook_dataset_scope_protocol() -> DatasetScopeProtocol:
    """Return the canonical Yearbook scope used by conference presets."""
    return DatasetScopeProtocol(
        name="yearbook",
        variant=YEARBOOK_SCOPE_VARIANT,
        input_description="aligned 3-channel face tensors downscaled to 32x32",
        label_description="gender with M mapped to 1",
        time_column="year",
        params={
            "image_source": YEARBOOK_IMAGE_SOURCE,
            "image_size": [32, 32],
            "input_channels": 3,
            "label_col": "gender",
            "positive_label": "M",
        },
    )


def yearbook_split_protocol() -> SplitProtocol:
    return SplitProtocol(
        name="stratified_temporal_train_test_val",
        split_seed=42,
        train_size=0.7,
        val_size=0.0,
        test_size=0.3,
        stratify_by=("year", "gender"),
    )


def yearbook_time_slice_protocol() -> TimeSliceProtocol:
    return TimeSliceProtocol(
        eval="simple_year_slices",
        min_time=None,
        max_time=None,
        time_unit="year",
    )


def yearbook_evaluation_protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        primary_metric="accuracy",
        secondary_metrics=("balanced_accuracy",),
        task="binary_classification",
    )
