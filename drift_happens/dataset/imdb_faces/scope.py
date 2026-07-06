"""Canonical IMDB faces dataset scope metadata."""

from __future__ import annotations

from drift_happens.configs import (
    DatasetScopeProtocol,
    EvaluationProtocol,
    SplitProtocol,
    TimeSliceProtocol,
)

IMDB_FACES_SCOPE_VARIANT = "faces_32x32_celeb_split_cumulative_v1"
IMDB_FACES_IMAGE_SOURCE = "imdb_crop"


def imdb_faces_dataset_scope_protocol() -> DatasetScopeProtocol:
    """Return the canonical IMDB faces scope used by conference presets."""
    return DatasetScopeProtocol(
        name="imdb_faces",
        variant=IMDB_FACES_SCOPE_VARIANT,
        input_description="aligned 3-channel face crops downscaled to 32x32",
        label_description="gender with male stored as 1",
        time_column="photo_taken",
        params={
            "image_source": IMDB_FACES_IMAGE_SOURCE,
            "image_size": [32, 32],
            "input_channels": 3,
            "label_col": "gender",
            "positive_label": 1,
            "instance_col": "celeb_id",
        },
    )


def imdb_faces_split_protocol() -> SplitProtocol:
    """Per-celebrity split: every photo of a person lands in one side only."""
    return SplitProtocol(
        name="instance_based_train_val_test",
        split_seed=42,
        train_size=0.7,
        val_size=0.0,
        test_size=0.3,
    )


def imdb_faces_time_slice_protocol() -> TimeSliceProtocol:
    return TimeSliceProtocol(
        eval="simple_photo_taken_slices",
        min_time=None,
        max_time=None,
        time_unit="year",
    )


def imdb_faces_evaluation_protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        primary_metric="accuracy",
        secondary_metrics=("balanced_accuracy",),
        task="binary_classification",
    )
