"""Tests for ConferenceProtocol consistency validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift_happens.configs.protocol import (
    CacheProtocol,
    ConferenceProtocol,
    DatasetScopeProtocol,
    EvaluationProtocol,
    ModelProtocol,
    SeedProtocol,
    SplitProtocol,
    TimeSliceProtocol,
)


def _conference_protocol(
    *, split_seed: int, seeds_split_seed: int
) -> ConferenceProtocol:
    return ConferenceProtocol(
        comparison_group="g",
        dataset_scope=DatasetScopeProtocol(
            name="n",
            variant="v",
            input_description="i",
            label_description="l",
            time_column="t",
        ),
        split=SplitProtocol(name="s", split_seed=split_seed),
        time_slices=TimeSliceProtocol(eval="e", time_unit="year"),
        evaluation=EvaluationProtocol(primary_metric="acc", task="cls"),
        seeds=SeedProtocol(split_seed=seeds_split_seed),
        cache=CacheProtocol(cache_kind="k", cache_scope="sc", producer="p"),
        model=ModelProtocol(identifier="m", family="f", input_kind="image"),
    )


def test_conference_protocol_accepts_matching_split_seeds() -> None:
    protocol = _conference_protocol(split_seed=42, seeds_split_seed=42)
    assert protocol.split.split_seed == protocol.seeds.split_seed == 42


def test_conference_protocol_rejects_mismatched_split_seeds() -> None:
    with pytest.raises(ValidationError, match="split_seed"):
        _conference_protocol(split_seed=42, seeds_split_seed=7)
