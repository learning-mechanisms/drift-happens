from __future__ import annotations

import pandas as pd
import pytest

from drift_happens.sample.splits import DatasetSplit, DatasetTimeSplitConfig
from drift_happens.sample.time.filter import (
    filter_dataset_by_time_slice,
    iter_evaluation_sets,
)


@pytest.mark.parametrize(
    ("lower_inc", "upper_inc", "expected"),
    [
        (True, True, [2, 3, 4]),
        (True, False, [2, 3]),
        (False, True, [3, 4]),
        (False, False, [3]),
    ],
)
def test_filter_dataset_by_time_slice_respects_inclusive_bounds(
    lower_inc: bool, upper_inc: bool, expected: list[int]
) -> None:
    df = pd.DataFrame({"year": [1, 2, 3, 4, 5]})
    split = DatasetTimeSplitConfig(
        lower_bound=2,
        upper_bound=4,
        lower_bound_inclusive=lower_inc,
        upper_bound_inclusive=upper_inc,
    )

    out = filter_dataset_by_time_slice(df, "year", split)

    assert out["year"].tolist() == expected


def test_filter_dataset_by_time_slice_handles_unbounded_above() -> None:
    df = pd.DataFrame({"year": [2000, 2001, 2002]})
    split = DatasetTimeSplitConfig(lower_bound=2001, upper_bound=None)

    out = filter_dataset_by_time_slice(df, "year", split)

    assert out["year"].tolist() == [2001, 2002]


def test_iter_evaluation_sets_uses_val_test_for_overlapping_training_slice() -> None:
    split = DatasetSplit(
        train_df=pd.DataFrame({"year": [2000]}, index=[0]),
        val_df=pd.DataFrame({"year": [2000]}, index=[1]),
        test_df=pd.DataFrame({"year": [2001]}, index=[2]),
    )
    train_slice = DatasetTimeSplitConfig(
        lower_bound=2000, upper_bound=2001, upper_bound_inclusive=False
    )
    eval_slices = {
        "in": DatasetTimeSplitConfig(
            lower_bound=2000, upper_bound=2001, upper_bound_inclusive=False
        ),
        "out": DatasetTimeSplitConfig(lower_bound=2001, upper_bound=2002),
    }

    rows = {
        key: df.index.tolist()
        for key, _, df in iter_evaluation_sets(split, "year", eval_slices, train_slice)
    }

    assert rows == {"in": [1], "out": [2]}


def test_filter_dataset_by_time_slice_handles_unbounded_below() -> None:
    df = pd.DataFrame({"year": [2000, 2001, 2002]})
    split = DatasetTimeSplitConfig(
        lower_bound=None, upper_bound=2001, upper_bound_inclusive=True
    )

    out = filter_dataset_by_time_slice(df, "year", split)

    assert out["year"].tolist() == [2000, 2001]
