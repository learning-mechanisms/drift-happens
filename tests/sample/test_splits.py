import pandas as pd
import pytest

from drift_happens.sample.splits import (
    DatasetTimeSplitConfig,
    create_cumulative_from_start_time_slices,
    create_instance_based_train_val_test_split,
    create_simple_time_slices,
    create_stratified_temporal_train_test_val_splits,
)

# ----------------------------- TRAIN / TEST / VAL SPLITS ---------------------------- #


def test_create_stratified_temporal_train_test_val_splits():
    df = pd.DataFrame(
        {
            "time": [2020] * 6 + [2021] * 6,
            "label": [0, 1, 0, 1, 0, 1] * 2,
            "value": range(12),
        }
    )

    split = create_stratified_temporal_train_test_val_splits(
        df,
        time_col="time",
        label_col="label",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=0,
    )

    # --- basic shape checks ---
    assert len(split.train_df) > 0
    assert len(split.val_df) > 0
    assert len(split.test_df) > 0

    # --- no rows lost or duplicated ---
    total_rows = len(split.train_df) + len(split.val_df) + len(split.test_df)
    assert total_rows == len(df)

    # --- all rows accounted for ---
    combined_idx = split.train_df.index.append(split.val_df.index).append(
        split.test_df.index
    )
    assert combined_idx.is_unique

    # --- both years contribute to every split (temporal structure) ---
    for part in (split.train_df, split.val_df, split.test_df):
        assert set(part["time"]) == {2020, 2021}

    # --- label proportions are preserved per split (stratification) ---
    for part in (split.train_df, split.val_df, split.test_df):
        counts = part["label"].value_counts()
        assert counts[0] == counts[1]


def test_stratified_temporal_split_supports_val_split_without_label_col():
    df = pd.DataFrame(
        {
            "time": [2020] * 8 + [2021] * 8,
            "value": range(16),
        }
    )

    split = create_stratified_temporal_train_test_val_splits(
        df,
        time_col="time",
        label_col=None,
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=0,
    )

    assert len(split.val_df) > 0
    total_rows = len(split.train_df) + len(split.val_df) + len(split.test_df)
    assert total_rows == len(df)


def test_instance_based_split_no_leakage():
    df = pd.DataFrame(
        {
            "instance": ["A", "A", "B", "B", "C", "C"],
            "time": [1, 2, 1, 2, 1, 2],
            "value": range(6),
        }
    )

    split = create_instance_based_train_val_test_split(
        df,
        instance_col="instance",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=0,
    )

    all_instances = (
        set(split.train_df["instance"])
        | set(split.val_df["instance"])
        | set(split.test_df["instance"])
    )

    assert all_instances == {"A", "B", "C"}
    train_inst = set(split.train_df["instance"])
    val_inst = set(split.val_df["instance"])
    test_inst = set(split.test_df["instance"])
    assert not (train_inst & val_inst)
    assert not (train_inst & test_inst)
    assert not (val_inst & test_inst)


# ------------------------------------------------------------------------------------ #
#                                  SIMPLE TIME SLICES                                  #
# ------------------------------------------------------------------------------------ #


def test_create_simple_time_slices_basic():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2],
            "value": [10, 20, 30, 40, 50],
        }
    )

    splits = create_simple_time_slices(df, time_col="time")

    # Expected keys = sorted unique times
    assert list(splits.keys()) == [1, 2, 3]

    # First interval: [1, 2)
    cfg_1 = splits[1]
    assert isinstance(cfg_1, DatasetTimeSplitConfig)
    assert cfg_1.lower_bound == 1
    assert cfg_1.upper_bound == 2
    assert cfg_1.lower_bound_inclusive is True
    assert cfg_1.upper_bound_inclusive is False

    # Last interval: [3, +inf)
    cfg_3 = splits[3]
    assert cfg_3.lower_bound == 3
    assert cfg_3.upper_bound is None
    assert cfg_3.lower_bound_inclusive is True
    assert cfg_3.upper_bound_inclusive is False


def test_create_simple_time_slices_with_min_time():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )

    splits = create_simple_time_slices(df, time_col="time", min_time=3)

    # Only splits with lower bound >= 3 remain
    assert list(splits.keys()) == [3, 4]

    # Check interval boundaries
    assert splits[3].upper_bound == 4
    assert splits[4].upper_bound is None


# ------------------------------------------------------------------------------------ #
#                                CUMULATIVE TIME SLICES                                #
# ------------------------------------------------------------------------------------ #


def test_create_cumulative_from_start_time_slices_basic():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2],
            "value": [10, 20, 30, 40, 50],
        }
    )

    splits = create_cumulative_from_start_time_slices(df, time_col="time")

    # Keys should be sorted unique times
    assert list(splits.keys()) == [1, 2, 3]

    # All intervals should start at the earliest time (1)
    cfg_1 = splits[1]
    assert isinstance(cfg_1, DatasetTimeSplitConfig)
    assert cfg_1.lower_bound == 1
    assert cfg_1.upper_bound == 1
    assert cfg_1.lower_bound_inclusive is True
    assert cfg_1.upper_bound_inclusive is True

    cfg_3 = splits[3]
    assert cfg_3.lower_bound == 1
    assert cfg_3.upper_bound == 3
    assert cfg_3.lower_bound_inclusive is True
    assert cfg_3.upper_bound_inclusive is True


def test_create_cumulative_from_start_time_slices_with_min_time():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )

    splits = create_cumulative_from_start_time_slices(df, time_col="time", min_time=3)

    # Only splits with upper bound >= min_time remain
    assert list(splits.keys()) == [3, 4]

    # Check that lower bound is still the dataset start
    assert splits[3].lower_bound == 1
    assert splits[4].lower_bound == 1


# ------------------------------------------------------------------------------------ #
#                                     OVERLAP TEST                                     #
# ------------------------------------------------------------------------------------ #


def cfg(lb, ub, lb_inc: bool = True, ub_inc: bool = True) -> DatasetTimeSplitConfig:
    """Helper to construct configs more concisely."""
    return DatasetTimeSplitConfig(
        lower_bound=lb,
        upper_bound=ub,
        lower_bound_inclusive=lb_inc,
        upper_bound_inclusive=ub_inc,
    )


@pytest.mark.parametrize(
    "cfg1,cfg2,expected",
    [
        # Clearly disjoint: [1, 2) and [3, 4)
        (cfg(1, 2, True, False), cfg(3, 4, True, False), False),
        # Overlapping ranges: [1, 3] and [2, 4]
        (cfg(1, 3), cfg(2, 4), True),
        # One inside another: [1, 5] and [2, 3]
        (cfg(1, 5), cfg(2, 3), True),
        # Touch at boundary, both inclusive: [1, 2] and [2, 3] → overlap
        (cfg(1, 2, True, True), cfg(2, 3, True, True), True),
        # Touch at boundary, first not inclusive at upper: [1, 2) and [2, 3] → no overlap
        (cfg(1, 2, True, False), cfg(2, 3, True, True), False),
        # Touch at boundary, second not inclusive at lower: [1, 2] and (2, 3] → no overlap
        (cfg(1, 2, True, True), cfg(2, 3, False, True), False),
        # Touch at boundary on the other side: [0, 1] and [1, 2], both inclusive
        (cfg(0, 1, True, True), cfg(1, 2, True, True), True),
        # Touch at boundary on the other side, one exclusive: [0, 1) and [1, 2] → no overlap
        (cfg(0, 1, True, False), cfg(1, 2, True, True), False),
        # Unbounded below vs finite, overlapping: (-inf, 5] & [0, 10]
        (cfg(None, 5, True, True), cfg(0, 10, True, True), True),
        # Unbounded below vs finite, disjoint: (-inf, 5] & [6, 10] → no overlap
        (cfg(None, 5, True, True), cfg(6, 10, True, True), False),
        # Unbounded above vs finite, always overlap if ranges intersect at all: [1, inf) & [10, 20]
        (cfg(1, None, True, True), cfg(10, 20, True, True), True),
        # Both unbounded → overlap
        (cfg(None, None), cfg(None, None), True),
        # Mixed inclusivity, overlapping interior: (1, 5) & [2, 4]
        (cfg(1, 5, False, False), cfg(2, 4, True, True), True),
    ],
)
def test_overlaps(
    cfg1: DatasetTimeSplitConfig, cfg2: DatasetTimeSplitConfig, expected: bool
):
    assert cfg1.overlaps(cfg2) == expected


@pytest.mark.parametrize(
    "cfg1,cfg2",
    [
        (cfg(1, 2, True, False), cfg(3, 4, True, False)),  # disjoint
        (cfg(1, 3), cfg(2, 4)),  # overlapping
        (cfg(1, 2, True, True), cfg(2, 3, True, True)),  # touching inclusive
        (cfg(1, 2, True, False), cfg(2, 3, True, True)),  # touching, non-overlap
        (cfg(None, 5, True, True), cfg(6, 10, True, True)),  # disjoint with None
        (cfg(1, None, True, True), cfg(10, 20, True, True)),  # overlap with None
        (cfg(None, None), cfg(5, 10)),  # overlap with fully unbounded
    ],
)
def test_overlaps_symmetric(cfg1: DatasetTimeSplitConfig, cfg2: DatasetTimeSplitConfig):
    """Overlaps should be symmetric: a.overlaps(b) == b.overlaps(a)."""
    assert cfg1.overlaps(cfg2) == cfg2.overlaps(cfg1)


def test_stratified_split_excludes_nan_time_without_crash():
    df = pd.DataFrame(
        {
            "time": [2020] * 6 + [2021] * 6 + [float("nan")],
            "label": [0, 1, 0, 1, 0, 1] * 2 + [0],
            "value": range(13),
        }
    )

    split = create_stratified_temporal_train_test_val_splits(
        df,
        time_col="time",
        label_col="label",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=0,
    )

    combined = pd.concat([split.train_df, split.val_df, split.test_df])
    assert combined["time"].notna().all()
    assert len(combined) == int(df["time"].notna().sum())


def test_stratified_split_falls_back_when_test_side_smaller_than_classes():
    df = pd.DataFrame(
        {
            "time": [2020] * 10,
            "label": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "value": range(10),
        }
    )

    split = create_stratified_temporal_train_test_val_splits(
        df,
        time_col="time",
        label_col="label",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=0,
    )

    assert len(split.train_df) + len(split.val_df) + len(split.test_df) == len(df)


def test_stratified_split_falls_back_when_a_class_drops_to_one_after_test():
    df = pd.DataFrame(
        {
            "time": [2020] * 6,
            "label": ["A", "A", "A", "A", "B", "B"],
            "value": range(6),
        }
    )

    split = create_stratified_temporal_train_test_val_splits(
        df,
        time_col="time",
        label_col="label",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=0,
    )

    assert len(split.train_df) + len(split.val_df) + len(split.test_df) == len(df)
