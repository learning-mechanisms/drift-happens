import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from drift_happens.sample.time.types import DatasetSplit as DatasetSplit
from drift_happens.sample.time.types import (
    DatasetTimeSplitConfig as DatasetTimeSplitConfig,
)
from drift_happens.utils.numpy import unwrap_np_type

# ----------------------------- TRAIN / TEST / VAL SPLITS ---------------------------- #


def _can_stratify(labels: pd.Series, split_size: float) -> bool:
    """Whether a stratified split of this fraction satisfies sklearn's bounds."""
    n_classes = labels.nunique()
    if n_classes < 2 or labels.value_counts().min() < 2:
        return False
    n_minor = math.ceil(split_size * len(labels))
    return n_minor >= n_classes and len(labels) - n_minor >= n_classes


def create_stratified_temporal_train_test_val_splits(
    df: pd.DataFrame,
    time_col: str,
    label_col: str | None = None,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int | None = None,
) -> DatasetSplit:
    """
    Create stratified temporal train/val/test splits.

    For each distinct value in `time_col` (e.g., each year), this function creates a
    stratified train/val/test split w.r.t. `label_col`, and then concatenates the per-
    time splits into global train/val/test DataFrames.

    `val_size` and `test_size` may be 0.0, in which case the corresponding split will be
    empty.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0, got {total}"
        )
    if train_size <= 0:
        raise ValueError("train_size must be > 0.")
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be >= 0.")

    # We'll use our own RNG to derive per-time seeds, so results are
    # reproducible but don't depend on year order.
    rng = np.random.RandomState(seed) if seed is not None else None

    df_sorted = df.sort_values(time_col)
    unique_times = df_sorted[time_col].dropna().unique()

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for t in unique_times:
        df_t = df_sorted[df_sorted[time_col] == t]

        if label_col is not None:
            y = df_t[label_col]

        # Too few samples; just assign everything to train to avoid
        # degenerate splits. Adjust if you want stricter behavior.
        if len(df_t) < 3:
            train_parts.append(df_t)
            continue

        # --- Split off test set (if any) ---
        if test_size > 0:
            rs_test = rng.randint(0, 2**32 - 1) if rng is not None else None
            stratify_arg = (
                y if label_col is not None and _can_stratify(y, test_size) else None
            )
            df_train_val_t, df_test_t = train_test_split(
                df_t,
                test_size=test_size,
                stratify=stratify_arg,
                random_state=rs_test,
            )
        else:
            df_train_val_t = df_t
            df_test_t = df_t.iloc[0:0].copy()

        # --- Split train vs val (if any val) ---
        if val_size > 0:
            rs_val = rng.randint(0, 2**32 - 1) if rng is not None else None
            # val share within the (train+val) portion
            val_fraction_within_train_val = val_size / (train_size + val_size)

            if label_col is not None:
                y_train_val = df_train_val_t[label_col]
                stratify_tv = (
                    y_train_val
                    if _can_stratify(y_train_val, val_fraction_within_train_val)
                    else None
                )
            else:
                stratify_tv = None

            df_train_t, df_val_t = train_test_split(
                df_train_val_t,
                test_size=val_fraction_within_train_val,
                stratify=stratify_tv,
                random_state=rs_val,
            )
        else:
            df_train_t = df_train_val_t
            df_val_t = df_train_val_t.iloc[0:0].copy()

        train_parts.append(df_train_t)
        if len(df_val_t) > 0:
            val_parts.append(df_val_t)
        if len(df_test_t) > 0:
            test_parts.append(df_test_t)

    train_df = pd.concat(train_parts, axis=0).sort_values(time_col)
    val_df = (
        pd.concat(val_parts, axis=0).sort_values(time_col)
        if val_parts
        else df.iloc[0:0].copy()
    )
    test_df = (
        pd.concat(test_parts, axis=0).sort_values(time_col)
        if test_parts
        else df.iloc[0:0].copy()
    )

    return DatasetSplit(train_df=train_df, val_df=val_df, test_df=test_df)


def create_instance_based_train_val_test_split(
    df: pd.DataFrame,
    *,
    instance_col: str,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int | None = None,
) -> DatasetSplit:
    """
    Create train/val/test splits based purely on instances.

    All rows belonging to the same instance (across all times) are assigned
    to the same split. No stratification or temporal logic is applied.

    Args:
        df: Input dataframe.
        instance_col: Column identifying instances/entities.
        train_size: Fraction of instances assigned to training.
        val_size: Fraction of instances assigned to validation.
        test_size: Fraction of instances assigned to testing.
        seed: Random seed for reproducibility.

    Returns:
        DatasetSplit with train_df, val_df, test_df.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0, got {total}"
        )
    if train_size <= 0:
        raise ValueError("train_size must be > 0.")
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be >= 0.")

    if instance_col not in df.columns:
        raise KeyError(f"instance_col '{instance_col}' not found in DataFrame")

    # Extract unique instances
    instances = df[[instance_col]].drop_duplicates()

    rng = np.random.RandomState(seed) if seed is not None else None

    # Split off test instances
    if test_size > 0:
        rs_test = rng.randint(0, 2**32 - 1) if rng is not None else None
        inst_train_val, inst_test = train_test_split(
            instances,
            test_size=test_size,
            random_state=rs_test,
        )
    else:
        inst_train_val = instances
        inst_test = instances.iloc[0:0]

    # Split train vs val instances
    if val_size > 0:
        rs_val = rng.randint(0, 2**32 - 1) if rng is not None else None
        val_fraction_within_train_val = val_size / (train_size + val_size)

        inst_train, inst_val = train_test_split(
            inst_train_val,
            test_size=val_fraction_within_train_val,
            random_state=rs_val,
        )
    else:
        inst_train = inst_train_val
        inst_val = inst_train_val.iloc[0:0]

    # Materialize row-level splits
    train_instances = set(inst_train[instance_col])
    val_instances = set(inst_val[instance_col])
    test_instances = set(inst_test[instance_col])

    train_df = df[df[instance_col].isin(train_instances)]
    val_df = df[df[instance_col].isin(val_instances)]
    test_df = df[df[instance_col].isin(test_instances)]

    return DatasetSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


# -------------------------------- TIME SLICE CONFIGS -------------------------------- #


def create_simple_time_slices(
    df: pd.DataFrame, time_col: str, min_time: Any | None = None
) -> dict[Any, DatasetTimeSplitConfig]:
    """
    Create simple NON cumulative time-wise split configurations.

    We take the sorted unique values of `time_col` and create one
    interval per unique value. Each interval is of the form
    [t_i, t_{i+1}) for all but the last, which is [t_last, +inf).

    Args:
        df: Input dataframe with time column.
        time_col: Name of the time column.
        min_time: If provided, only include splits with lower bound >= min_time.

    Returns:
        A dictionary mapping from the lower-bound time value to
        a DatasetTimeSplitConfig describing [lower, upper).
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in DataFrame columns")

    # Sort unique time values
    unique_times = pd.Index(df[time_col].dropna().unique()).sort_values()

    if len(unique_times) == 0:
        raise ValueError("No non-NA values found in time_col; cannot create splits.")

    splits: dict[Any, DatasetTimeSplitConfig] = {}

    for i, lower in enumerate(unique_times):
        if i < len(unique_times) - 1:
            upper = unique_times[i + 1]
            upper_inclusive = False  # enforce [lower, upper)
        else:
            # Last interval goes to +inf (no upper bound)
            upper = None
            upper_inclusive = False  # upper bound is None, exclusivity is irrelevant

        splits[lower] = DatasetTimeSplitConfig(
            lower_bound=unwrap_np_type(lower),
            upper_bound=unwrap_np_type(upper),
            lower_bound_inclusive=True,
            upper_bound_inclusive=upper_inclusive,
        )

    if min_time is None:
        return splits
    else:
        return {k: v for k, v in splits.items() if k >= unwrap_np_type(min_time)}


def create_cumulative_from_start_time_slices(
    df: pd.DataFrame, time_col: str, min_time: Any | None = None
) -> dict[Any, DatasetTimeSplitConfig]:
    """
    Create cumulative time-wise split configurations from the dataset start.

    For sorted unique time values t_0 < t_1 < ... < t_n, this creates one
    interval per unique value of the form [t_0, t_i], i.e., all intervals
    start at the first time in the dataset and end at t_i (inclusive).

    Args:
        df: Input dataframe with time column.
        time_col: Name of the time column.
        min_time: If provided, only include splits with upper bound >= min_time.

    Returns:
        A dictionary mapping from the upper-bound time value t_i to
        a DatasetTimeSplitConfig describing [t_0, t_i].
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in DataFrame columns")

    # Sort unique time values
    unique_times = pd.Index(df[time_col].dropna().unique()).sort_values()

    if len(unique_times) == 0:
        raise ValueError("No non-NA values found in time_col; cannot create splits.")

    # Earliest time in the dataset
    start_time = unique_times[0]

    splits: dict[Any, DatasetTimeSplitConfig] = {}

    for upper in unique_times:
        splits[unwrap_np_type(upper)] = DatasetTimeSplitConfig(
            lower_bound=unwrap_np_type(start_time),
            upper_bound=unwrap_np_type(upper),
            lower_bound_inclusive=True,  # [start_time, upper]
            upper_bound_inclusive=True,
        )

    if min_time is None:
        return splits
    else:
        return {k: v for k, v in splits.items() if k >= unwrap_np_type(min_time)}
