from collections.abc import Iterator
from typing import Any

import pandas as pd

from drift_happens.sample.time.types import DatasetSplit, DatasetTimeSplitConfig

# --------------------------------- FILTER TIME SLICE -------------------------------- #


def filter_dataset_by_time_slice(
    df: pd.DataFrame,
    time_col: str,
    split_config: DatasetTimeSplitConfig,
) -> pd.DataFrame:
    """
    Filter dataframe according to DatasetTimeSplitConfig.

    Args:
        df: Input dataframe with time column.
        time_col: Name of the time column.
        split_config: DatasetTimeSplitConfig defining the time slice.

    Returns:
        Filtered dataframe for the given time slice.
    """
    mask = pd.Series(True, index=df.index)

    if split_config.lower_bound is not None:
        mask &= (
            (df[time_col] >= split_config.lower_bound)
            if split_config.lower_bound_inclusive
            else (df[time_col] > split_config.lower_bound)
        )

    if split_config.upper_bound is not None:
        mask &= (
            (df[time_col] <= split_config.upper_bound)
            if split_config.upper_bound_inclusive
            else (df[time_col] < split_config.upper_bound)
        )

    return df[mask]


def iter_evaluation_sets(
    dataset_split: DatasetSplit,
    time_col: str,
    split_configs: dict[Any, DatasetTimeSplitConfig],
    training_time_split: DatasetTimeSplitConfig,
) -> Iterator[tuple[Any, DatasetTimeSplitConfig, pd.DataFrame]]:
    """
    Iterate over evaluation time intervals defined by time slice configurations.

    For intervals overlapping the training interval (in distribution),
    yield the corresponding filtered validation+test dataframe for evaluation.

    For all other intervals (out of distribution), yield the full train+val+test
    dataframe filtered to that time slice.

    Args:
        dataset_split: Train/val/test dataframes to draw evaluation rows from.
        time_col: Name of the time column.
        split_configs: Dictionary mapping from time value to DatasetTimeSplitConfig.
        training_time_split: DatasetTimeSplitConfig for the currently trained model.

    Yields:
        Tuples of (time value, slice config, filtered dataframe for that slice).
    """
    in_distribution_df = pd.concat(
        [dataset_split.val_df, dataset_split.test_df], axis=0
    )
    out_of_distribution_df = pd.concat(
        [dataset_split.train_df, dataset_split.val_df, dataset_split.test_df], axis=0
    )
    for time_value, split_config in split_configs.items():
        combined_df = (
            in_distribution_df
            if training_time_split.overlaps(split_config)
            else out_of_distribution_df
        )
        filtered_df = filter_dataset_by_time_slice(combined_df, time_col, split_config)
        yield time_value, split_config, filtered_df
