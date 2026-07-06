from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from drift_happens.sample.splits import (
    DatasetSplit,
    DatasetTimeSplitConfig,
)


@dataclass(frozen=True)
class PipelineContext:
    df: pd.DataFrame
    tensor_dataset: Dataset[Any]
    dataset_splits: DatasetSplit
    trainer_keys: list[str]
    train_time_slices: dict[Any, DatasetTimeSplitConfig]
    artifacts_dir: Path
