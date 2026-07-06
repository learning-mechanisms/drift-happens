from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel


@dataclass
class DatasetSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class DatasetTimeSplitConfig(BaseModel):
    lower_bound: Any
    upper_bound: Any

    lower_bound_inclusive: bool = True
    upper_bound_inclusive: bool = True

    def overlaps(self, other: "DatasetTimeSplitConfig") -> bool:
        """Check if this time split overlaps with another."""
        if self.upper_bound is not None and other.lower_bound is not None:
            if self.upper_bound < other.lower_bound:
                return False
            if self.upper_bound == other.lower_bound:
                return self.upper_bound_inclusive and other.lower_bound_inclusive

        if other.upper_bound is not None and self.lower_bound is not None:
            if other.upper_bound < self.lower_bound:
                return False
            if other.upper_bound == self.lower_bound:
                return other.upper_bound_inclusive and self.lower_bound_inclusive

        return True
