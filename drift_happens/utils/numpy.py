from typing import Any

import numpy as np
import pandas as pd


def unwrap_np_type(value: Any) -> Any:
    """Convert pandas/NumPy scalar types to plain Python types."""
    # Pandas Timestamp -> datetime
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    # NumPy scalar -> Python scalar
    if isinstance(value, np.generic):
        return value.item()

    return value
