from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from drift_happens.utils.numpy import unwrap_np_type


def test_unwrap_np_type_converts_numpy_scalars_and_preserves_none() -> None:
    int_result = unwrap_np_type(np.int64(3))
    assert int_result == 3
    assert type(int_result) is int
    assert isinstance(unwrap_np_type(np.float32(1.5)), float)
    assert unwrap_np_type(None) is None


def test_unwrap_np_type_converts_pandas_timestamp() -> None:
    value = unwrap_np_type(pd.Timestamp("2026-05-27T12:00:00"))

    assert value == datetime(2026, 5, 27, 12, 0, 0)
    assert type(value) is datetime
