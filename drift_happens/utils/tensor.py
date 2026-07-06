import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def to_tensor(X: ArrayLike, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert input to a torch tensor if it is not already one."""
    if isinstance(X, torch.Tensor):
        if dtype is not None:
            return X.to(dtype)
        return X
    return torch.as_tensor(X, dtype=dtype)
