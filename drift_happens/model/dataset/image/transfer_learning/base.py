from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, model_validator
from torch.utils.data import DataLoader, TensorDataset

from drift_happens.model.dataset.image.transfer_learning.head_only import (
    HeadOnlyStateDictMixin,
)
from drift_happens.utils.pytorch import device_manual_mps_or_cuda_if_available


class TransferLearningConfig(BaseModel):
    """Configuration for transfer learning models."""

    pretrained: bool = True
    """Load pretrained backbone weights when constructing the model."""

    fine_tune: bool = False
    """
    If False, freeze backbone weights (transfer learning).

    If True, allow backbone to be trained (fine-tuning).
    """

    needs_backend_fw_pass: bool = True
    """Precomputed embeddings allow to skip the forward pass."""

    @model_validator(mode="after")
    def check_backend_fw_pass(self):
        """When fine_tune = True, we always need a backbone forward pass."""
        fine_tune = self.fine_tune
        needs_backend_fw_pass = self.needs_backend_fw_pass
        if fine_tune and not needs_backend_fw_pass:
            raise ValueError(
                "When fine_tune is True, needs_backend_fw_pass must also be True."
            )
        return self


class TransferLearningModel(HeadOnlyStateDictMixin, ABC):
    """
    Mixin / abstract base for transfer-learning models that:

    - optionally skip backbone forward in `forward` (caller-controlled via config)
    - support offline embedding precomputation over a TensorDataset

    Requirements for subclasses:
    - must define `self.config` with attribute `needs_backend_fw_pass: bool`
    - must implement `forward_only_backend(x) -> Tensor`
    """

    config: TransferLearningConfig

    @abstractmethod
    def forward_only_backend(self, x: torch.Tensor) -> torch.Tensor:
        """Run ONLY the pretrained backbone and return embeddings."""
        raise NotImplementedError

    def maybe_forward_backend(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for forward() implementations."""
        return self.forward_only_backend(x) if self.config.needs_backend_fw_pass else x

    def forward_only_backend_batched(
        self,
        dataset: TensorDataset,
        batch_size: int = 1024,
        *,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> TensorDataset:
        """Always computes backbone embeddings for X in `dataset` and returns a new
        TensorDataset (embeddings, y?)."""
        if len(dataset.tensors) > 2:
            raise ValueError("dataset must hold (X) or (X, y), got extra tensors.")

        device = device_manual_mps_or_cuda_if_available()

        self.eval()
        self.to(device)

        X = dataset.tensors[0]
        y = dataset.tensors[1] if len(dataset.tensors) > 1 else None

        new_features: list[torch.Tensor] = []

        loader: DataLoader = DataLoader(
            TensorDataset(X),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and device == "cuda",
        )

        with torch.no_grad():
            for batch_tensor_dataset in loader:
                batch = batch_tensor_dataset[0].to(device)
                out = self.forward_only_backend(batch)
                new_features.append(out.cpu())

        new_X = torch.cat(new_features, dim=0)

        if y is not None:
            return TensorDataset(new_X, y)
        return TensorDataset(new_X)
