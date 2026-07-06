"""Tests for cached image embedding trainer handoff."""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
    TransferLearningModel,
)
from drift_happens.model.dataset.image.transfer_learning.embedding_head import (
    CachedEmbeddingHead,
)
from drift_happens.model.trainer.pytorch import PytorchTrainer, PytorchTrainerConfig
from drift_happens.pipeline.context import PipelineContext
from drift_happens.pipeline.image.run import embed_dataset_if_needed
from drift_happens.sample.splits import DatasetSplit


class FakeFrozenTransferModel(TransferLearningModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = TransferLearningConfig(
            fine_tune=False,
            needs_backend_fw_pass=False,
        )
        self.classifier = nn.Linear(4, 2)

    def forward_only_backend(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


def test_cached_embedding_replaces_future_resets_with_head_only_model(
    tmp_path: Path,
) -> None:
    factory_calls = 0

    def model_factory() -> nn.Module:
        nonlocal factory_calls
        factory_calls += 1
        return FakeFrozenTransferModel()

    trainer = PytorchTrainer(
        model_factory=model_factory,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
        criterion=nn.CrossEntropyLoss(),
        config=PytorchTrainerConfig(num_epochs=1, batch_size=2, device="cpu"),
    )
    ctx = PipelineContext(
        df=pd.DataFrame({"year": [2000, 2001]}),
        tensor_dataset=TensorDataset(
            torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
            torch.tensor([0, 1]),
        ),
        dataset_splits=DatasetSplit(
            train_df=pd.DataFrame(index=[0]),
            val_df=pd.DataFrame(index=[]),
            test_df=pd.DataFrame(index=[1]),
        ),
        trainer_keys=["fake_frozen"],
        train_time_slices={},
        artifacts_dir=tmp_path,
    )

    embedded = embed_dataset_if_needed(
        ctx,
        trainer,
        "fake_frozen",
        dataset_cache_dir=tmp_path / "cache",
        dataset_id="unit",
    )
    trainer.reset_model()

    assert embedded.tensors[0].shape == (2, 4)
    assert isinstance(trainer._model, CachedEmbeddingHead)
    assert factory_calls == 1
    head = trainer._model
    assert head.classifier.in_features == 4
    assert head.classifier.out_features == 2
