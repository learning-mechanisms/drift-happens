from typing import cast

import torch
from torch.utils.data import TensorDataset

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningModel,
)
from drift_happens.model.dataset.image.transfer_learning.dino import DinoModelSize
from drift_happens.model.dataset.image.transfer_learning.dino_v2 import DinoV2Config
from drift_happens.pipeline.image.run import _embedding_cache_id


class _StubModel:
    """Stand-in exposing the `.config` the cache id reads, without a backbone."""

    def __init__(self, config: DinoV2Config) -> None:
        self.config = config


class _OtherStubModel(_StubModel):
    """A second class, to exercise the producer (model class name) field."""


def _model(
    size: DinoModelSize = "small", cls: type[_StubModel] = _StubModel
) -> TransferLearningModel:
    return cast(TransferLearningModel, cls(DinoV2Config(model_size=size)))


def _dataset(fill: float) -> TensorDataset:
    return TensorDataset(torch.full((4, 3, 8, 8), fill), torch.zeros(4))


def test_cache_id_is_deterministic() -> None:
    model, dataset = _model(), _dataset(0.0)
    first = _embedding_cache_id("yearbook", "k", model, dataset)
    second = _embedding_cache_id("yearbook", "k", model, dataset)
    assert first == second


def test_cache_id_changes_with_backbone_config() -> None:
    dataset = _dataset(0.0)
    small = _embedding_cache_id("yearbook", "k", _model("small"), dataset)
    base = _embedding_cache_id("yearbook", "k", _model("base"), dataset)
    assert small != base


def test_cache_id_changes_with_input_data() -> None:
    model = _model()
    zeros = _embedding_cache_id("yearbook", "k", model, _dataset(0.0))
    ones = _embedding_cache_id("yearbook", "k", model, _dataset(1.0))
    assert zeros != ones


def test_cache_id_changes_with_labels() -> None:
    # Same images, different labels: the id must rotate so a relabeled dataset
    # never reuses embeddings cached under the old labels.
    model = _model()
    images = torch.full((4, 3, 8, 8), 0.0)
    zeros = _embedding_cache_id(
        "yearbook", "k", model, TensorDataset(images, torch.zeros(4))
    )
    ones = _embedding_cache_id(
        "yearbook", "k", model, TensorDataset(images, torch.ones(4))
    )
    assert zeros != ones


def test_cache_id_changes_with_model_class() -> None:
    # DINOv2 and DINOv3 share an empty config subclass, so the model class name
    # is the only field in the id that tells them apart.
    dataset = _dataset(0.0)
    one = _embedding_cache_id("yearbook", "k", _model("small", _StubModel), dataset)
    other = _embedding_cache_id(
        "yearbook", "k", _model("small", _OtherStubModel), dataset
    )
    assert one != other
