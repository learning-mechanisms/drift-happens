from __future__ import annotations

import json

import torch

from drift_happens.pipeline.amazon_reviews_23.trainers import (
    AmazonReviewsTrainingConfig,
)
from drift_happens.pipeline.arxiv.trainers import ArxivTrainingConfig

_BASE_KWARGS = dict(
    architecture_name="ffn_s", batch_size=64, learning_rate=5e-4, num_epochs=10
)


def _dump(config) -> dict:
    return json.loads(config.model_dump_json())


def test_arxiv_config_serializes_pos_weight_tensor() -> None:
    config = ArxivTrainingConfig(
        **_BASE_KWARGS,
        category_to_idx={"cs": 0, "math": 1},
        pos_weight=torch.tensor([1.5, 2.0]),
    )

    assert _dump(config)["pos_weight"] == [1.5, 2.0]


def test_arxiv_config_serializes_missing_pos_weight_as_null() -> None:
    config = ArxivTrainingConfig(**_BASE_KWARGS, category_to_idx={"cs": 0})

    assert _dump(config)["pos_weight"] is None


def test_amazon_config_serializes_class_weights_tensor() -> None:
    config = AmazonReviewsTrainingConfig(
        **_BASE_KWARGS,
        class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )

    assert _dump(config)["class_weights"] == [1.0, 2.0, 3.0, 4.0, 5.0]
