"""
Streaming chunk-wise evaluation must match the gather-then-compute path.

At production scale the sequence-embedding caches cannot be gathered whole, so
evaluation streams each slice one cache chunk at a time. These tests pin that the
streamed probabilities, the optimal multilabel thresholds, and the written metric cells
match gathering the slice into RAM first, within floating-point tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from drift_happens.dataset.cache import (
    FeatureCacheManifest,
    load_feature_cache,
    write_feature_cache_manifest,
    write_tensor_chunks,
)
from drift_happens.model.trainer.pytorch import PytorchTrainer, PytorchTrainerConfig
from drift_happens.pipeline.evaluation import (
    _predict_proba_and_labels,
    _slice_inputs_and_labels,
)


def _write_chunked_cache(
    root: Path, tensors: tuple[torch.Tensor, ...], *, chunk_size: int, output: str
) -> None:
    """Write a ≥2-chunk feature cache so streaming crosses chunk boundaries."""
    chunks = write_tensor_chunks(root, tensors, chunk_size=chunk_size)
    assert len(chunks) >= 2, "the equivalence test needs a multi-chunk cache"
    write_feature_cache_manifest(
        root,
        FeatureCacheManifest(
            kind="pooled_embedding_dataset",
            cache_id="unit",
            dataset="arxiv",
            dataset_variant="unit",
            input_version="unit:v1",
            producer="roberta-base",
            output=output,
            params={},
            row_count=tensors[0].shape[0],
            content_hash="rows",
            label_schema_hash="labels",
            chunks=chunks,
        ),
    )


def _linear_trainer(in_features: int, out_features: int, *, multi_label: bool):
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features)

    def factory() -> nn.Module:
        clone = nn.Linear(in_features, out_features)
        clone.load_state_dict(linear.state_dict())
        return clone

    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    # The streaming path re-batches within each chunk; the shuffled eval indices
    # exercise the row-order reconstruction, proving the result is independent of
    # batching and chunk layout.
    config = PytorchTrainerConfig(num_epochs=0, batch_size=3, device="cpu")
    return PytorchTrainer(
        model_factory=factory,
        optimizer_factory=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=criterion,
        config=config,
        multi_label=multi_label,
    )


def test_streamed_probs_and_labels_match_gather(tmp_path: Path) -> None:
    rows, features = 7, 4
    embeddings = torch.randn(rows, features, generator=torch.manual_seed(1))
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    _write_chunked_cache(
        tmp_path, (embeddings, labels), chunk_size=3, output="pooled_embedding"
    )
    dataset = load_feature_cache(tmp_path)
    trainer = _linear_trainer(features, 2, multi_label=False)

    indices = [6, 0, 4, 1, 5, 2, 3]
    streamed_probs, streamed_labels = _predict_proba_and_labels(
        trainer, dataset, indices
    )

    gathered_inputs, gathered_labels = _slice_inputs_and_labels(dataset, indices)
    reference_probs = trainer.predict_proba(gathered_inputs)

    torch.testing.assert_close(streamed_probs, reference_probs)
    torch.testing.assert_close(streamed_labels, gathered_labels)


def test_multilabel_streaming_thresholds_and_auc_match_gather(tmp_path: Path) -> None:
    from drift_happens.evaluation.metrics import MultiLabelROCAUCTracker

    rows, features, classes = 7, 4, 3
    embeddings = torch.randn(rows, features, generator=torch.manual_seed(2))
    # Every class carries both a positive and a negative so the AUC is defined.
    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    _write_chunked_cache(
        tmp_path, (embeddings, labels), chunk_size=3, output="pooled_embedding"
    )
    chunked = load_feature_cache(tmp_path)
    indices = [6, 0, 4, 1, 5, 2, 3]

    streamer = _linear_trainer(features, classes, multi_label=True)
    streamed_probs, streamed_labels = _predict_proba_and_labels(
        streamer, chunked, indices
    )
    streamed_thresholds = streamer.find_optimal_threshold(
        streamed_probs, streamed_labels
    )

    gatherer = _linear_trainer(features, classes, multi_label=True)
    gathered_inputs, gathered_labels = _slice_inputs_and_labels(chunked, indices)
    gathered_probs = gatherer.predict_proba(gathered_inputs)
    gathered_thresholds = gatherer.find_optimal_threshold(
        gathered_probs, gathered_labels
    )

    torch.testing.assert_close(streamed_probs, gathered_probs)
    torch.testing.assert_close(streamed_labels, gathered_labels)
    np.testing.assert_array_equal(streamed_thresholds, gathered_thresholds)

    # The metric tracker built from the streamed scores must report the same AUC.
    streamed_tracker = MultiLabelROCAUCTracker.from_predictions(
        classes, streamed_labels, None, streamed_probs
    )
    gathered_tracker = MultiLabelROCAUCTracker.from_predictions(
        classes, gathered_labels, None, gathered_probs
    )
    assert streamed_tracker.scalar_metrics() == gathered_tracker.scalar_metrics()


def test_regression_eval_streaming_matches_gather(tmp_path: Path) -> None:
    from drift_happens.model.text.weighted_mse_loss import WeightedMSELoss

    rows, features = 7, 4
    embeddings = torch.randn(rows, features, generator=torch.manual_seed(3))
    labels = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0])

    chunked_root = tmp_path / "chunked"
    chunked_root.mkdir()
    _write_chunked_cache(
        chunked_root, (embeddings, labels), chunk_size=3, output="pooled_embedding"
    )
    chunked = load_feature_cache(chunked_root)

    def trainer() -> PytorchTrainer:
        torch.manual_seed(0)
        linear = nn.Linear(features, 1)

        def factory() -> nn.Module:
            clone = nn.Linear(features, 1)
            clone.load_state_dict(linear.state_dict())
            return clone

        return PytorchTrainer(
            model_factory=factory,
            optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
            criterion=WeightedMSELoss(torch.ones(5)),
            config=PytorchTrainerConfig(num_epochs=0, batch_size=3, device="cpu"),
        )

    streamed = trainer()
    indices = [6, 0, 4, 1, 5, 2, 3]
    streamed_probs, streamed_labels = _predict_proba_and_labels(
        streamed, chunked, indices
    )

    gathered = trainer()
    gathered_inputs, gathered_labels = _slice_inputs_and_labels(chunked, indices)
    reference_probs = gathered.predict_proba(gathered_inputs)

    torch.testing.assert_close(streamed_probs, reference_probs)
    torch.testing.assert_close(streamed_labels, gathered_labels)
