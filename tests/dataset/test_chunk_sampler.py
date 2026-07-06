"""Tests for the chunk-blocked batch sampler used by out-of-core sequence caches."""

from __future__ import annotations

from pathlib import Path

import torch

from drift_happens.dataset.cache import (
    FeatureCacheManifest,
    load_feature_cache,
    write_feature_cache_manifest,
    write_tensor_chunks,
)
from drift_happens.dataset.chunk_sampler import ChunkBlockedBatchSampler


def _chunked(root: Path, *, rows: int, chunk_size: int):
    values = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    labels = torch.arange(rows)
    chunks = write_tensor_chunks(root, (values, labels), chunk_size=chunk_size)
    write_feature_cache_manifest(
        root,
        FeatureCacheManifest(
            kind="sequence_embedding_dataset",
            cache_id="unit",
            dataset="arxiv",
            dataset_variant="unit",
            input_version="unit:v1",
            producer="roberta-base",
            output="last_hidden_state",
            params={},
            row_count=rows,
            content_hash="rows",
            label_schema_hash="labels",
            chunks=chunks,
        ),
    )
    return load_feature_cache(root)


def test_every_batch_stays_within_one_chunk(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=10, chunk_size=4)  # chunks: [0,4) [4,8) [8,10)
    row_indices = list(range(10))
    sampler = ChunkBlockedBatchSampler(
        dataset, row_indices, batch_size=3, drop_last=False, seed=0
    )

    for batch in sampler:
        chunks = {dataset._chunk_index_for_row(row_indices[pos]) for pos in batch}
        assert len(chunks) == 1, "a batch must not span more than one chunk"


def test_covers_every_position_exactly_once(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=10, chunk_size=4)
    row_indices = list(range(10))
    sampler = ChunkBlockedBatchSampler(
        dataset, row_indices, batch_size=3, drop_last=False, seed=1
    )

    seen = [pos for batch in sampler for pos in batch]
    assert sorted(seen) == list(range(10))
    assert len(sampler) == len(list(sampler))


def test_is_deterministic_for_a_seed_and_reshuffles_per_epoch(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=12, chunk_size=4)
    row_indices = list(range(12))

    def batches(seed: int) -> list[list[int]]:
        return list(
            ChunkBlockedBatchSampler(
                dataset, row_indices, batch_size=3, drop_last=False, seed=seed
            )
        )

    assert batches(5) == batches(5)  # reproducible from the seed
    assert batches(5) != batches(6)  # different epoch seed reshuffles


def test_drop_last_drops_short_trailing_batches_per_chunk(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=10, chunk_size=4)  # chunk sizes 4, 4, 2
    row_indices = list(range(10))
    sampler = ChunkBlockedBatchSampler(
        dataset, row_indices, batch_size=3, drop_last=True, seed=0
    )

    batches = list(sampler)
    # Each chunk drops its short tail: chunk of 4 -> one batch of 3; chunk of 2 ->
    # nothing. So 1 + 1 + 0 = 2 full batches, each of size 3.
    assert len(batches) == 2
    assert all(len(batch) == 3 for batch in batches)
    assert len(sampler) == 2


def test_handles_a_subset_of_rows_across_chunks(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=10, chunk_size=4)
    # A non-contiguous subset, as a time slice would produce.
    row_indices = [9, 1, 5, 2, 8, 0]
    sampler = ChunkBlockedBatchSampler(
        dataset, row_indices, batch_size=2, drop_last=False, seed=3
    )

    seen = [pos for batch in sampler for pos in batch]
    assert sorted(seen) == list(range(len(row_indices)))
    for batch in sampler:
        chunks = {dataset._chunk_index_for_row(row_indices[pos]) for pos in batch}
        assert len(chunks) == 1


def test_window_wider_than_one_chunk_mixes_chunks_in_a_batch(tmp_path: Path) -> None:
    dataset = _chunked(tmp_path, rows=12, chunk_size=4)  # 3 chunks of 4
    row_indices = list(range(12))
    sampler = ChunkBlockedBatchSampler(
        dataset, row_indices, batch_size=4, drop_last=False, seed=0, shuffle_window=3
    )

    spanning = [
        batch
        for batch in sampler
        if len({dataset._chunk_index_for_row(row_indices[pos]) for pos in batch}) > 1
    ]
    assert spanning, "a window wider than one chunk must pool rows across chunks"

    seen = [pos for batch in sampler for pos in batch]
    assert sorted(seen) == list(range(12))
    assert len(sampler) == len(list(sampler))
