"""Tests for manifest-backed feature caches."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

from drift_happens.dataset.cache import (
    CacheIdentity,
    ChunkedTensorDataset,
    FeatureCacheChunk,
    FeatureCacheManifest,
    _payload_to_tensors,
    content_fingerprint,
    load_feature_cache,
    write_feature_cache_manifest,
    write_tensor_chunks,
)


def _identity(manifest: FeatureCacheManifest, **overrides: Any) -> CacheIdentity:
    base = {
        f.name: getattr(manifest, f.name) for f in dataclasses.fields(CacheIdentity)
    }
    base.update(overrides)
    return CacheIdentity(**base)


def _manifest(chunks, row_count: int) -> FeatureCacheManifest:
    return FeatureCacheManifest(
        kind="sequence_embedding_dataset",
        cache_id="test-cache",
        dataset="arxiv",
        dataset_variant="unit",
        input_version="unit:v1",
        producer="roberta-base",
        output="last_hidden_state",
        params={"max_length": 4},
        row_count=row_count,
        content_hash="rows",
        label_schema_hash="labels",
        chunks=chunks,
    )


def test_chunked_feature_cache_round_trips(tmp_path) -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    mask = torch.ones(3, 2, dtype=torch.bool)
    labels = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    chunks = write_tensor_chunks(tmp_path, (embeddings, mask, labels), chunk_size=2)
    write_feature_cache_manifest(tmp_path, _manifest(chunks, row_count=3))

    dataset = load_feature_cache(tmp_path)

    row = dataset[2]
    assert len(dataset) == 3
    assert torch.equal(row[0], embeddings[2])
    assert torch.equal(row[1], mask[2])
    assert torch.equal(row[2], labels[2])


def _write_simple_cache(tmp_path) -> FeatureCacheManifest:
    chunks = write_tensor_chunks(
        tmp_path, (torch.arange(6, dtype=torch.float32).reshape(3, 2),), chunk_size=2
    )
    manifest = _manifest(chunks, row_count=3)
    write_feature_cache_manifest(tmp_path, manifest)
    return manifest


def test_load_feature_cache_accepts_matching_identity(tmp_path) -> None:
    manifest = _write_simple_cache(tmp_path)
    dataset = load_feature_cache(tmp_path, expected=_identity(manifest))
    assert len(dataset) == 3


def test_load_feature_cache_none_expected_skips_identity_check(tmp_path) -> None:
    _write_simple_cache(tmp_path)
    assert len(load_feature_cache(tmp_path, expected=None)) == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"producer": "bert-base"},
        {"output": "pooled_embedding"},
        {"params": {"max_length": 512}},
        {"row_count": 5},
        {"content_hash": "tampered"},
        {"cache_schema_version": 2},
    ],
)
def test_load_feature_cache_rejects_mismatched_identity(tmp_path, overrides) -> None:
    manifest = _write_simple_cache(tmp_path)
    field = next(iter(overrides))
    with pytest.raises(ValueError, match=field):
        load_feature_cache(tmp_path, expected=_identity(manifest, **overrides))


def test_manifest_rejects_empty_chunks_with_nonzero_row_count(tmp_path) -> None:
    manifest = _manifest((), row_count=1)

    with pytest.raises(ValueError, match="must list chunks"):
        write_feature_cache_manifest(tmp_path, manifest)


def test_load_feature_cache_raises_on_missing_chunk_file(tmp_path) -> None:
    # Write a valid cache then delete a chunk file to simulate corruption.
    _write_simple_cache(tmp_path)
    for chunk_file in tmp_path.glob("chunk-*.pt"):
        chunk_file.unlink()

    with pytest.raises(FileNotFoundError, match="missing cache chunk"):
        load_feature_cache(tmp_path)


def test_chunked_tensor_dataset_supports_negative_index(tmp_path) -> None:
    values = torch.arange(6).reshape(3, 2)
    chunks = write_tensor_chunks(tmp_path, (values,), chunk_size=2)

    row = ChunkedTensorDataset(tmp_path, _manifest(chunks, row_count=3))[-1]

    torch.testing.assert_close(row[0], values[-1])


def test_chunked_tensor_dataset_rejects_out_of_range_index(tmp_path) -> None:
    chunks = write_tensor_chunks(tmp_path, (torch.arange(3),), chunk_size=2)
    dataset = ChunkedTensorDataset(tmp_path, _manifest(chunks, row_count=3))

    with pytest.raises(IndexError):
        dataset[3]


def test_manifest_rejects_gap_or_overlapping_chunks(tmp_path) -> None:
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"x")
    gap = _manifest((FeatureCacheChunk(path="a.pt", start=1, stop=2),), row_count=2)
    overlap = _manifest(
        (
            FeatureCacheChunk(path="a.pt", start=0, stop=2),
            FeatureCacheChunk(path="b.pt", start=1, stop=3),
        ),
        row_count=3,
    )

    with pytest.raises(ValueError, match="expected 0"):
        gap.assert_complete(tmp_path)
    with pytest.raises(ValueError, match="expected 2"):
        overlap.assert_complete(tmp_path)


def test_write_tensor_chunks_rejects_mismatched_leading_dim(tmp_path) -> None:
    with pytest.raises(ValueError, match="same leading dimension"):
        write_tensor_chunks(tmp_path, (torch.zeros(2), torch.zeros(3)), chunk_size=2)


def test_payload_to_tensors_accepts_tensor_dataset_and_dict_payload() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(2), torch.ones(2))
    payload = {"tensors": [torch.zeros(2, 1), torch.ones(2, 1)]}

    assert len(_payload_to_tensors(dataset)) == 2
    assert len(_payload_to_tensors(payload)) == 2


def test_load_chunk_loads_tensor_dataset_payload(tmp_path) -> None:
    # A chunk pickled as a TensorDataset must load through _load_chunk, not only
    # through _payload_to_tensors directly: torch.load(weights_only=True) rejects
    # TensorDataset unless the load is wrapped in safe_globals([TensorDataset]).
    embeddings = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    labels = torch.arange(3, dtype=torch.float32)
    torch.save(
        torch.utils.data.TensorDataset(embeddings, labels), tmp_path / "chunk0.pt"
    )
    chunks = (FeatureCacheChunk(path="chunk0.pt", start=0, stop=3),)
    dataset = ChunkedTensorDataset(tmp_path, _manifest(chunks, row_count=3))

    gathered = dataset.gather([2, 0, 1])

    assert torch.equal(gathered[0], embeddings[[2, 0, 1]])
    assert torch.equal(gathered[1], labels[[2, 0, 1]])


def test_gather_preserves_order_across_chunks(tmp_path) -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    labels = torch.arange(6, dtype=torch.float32)
    chunks = write_tensor_chunks(tmp_path, (embeddings, labels), chunk_size=2)
    write_feature_cache_manifest(tmp_path, _manifest(chunks, row_count=6))
    dataset = load_feature_cache(tmp_path)

    gathered = dataset.gather([5, 0, 3, 1])

    assert torch.equal(gathered[0], embeddings[[5, 0, 3, 1]])
    assert torch.equal(gathered[1], labels[[5, 0, 3, 1]])


def test_gather_repeats_and_reorders_within_a_chunk(tmp_path) -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    labels = torch.arange(6, dtype=torch.float32)
    chunks = write_tensor_chunks(tmp_path, (embeddings, labels), chunk_size=2)
    write_feature_cache_manifest(tmp_path, _manifest(chunks, row_count=6))
    dataset = load_feature_cache(tmp_path)

    order = [4, 4, 1, 5, 0, 1]
    gathered = dataset.gather(order)

    assert torch.equal(gathered[0], embeddings[order])
    assert torch.equal(gathered[1], labels[order])


def test_gather_preserves_dtype_per_column(tmp_path) -> None:
    embeddings = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    masks = torch.ones(6, 2, dtype=torch.bool)
    labels = torch.arange(6, dtype=torch.long)
    chunks = write_tensor_chunks(tmp_path, (embeddings, masks, labels), chunk_size=2)
    write_feature_cache_manifest(tmp_path, _manifest(chunks, row_count=6))
    dataset = load_feature_cache(tmp_path)

    gathered = dataset.gather([5, 0, 2])

    assert gathered[0].dtype == torch.float16
    assert gathered[1].dtype == torch.bool
    assert gathered[2].dtype == torch.long
    assert torch.equal(gathered[0], embeddings[[5, 0, 2]])
    assert torch.equal(gathered[1], masks[[5, 0, 2]])
    assert torch.equal(gathered[2], labels[[5, 0, 2]])


def test_gather_loads_each_chunk_at_most_once(tmp_path) -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    labels = torch.arange(6, dtype=torch.float32)
    chunks = write_tensor_chunks(tmp_path, (embeddings, labels), chunk_size=2)
    write_feature_cache_manifest(tmp_path, _manifest(chunks, row_count=6))
    dataset = load_feature_cache(tmp_path)

    loaded: list[int] = []
    original = dataset._load_chunk

    def _counting_load(chunk_index: int):
        loaded.append(chunk_index)
        return original(chunk_index)

    dataset._load_chunk = _counting_load  # type: ignore[method-assign]
    gathered = dataset.gather([5, 0, 3, 1])

    # Each of the three touched chunks is read exactly once, even though the
    # requested rows interleave across them.
    assert sorted(loaded) == [0, 1, 2]
    assert torch.equal(gathered[0], embeddings[[5, 0, 3, 1]])


def test_gather_handles_empty_and_rejects_out_of_range(tmp_path) -> None:
    _write_simple_cache(tmp_path)
    dataset = load_feature_cache(tmp_path)

    empty = dataset.gather([])
    assert empty[0].shape == (0, 2)

    with pytest.raises(IndexError):
        dataset.gather([3])


def test_gather_on_empty_cache_raises_clear_error(tmp_path) -> None:
    # A zero-row cache is a valid manifest (assert_complete blesses chunks=()),
    # but it has no column schema, so gather([]) must report that clearly
    # instead of an IndexError from indexing the empty chunk list.
    write_feature_cache_manifest(tmp_path, _manifest((), row_count=0))
    dataset = load_feature_cache(tmp_path)

    with pytest.raises(ValueError, match="empty cache"):
        dataset.gather([])


def test_content_fingerprint_is_deterministic() -> None:
    texts = ["hello", "world"]
    labels = torch.tensor([0, 1])
    assert content_fingerprint(texts, labels) == content_fingerprint(texts, labels)


def test_content_fingerprint_changes_with_texts() -> None:
    labels = torch.tensor([0, 1])
    assert content_fingerprint(["hello", "world"], labels) != content_fingerprint(
        ["hello", "there"], labels
    )


def test_content_fingerprint_changes_with_labels() -> None:
    texts = ["hello", "world"]
    assert content_fingerprint(texts, torch.tensor([0, 1])) != content_fingerprint(
        texts, torch.tensor([1, 0])
    )


def test_content_fingerprint_is_order_sensitive() -> None:
    labels = torch.tensor([0, 1])
    assert content_fingerprint(["a", "b"], labels) != content_fingerprint(
        ["b", "a"], labels
    )


def test_content_fingerprint_separates_text_boundaries() -> None:
    labels = torch.tensor([0])
    assert content_fingerprint(["a", "b"], labels) != content_fingerprint(
        ["ab"], labels
    )


def test_content_fingerprint_separates_boundaries_at_fixed_count() -> None:
    # Same text count, same concatenation — only per-text framing can distinguish.
    labels = torch.tensor([0, 1])
    assert content_fingerprint(["aa", "b"], labels) != content_fingerprint(
        ["a", "ab"], labels
    )


def test_content_fingerprint_separates_null_boundaries_at_fixed_count() -> None:
    labels = torch.tensor([0, 1])
    assert content_fingerprint(["a\x00", "b"], labels) != content_fingerprint(
        ["a", "\x00b"], labels
    )


def test_content_fingerprint_distinguishes_label_dtype() -> None:
    texts = ["a", "b"]
    int_fp = content_fingerprint(texts, torch.tensor([1, 0], dtype=torch.int64))
    float_fp = content_fingerprint(texts, torch.tensor([1.0, 0.0], dtype=torch.float32))
    assert int_fp != float_fp


def test_content_fingerprint_is_short_hex() -> None:
    fingerprint = content_fingerprint(["hello"], torch.tensor([0]))
    assert len(fingerprint) == 16
    assert all(char in "0123456789abcdef" for char in fingerprint)


def test_content_fingerprint_handles_embedded_null_in_text() -> None:
    labels = torch.tensor([0])
    assert content_fingerprint(["a\x00b"], labels) != content_fingerprint(
        ["a", "b"], labels
    )


def test_content_fingerprint_distinguishes_label_shape() -> None:
    texts = ["row"]
    flat = content_fingerprint(texts, torch.zeros(6))
    matrix = content_fingerprint(texts, torch.zeros(2, 3))
    assert flat != matrix
