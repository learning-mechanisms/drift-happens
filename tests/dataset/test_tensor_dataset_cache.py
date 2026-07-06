"""Tests for TensorDataset cache loading."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from drift_happens.dataset.dataset import TensorDatasetCache


def test_tensor_dataset_cache_loads_existing_tensor_dataset(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    expected = TensorDataset(
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([0, 1]),
    )
    torch.save(expected, tmp_path / "sample_embed_cached.pt")

    def fail() -> TensorDataset:
        raise AssertionError("embedding function should not run for an existing cache")

    loaded = cache.get_create_cache_embedding("cached", fail)

    assert isinstance(loaded, TensorDataset)
    assert len(loaded.tensors) == len(expected.tensors)
    for actual_tensor, expected_tensor in zip(loaded.tensors, expected.tensors):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def _fresh() -> TensorDataset:
    return TensorDataset(torch.tensor([[9, 9]]), torch.tensor([1]))


def _write_stale(path) -> None:
    torch.save(TensorDataset(torch.tensor([[0, 0]]), torch.tensor([0])), path)


def test_reuse_loads_existing_and_skips_recompute(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    _write_stale(tmp_path / "sample_embed_e.pt")

    def fail() -> TensorDataset:
        raise AssertionError("reuse must not recompute when a cache exists")

    loaded = cache.get_create_cache_embedding("e", fail, reuse_policy="reuse")

    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[0, 0]]))


def test_reuse_computes_and_writes_on_first_call(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    path = tmp_path / "sample_embed_e.pt"

    loaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="reuse")

    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[9, 9]]))
    assert path.exists()

    def fail() -> TensorDataset:
        raise AssertionError("the written cache must be reused on the next call")

    reused = cache.get_create_cache_embedding("e", fail, reuse_policy="reuse")
    torch.testing.assert_close(reused.tensors[0], torch.tensor([[9, 9]]))


def test_refresh_recomputes_and_overwrites(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    path = tmp_path / "sample_embed_e.pt"
    _write_stale(path)

    loaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="refresh")

    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[9, 9]]))
    # The stale file is overwritten with the recomputed embedding.
    reloaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="reuse")
    torch.testing.assert_close(reloaded.tensors[0], torch.tensor([[9, 9]]))


def test_interrupted_write_leaves_no_loadable_cache_file(tmp_path, monkeypatch) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    path = tmp_path / "sample_embed_e.pt"

    def crash(obj, target) -> None:
        # Simulate partial bytes landing at the write target before failure.
        Path(target).write_bytes(b"partial")
        raise RuntimeError("interrupted mid-write")

    monkeypatch.setattr(torch, "save", crash)
    with pytest.raises(RuntimeError, match="interrupted mid-write"):
        cache.get_create_cache_embedding("e", _fresh, reuse_policy="reuse")

    # The write goes to a .tmp sibling; on failure the .tmp is unlinked, so
    # neither the final path nor a stale .tmp survives the interrupted write.
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    monkeypatch.undo()

    loaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="reuse")
    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[9, 9]]))
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_disabled_ignores_cache_and_writes_nothing(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    path = tmp_path / "sample_embed_e.pt"

    loaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="disabled")

    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[9, 9]]))
    assert not path.exists()


def test_disabled_does_not_read_existing_cache(tmp_path) -> None:
    cache = TensorDatasetCache(cache_dir=tmp_path, dataset_id="sample")
    path = tmp_path / "sample_embed_e.pt"
    _write_stale(path)

    loaded = cache.get_create_cache_embedding("e", _fresh, reuse_policy="disabled")

    # The recomputed value is returned and the existing file is left untouched.
    torch.testing.assert_close(loaded.tensors[0], torch.tensor([[9, 9]]))
    on_disk = cache.get_create_cache_embedding("e", _fresh, reuse_policy="reuse")
    torch.testing.assert_close(on_disk.tensors[0], torch.tensor([[0, 0]]))
