"""Manifest-backed chunked tensor caches keyed by row and label content."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch.utils.data import Dataset

from drift_happens.configs.base import BaseConfig

FeatureCacheKind = Literal[
    "sequence_embedding_dataset",
    "pooled_embedding_dataset",
    "prediction_table",
]


class FeatureCacheChunk(BaseConfig):
    """One tensor chunk in a feature cache."""

    path: str = Field(min_length=1)
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


class FeatureCacheManifest(BaseConfig):
    """Portable manifest describing reusable cached model outputs."""

    kind: FeatureCacheKind
    cache_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_variant: str = Field(min_length=1)
    input_version: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    producer_revision: str | None = None
    output: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    row_count: int
    content_hash: str
    label_schema_hash: str
    chunks: tuple[FeatureCacheChunk, ...]
    cache_schema_version: int = 1
    producer_adapter_version: int = 1

    def assert_complete(self, root: Path) -> None:
        """Raise if the manifest points to missing or inconsistent chunks."""
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if not self.chunks and self.row_count != 0:
            raise ValueError("non-empty cache manifests must list chunks")

        expected_start = 0
        for chunk in self.chunks:
            if chunk.start != expected_start:
                raise ValueError(
                    f"chunk {chunk.path} starts at {chunk.start}, expected {expected_start}"
                )
            if chunk.stop < chunk.start:
                raise ValueError(f"chunk {chunk.path} has stop < start")
            path = root / chunk.path
            if not path.exists():
                raise FileNotFoundError(f"missing cache chunk: {path}")
            expected_start = chunk.stop

        if expected_start != self.row_count:
            raise ValueError(
                f"chunk rows end at {expected_start}, manifest row_count is {self.row_count}"
            )


@dataclass(frozen=True)
class CacheIdentity:
    """
    The fields of a feature cache that define its content.

    Two caches with the same identity hold interchangeable tensors; ``chunks`` is
    excluded because the on-disk chunking is a storage detail, not content.
    """

    kind: FeatureCacheKind
    cache_id: str
    dataset: str
    dataset_variant: str
    input_version: str
    producer: str
    producer_revision: str | None
    output: str
    params: dict[str, Any]
    row_count: int
    content_hash: str
    label_schema_hash: str
    cache_schema_version: int
    producer_adapter_version: int


def _identity_mismatches(
    manifest: FeatureCacheManifest, expected: CacheIdentity
) -> list[str]:
    """Field names where ``manifest`` differs from ``expected`` (empty when equal)."""
    # Every CacheIdentity field is an identity field; the manifest carries each
    # under the same name, so adding one tightens this check automatically.
    return [
        f"{field.name}: cached {got!r} != expected {want!r}"
        for field in fields(expected)
        if (got := getattr(manifest, field.name))
        != (want := getattr(expected, field.name))
    ]


def content_fingerprint(texts: Sequence[str], labels: torch.Tensor) -> str:
    """
    Short content hash over the rows a cached dataset is built from.

    Folds the row texts (in order) and the label tensor (dtype, shape and bytes) into
    one digest, so a change to either yields a different cache id instead of reusing
    stale rows. Every field is length-framed, so no text content — an embedded NUL byte
    included — can make two different inputs collide.
    """
    hasher = hashlib.sha256()
    hasher.update(len(texts).to_bytes(8, "big"))
    for text in texts:
        encoded = text.encode()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    array = labels.detach().cpu().contiguous().numpy()
    descriptor = f"{array.dtype}:{array.shape}".encode()
    hasher.update(len(descriptor).to_bytes(8, "big"))
    hasher.update(descriptor)
    hasher.update(array.tobytes())
    return hasher.hexdigest()[:16]


class ChunkedTensorDataset(Dataset):
    """Lazy ``Dataset`` over a complete chunked tensor cache."""

    def __init__(
        self, root: Path, manifest: FeatureCacheManifest, chunk_cache_size: int = 1
    ):
        self.root = Path(root)
        self.manifest = manifest
        self.manifest.assert_complete(self.root)
        self._chunk_cache: OrderedDict[int, tuple[torch.Tensor, ...]] = OrderedDict()
        self._chunk_cache_size = max(1, chunk_cache_size)

    def ensure_chunk_cache_size(self, size: int) -> None:
        """Keep at least ``size`` recently used chunks resident across reads."""
        self._chunk_cache_size = max(self._chunk_cache_size, size)

    def __len__(self) -> int:
        return self.manifest.row_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        chunk_index = self._chunk_index_for_row(index)
        chunk = self.manifest.chunks[chunk_index]
        tensors = self._load_chunk(chunk_index)
        local_index = index - chunk.start
        return tuple(tensor[local_index] for tensor in tensors)

    def gather(self, indices: Sequence[int]) -> tuple[torch.Tensor, ...]:
        """
        Materialize the given rows as stacked column tensors.

        Rows are grouped per chunk so every chunk file is loaded at most once, and the
        output preserves the order of ``indices``. Each chunk's rows are copied straight
        into the preallocated output and the chunk is released before the next one
        loads, so peak memory stays near one chunk plus the output rather than every
        touched chunk at once.
        """
        if not indices:
            if not self.manifest.chunks:
                raise ValueError(
                    "cannot gather from an empty cache: a zero-row cache has no "
                    "chunks and therefore no column schema to materialize"
                )
            empty = self._load_chunk(0)
            return tuple(tensor[:0] for tensor in empty)

        positions_by_chunk: dict[int, list[int]] = {}
        for position, index in enumerate(indices):
            if index < 0 or index >= len(self):
                raise IndexError(index)
            positions_by_chunk.setdefault(self._chunk_index_for_row(index), []).append(
                position
            )

        row_count = len(indices)
        outputs: tuple[torch.Tensor, ...] | None = None
        for chunk_index, positions in positions_by_chunk.items():
            chunk = self.manifest.chunks[chunk_index]
            tensors = self._load_chunk(chunk_index)
            if outputs is None:
                outputs = tuple(
                    torch.empty((row_count, *tensor.shape[1:]), dtype=tensor.dtype)
                    for tensor in tensors
                )
            local_indices = torch.as_tensor(
                [indices[position] - chunk.start for position in positions],
                dtype=torch.long,
            )
            dest = torch.as_tensor(positions, dtype=torch.long)
            for output, tensor in zip(outputs, tensors, strict=True):
                # index_copy_ writes this chunk's rows into their final slots and
                # holds no reference to ``tensor``, so it can be freed before the
                # next chunk is loaded.
                output.index_copy_(0, dest, tensor.index_select(0, local_indices))

        assert outputs is not None  # non-empty indices guarantee at least one chunk
        return outputs

    def iter_chunk_gathers(
        self, indices: Sequence[int]
    ) -> Iterator[tuple[list[int], tuple[torch.Tensor, ...]]]:
        """
        Yield ``(positions, columns)`` for the requested rows, one chunk at a time.

        Only the rows that fall in the current chunk are materialized before the chunk
        is released, so the caller can stream a slice through a model without ever
        holding more than one chunk plus its own small per-chunk outputs in memory.
        ``positions`` are the indices into ``indices`` that the yielded rows came from,
        letting the caller scatter per-row results back into the original order;
        ``columns`` are the stacked column tensors for those rows, in the order
        ``positions`` lists them.
        """
        if not indices:
            return

        positions_by_chunk: dict[int, list[int]] = {}
        for position, index in enumerate(indices):
            if index < 0 or index >= len(self):
                raise IndexError(index)
            positions_by_chunk.setdefault(self._chunk_index_for_row(index), []).append(
                position
            )

        for chunk_index in sorted(positions_by_chunk):
            positions = positions_by_chunk[chunk_index]
            chunk = self.manifest.chunks[chunk_index]
            tensors = self._load_chunk(chunk_index)
            local_indices = torch.as_tensor(
                [indices[position] - chunk.start for position in positions],
                dtype=torch.long,
            )
            columns = tuple(tensor.index_select(0, local_indices) for tensor in tensors)
            yield positions, columns

    def _chunk_index_for_row(self, index: int) -> int:
        for chunk_index, chunk in enumerate(self.manifest.chunks):
            if chunk.start <= index < chunk.stop:
                return chunk_index
        raise IndexError(index)

    def _load_chunk(self, chunk_index: int) -> tuple[torch.Tensor, ...]:
        cached = self._chunk_cache.get(chunk_index)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk_index)
            return cached

        chunk = self.manifest.chunks[chunk_index]
        # Load through the safe unpickler, allowing the TensorDataset payload
        # form (mirrors _load_tensor_dataset).
        path = self.root / chunk.path
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            payload = torch.load(path, map_location="cpu")
        else:
            with safe_globals([torch.utils.data.TensorDataset]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        tensors = _payload_to_tensors(payload)
        if not tensors:
            raise ValueError(f"cache chunk {chunk.path} did not contain tensors")
        for tensor in tensors:
            if tensor.shape[0] != chunk.length:
                raise ValueError(
                    f"chunk {chunk.path} tensor has {tensor.shape[0]} rows, expected {chunk.length}"
                )
        self._chunk_cache[chunk_index] = tensors
        while len(self._chunk_cache) > self._chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return tensors


def load_feature_cache(
    root: Path, expected: CacheIdentity | None = None
) -> ChunkedTensorDataset:
    """
    Load and validate a chunked feature cache from ``root``.

    When ``expected`` is given, raise ``ValueError`` if the stored manifest does not
    have that content identity, so a stale or wrong cache is never served.
    """
    manifest = FeatureCacheManifest.model_validate_json(
        (Path(root) / "manifest.json").read_text()
    )
    if expected is not None:
        mismatches = _identity_mismatches(manifest, expected)
        if mismatches:
            raise ValueError(
                f"feature cache at {root} does not match the expected identity "
                f"({'; '.join(mismatches)})"
            )
    return ChunkedTensorDataset(Path(root), manifest)


def write_feature_cache_manifest(root: Path, manifest: FeatureCacheManifest) -> Path:
    """Atomically write ``manifest.json`` after all chunks are present."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest.assert_complete(root)
    target = root / "manifest.json"
    tmp = root / f".manifest.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return target


def write_atomic_torch_file(payload: object, target: Path) -> Path:
    """Write a torch payload through a unique temp file, then publish atomically."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return target


def write_tensor_chunks(
    root: Path,
    tensors: Sequence[torch.Tensor],
    *,
    chunk_size: int,
    prefix: str = "chunk",
) -> tuple[FeatureCacheChunk, ...]:
    """Write tensors with shared leading dimension into chunk files."""
    if not tensors:
        raise ValueError("at least one tensor is required")
    row_count = tensors[0].shape[0]
    for tensor in tensors:
        if tensor.shape[0] != row_count:
            raise ValueError("all tensors must share the same leading dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[FeatureCacheChunk] = []
    for chunk_index, start in enumerate(range(0, row_count, chunk_size)):
        stop = min(start + chunk_size, row_count)
        rel_path = f"{prefix}-{chunk_index:05d}.pt"
        write_atomic_torch_file(
            tuple(tensor[start:stop].cpu() for tensor in tensors),
            root / rel_path,
        )
        chunks.append(FeatureCacheChunk(path=rel_path, start=start, stop=stop))
    return tuple(chunks)


def _payload_to_tensors(payload: object) -> tuple[torch.Tensor, ...]:
    if isinstance(payload, torch.Tensor):
        return (payload,)
    if isinstance(payload, torch.utils.data.TensorDataset):
        return tuple(payload.tensors)
    if isinstance(payload, (tuple, list)):
        if all(isinstance(item, torch.Tensor) for item in payload):
            return tuple(payload)
    if isinstance(payload, dict):
        tensors = payload.get("tensors")
        if isinstance(tensors, (tuple, list)) and all(
            isinstance(item, torch.Tensor) for item in tensors
        ):
            return tuple(tensors)
    raise TypeError(f"unsupported cache chunk payload type: {type(payload)!r}")
