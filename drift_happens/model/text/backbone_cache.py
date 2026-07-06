"""Build chunked caches from frozen Hugging Face text backbones."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, ValidationError
from transformers import AutoModel, AutoTokenizer

from drift_happens.configs.base import BaseConfig
from drift_happens.dataset.cache import (
    CacheIdentity,
    FeatureCacheChunk,
    FeatureCacheManifest,
    load_feature_cache,
    write_atomic_torch_file,
    write_feature_cache_manifest,
)
from drift_happens.model.text.feature_models import masked_mean_pool
from drift_happens.utils.log import get_logger
from drift_happens.utils.pytorch import device_manual_mps_or_cuda_if_available

logger = get_logger()

TextBackboneCacheKind = Literal[
    "sequence_embedding_dataset",
    "pooled_embedding_dataset",
]

# A sequence chunk holds [rows, max_length, hidden] tensors, so the pooled
# default of 8192 rows would build multi-GiB chunks. Cap sequence rows per chunk.
_SEQUENCE_CHUNK_SIZE = 512


class TextBackboneCacheRequest(BaseConfig):
    """Stable request fields for frozen text-backbone feature caches."""

    kind: TextBackboneCacheKind
    cache_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_variant: str = Field(min_length=1)
    input_version: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    producer_revision: str | None = None
    max_length: int = Field(gt=0)
    text_col: str
    output: Literal["last_hidden_state", "pooled_embedding"]
    content_hash: str
    label_schema_hash: str
    pooling_strategy: Literal["masked_mean"] | None = None
    embedding_dtype: Literal["float16", "float32"] = "float16"
    batch_size: int = 32
    chunk_size: int = 8192
    cache_schema_version: int = 1
    producer_adapter_version: int = 1


def cache_text_backbone_outputs(
    *,
    texts: Sequence[str],
    labels: torch.Tensor,
    cache_root: Path,
    request: TextBackboneCacheRequest,
    device: str | torch.device | None = None,
) -> FeatureCacheManifest:
    """
    Compute and cache frozen text-backbone outputs.

    Sequence caches store ``(last_hidden_state, attention_mask, labels)``. Pooled caches
    store ``(pooled_embedding, labels)`` using masked mean pooling.
    """
    if len(texts) != labels.shape[0]:
        raise ValueError("texts and labels must have the same row count")
    if (
        request.kind == "sequence_embedding_dataset"
        and request.output != "last_hidden_state"
    ):
        raise ValueError("sequence caches must request last_hidden_state output")
    if (
        request.kind == "pooled_embedding_dataset"
        and request.output != "pooled_embedding"
    ):
        raise ValueError("pooled caches must request pooled_embedding output")
    if request.kind == "pooled_embedding_dataset" and request.pooling_strategy is None:
        raise ValueError("pooled caches must specify a pooling strategy")
    if (
        request.kind == "sequence_embedding_dataset"
        and request.pooling_strategy is not None
    ):
        raise ValueError("sequence caches must not specify a pooling strategy")
    if request.batch_size <= 0 or request.chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive")

    chunk_size = request.chunk_size
    if (
        request.kind == "sequence_embedding_dataset"
        and "chunk_size" not in request.model_fields_set
    ):
        chunk_size = _SEQUENCE_CHUNK_SIZE

    cache_root = Path(cache_root)
    row_count = len(texts)
    identity = cache_identity_from_request(request, row_count)
    reused = _reuse_existing_cache(cache_root, request, row_count)
    if reused is not None:
        return reused
    cache_root.mkdir(parents=True, exist_ok=True)
    chunk_root = _build_chunk_root(cache_root, identity)
    torch_dtype = (
        torch.float16 if request.embedding_dtype == "float16" else torch.float32
    )
    run_device = torch.device(
        device
        if device is not None
        else (device_manual_mps_or_cuda_if_available() or "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        request.producer,
        revision=request.producer_revision,
    )
    model = AutoModel.from_pretrained(
        request.producer,
        revision=request.producer_revision,
    )
    model.eval()
    model.to(run_device)

    chunks: list[FeatureCacheChunk] = []
    chunk_payload: list[tuple[torch.Tensor, ...]] = []
    chunk_start = 0

    # Sequence caches stack fixed-length tensors across chunks, so every batch
    # must pad to ``max_length``. Pooled caches reduce each row with a masked mean
    # that ignores padding, so padding only to the longest row in the batch is
    # output-identical and avoids embedding the padding positions at all.
    padding = (
        "max_length" if request.kind == "sequence_embedding_dataset" else "longest"
    )

    with torch.no_grad():
        for start in range(0, len(texts), request.batch_size):
            stop = min(start + request.batch_size, len(texts))
            tokenized = tokenizer(
                list(texts[start:stop]),
                padding=padding,
                truncation=True,
                max_length=request.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(run_device) for key, value in tokenized.items()}
            outputs = model(**tokenized)
            mask = tokenized["attention_mask"].bool()
            hidden = outputs.last_hidden_state

            batch_payload: tuple[torch.Tensor, ...]
            if request.kind == "sequence_embedding_dataset":
                batch_payload = (
                    hidden.to(dtype=torch_dtype).cpu(),
                    mask.cpu(),
                    labels[start:stop].cpu(),
                )
            else:
                pooled = masked_mean_pool(hidden.float(), mask)
                batch_payload = (
                    pooled.to(dtype=torch_dtype).cpu(),
                    labels[start:stop].cpu(),
                )

            chunk_payload.append(batch_payload)
            payload_rows = sum(batch[0].shape[0] for batch in chunk_payload)
            if payload_rows >= chunk_size or stop == row_count:
                chunk_stop = chunk_start + payload_rows
                rel_path = f"chunk-{len(chunks):05d}.pt"
                chunk_path = chunk_root / rel_path
                write_atomic_torch_file(_concat_payload(chunk_payload), chunk_path)
                chunks.append(
                    FeatureCacheChunk(
                        path=str(chunk_path.relative_to(cache_root)),
                        start=chunk_start,
                        stop=chunk_stop,
                    )
                )
                chunk_payload = []
                chunk_start = chunk_stop

    manifest = FeatureCacheManifest(
        **dataclasses.asdict(identity), chunks=tuple(chunks)
    )
    write_feature_cache_manifest(cache_root, manifest)
    return manifest


def _concat_payload(
    payload: Sequence[tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...]:
    if not payload:
        raise ValueError("cannot concatenate an empty payload")
    columns = tuple(zip(*payload))
    return tuple(torch.cat(column, dim=0) for column in columns)


def cache_identity_from_request(
    request: TextBackboneCacheRequest, row_count: int
) -> CacheIdentity:
    """The content identity a cache built from ``request`` must have."""
    return CacheIdentity(
        kind=request.kind,
        cache_id=request.cache_id,
        dataset=request.dataset,
        dataset_variant=request.dataset_variant,
        input_version=request.input_version,
        producer=request.producer,
        producer_revision=request.producer_revision,
        output=request.output,
        params={
            "dtype": request.embedding_dtype,
            "max_length": request.max_length,
            "pooling_strategy": request.pooling_strategy,
            "text_col": request.text_col,
        },
        row_count=row_count,
        content_hash=request.content_hash,
        label_schema_hash=request.label_schema_hash,
        cache_schema_version=request.cache_schema_version,
        producer_adapter_version=request.producer_adapter_version,
    )


def _reuse_existing_cache(
    cache_root: Path,
    request: TextBackboneCacheRequest,
    row_count: int,
) -> FeatureCacheManifest | None:
    """
    Return a valid existing cache manifest that matches ``request``, if any.

    A cache is reused only when its manifest loads, its chunks are present and
    consistent, and its content identity matches the request. A present-but-mismatching
    or corrupt cache is reported and rebuilt, so a stale cache is never silently served.
    """
    if not (cache_root / "manifest.json").exists():
        return None
    try:
        cached = load_feature_cache(
            cache_root, expected=cache_identity_from_request(request, row_count)
        )
        _assert_torch_chunks_are_readable(cache_root, cached.manifest)
    except (OSError, ValueError, ValidationError) as error:
        logger.info(f"Ignoring unusable or mismatched cache at {cache_root}: {error}")
        return None
    logger.info(f"Reusing cached text-backbone outputs at {cache_root}")
    return cached.manifest


def _build_chunk_root(cache_root: Path, identity: CacheIdentity) -> Path:
    """Return a unique directory for this build's immutable chunk files."""
    identity_payload = json.dumps(
        dataclasses.asdict(identity), sort_keys=True, separators=(",", ":")
    ).encode()
    identity_digest = hashlib.sha256(identity_payload).hexdigest()[:16]
    build_id = f"{identity_digest}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    return cache_root / "chunks" / build_id


def _assert_torch_chunks_are_readable(
    cache_root: Path, manifest: FeatureCacheManifest
) -> None:
    """Reject chunks that are present but not complete torch zip archives."""
    for chunk in manifest.chunks:
        path = cache_root / chunk.path
        if not zipfile.is_zipfile(path):
            raise ValueError(f"cache chunk is not a complete torch file: {path}")
