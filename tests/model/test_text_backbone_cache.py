"""Tests for the text-backbone cache reuse guard."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drift_happens.dataset.cache import (
    FeatureCacheChunk,
    FeatureCacheManifest,
    load_feature_cache,
    write_feature_cache_manifest,
)
from drift_happens.model.text import backbone_cache
from drift_happens.model.text.backbone_cache import (
    TextBackboneCacheRequest,
    _reuse_existing_cache,
    cache_identity_from_request,
    cache_text_backbone_outputs,
)
from drift_happens.model.text.feature_models import masked_mean_pool


def _request(**overrides: Any) -> TextBackboneCacheRequest:
    fields: dict[str, Any] = {
        "kind": "pooled_embedding_dataset",
        "cache_id": "cache-id-sentinel",
        "dataset": "arxiv",
        "dataset_variant": "dataset-variant-sentinel",
        "input_version": "input-version-sentinel:v1",
        "producer": "roberta-base",
        "max_length": 256,
        "text_col": "title_abstract",
        "output": "pooled_embedding",
        "pooling_strategy": "masked_mean",
        "content_hash": "rowhash",
        "label_schema_hash": "labelhash",
    }
    fields.update(overrides)
    return TextBackboneCacheRequest(**fields)


def _write_cache(root, request: TextBackboneCacheRequest, row_count: int = 4) -> None:
    torch.save(
        (torch.zeros(row_count, 3), torch.zeros(row_count, dtype=torch.long)),
        root / "chunk-00000.pt",
    )
    manifest = FeatureCacheManifest(
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
        chunks=(FeatureCacheChunk(path="chunk-00000.pt", start=0, stop=row_count),),
    )
    write_feature_cache_manifest(root, manifest)


def test_reuse_returns_manifest_for_matching_cache(tmp_path) -> None:
    request = _request()
    _write_cache(tmp_path, request)

    reused = _reuse_existing_cache(tmp_path, request, 4)

    assert reused is not None
    assert reused.cache_id == request.cache_id


def test_reuse_none_when_no_manifest(tmp_path) -> None:
    assert _reuse_existing_cache(tmp_path, _request(), 4) is None


def test_reuse_none_when_request_mismatches(tmp_path) -> None:
    _write_cache(tmp_path, _request(max_length=256))
    assert _reuse_existing_cache(tmp_path, _request(max_length=512), 4) is None


def test_reuse_none_when_row_count_differs(tmp_path) -> None:
    request = _request()
    _write_cache(tmp_path, request, row_count=4)
    assert _reuse_existing_cache(tmp_path, request, 5) is None


def test_reuse_none_when_chunk_missing(tmp_path) -> None:
    request = _request()
    _write_cache(tmp_path, request)
    (tmp_path / "chunk-00000.pt").unlink()
    assert _reuse_existing_cache(tmp_path, request, 4) is None


def test_cache_identity_from_request_carries_all_fields() -> None:
    request = _request()
    identity = cache_identity_from_request(request, 4)
    assert identity.kind == request.kind
    assert identity.cache_id == request.cache_id
    assert identity.dataset == request.dataset
    assert identity.dataset_variant == request.dataset_variant
    assert identity.input_version == request.input_version
    assert identity.producer == request.producer
    assert identity.producer_revision == request.producer_revision
    assert identity.output == request.output
    assert identity.content_hash == request.content_hash
    assert identity.label_schema_hash == request.label_schema_hash
    assert identity.cache_schema_version == request.cache_schema_version
    assert identity.producer_adapter_version == request.producer_adapter_version
    assert identity.row_count == 4
    assert identity.params["max_length"] == request.max_length
    assert identity.params["dtype"] == request.embedding_dtype
    assert identity.params["pooling_strategy"] == request.pooling_strategy
    assert identity.params["text_col"] == request.text_col


def test_load_feature_cache_detects_content_change(tmp_path) -> None:
    request = _request()
    _write_cache(tmp_path, request)
    # Matching identity loads; a changed content hash is rejected as stale.
    load_feature_cache(tmp_path, expected=cache_identity_from_request(request, 4))
    with pytest.raises(ValueError, match="content_hash"):
        load_feature_cache(
            tmp_path,
            expected=cache_identity_from_request(_request(content_hash="other"), 4),
        )


def test_reuse_none_when_producer_mismatches(tmp_path) -> None:
    _write_cache(tmp_path, _request(producer="roberta-base"))
    assert _reuse_existing_cache(tmp_path, _request(producer="bert-base"), 4) is None


def test_reuse_none_when_producer_revision_mismatches(tmp_path) -> None:
    _write_cache(tmp_path, _request(producer_revision="rev-a"))
    assert (
        _reuse_existing_cache(tmp_path, _request(producer_revision="rev-b"), 4) is None
    )


def test_reuse_none_when_dtype_mismatches(tmp_path) -> None:
    _write_cache(tmp_path, _request(embedding_dtype="float16"))
    assert (
        _reuse_existing_cache(tmp_path, _request(embedding_dtype="float32"), 4) is None
    )


class _TinyTokenizer:
    def __call__(self, texts, **kwargs):
        rows = len(texts)
        length = kwargs["max_length"]
        return {
            "input_ids": torch.zeros((rows, length), dtype=torch.long),
            "attention_mask": torch.ones((rows, length), dtype=torch.long),
        }


class _TinyModel:
    def eval(self) -> "_TinyModel":
        return self

    def to(self, device) -> "_TinyModel":
        return self

    def __call__(self, **tokenized):
        rows, length = tokenized["input_ids"].shape
        return SimpleNamespace(last_hidden_state=torch.ones((rows, length, 3)))


def test_interrupted_rebuild_leaves_existing_cache_usable(
    tmp_path, monkeypatch
) -> None:
    old_request = _request()
    _write_cache(tmp_path, old_request)
    old_dataset = load_feature_cache(
        tmp_path, expected=cache_identity_from_request(old_request, 4)
    )

    monkeypatch.setattr(
        backbone_cache,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda name, revision=None: _TinyTokenizer()),
    )
    monkeypatch.setattr(
        backbone_cache,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda name, revision=None: _TinyModel()),
    )

    def crash_before_manifest(root, manifest):
        raise RuntimeError("crashed before the manifest write")

    monkeypatch.setattr(
        backbone_cache, "write_feature_cache_manifest", crash_before_manifest
    )

    # New content forces a rebuild that dies after the chunk writes.
    with pytest.raises(RuntimeError, match="before the manifest write"):
        cache_text_backbone_outputs(
            texts=["a", "b", "c", "d"],
            labels=torch.zeros(4, dtype=torch.long),
            cache_root=tmp_path,
            request=_request(content_hash="newhash"),
            device="cpu",
        )

    # Rebuilds publish into a unique chunk directory, so a crash before the new
    # manifest is published must not delete or corrupt readers of the old cache.
    assert torch.equal(old_dataset[0][0], torch.zeros(3))
    reloaded = load_feature_cache(
        tmp_path, expected=cache_identity_from_request(old_request, 4)
    )
    assert torch.equal(reloaded[0][0], torch.zeros(3))


def test_rebuild_uses_immutable_chunk_directory(tmp_path, monkeypatch) -> None:
    old_request = _request()
    _write_cache(tmp_path, old_request)
    old_dataset = load_feature_cache(
        tmp_path, expected=cache_identity_from_request(old_request, 4)
    )
    _patch_tiny_backbone(monkeypatch)

    new_manifest = cache_text_backbone_outputs(
        texts=["a", "b", "c", "d"],
        labels=torch.zeros(4, dtype=torch.long),
        cache_root=tmp_path,
        request=_request(content_hash="newhash"),
        device="cpu",
    )

    assert (tmp_path / "chunk-00000.pt").exists()
    assert torch.equal(old_dataset[0][0], torch.zeros(3))
    assert all(chunk.path.startswith("chunks/") for chunk in new_manifest.chunks)


def test_reuse_rebuilds_corrupt_chunk(tmp_path, monkeypatch) -> None:
    request = _request()
    _write_cache(tmp_path, request)
    (tmp_path / "chunk-00000.pt").write_bytes(b"not a complete torch file")
    _patch_tiny_backbone(monkeypatch)

    manifest = cache_text_backbone_outputs(
        texts=["a", "b", "c", "d"],
        labels=torch.zeros(4, dtype=torch.long),
        cache_root=tmp_path,
        request=request,
        device="cpu",
    )

    assert all(chunk.path.startswith("chunks/") for chunk in manifest.chunks)
    dataset = load_feature_cache(
        tmp_path, expected=cache_identity_from_request(request, 4)
    )
    assert torch.equal(dataset[0][0], torch.ones(3, dtype=torch.float16))


def _patch_tiny_backbone(monkeypatch) -> None:
    monkeypatch.setattr(
        backbone_cache,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda name, revision=None: _TinyTokenizer()),
    )
    monkeypatch.setattr(
        backbone_cache,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda name, revision=None: _TinyModel()),
    )


def _sequence_request(**overrides: Any) -> TextBackboneCacheRequest:
    fields: dict[str, Any] = {
        "kind": "sequence_embedding_dataset",
        "cache_id": "seq-v2",
        "dataset": "arxiv",
        "dataset_variant": "seq-v2",
        "input_version": "seq-v2:v1",
        "producer": "roberta-base",
        "max_length": 8,
        "text_col": "title_abstract",
        "output": "last_hidden_state",
        "content_hash": "rowhash",
        "label_schema_hash": "labelhash",
    }
    fields.update(overrides)
    return TextBackboneCacheRequest(**fields)


def test_sequence_cache_caps_chunk_rows_by_default(tmp_path, monkeypatch) -> None:
    _patch_tiny_backbone(monkeypatch)
    rows = backbone_cache._SEQUENCE_CHUNK_SIZE + 16
    manifest = cache_text_backbone_outputs(
        texts=["x"] * rows,
        labels=torch.zeros(rows, dtype=torch.long),
        cache_root=tmp_path,
        request=_sequence_request(),
        device="cpu",
    )

    assert len(manifest.chunks) > 1
    assert manifest.chunks[0].length == backbone_cache._SEQUENCE_CHUNK_SIZE


def test_sequence_cache_honors_explicit_chunk_size(tmp_path, monkeypatch) -> None:
    _patch_tiny_backbone(monkeypatch)
    rows = backbone_cache._SEQUENCE_CHUNK_SIZE + 16
    manifest = cache_text_backbone_outputs(
        texts=["x"] * rows,
        labels=torch.zeros(rows, dtype=torch.long),
        cache_root=tmp_path,
        request=_sequence_request(chunk_size=rows),
        device="cpu",
    )

    assert len(manifest.chunks) == 1


class _PaddingAwareTokenizer:
    """
    Tokenizer whose output length depends on the padding mode.

    Each text's real length is its character count (capped at ``max_length``).
    ``padding="longest"`` pads to the longest real length in the batch;
    ``padding="max_length"`` pads to ``max_length``. Padding positions get a distinct
    token id so a model can give them non-zero embeddings, letting the test prove the
    masked mean ignores them.
    """

    pad_id = 0

    def __call__(self, texts, **kwargs):
        max_length = kwargs["max_length"]
        real = [min(len(text), max_length) for text in texts]
        if kwargs["padding"] == "longest":
            width = max(real)
        else:
            width = max_length
        input_ids = torch.full((len(texts), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, length in enumerate(real):
            # Real tokens carry id == position+1 so padding (id 0) is distinct.
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _IdEmbeddingModel:
    """Model whose hidden state for a token is its id broadcast over the width."""

    def eval(self) -> "_IdEmbeddingModel":
        return self

    def to(self, device) -> "_IdEmbeddingModel":
        return self

    def __call__(self, **tokenized):
        ids = tokenized["input_ids"].float()
        # Padding tokens (id 0) embed to zero here; bump them so a leak would show.
        hidden = torch.where(ids == 0.0, torch.full_like(ids, 99.0), ids)
        return SimpleNamespace(last_hidden_state=hidden.unsqueeze(-1).repeat(1, 1, 3))


def test_pooled_cache_longest_padding_matches_max_length(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        backbone_cache,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda name, revision=None: _PaddingAwareTokenizer()
        ),
    )
    monkeypatch.setattr(
        backbone_cache,
        "AutoModel",
        SimpleNamespace(
            from_pretrained=lambda name, revision=None: _IdEmbeddingModel()
        ),
    )

    texts = ["a", "abc", "abcdef", "ab"]
    labels = torch.zeros(len(texts), dtype=torch.long)

    # Production now builds pooled caches with padding="longest".
    cache_root = tmp_path / "pooled"
    cache_root.mkdir()
    cache_text_backbone_outputs(
        texts=texts,
        labels=labels,
        cache_root=cache_root,
        request=_request(max_length=8, content_hash="pad-pool"),
        device="cpu",
    )
    pooled = load_feature_cache(cache_root).gather(list(range(len(texts))))[0]

    # Hand-compute the masked-mean pooling under the legacy max_length padding;
    # because the mean ignores padding, the two must be bit-for-bit identical.
    tokenized = _PaddingAwareTokenizer()(
        texts, padding="max_length", truncation=True, max_length=8
    )
    ids = tokenized["input_ids"].float()
    hidden = torch.where(ids == 0.0, torch.full_like(ids, 99.0), ids)
    hidden = hidden.unsqueeze(-1).repeat(1, 1, 3)
    expected = masked_mean_pool(hidden, tokenized["attention_mask"].bool()).to(
        torch.float16
    )

    assert torch.equal(pooled, expected)


def test_writer_reuses_without_loading_model(tmp_path, monkeypatch) -> None:
    request = _request()
    _write_cache(tmp_path, request)

    class _NoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("model must not load when a valid cache exists")

    monkeypatch.setattr(backbone_cache, "AutoModel", _NoModel)
    monkeypatch.setattr(backbone_cache, "AutoTokenizer", _NoModel)

    result = cache_text_backbone_outputs(
        texts=["a", "b", "c", "d"],
        labels=torch.zeros(4, dtype=torch.long),
        cache_root=tmp_path,
        request=request,
    )

    assert result.cache_id == request.cache_id
