"""
End-to-end tests for the conference text path with a faked backbone.

These drive the real cache write/read path (tokenization loop, masked mean pooling,
batch concatenation, chunking, manifest, identity-checked load), the per-dataset wiring
in ``_conference_text_dataset`` and ``setup()``, and the handoff from a cached embedding
into the conference model head. A tiny deterministic stand-in replaces the HuggingFace
tokenizer and model so no weights are downloaded; the fake emits each frozen backbone's
real embedding dimension and marks the trailing positions as padding with a distinct
hidden value, so pooling that ignored the attention mask or a head sized for the wrong
dimension would fail the assertions.
"""

from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest
import torch

from drift_happens.dataset.arxiv.scope import (
    ARXIV_SCOPE_VARIANT,
    ARXIV_TARGET_LABELS,
)
from drift_happens.model.text import backbone_cache
from drift_happens.model.text.frozen_backbone import (
    FROZEN_TEXT_BACKBONE_DIMS,
    FROZEN_TEXT_BACKBONE_PRODUCERS,
)
from drift_happens.pipeline.amazon_reviews_23 import run as amazon_run
from drift_happens.pipeline.arxiv import run as arxiv_run
from drift_happens.pipeline.text.trainers import model_factory

_PADDING = 2
_PADDING_HIDDEN = 9.0
_BERT_DIM = FROZEN_TEXT_BACKBONE_DIMS["bert_base_frozen"]
_ROBERTA_DIM = FROZEN_TEXT_BACKBONE_DIMS["roberta_base_frozen"]
_MINILM_DIM = FROZEN_TEXT_BACKBONE_DIMS["minilm_l6_frozen"]
_PRODUCER_DIM = {
    producer: FROZEN_TEXT_BACKBONE_DIMS[key]
    for key, producer in FROZEN_TEXT_BACKBONE_PRODUCERS.items()
}


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        rows = len(texts)
        length = kwargs["max_length"]
        attention_mask = torch.ones((rows, length), dtype=torch.long)
        attention_mask[:, length - _PADDING :] = 0
        return {
            "input_ids": torch.zeros((rows, length), dtype=torch.long),
            "attention_mask": attention_mask,
        }


class _FakeModel:
    def __init__(self, hidden_dim: int) -> None:
        self._hidden_dim = hidden_dim

    def eval(self) -> "_FakeModel":
        return self

    def to(self, device) -> "_FakeModel":
        return self

    def __call__(self, **tokenized):
        input_ids = tokenized["input_ids"]
        rows, length = input_ids.shape
        # Real positions carry 1.0; padded positions carry a distinct value that
        # masked mean pooling must drop. Same device as the input so the real
        # pooling/cast code runs unchanged on cpu, mps, or cuda.
        hidden = torch.ones(
            (rows, length, self._hidden_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )
        hidden[:, length - _PADDING :, :] = _PADDING_HIDDEN
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(name, revision=None) -> _FakeTokenizer:
        return _FakeTokenizer()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(name, revision=None) -> _FakeModel:
        return _FakeModel(_PRODUCER_DIM[name])


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch) -> None:
    monkeypatch.setattr(backbone_cache, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(backbone_cache, "AutoModel", _FakeAutoModel)
    # Stub out the live HuggingFace revision lookup so tests are hermetic.
    monkeypatch.setattr(
        arxiv_run, "resolve_huggingface_revision", lambda _: "test-revision"
    )
    monkeypatch.setattr(
        amazon_run, "resolve_huggingface_revision", lambda _: "test-revision"
    )


def _arxiv_df() -> pd.DataFrame:
    # Distinct per-row labels so a row permutation cannot pass the alignment check.
    return pd.DataFrame(
        {
            "title_abstract": ["paper one", "paper two", "paper three"],
            "top_subjects": [["cs.LG"], ["hep-ph", "cs.CV"], ["gr-qc"]],
        }
    )


def _arxiv_setup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title_abstract": [f"paper {index}" for index in range(6)],
            "top_subjects": [
                ["cs.LG"],
                ["hep-ph"],
                ["cs.CV"],
                ["cs.AI"],
                ["hep-th"],
                ["quant-ph"],
            ],
            "year": [2000, 2000, 2001, 2001, 2002, 2002],
        }
    )


def test_arxiv_conference_pooled_glue(tmp_path) -> None:
    dataset, category_to_idx, labels = arxiv_run._conference_text_dataset(
        _arxiv_df(), model_key="bert_base_frozen", max_seq_len=8, cache_dir=tmp_path
    )

    assert len(dataset) == 3
    embedding, label = dataset[0]
    assert embedding.shape == (_BERT_DIM,)  # pooled, no sequence dimension
    # Masked mean over the real positions (all 1.0) ignores the padded values.
    assert torch.allclose(embedding.float(), torch.ones(_BERT_DIM), atol=1e-3)
    assert label.shape == (len(ARXIV_TARGET_LABELS),)
    assert set(category_to_idx) == set(ARXIV_TARGET_LABELS)
    assert labels.shape == (3, len(ARXIV_TARGET_LABELS))
    for row in range(3):
        assert torch.equal(dataset[row][-1], labels[row])

    cache_root = tmp_path / "bert-base-uncased" / "pooled_embedding_dataset"
    assert (cache_root / "manifest.json").exists()
    assert list(cache_root.glob("chunks/*/chunk-*.pt"))


class _ForbiddenBackbone:
    @staticmethod
    def from_pretrained(name, revision=None):
        raise AssertionError("the backbone was loaded instead of reusing the cache")


def test_arxiv_conference_pooled_reuses_existing_cache(tmp_path, monkeypatch) -> None:
    first, _category_to_idx, _labels = arxiv_run._conference_text_dataset(
        _arxiv_df(), model_key="bert_base_frozen", max_seq_len=8, cache_dir=tmp_path
    )

    # A second call with the same inputs must reuse the cache without touching the
    # backbone; loading either fake here would raise.
    monkeypatch.setattr(backbone_cache, "AutoTokenizer", _ForbiddenBackbone)
    monkeypatch.setattr(backbone_cache, "AutoModel", _ForbiddenBackbone)

    second, _category_to_idx_again, _labels_again = arxiv_run._conference_text_dataset(
        _arxiv_df(), model_key="bert_base_frozen", max_seq_len=8, cache_dir=tmp_path
    )

    assert len(second) == len(first)
    assert torch.equal(second[0][0], first[0][0])
    assert torch.equal(second[0][-1], first[0][-1])


def _distinct_subjects(index: int) -> list[str]:
    # A unique subject set per row, so a row swap across batches cannot pass the
    # alignment check: a single subject below the subject count, a distinct pair
    # above it.
    count = len(ARXIV_TARGET_LABELS)
    primary = ARXIV_TARGET_LABELS[index % count]
    if index < count:
        return [primary]
    return [primary, ARXIV_TARGET_LABELS[(index + 1) % count]]


def test_arxiv_conference_pooled_spans_multiple_batches(tmp_path) -> None:
    # More rows than the cache batch size, so the across-batch concatenation runs.
    rows = (
        backbone_cache.TextBackboneCacheRequest.model_fields["batch_size"].default + 8
    )
    df = pd.DataFrame(
        {
            "title_abstract": [f"paper {index}" for index in range(rows)],
            "top_subjects": [_distinct_subjects(index) for index in range(rows)],
        }
    )

    dataset, _category_to_idx, labels = arxiv_run._conference_text_dataset(
        df, model_key="bert_base_frozen", max_seq_len=8, cache_dir=tmp_path
    )

    assert len(dataset) == rows
    for row in range(rows):
        embedding, _label = dataset[row]
        assert embedding.shape == (_BERT_DIM,)
        assert torch.equal(dataset[row][-1], labels[row])


def test_arxiv_conference_sequence_glue(tmp_path) -> None:
    dataset, _category_to_idx, _labels = arxiv_run._conference_text_dataset(
        _arxiv_df(), model_key="from_scratch", max_seq_len=8, cache_dir=tmp_path
    )

    assert len(dataset) == 3
    hidden, mask, label = dataset[0]
    assert hidden.shape == (8, _ROBERTA_DIM)  # sequence (length, hidden)
    assert mask.shape == (8,)
    assert int(mask[: 8 - _PADDING].sum()) == 8 - _PADDING
    assert int(mask[8 - _PADDING :].sum()) == 0
    assert torch.allclose(hidden[0].float(), torch.ones(_ROBERTA_DIM), atol=1e-3)
    assert torch.allclose(
        hidden[-1].float(), torch.full((_ROBERTA_DIM,), _PADDING_HIDDEN), atol=1e-3
    )
    assert label.shape == (len(ARXIV_TARGET_LABELS),)

    cache_root = tmp_path / "roberta-base" / "sequence_embedding_dataset"
    assert (cache_root / "manifest.json").exists()


def test_amazon_conference_pooled_glue(tmp_path) -> None:
    df = pl.DataFrame(
        {"text": ["a review", "b review", "c review"], "rating": [1, 5, 3]}
    )

    dataset, labels = amazon_run._conference_text_dataset(
        df, model_key="minilm_l6_frozen", max_seq_len=8, cache_dir=tmp_path
    )

    assert len(dataset) == 3
    embedding, label = dataset[0]
    assert embedding.shape == (_MINILM_DIM,)
    assert torch.allclose(embedding.float(), torch.ones(_MINILM_DIM), atol=1e-3)
    assert labels.shape == (3,)
    assert labels.dtype == torch.long
    for row in range(3):
        assert torch.equal(dataset[row][-1], labels[row])

    cache_root = (
        tmp_path / "sentence-transformers_all-MiniLM-L6-v2" / "pooled_embedding_dataset"
    )
    assert (cache_root / "manifest.json").exists()


def test_arxiv_setup_accepts_keys_sharing_the_sequence_cache(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(arxiv_run, "load_arxiv", lambda: _arxiv_setup_df())
    monkeypatch.setattr(arxiv_run, "ARTIFACTS_DIR", tmp_path)

    context = arxiv_run.setup(trainer_keys=["ffn_s", "tx_s"])

    assert context.trainer_keys == ["ffn_s", "tx_s"]
    cache_root = (
        tmp_path
        / "cache"
        / "arxiv"
        / ARXIV_SCOPE_VARIANT
        / "roberta-base"
        / "sequence_embedding_dataset"
    )
    assert (cache_root / "manifest.json").exists()


def test_arxiv_setup_rejects_keys_needing_different_caches(monkeypatch) -> None:
    # The guard must fire before any data is touched.
    monkeypatch.setattr(
        arxiv_run,
        "load_arxiv",
        lambda: pytest.fail("setup loaded data despite mismatched cache keys"),
    )

    with pytest.raises(ValueError, match="bert_base_frozen"):
        arxiv_run.setup(trainer_keys=["ffn_s", "bert_base_frozen"])


def test_amazon_setup_rejects_keys_needing_different_caches(monkeypatch) -> None:
    monkeypatch.setattr(
        amazon_run,
        "load_amazon_reviews_23",
        lambda: pytest.fail("setup loaded data despite mismatched cache keys"),
    )

    with pytest.raises(ValueError, match="minilm_l6_frozen"):
        amazon_run.setup(trainer_keys=["bert_base_frozen", "minilm_l6_frozen"])


def test_arxiv_setup_conference_assembles_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(arxiv_run, "load_arxiv", lambda: _arxiv_setup_df())
    monkeypatch.setattr(arxiv_run, "ARTIFACTS_DIR", tmp_path)

    context = arxiv_run.setup(trainer_keys=["bert_base_frozen"])

    assert len(context.tensor_dataset) == 6
    assert set(context.category_to_idx) == set(ARXIV_TARGET_LABELS)
    assert context.pos_weight.shape == (len(ARXIV_TARGET_LABELS),)
    assert context.trainer_keys == ["bert_base_frozen"]
    assert context.trainer_configs["bert_base_frozen"].feature_input_dim == _BERT_DIM
    assert len(context.train_time_slices) >= 1
    assert len(context.dataset_splits.train_df) >= 1
    assert (tmp_path / "cache" / "arxiv").exists()


def test_arxiv_conference_model_consumes_cached_embeddings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(arxiv_run, "load_arxiv", lambda: _arxiv_setup_df())
    monkeypatch.setattr(arxiv_run, "ARTIFACTS_DIR", tmp_path)

    context = arxiv_run.setup(trainer_keys=["bert_base_frozen"])

    config = context.trainer_configs["bert_base_frozen"]
    model = model_factory(config=config, dim_output=config.num_classes)
    embedding, _label = context.tensor_dataset[0]

    # The frozen head must be sized to the cached embedding it actually receives.
    assert model.classifier.in_features == embedding.shape[-1]

    logits = model(embedding.unsqueeze(0))

    assert logits.shape == (1, len(ARXIV_TARGET_LABELS))
    assert torch.isfinite(logits).all()


def test_amazon_setup_conference_assembles_context(tmp_path, monkeypatch) -> None:
    # Every rating in 1..5 is present: setup() builds the class weights over the
    # full rating range, so a missing rating would leave a weight undefined.
    df = pl.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "text": [f"review {index}" for index in range(5)],
            "rating": [1, 2, 3, 4, 5],
            "half_year": [28, 28, 29, 29, 30],
        }
    )
    monkeypatch.setattr(amazon_run, "load_amazon_reviews_23", lambda: df.clone())
    monkeypatch.setattr(amazon_run, "ARTIFACTS_DIR", tmp_path)

    context = amazon_run.setup(trainer_keys=["minilm_l6_frozen"])

    assert len(context.tensor_dataset) == 5
    assert context.class_weights.shape == (5,)
    assert context.trainer_keys == ["minilm_l6_frozen"]
    assert context.trainer_configs["minilm_l6_frozen"].feature_input_dim == _MINILM_DIM
    assert len(context.train_time_slices) >= 1
    assert len(context.dataset_splits.train_df) >= 1
    assert (tmp_path / "cache" / "amazon_reviews_23").exists()


def test_amazon_class_weights_come_from_the_training_split(
    tmp_path, monkeypatch
) -> None:
    # Eight rows per half_year so the 70/30 split actually carves off a test
    # set with a label balance that differs from the full data.
    rows = 16
    df = pl.DataFrame(
        {
            "row_id": list(range(rows)),
            "text": [f"review {index}" for index in range(rows)],
            "rating": [1, 2, 3, 4, 5, 1, 2, 3] * 2,
            "half_year": [28] * 8 + [29] * 8,
        }
    )
    monkeypatch.setattr(amazon_run, "load_amazon_reviews_23", lambda: df.clone())
    monkeypatch.setattr(amazon_run, "ARTIFACTS_DIR", tmp_path)

    context = amazon_run.setup(trainer_keys=["minilm_l6_frozen"])

    train_counts = (
        context.dataset_splits.train_df["rating"].value_counts().reindex(range(1, 6))
    )
    expected = torch.tensor(
        [train_counts.sum() / train_counts[i] for i in range(1, 6)],
        dtype=torch.float32,
    )
    full_counts = df.to_pandas()["rating"].value_counts().reindex(range(1, 6))
    full_dataset_weights = torch.tensor(
        [full_counts.sum() / full_counts[i] for i in range(1, 6)],
        dtype=torch.float32,
    )

    torch.testing.assert_close(context.class_weights, expected)
    assert not torch.allclose(context.class_weights, full_dataset_weights)


def test_amazon_setup_rejects_a_training_split_missing_a_rating(
    tmp_path, monkeypatch
) -> None:
    # All five ratings exist in the data, but rating 1 appears once and the
    # seed-42 split sends that row to the test set, so the guard must fire.
    ratings = [1, 3, 5, 4, 4, 3, 4, 3, 5, 2, 5, 2, 3, 2]
    rows = len(ratings)
    df = pl.DataFrame(
        {
            "row_id": list(range(rows)),
            "text": [f"review {index}" for index in range(rows)],
            "rating": ratings,
            "half_year": [28] * 7 + [29] * 7,
        }
    )
    monkeypatch.setattr(amazon_run, "load_amazon_reviews_23", lambda: df.clone())
    monkeypatch.setattr(amazon_run, "ARTIFACTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="no rows for rating"):
        amazon_run.setup(trainer_keys=["minilm_l6_frozen"])


def test_arxiv_pos_weight_comes_from_the_training_split(tmp_path, monkeypatch) -> None:
    rows = 16
    df = pd.DataFrame(
        {
            "year": [2000] * 8 + [2001] * 8,
            "title_abstract": [f"paper {index}" for index in range(rows)],
            "top_subjects": [
                ["cs.LG"] if index % 4 else ["hep-th"] for index in range(rows)
            ],
        }
    )
    monkeypatch.setattr(arxiv_run, "load_arxiv", lambda: df.copy())
    monkeypatch.setattr(arxiv_run, "ARTIFACTS_DIR", tmp_path)

    context = arxiv_run.setup(trainer_keys=["minilm_l6_frozen"])

    train_index = torch.as_tensor(
        context.dataset_splits.train_df.index.to_numpy(), dtype=torch.long
    )
    labels = torch.zeros((rows, len(context.category_to_idx)))
    for row, subjects in enumerate(df["top_subjects"]):
        for subject in subjects:
            labels[row, context.category_to_idx[subject]] = 1.0
    train_labels = labels[train_index]
    positives = train_labels.sum(dim=0)
    expected = torch.where(
        positives > 0,
        (train_labels.shape[0] - positives) / positives,
        torch.ones_like(positives),
    )

    torch.testing.assert_close(context.pos_weight, expected)
