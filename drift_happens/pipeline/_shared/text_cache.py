"""Backbone cache planning shared by the conference text pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from drift_happens.model.text.frozen_backbone import FROZEN_TEXT_BACKBONE_PRODUCERS


@dataclass(frozen=True, slots=True)
class ConferenceTextCachePlan:
    """The backbone feature cache one conference text trainer key consumes."""

    producer: str
    kind: Literal["sequence_embedding_dataset", "pooled_embedding_dataset"]
    output: Literal["last_hidden_state", "pooled_embedding"]
    pooling_strategy: Literal["masked_mean"] | None


def conference_text_cache_plan(model_key: str) -> ConferenceTextCachePlan:
    """
    Cache plan for ``model_key``.

    Frozen-backbone heads read pooled embeddings produced by their own backbone; every
    scratch sequence architecture reads the shared RoBERTa sequence cache.
    """
    if model_key in FROZEN_TEXT_BACKBONE_PRODUCERS:
        return ConferenceTextCachePlan(
            producer=FROZEN_TEXT_BACKBONE_PRODUCERS[model_key],
            kind="pooled_embedding_dataset",
            output="pooled_embedding",
            pooling_strategy="masked_mean",
        )
    return ConferenceTextCachePlan(
        producer="roberta-base",
        kind="sequence_embedding_dataset",
        output="last_hidden_state",
        pooling_strategy=None,
    )


def single_text_cache_plan(trainer_keys: Sequence[str]) -> ConferenceTextCachePlan:
    """
    The one cache plan shared by every key in ``trainer_keys``.

    The text CLIs build a single feature cache per invocation, so all requested keys
    must consume the same producer and cache kind; running a key against another key's
    cache would feed it the wrong features.
    """
    if not trainer_keys:
        raise ValueError("at least one trainer key is required")
    expected = conference_text_cache_plan(trainer_keys[0])
    mismatched = sorted(
        {key for key in trainer_keys if conference_text_cache_plan(key) != expected}
    )
    if mismatched:
        raise ValueError(
            f"trainer key(s) {', '.join(mismatched)} need a different backbone "
            f"cache than {trainer_keys[0]!r} "
            f"({expected.producer}/{expected.kind}); run keys with different "
            "cache needs in separate invocations"
        )
    return expected
