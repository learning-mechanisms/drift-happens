"""Tests for the shared conference text cache plan and its multi-key guard."""

import pytest

from drift_happens.model.text.frozen_backbone import FROZEN_TEXT_BACKBONE_PRODUCERS
from drift_happens.pipeline._shared.text_cache import (
    conference_text_cache_plan,
    single_text_cache_plan,
)


def test_frozen_key_plans_a_pooled_cache_from_its_own_producer() -> None:
    plan = conference_text_cache_plan("bert_base_frozen")

    assert plan.producer == FROZEN_TEXT_BACKBONE_PRODUCERS["bert_base_frozen"]
    assert plan.kind == "pooled_embedding_dataset"
    assert plan.output == "pooled_embedding"
    assert plan.pooling_strategy == "masked_mean"


def test_sequence_key_plans_the_shared_roberta_cache() -> None:
    plan = conference_text_cache_plan("ffn_s")

    assert plan.producer == "roberta-base"
    assert plan.kind == "sequence_embedding_dataset"
    assert plan.output == "last_hidden_state"
    assert plan.pooling_strategy is None


def test_single_plan_accepts_keys_sharing_one_cache() -> None:
    plan = single_text_cache_plan(["ffn_s", "tx_m", "bilstm_m"])

    assert plan == conference_text_cache_plan("ffn_s")


def test_single_plan_rejects_mixed_sequence_and_frozen_keys() -> None:
    with pytest.raises(ValueError, match="bert_base_frozen"):
        single_text_cache_plan(["ffn_s", "bert_base_frozen"])


def test_single_plan_rejects_frozen_keys_with_distinct_producers() -> None:
    # Every frozen head reads its own producer's pooled cache, so even two
    # frozen keys cannot share one invocation.
    with pytest.raises(ValueError, match="minilm_l6_frozen"):
        single_text_cache_plan(["bert_base_frozen", "minilm_l6_frozen"])


def test_single_plan_rejects_an_empty_key_list() -> None:
    with pytest.raises(ValueError, match="at least one trainer key"):
        single_text_cache_plan([])
