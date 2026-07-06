from __future__ import annotations

import pytest
import torch

from drift_happens.experiments.common import TIER_TARGETS
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES,
    TEXT_SCRATCH_FAMILIES,
    text_model_factory,
)
from drift_happens.model.text.feature_models import (
    EmbeddingFeatureRNN,
    EmbeddingFeatureRNNConfig,
    masked_mean_pool,
)
from drift_happens.model.text.frozen_backbone import (
    FROZEN_TEXT_BACKBONE_DIMS,
    FROZEN_TEXT_BACKBONE_IDS,
    pooled_embedding_head_for_backbone,
)


def test_sequence_feature_models_accept_embeddings_and_masks() -> None:
    embeddings = torch.randn(2, 8, 768)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, -2:] = False

    for architecture in CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES:
        model = text_model_factory(
            architecture,
            dim_output=20,
            feature_input_dim=768,
        )
        out = model(embeddings, mask)
        assert out.shape == (2, 20), architecture


def test_frozen_text_heads_accept_pooled_embeddings() -> None:
    for backbone_id in FROZEN_TEXT_BACKBONE_IDS:
        model = pooled_embedding_head_for_backbone(backbone_id, dim_output=1)
        pooled = torch.randn(3, FROZEN_TEXT_BACKBONE_DIMS[backbone_id])
        out = model(pooled)
        assert out.shape == (3, 1), backbone_id


def test_masked_mean_pool_ignores_masked_tokens_and_handles_all_masked() -> None:
    embeddings = torch.tensor([[[1.0, 3.0], [9.0, 9.0]], [[5.0, 7.0], [8.0, 10.0]]])
    mask = torch.tensor([[True, False], [False, False]])

    pooled = masked_mean_pool(embeddings, mask)

    torch.testing.assert_close(pooled, torch.tensor([[1.0, 3.0], [0.0, 0.0]]))


def test_text_model_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unknown text architecture"):
        text_model_factory("unknown", 10)  # type: ignore[arg-type]


def test_text_scratch_families_keys_match_conference_architectures() -> None:
    assert tuple(TEXT_SCRATCH_FAMILIES) == CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES, (
        "TEXT_SCRATCH_FAMILIES keys must stay in 1:1 order-matched sync with "
        "CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES"
    )


def test_text_scratch_families_tiers_are_known_targets() -> None:
    tiers = {tier for _, tier in TEXT_SCRATCH_FAMILIES.values()}
    unknown = tiers - set(TIER_TARGETS)
    assert not unknown, (
        f"TEXT_SCRATCH_FAMILIES tiers {sorted(unknown)} are not in TIER_TARGETS; "
        "TIER_TARGETS.get() would silently return None for them"
    )


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_embedding_feature_rnn_is_invariant_to_padding_length(
    rnn_type: str, use_attention: bool, bidirectional: bool
) -> None:
    torch.manual_seed(0)
    config = EmbeddingFeatureRNNConfig(
        projection_dim=16,
        hidden_dim=16,
        num_layers=2,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        use_attention=use_attention,
        dropout=0.0,
    )
    model = EmbeddingFeatureRNN(input_dim=8, dim_output=3, config=config).eval()

    lengths = torch.tensor([5, 3, 7])
    short_len, long_len = 7, 19
    mask_short = torch.arange(short_len).unsqueeze(0) < lengths.unsqueeze(1)
    PAD_POISON = 9.0  # conspicuous value that must never leak into the output
    base = torch.randn(3, short_len, 8).masked_fill(
        ~mask_short.unsqueeze(-1), PAD_POISON
    )
    padded = torch.full((3, long_len, 8), PAD_POISON)
    padded[:, :short_len] = base
    mask_long = torch.zeros(3, long_len, dtype=torch.bool)
    mask_long[:, :short_len] = mask_short

    torch.testing.assert_close(model(base, mask_short), model(padded, mask_long))


def test_embedding_feature_rnn_handles_an_all_pad_row() -> None:
    torch.manual_seed(0)
    config = EmbeddingFeatureRNNConfig(
        projection_dim=16, hidden_dim=16, num_layers=1, rnn_type="gru"
    )
    model = EmbeddingFeatureRNN(input_dim=8, dim_output=3, config=config).eval()
    embeddings = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1] = False

    # Also covers dtype tolerance: float16 embeddings and int64 mask must be accepted.
    out = model(embeddings.half(), mask.long())

    assert torch.isfinite(out).all()
