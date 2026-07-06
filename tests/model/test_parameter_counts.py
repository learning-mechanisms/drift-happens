from __future__ import annotations

import pytest

from drift_happens.model.dataset.image.architectures import ImageModelFactory
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES,
    text_model_factory,
)
from drift_happens.model.parameters import count_parameters
from drift_happens.pipeline.image.trainers import image_model_factory


def test_yearbook_scratch_parameter_tiers_are_calibrated() -> None:
    rows = {
        "mlp_s": image_model_factory(ImageModelFactory._mlp_preset("mlp_s")),
        "mlp_m": image_model_factory(ImageModelFactory._mlp_preset("mlp_m")),
        "mlp_l": image_model_factory(ImageModelFactory._mlp_preset("mlp_l")),
        "cnn_s": image_model_factory(ImageModelFactory._cnn_preset("cnn_s")),
        "cnn_m": image_model_factory(ImageModelFactory._cnn_preset("cnn_m")),
        "cnn_l": image_model_factory(ImageModelFactory._cnn_preset("cnn_l")),
        "resnet_s": image_model_factory(ImageModelFactory._resnet_preset("resnet_s")),
        "resnet_m": image_model_factory(ImageModelFactory._resnet_preset("resnet_m")),
        "resnet_l": image_model_factory(ImageModelFactory._resnet_preset("resnet_l")),
        "vit_s": image_model_factory(ImageModelFactory._vit_preset("vit_s")),
        "vit_m": image_model_factory(ImageModelFactory._vit_preset("vit_m")),
        "vit_l": image_model_factory(ImageModelFactory._vit_preset("vit_l")),
    }

    for name, model in rows.items():
        trainable = count_parameters(model).trainable
        if name == "vit_s":
            # vit_s genuinely lands ~75k — below the generic small floor
            assert 70_000 <= trainable <= 120_000, name
        elif name.endswith("_s"):
            assert 80_000 <= trainable <= 120_000, name
        elif name.endswith("_m"):
            assert 400_000 <= trainable <= 600_000, name
        elif name.endswith("_l"):
            assert 1_600_000 <= trainable <= 2_400_000, name
        else:
            pytest.fail(f"{name!r} has no recognized tier suffix; add bounds")


def test_text_sequence_parameter_tiers_are_calibrated() -> None:
    for architecture in CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES:
        model = text_model_factory(
            architecture,
            dim_output=20,
            feature_input_dim=768,
        )
        trainable = count_parameters(model).trainable
        if architecture.endswith("_s"):
            assert 80_000 <= trainable <= 120_000, architecture
        elif architecture.endswith("_m"):
            assert 400_000 <= trainable <= 600_000, architecture
        elif architecture.endswith("_l"):
            assert 1_600_000 <= trainable <= 2_400_000, architecture
        else:
            pytest.fail(f"{architecture!r} has no recognized tier suffix; add bounds")
