from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
)
from drift_happens.pipeline.image.trainers import (
    conference_image_model_configs,
    is_frozen_conference_model,
)
from drift_happens.pipeline.imdb_faces.trainers import (
    imdb_faces_conference_trainer_configs,
)
from drift_happens.pipeline.yearbook.trainers import (
    yearbook_conference_trainer_configs,
)

_EXPECTED_TOTAL = 21
_EXPECTED_FROZEN = 9


def test_frozen_conference_models_precompute_their_embeddings() -> None:
    # A frozen backbone must declare needs_backend_fw_pass=False so the
    # pipeline precomputes embeddings once instead of forwarding it every batch.
    configs = conference_image_model_configs()
    frozen = {
        key: config
        for key, config in configs.items()
        if is_frozen_conference_model(key)
    }
    transfer_keys = {
        key
        for key, config in configs.items()
        if isinstance(config, TransferLearningConfig)
    }

    # Every TransferLearningConfig key must carry the _frozen suffix and vice versa.
    assert transfer_keys == set(frozen), (
        f"frozen-suffix keys {set(frozen)} != TransferLearningConfig keys {transfer_keys}"
    )
    assert len(configs) == _EXPECTED_TOTAL
    assert len(frozen) == _EXPECTED_FROZEN
    for config in frozen.values():
        assert isinstance(config, TransferLearningConfig)
        assert config.fine_tune is False
        assert config.needs_backend_fw_pass is False


def test_image_datasets_serve_the_same_conference_lineup() -> None:
    # yearbook_conference_trainer_configs and imdb_faces_conference_trainer_configs
    # are module-level aliases of conference_image_trainer_configs; this test
    # catches future divergence if the aliases are ever replaced with independent
    # implementations.
    shared = set(conference_image_model_configs())

    assert set(yearbook_conference_trainer_configs()) == shared
    assert set(imdb_faces_conference_trainer_configs()) == shared
