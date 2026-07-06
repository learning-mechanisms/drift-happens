"""Opt-in smoke and download tests for every dataset/model trainer matrix."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from drift_happens.experiments.plans import build_plan_stages, materialized_presets
from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
)
from drift_happens.model.dataset.text.architectures import (
    CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES,
)
from drift_happens.model.text.frozen_backbone import (
    FROZEN_TEXT_BACKBONE_DIMS,
    FROZEN_TEXT_BACKBONE_PRODUCERS,
)
from drift_happens.model.trainer.pytorch import PytorchTrainer
from drift_happens.pipeline.amazon_reviews_23.trainers import (
    amazon_reviews_conference_trainer_configs,
)
from drift_happens.pipeline.amazon_reviews_23.trainers import (
    build_trainers_from_configs as build_amazon_trainers,
)
from drift_happens.pipeline.arxiv.trainers import (
    arxiv_conference_trainer_configs,
)
from drift_happens.pipeline.arxiv.trainers import (
    build_trainers_from_configs as build_arxiv_trainers,
)
from drift_happens.pipeline.context import PipelineContext
from drift_happens.pipeline.image.run import embed_dataset_if_needed
from drift_happens.pipeline.imdb_faces.trainers import (
    build_trainers_from_configs as build_imdb_trainers,
)
from drift_happens.pipeline.imdb_faces.trainers import (
    imdb_faces_conference_trainer_configs,
)
from drift_happens.pipeline.yearbook.trainers import (
    build_trainers_from_configs as build_yearbook_trainers,
)
from drift_happens.pipeline.yearbook.trainers import (
    yearbook_conference_trainer_configs,
)
from drift_happens.sample.splits import DatasetSplit

pytestmark = pytest.mark.integration

_RUN_MODEL_MATRIX_SMOKE = pytest.mark.skipif(
    os.environ.get("DRIFT_RUN_MODEL_MATRIX_SMOKE") != "1",
    reason="set DRIFT_RUN_MODEL_MATRIX_SMOKE=1 to run external model matrix smoke",
)
_RUN_MODEL_DOWNLOAD_SMOKE = pytest.mark.skipif(
    os.environ.get("DRIFT_RUN_MODEL_DOWNLOAD_SMOKE") != "1",
    reason="set DRIFT_RUN_MODEL_DOWNLOAD_SMOKE=1 to run external download smoke",
)

SampleTask = Literal["image_classification", "arxiv_multilabel", "amazon_regression"]
TrainerBuilder = Callable[[dict[str, Any]], dict[str, PytorchTrainer]]

SAMPLE_COUNT = 2
TEXT_SEQUENCE_LENGTH = 8
ARXIV_CATEGORY_TO_IDX = {"cs": 0, "math": 1, "stat": 2}
FULL_MATRIX_PLAN_STAGES = (
    "p40_amazon_reviews_23_all_seeds",
    "p50_arxiv_all_seeds",
    "p60_yearbook_all_seeds",
    "p70_imdb_faces_all_seeds",
)


@dataclass(frozen=True, slots=True)
class SmokeCase:
    """One dataset/model trainer configuration to instantiate and train once."""

    dataset: str
    matrix: str
    trainer_key: str
    trainer_config: Any
    build_trainers: TrainerBuilder
    task: SampleTask

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.matrix}/{self.trainer_key}"


def _all_smoke_cases() -> tuple[SmokeCase, ...]:
    class_weights = torch.ones(5)
    arxiv_pos_weight = torch.ones(len(ARXIV_CATEGORY_TO_IDX))

    return (
        *_cases_from_configs(
            dataset="yearbook",
            matrix="conference",
            configs=yearbook_conference_trainer_configs(),
            build_trainers=build_yearbook_trainers,
            task="image_classification",
        ),
        *_cases_from_configs(
            dataset="imdb_faces",
            matrix="conference",
            configs=imdb_faces_conference_trainer_configs(),
            build_trainers=build_imdb_trainers,
            task="image_classification",
        ),
        *_cases_from_configs(
            dataset="arxiv",
            matrix="conference",
            configs=arxiv_conference_trainer_configs(
                category_to_idx=ARXIV_CATEGORY_TO_IDX,
                pos_weight=arxiv_pos_weight,
            ),
            build_trainers=build_arxiv_trainers,
            task="arxiv_multilabel",
        ),
        *_cases_from_configs(
            dataset="amazon_reviews_23",
            matrix="conference",
            configs=amazon_reviews_conference_trainer_configs(
                class_weights=class_weights,
            ),
            build_trainers=build_amazon_trainers,
            task="amazon_regression",
        ),
    )


def _filter_by_env_pattern(
    items: tuple, env_var: str, key: Callable[[Any], str]
) -> tuple:
    pattern = os.environ.get(env_var)
    if pattern is None:
        return items
    return tuple(item for item in items if pattern in key(item))


def _smoke_cases() -> tuple[SmokeCase, ...]:
    return _filter_by_env_pattern(
        _all_smoke_cases(), "DRIFT_MODEL_MATRIX_SMOKE_PATTERN", lambda c: c.id
    )


def _all_image_backbone_download_cases() -> tuple[SmokeCase, ...]:
    return tuple(
        case
        for case in _all_smoke_cases()
        if isinstance(
            getattr(case.trainer_config, "architecture_specific_config", None),
            TransferLearningConfig,
        )
    )


def _image_backbone_download_cases() -> tuple[SmokeCase, ...]:
    return _filter_by_env_pattern(
        _all_image_backbone_download_cases(),
        "DRIFT_MODEL_DOWNLOAD_SMOKE_PATTERN",
        lambda c: c.id,
    )


def _all_text_backbone_download_producers() -> tuple[str, ...]:
    # "roberta-base" is also the shared sequence-cache producer for scratch text models
    # (drift_happens/pipeline/_shared/text_cache.py); included explicitly to cover that path
    # even if FROZEN_TEXT_BACKBONE_PRODUCERS ever diverges.
    return tuple(sorted({"roberta-base", *FROZEN_TEXT_BACKBONE_PRODUCERS.values()}))


def _text_backbone_download_producers() -> tuple[str, ...]:
    return _filter_by_env_pattern(
        _all_text_backbone_download_producers(),
        "DRIFT_MODEL_DOWNLOAD_SMOKE_PATTERN",
        lambda p: p,
    )


def _cases_from_configs(
    *,
    dataset: str,
    matrix: str,
    configs: dict[str, Any],
    build_trainers: TrainerBuilder,
    task: SampleTask,
) -> tuple[SmokeCase, ...]:
    return tuple(
        SmokeCase(
            dataset=dataset,
            matrix=matrix,
            trainer_key=trainer_key,
            trainer_config=trainer_config.model_copy(
                update={"batch_size": SAMPLE_COUNT, "num_epochs": 1}
            ),
            build_trainers=build_trainers,
            task=task,
        )
        for trainer_key, trainer_config in sorted(configs.items())
    )


def test_full_matrix_plan_models_are_covered_by_smoke_cases() -> None:
    smoke_case_ids = {case.id for case in _all_smoke_cases()}
    planned_case_ids = {
        _plan_job_smoke_case_id(job) for job in _full_matrix_plan_jobs()
    }

    assert planned_case_ids <= smoke_case_ids


def _plan_stages_by_name() -> dict[str, Any]:
    return {stage.name: stage for stage in build_plan_stages(materialized_presets())}


def test_full_matrix_plan_model_counts_match_expected_lineup() -> None:
    stages = _plan_stages_by_name()

    assert {
        stage_name: len({job.label for job in stages[stage_name].jobs})
        for stage_name in FULL_MATRIX_PLAN_STAGES
    } == {
        "p40_amazon_reviews_23_all_seeds": 20,
        "p50_arxiv_all_seeds": 21,
        "p60_yearbook_all_seeds": 22,
        "p70_imdb_faces_all_seeds": 21,
    }


def _plan_job_smoke_case_id(job: Any) -> str:
    config = _plan_job_payload(job)["config"]
    return f"{config['dataset']['name']}/conference/{config['trainer']['key']}"


def test_download_smoke_cases_cover_full_matrix_external_backbones() -> None:
    image_download_case_ids = {case.id for case in _all_image_backbone_download_cases()}
    text_download_producers = set(_all_text_backbone_download_producers())
    planned_image_case_ids: set[str] = set()
    planned_text_producers: set[str] = set()

    for job in _full_matrix_plan_jobs():
        payload = _plan_job_payload(job)
        dataset_name = payload["config"]["dataset"]["name"]
        tags = set(payload["tags"])
        cache = payload["config"]["preprocessing"].get("cache")
        if dataset_name in {"imdb_faces", "yearbook"} and (
            "embedding-cache" in tags
            or "frozen-pretrained" in tags
            or "transfer-learning" in tags
        ):
            planned_image_case_ids.add(_plan_job_smoke_case_id(job))
        if dataset_name in {"arxiv", "amazon_reviews_23"} and cache is not None:
            planned_text_producers.add(cache["producer"])

    assert planned_image_case_ids <= image_download_case_ids
    assert planned_text_producers <= text_download_producers


def _full_matrix_plan_jobs() -> tuple[Any, ...]:
    stages = _plan_stages_by_name()
    return tuple(
        job for stage_name in FULL_MATRIX_PLAN_STAGES for job in stages[stage_name].jobs
    )


def _plan_job_payload(job: Any) -> dict[str, Any]:
    return json.loads(job.config_path.read_text())


@_RUN_MODEL_DOWNLOAD_SMOKE
@pytest.mark.parametrize(
    "case",
    _image_backbone_download_cases(),
    ids=lambda case: case.id,
)
def test_image_backbone_from_matrix_is_downloadable(case: SmokeCase) -> None:
    trainer = case.build_trainers({case.trainer_key: case.trainer_config})[
        case.trainer_key
    ]

    assert isinstance(trainer._model, nn.Module)


@_RUN_MODEL_DOWNLOAD_SMOKE
@pytest.mark.parametrize(
    "producer",
    _text_backbone_download_producers(),
    ids=lambda producer: producer.replace("/", "__"),
)
def test_text_backbone_producer_from_matrix_is_downloadable(producer: str) -> None:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(producer)
    model = AutoModel.from_pretrained(producer)

    assert tokenizer is not None
    assert isinstance(model, nn.Module)


@_RUN_MODEL_MATRIX_SMOKE
@pytest.mark.cuda_allowed
@pytest.mark.parametrize("case", _smoke_cases(), ids=lambda case: case.id)
def test_model_dataset_matrix_trains_one_epoch(case: SmokeCase, tmp_path: Path) -> None:
    torch.manual_seed(0)

    trainer = case.build_trainers({case.trainer_key: case.trainer_config})[
        case.trainer_key
    ]
    forced_device = _smoke_device()
    if forced_device is not None:
        trainer._config = trainer._config.model_copy(update={"device": forced_device})
    dataset = _dataset_for_case(case, trainer, tmp_path)

    history = trainer.fit(dataset)

    assert len(history.train_epochs) == 1


def _smoke_device() -> str | None:
    """Return explicit device override, or None to keep the builder-resolved device."""
    raw_device = os.environ.get("DRIFT_MODEL_MATRIX_SMOKE_DEVICE", "auto")
    return None if raw_device == "auto" else raw_device


def _dataset_for_case(
    case: SmokeCase, trainer: PytorchTrainer, tmp_path: Path
) -> TensorDataset:
    if case.task == "image_classification":
        return _image_classification_dataset(case, trainer, tmp_path)
    if case.task == "arxiv_multilabel":
        return _text_dataset(
            case.trainer_config,
            labels=torch.tensor(
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
        )
    if case.task == "amazon_regression":
        return _text_dataset(
            case.trainer_config,
            labels=torch.tensor([1, 5], dtype=torch.long),
        )
    raise ValueError(f"Unknown smoke task: {case.task}")


def _image_classification_dataset(
    case: SmokeCase,
    trainer: PytorchTrainer,
    tmp_path: Path,
) -> TensorDataset:
    labels = torch.tensor([0, 1], dtype=torch.long)
    images = TensorDataset(torch.randn(SAMPLE_COUNT, 3, 32, 32), labels)
    model_config = case.trainer_config.architecture_specific_config
    if not (
        isinstance(model_config, TransferLearningConfig)
        and not model_config.needs_backend_fw_pass
    ):
        return images

    ctx = PipelineContext(
        df=pd.DataFrame({"year": [2000, 2001]}),
        tensor_dataset=images,
        dataset_splits=DatasetSplit(
            train_df=pd.DataFrame(index=[0]),
            val_df=pd.DataFrame(index=[]),
            test_df=pd.DataFrame(index=[1]),
        ),
        trainer_keys=[case.trainer_key],
        train_time_slices={},
        artifacts_dir=tmp_path,
    )
    embedded = embed_dataset_if_needed(
        ctx,
        trainer,
        case.trainer_key,
        dataset_cache_dir=tmp_path / "embedding-cache",
        dataset_id="smoke",
    )
    trainer.reset_model()
    return embedded


def _text_dataset(trainer_config: Any, *, labels: torch.Tensor) -> TensorDataset:
    architecture_name = trainer_config.architecture_name
    if architecture_name in CONFERENCE_SEQUENCE_TEXT_ARCHITECTURES:
        attention_mask = torch.ones(
            SAMPLE_COUNT, TEXT_SEQUENCE_LENGTH, dtype=torch.bool
        )
        sequence_embeddings = torch.randn(
            SAMPLE_COUNT,
            TEXT_SEQUENCE_LENGTH,
            trainer_config.feature_input_dim,
        )
        return TensorDataset(sequence_embeddings, attention_mask, labels)

    input_dim = FROZEN_TEXT_BACKBONE_DIMS[architecture_name]
    return TensorDataset(torch.randn(SAMPLE_COUNT, input_dim), labels)
