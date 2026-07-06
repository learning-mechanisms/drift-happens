"""Freeze model parameter counts into the params parquet."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
import torch.nn as nn

from drift_happens.analysis.datasets import schema
from drift_happens.analysis.datasets.locations import PARAMS_PARQUET
from drift_happens.analysis.plots.names import trainer_family

_DEFAULT_TEXT_DIM = 768
_QUIET_MODEL_LOGGERS = (
    "dinov2",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "py.warnings",
    "timm",
    "torch.hub",
)


def freeze_params(
    output: Path = PARAMS_PARQUET, *, local_files_only: bool = False
) -> Path:
    """Instantiate every conference model, count its parameters, and freeze them."""
    from drift_happens.experiments import amazon_reviews_23, arxiv, yearbook

    with _huggingface_model_loading(local_files_only=local_files_only):
        rows = (
            _text_rows("arxiv", arxiv, local_files_only=local_files_only)
            + _text_rows(
                "amazon_reviews_23",
                amazon_reviews_23,
                local_files_only=local_files_only,
            )
            + _yearbook_rows(yearbook)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    schema.check_params(pl.DataFrame(rows)).write_parquet(output)
    return output


@contextmanager
def _huggingface_model_loading(*, local_files_only: bool) -> Iterator[None]:
    """Build pretrained backbones with Hugging Face logging silenced."""
    import transformers

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved_env = {key: os.environ.get(key) for key in keys}
    saved_log = transformers.logging.get_verbosity()
    saved_logger_levels = {
        name: logging.getLogger(name).level for name in _QUIET_MODEL_LOGGERS
    }
    if local_files_only:
        os.environ.update(dict.fromkeys(keys, "1"))
    for name in _QUIET_MODEL_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers.logging.set_verbosity(saved_log)
        for name, level in saved_logger_levels.items():
            logging.getLogger(name).setLevel(level)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _text_rows(
    dataset: str, module: Any, *, local_files_only: bool
) -> list[dict[str, object]]:
    from drift_happens.model.parameters import count_parameters

    rows = []
    for trainer in _conference_trainers(module):
        counts = count_parameters(_text_model(trainer.model))
        backbone = _backbone_params(
            trainer.model.get("producer"), local_files_only=local_files_only
        )
        rows.append(_row(dataset, trainer, counts.trainable, counts.total + backbone))
    return rows


def _backbone_params(producer: str | None, *, local_files_only: bool = False) -> int:
    """Parameters of the frozen feature extractor behind a text trainer."""
    if not producer:
        return 0
    from transformers import AutoModel

    if local_files_only:
        backbone = AutoModel.from_pretrained(producer, local_files_only=True)
    else:
        try:
            backbone = AutoModel.from_pretrained(producer, local_files_only=True)
        except OSError:
            backbone = AutoModel.from_pretrained(producer, local_files_only=False)
    return sum(parameter.numel() for parameter in backbone.parameters())


def _yearbook_rows(module: Any) -> list[dict[str, object]]:
    from drift_happens.model.dataset.image.transfer_learning.base import (
        TransferLearningConfig,
    )
    from drift_happens.model.parameters import count_parameters
    from drift_happens.pipeline.image.trainers import image_model_factory
    from drift_happens.pipeline.yearbook.trainers import (
        yearbook_conference_trainer_configs,
    )

    trainers = {trainer.key: trainer for trainer in _conference_trainers(module)}
    rows = []
    for key, config in yearbook_conference_trainer_configs().items():
        model_config = config.architecture_specific_config
        if isinstance(model_config, TransferLearningConfig):
            model_config = model_config.model_copy(update={"pretrained": False})
        model = image_model_factory(model_config)
        counts = count_parameters(model)
        rows.append(_row("yearbook", trainers[key], counts.trainable, counts.total))
    return rows


def _conference_trainers(module: Any) -> list[Any]:
    return [
        preset.build().trainer
        for preset in module.presets()
        if "conference" in preset.group
    ]


def _row(dataset: str, trainer: Any, trainable: int, total: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "trainer": trainer.key,
        "trainer_family": trainer_family(dataset, trainer.key),
        "trainable": trainable,
        "total": total,
    }


def _text_model(model: dict[str, Any]) -> nn.Module:
    from drift_happens.model.dataset.text.architectures import text_model_factory
    from drift_happens.model.text.frozen_backbone import FROZEN_TEXT_BACKBONE_DIMS

    name = model["architecture"]
    dim = FROZEN_TEXT_BACKBONE_DIMS.get(name, model.get("input_dim", _DEFAULT_TEXT_DIM))
    return text_model_factory(
        architecture_name=name, dim_output=model["output_dim"], feature_input_dim=dim
    )
