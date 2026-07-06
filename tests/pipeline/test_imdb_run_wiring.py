from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from drift_happens.pipeline.imdb_faces import run


def _stub_context() -> SimpleNamespace:
    return SimpleNamespace(
        trainer_configs={"mlp_s": object()},
        dataset_splits=object(),
        train_time_slices={},
        artifacts_dir=Path("unused"),
    )


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr(
        run,
        "build_trainers_from_configs",
        lambda configs, device=None: {"mlp_s": object()},
    )
    monkeypatch.setattr(run, "embed_dataset_if_needed", lambda *a, **kw: object())


def test_train_single_model_passes_photo_taken_time_col(monkeypatch) -> None:
    recorded: dict = {}
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        run, "train_models_on_time_slices", lambda **kwargs: recorded.update(kwargs)
    )

    run.train_single_model(_stub_context(), "mlp_s")

    assert recorded["time_col"] == "photo_taken"


def test_eval_single_model_passes_photo_taken_time_col(monkeypatch) -> None:
    recorded: dict = {}
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        run, "eval_models_on_time_slices", lambda **kwargs: recorded.update(kwargs)
    )

    run.eval_single_model(_stub_context(), "mlp_s", eval_time_slices={})

    assert recorded["time_col"] == "photo_taken"
