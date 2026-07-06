from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import BaseModel
from torch.utils.data import TensorDataset

from drift_happens.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainerConfig,
)
from drift_happens.dataset.cache import (
    FeatureCacheManifest,
    load_feature_cache,
    write_feature_cache_manifest,
    write_tensor_chunks,
)
from drift_happens.evaluation.metrics import ClassificationMetrics
from drift_happens.pipeline.evaluation import (
    _classification_num_classes,
    _metric_values,
    _slice_inputs_and_labels,
    _write_eval_cell_completion,
    eval_models_on_time_slices,
)
from drift_happens.runtime.metrics import MetricRecord
from drift_happens.runtime.progress import SWEEP_PROGRESS_FILE_ENV
from drift_happens.sample.splits import DatasetSplit, DatasetTimeSplitConfig


class TinyTrainerConfig(BaseModel):
    name: str = "fake"


class FakeTrainer:
    task_type = "classification"

    def __init__(self, *, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.load_calls = 0
        self.predict_calls = 0
        self.proba_calls = 0

    def load_model(self, path: Path) -> None:
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError("load failed")

    def predict_proba(self, X) -> torch.Tensor:
        self.proba_calls += 1
        n = X[0].shape[0] if isinstance(X, tuple) else X.shape[0]
        labels = torch.arange(n) % 2
        return torch.stack(
            [1.0 - 0.7 * labels.float(), 0.3 + 0.5 * labels.float()], dim=1
        )

    def predict(self, X) -> torch.Tensor:
        self.predict_calls += 1
        n = X[0].shape[0] if isinstance(X, tuple) else X.shape[0]
        return torch.arange(n) % 2


class DeletingOutputTrainer(FakeTrainer):
    def __init__(self, delete_dir: Path) -> None:
        super().__init__()
        self.delete_dir = delete_dir
        self.deleted = False

    def predict_proba(self, X) -> torch.Tensor:
        probs = super().predict_proba(X)
        if not self.deleted:
            shutil.rmtree(self.delete_dir)
            self.deleted = True
        return probs


class MixedMetrics:
    def scalar_metrics(self) -> dict[str, float]:
        return {"good": 1.0, "loss": 0.25}


class RecordingMetricSink:
    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)

    def close(self, exit_code: int | None = None) -> None:
        return None


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        seed=7,
        dataset=DatasetConfig(name="synthetic"),
        trainer=TrainerConfig(key="fake"),
        evaluation=EvaluationConfig(metric="accuracy"),
    )


def _slice() -> DatasetTimeSplitConfig:
    return DatasetTimeSplitConfig(
        lower_bound=2000,
        upper_bound=2001,
        lower_bound_inclusive=True,
        upper_bound_inclusive=False,
    )


def _empty_slice() -> DatasetTimeSplitConfig:
    # A slice with no matching rows in _split() / _tensor_dataset().
    return DatasetTimeSplitConfig(
        lower_bound=2010,
        upper_bound=2011,
        lower_bound_inclusive=True,
        upper_bound_inclusive=False,
    )


def _split() -> DatasetSplit:
    return DatasetSplit(
        train_df=pd.DataFrame({"year": [2000, 2000]}, index=[0, 1]),
        val_df=pd.DataFrame({"year": [2000]}, index=[2]),
        test_df=pd.DataFrame({"year": [2000]}, index=[3]),
    )


def _tensor_dataset() -> TensorDataset:
    x = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    y = torch.tensor([0, 1, 0, 1])
    return TensorDataset(x, y)


def test_slice_inputs_and_labels_handles_tensor_dataset_multi_input() -> None:
    dataset = TensorDataset(
        torch.arange(6).reshape(3, 2), torch.ones(3, 2), torch.arange(3)
    )

    inputs, labels = _slice_inputs_and_labels(dataset, [0, 2])

    assert isinstance(inputs, tuple)
    assert len(inputs) == 2
    torch.testing.assert_close(inputs[0], torch.tensor([[0, 1], [4, 5]]))
    torch.testing.assert_close(labels, torch.tensor([0, 2]))


def test_slice_inputs_and_labels_rejects_non_tensor_dataset() -> None:
    with pytest.raises(TypeError, match="TensorDataset-like"):
        _slice_inputs_and_labels(object(), [0, 1])


def test_slice_inputs_and_labels_handles_chunked_tensor_dataset(
    tmp_path: Path,
) -> None:
    embeddings = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    labels = torch.arange(6)
    chunks = write_tensor_chunks(tmp_path, (embeddings, labels), chunk_size=2)
    write_feature_cache_manifest(
        tmp_path,
        FeatureCacheManifest(
            kind="pooled_embedding_dataset",
            cache_id="unit",
            dataset="arxiv",
            dataset_variant="unit",
            input_version="unit:v1",
            producer="roberta-base",
            output="pooled_embedding",
            params={},
            row_count=6,
            content_hash="rows",
            label_schema_hash="labels",
            chunks=chunks,
        ),
    )
    dataset = load_feature_cache(tmp_path)

    inputs, sliced_labels = _slice_inputs_and_labels(dataset, [4, 1])

    assert isinstance(inputs, torch.Tensor)
    torch.testing.assert_close(inputs, embeddings[[4, 1]])
    torch.testing.assert_close(sliced_labels, labels[[4, 1]])


def test_eval_models_writes_prediction_cell_metrics_and_completion(
    tmp_path: Path,
) -> None:
    trainer = FakeTrainer()
    sink = RecordingMetricSink()

    eval_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
        experiment_config=_config(),
        metric_sink=sink,
        save_predictions=True,
    )

    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"
    assert (slice_dir / "eval_slice=2000.json").exists()
    assert (slice_dir / "eval_slice=2000.completion.json").exists()
    assert (slice_dir / "evaluation_results_on_all_slices.json").exists()
    assert (slice_dir / "predictions" / "eval_slice_2000" / "predictions.pt").exists()
    assert [record.metric for record in sink.records] == [
        "eval/precision",
        "eval/recall",
        "eval/f1_score",
        "eval/accuracy",
        "eval/balanced_accuracy",
        "eval/precision_balanced",
        "eval/recall_balanced",
        "eval/f1_score_macro",
        "eval/loss",
    ]


def test_eval_models_skips_prediction_dump_by_default(tmp_path: Path) -> None:
    eval_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
    )

    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"
    # Scalar results still land; the heavy per-row predictions.pt does not.
    assert (slice_dir / "eval_slice=2000.json").exists()
    assert not (slice_dir / "predictions").exists()


def test_eval_models_emits_sweep_progress_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setenv(SWEEP_PROGRESS_FILE_ENV, str(progress_path))

    eval_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice(), "2010": _empty_slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
    )

    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "eval_cells_started",
        "eval_train_slice_started",
        "eval_cell_started",
        "eval_cell_finished",
        "eval_cell_started",
        "eval_cell_finished",
    ]
    assert {event["total_cells"] for event in events} == {2}
    assert events[2]["train_slice"] == "2000"
    assert events[2]["eval_slice"] == "2000"
    assert events[4]["train_slice"] == "2000"
    assert events[4]["eval_slice"] == "2010"


def test_eval_models_recreates_missing_output_dir_before_writing(
    tmp_path: Path,
) -> None:
    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"

    eval_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        DeletingOutputTrainer(slice_dir),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
    )

    assert (slice_dir / "eval_slice=2000.json").exists()
    assert (slice_dir / "eval_slice=2000.completion.json").exists()
    assert (slice_dir / "evaluation_results_on_all_slices.json").exists()


def test_eval_models_resume_loads_existing_cell_without_predicting(
    tmp_path: Path,
) -> None:
    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"
    slice_dir.mkdir(parents=True)
    metrics = ClassificationMetrics.from_predictions(
        np.array([0, 1]), np.array([0, 1]), num_classes=2
    )
    (slice_dir / "eval_slice=2000.json").write_text(metrics.model_dump_json())
    _write_eval_cell_completion(
        slice_dir / "eval_slice=2000.completion.json",
        trainer_key="fake",
        train_slice="2000",
        eval_slice="2000",
        identity=None,
        metrics=metrics,
    )
    trainer = FakeTrainer()

    eval_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
    )

    assert trainer.load_calls == 1
    assert trainer.predict_calls == 0
    assert trainer.proba_calls == 0
    assert (
        "2000"
        in json.loads(
            (slice_dir / "evaluation_results_on_all_slices.json").read_text()
        )["results"]
    )


def test_eval_resume_does_not_duplicate_ledger_rows(tmp_path: Path) -> None:
    sink = RecordingMetricSink()
    kwargs = dict(
        dataset_splits=_split(),
        training_time_slices={"2000": _slice()},
        eval_time_slices={"2000": _slice()},
        trainer_key="fake",
        trainer_config=TinyTrainerConfig(),
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
        experiment_config=_config(),
        metric_sink=sink,
    )

    eval_models_on_time_slices(
        _tensor_dataset(),
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        **kwargs,
    )
    eval_models_on_time_slices(
        _tensor_dataset(),
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        **kwargs,
    )

    rows = [
        (record.metric, record.train_slice, record.eval_slice)
        for record in sink.records
    ]
    assert len(rows) == len(set(rows))


def test_resumed_run_restores_class_count_for_a_later_empty_slice(
    tmp_path: Path,
) -> None:
    empty_slice = _empty_slice()

    def run(root: Path, eval_time_slices: dict) -> None:
        eval_models_on_time_slices(
            _tensor_dataset(),
            _split(),
            {"2000": _slice()},
            eval_time_slices,
            "fake",
            TinyTrainerConfig(),
            FakeTrainer(),  # type: ignore[arg-type]
            artifacts_dir=root / "eval",
            model_artifacts_dir=root / "train",
            experiment_config=_config(),
        )

    # Uninterrupted run over both slices as the reference.
    run(tmp_path / "full", {"2000": _slice(), "2010": empty_slice})
    # Interrupted run: the non-empty cell completes, then the empty cell is
    # evaluated only after a resume that skips the completed cell.
    run(tmp_path / "resumed", {"2000": _slice()})
    run(tmp_path / "resumed", {"2000": _slice(), "2010": empty_slice})

    def empty_cell_matrix(root: Path) -> list:
        cell = root / "eval" / "fake" / "train_slice_2000" / "eval_slice=2010.json"
        return json.loads(cell.read_text())["confusion_matrix"]

    reference = empty_cell_matrix(tmp_path / "full")
    assert len(reference) == 2  # class count remembered from the 2000 cell
    assert empty_cell_matrix(tmp_path / "resumed") == reference


def test_eval_models_raises_aggregated_error_when_model_load_fails(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match=r"fake \(train slice 2000\)"):
        eval_models_on_time_slices(
            _tensor_dataset(),
            _split(),
            {"2000": _slice()},
            {"2000": _slice()},
            "fake",
            TinyTrainerConfig(),
            FakeTrainer(fail_load=True),  # type: ignore[arg-type]
            artifacts_dir=tmp_path / "eval",
            model_artifacts_dir=tmp_path / "train",
        )


def test_metric_values_delegates_to_scalar_metrics() -> None:
    assert _metric_values(MixedMetrics()) == {"good": 1.0, "loss": 0.25}  # type: ignore[arg-type]


def test_classification_num_classes_infers_from_softmax_width() -> None:
    # Binary and multi-class are both inferred from the probability width, so the
    # eval pipeline no longer hard-codes num_classes=2.
    assert _classification_num_classes(torch.zeros(4, 2)) == 2
    assert _classification_num_classes(torch.zeros(4, 5)) == 5


def test_classification_num_classes_none_for_empty_slice() -> None:
    assert _classification_num_classes(torch.tensor([])) is None


class FakeMultilabelTrainer:
    task_type = "multilabel"

    def load_model(self, path: Path) -> None:
        pass

    def predict_proba(self, X) -> torch.Tensor:
        n = X[0].shape[0] if isinstance(X, tuple) else X.shape[0]
        return torch.full((n, 2), 0.6)

    def find_optimal_threshold(self, probs, y) -> np.ndarray:
        return np.array([0.5, 0.5])


def _multilabel_dataset() -> TensorDataset:
    x = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return TensorDataset(x, y)


def test_multilabel_eval_survives_an_empty_eval_slice(tmp_path: Path) -> None:
    empty_slice = _empty_slice()

    eval_models_on_time_slices(
        _multilabel_dataset(),
        _split(),
        {"2000": _slice()},
        {"2000": _slice(), "2010": empty_slice},
        "fake",
        TinyTrainerConfig(),
        FakeMultilabelTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
        experiment_config=_config(),
    )

    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"
    assert (slice_dir / "eval_slice=2000.json").exists()
    assert (slice_dir / "eval_slice=2010.json").exists()
    assert (slice_dir / "evaluation_results_on_all_slices.json").exists()


def test_multilabel_eval_resumes_over_an_empty_cell(tmp_path: Path) -> None:
    empty_slice = _empty_slice()
    kwargs = dict(
        dataset_splits=_split(),
        training_time_slices={"2000": _slice()},
        eval_time_slices={"2000": _slice(), "2010": empty_slice},
        trainer_key="fake",
        trainer_config=TinyTrainerConfig(),
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
        experiment_config=_config(),
    )

    eval_models_on_time_slices(
        _multilabel_dataset(),
        trainer=FakeMultilabelTrainer(),  # type: ignore[arg-type]
        **kwargs,
    )
    sink = RecordingMetricSink()
    eval_models_on_time_slices(
        _multilabel_dataset(),
        trainer=FakeMultilabelTrainer(),  # type: ignore[arg-type]
        metric_sink=sink,
        **kwargs,
    )

    # Every cell resumed from disk: the append-only ledger already holds their
    # rows, so the second run must not log them again.
    assert sink.records == []
    slice_dir = tmp_path / "eval" / "fake" / "train_slice_2000"
    assert set(
        json.loads((slice_dir / "evaluation_results_on_all_slices.json").read_text())[
            "results"
        ]
    ) == {"2000", "2010"}


def _boom_log(*args, **kwargs) -> None:
    raise RuntimeError("sink down")


def test_eval_resume_relogs_a_cell_whose_marker_was_not_reached(
    tmp_path: Path, monkeypatch
) -> None:
    import drift_happens.pipeline.evaluation as ev

    kwargs = dict(
        dataset_splits=_split(),
        training_time_slices={"2000": _slice()},
        eval_time_slices={"2000": _slice()},
        trainer_key="fake",
        trainer_config=TinyTrainerConfig(),
        artifacts_dir=tmp_path / "eval",
        model_artifacts_dir=tmp_path / "train",
        experiment_config=_config(),
    )

    # First attempt: logging fails after the cell JSON but before the marker.
    monkeypatch.setattr(ev, "_log_eval_cell", _boom_log)
    with pytest.raises(RuntimeError):
        eval_models_on_time_slices(
            _tensor_dataset(),
            trainer=FakeTrainer(),  # type: ignore[arg-type]
            metric_sink=RecordingMetricSink(),
            **kwargs,
        )

    # Resume with logging restored: the cell must be recomputed and re-logged,
    # not skipped on a marker that was never written.
    monkeypatch.undo()
    sink = RecordingMetricSink()
    trainer = FakeTrainer()
    eval_models_on_time_slices(
        _tensor_dataset(),
        trainer=trainer,  # type: ignore[arg-type]
        metric_sink=sink,
        **kwargs,
    )

    assert trainer.proba_calls > 0
    assert sink.records


def test_eval_models_chains_the_failure_cause(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"train slice 2000") as excinfo:
        eval_models_on_time_slices(
            _tensor_dataset(),
            _split(),
            {"2000": _slice()},
            {"2000": _slice()},
            "fake",
            TinyTrainerConfig(),
            FakeTrainer(fail_load=True),  # type: ignore[arg-type]
            artifacts_dir=tmp_path / "eval",
            model_artifacts_dir=tmp_path / "train",
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "load failed" in str(excinfo.value.__cause__)


def test_trainer_evaluation_results_keys_are_strings() -> None:
    from pydantic import ValidationError

    from drift_happens.pipeline.models import TrainerEvaluationResults

    metrics = ClassificationMetrics.from_predictions(
        np.array([0, 1]), np.array([0, 1]), num_classes=2
    )
    # JSON object keys are always strings, so the contract is str-keyed; a
    # non-string key must be rejected rather than silently advertised.
    with pytest.raises(ValidationError):
        TrainerEvaluationResults(results={2015: metrics})

    ter = TrainerEvaluationResults(results={"2015": metrics})
    reloaded = TrainerEvaluationResults.model_validate_json(ter.model_dump_json())
    assert list(reloaded.results) == ["2015"]
