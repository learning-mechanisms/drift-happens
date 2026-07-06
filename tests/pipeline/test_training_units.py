from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Subset, TensorDataset

import drift_happens.pipeline.training as training_module
from drift_happens.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    RunIdentity,
    TrainerConfig,
)
from drift_happens.dataset.cache import (
    ChunkedTensorDataset,
    FeatureCacheManifest,
    load_feature_cache,
    write_feature_cache_manifest,
    write_tensor_chunks,
)
from drift_happens.pipeline.training import (
    _checkpoint_identity_token,
    _prepare_resumable_checkpoint,
    _slice_seed,
    _train_slice_complete,
    _trainer_config_json,
    _write_train_slice_completion,
    train_models_on_time_slices,
)
from drift_happens.runtime.metrics import MetricRecord
from drift_happens.runtime.progress import SWEEP_PROGRESS_FILE_ENV
from drift_happens.runtime.stages import WorkUnitCompletion, write_json_atomic
from drift_happens.sample.splits import DatasetSplit, DatasetTimeSplitConfig
from drift_happens.utils.env import RESUME_CHECKPOINTS_ENV
from drift_happens.utils.pytorch import seed_everything


class TinyTrainerConfig(BaseModel):
    name: str = "fake"


class TensorTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: torch.Tensor


class FakeHistory(BaseModel):
    epochs: list[int] = Field(default_factory=lambda: [1])


class FakeTrainer:
    batch_size = 2

    def __init__(self, *, save_mode: str = "exact", fail_fit: bool = False) -> None:
        self.save_mode = save_mode
        self.fail_fit = fail_fit
        self.reset_calls = 0
        self.fit_calls = 0
        self.save_calls: list[Path] = []

    def reset_model(self) -> None:
        self.reset_calls += 1

    def fit(
        self, *, train, val, train_batch_sampler_factory=None, checkpoint_dir=None
    ) -> FakeHistory:
        self.fit_calls += 1
        self.train_batch_sampler_factory = train_batch_sampler_factory
        self.checkpoint_dir = checkpoint_dir
        if self.fail_fit:
            raise RuntimeError("boom")
        self.train_len = len(train)
        self.val_len = len(val)
        return FakeHistory()

    def save_model(self, path: Path) -> None:
        self.save_calls.append(path)
        target = path if self.save_mode == "exact" else path.with_suffix(".model")
        target.write_text("model")


class RecordingMetricSink:
    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)

    def close(self, exit_code: int | None = None) -> None:
        return None


def _identity(config_hash: str = "cfg") -> RunIdentity:
    return RunIdentity(
        source_identity="src",
        config_hash=config_hash,
        snapshot_sha256="snap",
        wandb_group="group",
        wandb_run_name="run",
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        seed=7,
        dataset=DatasetConfig(name="synthetic"),
        trainer=TrainerConfig(key="fake"),
        evaluation=EvaluationConfig(metric="accuracy"),
    )


def _split() -> DatasetSplit:
    return DatasetSplit(
        train_df=pd.DataFrame({"year": [2000, 2000], "label": [0, 1]}, index=[0, 1]),
        val_df=pd.DataFrame({"year": [2000], "label": [0]}, index=[2]),
        test_df=pd.DataFrame({"year": [2000], "label": [1]}, index=[3]),
    )


def _tensor_dataset() -> TensorDataset:
    return TensorDataset(
        torch.arange(4, dtype=torch.float32).reshape(4, 1), torch.arange(4)
    )


def _slice() -> DatasetTimeSplitConfig:
    return DatasetTimeSplitConfig(
        lower_bound=2000,
        upper_bound=2001,
        lower_bound_inclusive=True,
        upper_bound_inclusive=False,
    )


def _multi_split() -> DatasetSplit:
    return DatasetSplit(
        train_df=pd.DataFrame(
            {"year": [2000, 2001, 2002], "label": [0, 1, 0]}, index=[0, 1, 2]
        ),
        val_df=pd.DataFrame({"year": [2000], "label": [0]}, index=[3]),
        test_df=pd.DataFrame({"year": [2002], "label": [1]}, index=[4]),
    )


def _multi_tensor_dataset() -> TensorDataset:
    return TensorDataset(
        torch.arange(5, dtype=torch.float32).reshape(5, 1), torch.arange(5)
    )


def _cumulative_slices() -> dict[str, DatasetTimeSplitConfig]:
    return {
        year: DatasetTimeSplitConfig(
            lower_bound=2000,
            upper_bound=upper,
            lower_bound_inclusive=True,
            upper_bound_inclusive=True,
        )
        for year, upper in (("2000", 2000), ("2001", 2001), ("2002", 2002))
    }


def test_train_models_writes_slice_artifacts_and_completion(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    sink = RecordingMetricSink()

    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
        experiment_config=_config(),
        metric_sink=sink,
    )

    slice_dir = tmp_path / "fake" / "train_slice_2000"
    assert (tmp_path / "fake" / "config.json").exists()
    assert (slice_dir / "time_slice_config.json").exists()
    assert (slice_dir / "training_history.json").exists()
    assert (slice_dir / "trained_model.pt").exists()
    assert (
        json.loads((slice_dir / "completion.json").read_text())["exit_status"] == "ok"
    )
    assert (trainer.train_len, trainer.val_len) == (2, 1)
    assert [record.metric for record in sink.records] == ["train/slice_completed"]


def test_train_models_emits_sweep_progress_events(tmp_path: Path, monkeypatch) -> None:
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setenv(SWEEP_PROGRESS_FILE_ENV, str(progress_path))

    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "artifacts",
        experiment_config=_config(),
        metric_sink=RecordingMetricSink(),
    )

    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "train_slices_started",
        "train_slice_started",
        "train_slice_finished",
    ]
    assert {event["total_slices"] for event in events} == {1}
    assert events[1]["train_slice"] == "2000"
    assert events[2]["train_slice"] == "2000"


def test_trainer_config_json_serializes_tensor_fields() -> None:
    payload = json.loads(
        _trainer_config_json(TensorTrainerConfig(weights=torch.tensor([1.0, 2.5])))
    )

    assert payload == {"weights": [1.0, 2.5]}


def test_train_models_resume_skips_completed_slice(tmp_path: Path) -> None:
    slice_dir = tmp_path / "fake" / "train_slice_2000"
    slice_dir.mkdir(parents=True)
    (slice_dir / "training_history.json").write_text("{}")
    (slice_dir / "trained_model.pt").write_text("model")
    _write_train_slice_completion(
        slice_dir, trainer_key="fake", train_slice="2000", identity=None
    )
    trainer = FakeTrainer()
    sink = RecordingMetricSink()

    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
        experiment_config=_config(),
        metric_sink=sink,
    )

    assert trainer.fit_calls == 0
    # The first attempt already wrote this slice's ledger row; the skip path
    # must not duplicate it in the append-only sink.
    assert sink.records == []


def test_train_resume_does_not_duplicate_ledger_rows(tmp_path: Path) -> None:
    sink = RecordingMetricSink()
    kwargs = dict(
        dataset_splits=_multi_split(),
        training_time_slices=_cumulative_slices(),
        trainer_key="fake",
        trainer_config=TinyTrainerConfig(),
        artifacts_dir=tmp_path,
        experiment_config=_config(),
        metric_sink=sink,
    )

    train_models_on_time_slices(
        _multi_tensor_dataset(),
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        **kwargs,
    )
    last_slice_dir = tmp_path / "fake" / "train_slice_2002"
    for name in ("trained_model.pt", "training_history.json", "completion.json"):
        (last_slice_dir / name).unlink()
    train_models_on_time_slices(
        _multi_tensor_dataset(),
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        resume=True,
        **kwargs,
    )

    # One row per slice plus exactly one more for the retrained slice.
    rows = [(record.metric, record.train_slice) for record in sink.records]
    assert sorted(rows) == [
        ("train/slice_completed", "2000"),
        ("train/slice_completed", "2001"),
        ("train/slice_completed", "2002"),
        ("train/slice_completed", "2002"),
    ]


def test_train_models_resume_retrains_when_identity_mismatches(tmp_path: Path) -> None:
    identity = _identity()
    slice_dir = tmp_path / "fake" / "train_slice_2000"
    slice_dir.mkdir(parents=True)
    (slice_dir / "training_history.json").write_text("stale")
    (slice_dir / "trained_model.pt").write_text("stale")
    write_json_atomic(
        slice_dir / "completion.json",
        WorkUnitCompletion(
            kind="train_slice",
            stage="train",
            exit_status="ok",
            source_identity=identity.source_identity,
            config_hash="wrong",
            snapshot_sha256=identity.snapshot_sha256,
            trainer_key="fake",
            train_slice="2000",
            ended_at=datetime(2026, 1, 1, tzinfo=UTC),
        ).model_dump(mode="json"),
    )
    trainer = FakeTrainer()

    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
        run_identity=identity,
    )

    completion = json.loads((slice_dir / "completion.json").read_text())
    assert trainer.fit_calls == 1
    assert completion["config_hash"] == identity.config_hash
    assert (slice_dir / "training_history.json").read_text() != "stale"


def test_train_models_raises_aggregated_error_after_failed_slice(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match=r"fake \(slice 2000\)"):
        train_models_on_time_slices(
            _tensor_dataset(),
            _split(),
            {"2000": _slice()},
            "fake",
            TinyTrainerConfig(),
            FakeTrainer(fail_fit=True),  # type: ignore[arg-type]
            artifacts_dir=tmp_path,
        )


def test_train_slice_completion_accepts_legacy_model_suffix(
    tmp_path: Path,
) -> None:
    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(save_mode="legacy_suffix"),  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
    )

    assert _train_slice_complete(
        tmp_path / "fake" / "train_slice_2000",
        trainer_key="fake",
        train_slice="2000",
        identity=None,
    )


class _RngCapturingTrainer:
    """Records the global torch RNG draw at each reset and consumes RNG when fitting."""

    def __init__(self) -> None:
        self.reset_values: list[int] = []

    def reset_model(self) -> None:
        self.reset_values.append(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    def fit(self, *, train, val, checkpoint_dir=None) -> FakeHistory:
        torch.rand(
            8
        )  # mimic training so slices diverge in RNG position without the fix
        return FakeHistory()

    def save_model(self, path: Path) -> None:
        path.write_text("model")


def test_slice_seed_is_deterministic_distinct_and_in_range() -> None:
    base = _slice_seed(7, "fake", "2000")
    assert base == _slice_seed(7, "fake", "2000")  # deterministic
    assert base != _slice_seed(7, "fake", "2001")  # distinct slice
    assert base != _slice_seed(7, "other", "2000")  # distinct trainer
    assert base != _slice_seed(8, "fake", "2000")  # distinct base seed
    assert 0 <= base < 2**32  # valid for numpy's seed range


def test_resumed_slice_init_matches_uninterrupted_run(tmp_path: Path) -> None:
    dataset, split, slices = (
        _multi_tensor_dataset(),
        _multi_split(),
        _cumulative_slices(),
    )

    seed_everything(123)
    uninterrupted = _RngCapturingTrainer()
    train_models_on_time_slices(
        dataset,
        split,
        slices,
        "fake",
        TinyTrainerConfig(),
        uninterrupted,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "a",
        experiment_config=_config(),
    )

    seed_everything(123)
    train_models_on_time_slices(
        dataset,
        split,
        slices,
        "fake",
        TinyTrainerConfig(),
        _RngCapturingTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "b",
        experiment_config=_config(),
    )
    last_slice_dir = tmp_path / "b" / "fake" / "train_slice_2002"
    for name in ("trained_model.pt", "training_history.json", "completion.json"):
        (last_slice_dir / name).unlink()

    seed_everything(123)
    resumed = _RngCapturingTrainer()
    train_models_on_time_slices(
        dataset,
        split,
        slices,
        "fake",
        TinyTrainerConfig(),
        resumed,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "b",
        experiment_config=_config(),
        resume=True,
    )

    assert len(resumed.reset_values) == 1  # only the dropped slice retrains
    assert resumed.reset_values[0] == uninterrupted.reset_values[-1]


def test_prepare_resumable_checkpoint_starts_fresh_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(RESUME_CHECKPOINTS_ENV, raising=False)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_json_atomic(
        checkpoint_dir / "identity.json", _checkpoint_identity_token(None)
    )
    (checkpoint_dir / "epoch.pt").write_text("stale")

    _prepare_resumable_checkpoint(checkpoint_dir, identity=None)

    # Epoch resume is opt-in: an unfinished slice restarts from epoch 0, so the
    # leftover checkpoint is cleared even though its identity matches.
    assert not (checkpoint_dir / "epoch.pt").exists()


def test_prepare_resumable_checkpoint_keeps_matching_when_opted_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(RESUME_CHECKPOINTS_ENV, "1")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_json_atomic(
        checkpoint_dir / "identity.json", _checkpoint_identity_token(None)
    )
    (checkpoint_dir / "epoch.pt").write_text("keep")

    _prepare_resumable_checkpoint(checkpoint_dir, identity=None)

    assert (checkpoint_dir / "epoch.pt").read_text() == "keep"


def test_prepare_resumable_checkpoint_clears_mismatch_when_opted_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(RESUME_CHECKPOINTS_ENV, "1")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_json_atomic(checkpoint_dir / "identity.json", {"config_hash": "other"})
    (checkpoint_dir / "epoch.pt").write_text("stale")

    _prepare_resumable_checkpoint(checkpoint_dir, identity=None)

    # A checkpoint from a different run identity is discarded even when opted in.
    assert not (checkpoint_dir / "epoch.pt").exists()


def test_distinct_slices_get_distinct_inits(tmp_path: Path) -> None:
    seed_everything(123)
    trainer = _RngCapturingTrainer()
    train_models_on_time_slices(
        _multi_tensor_dataset(),
        _multi_split(),
        _cumulative_slices(),
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
        experiment_config=_config(),
    )

    assert len(set(trainer.reset_values)) == 3  # each slice reseeds independently


def test_train_models_records_effective_slice_seed(tmp_path: Path) -> None:
    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
        experiment_config=_config(),
    )
    completion = json.loads(
        (tmp_path / "fake" / "train_slice_2000" / "completion.json").read_text()
    )
    assert completion["seed"] == _slice_seed(7, "fake", "2000")


def test_train_models_without_experiment_config_seeds_with_the_default(
    tmp_path: Path,
) -> None:
    # The direct module-CLI path passes no experiment config; it must still seed each slice with the
    # default base seed (0, matching ExperimentConfig.seed) rather than leaving it unseeded.
    train_models_on_time_slices(
        _tensor_dataset(),
        _split(),
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        FakeTrainer(),  # type: ignore[arg-type]
        artifacts_dir=tmp_path,
    )
    completion = json.loads(
        (tmp_path / "fake" / "train_slice_2000" / "completion.json").read_text()
    )
    assert completion["seed"] == _slice_seed(0, "fake", "2000")


def test_direct_cli_initialization_is_deterministic(tmp_path: Path) -> None:
    # Two direct-CLI runs (no experiment_config) start from DIFFERENT global RNG states; the per-slice
    # reseed with the default base seed must still make their initializations identical, while distinct
    # slices keep distinct inits.
    seed_everything(1)
    first = _RngCapturingTrainer()
    train_models_on_time_slices(
        _multi_tensor_dataset(),
        _multi_split(),
        _cumulative_slices(),
        "fake",
        TinyTrainerConfig(),
        first,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "a",
    )

    seed_everything(2)
    second = _RngCapturingTrainer()
    train_models_on_time_slices(
        _multi_tensor_dataset(),
        _multi_split(),
        _cumulative_slices(),
        "fake",
        TinyTrainerConfig(),
        second,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "b",
    )

    assert first.reset_values == second.reset_values
    assert len(set(first.reset_values)) == len(first.reset_values)


def _chunked_dataset(
    root: Path, rows: int = 5, chunk_size: int = 2
) -> ChunkedTensorDataset:
    features = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    labels = torch.arange(rows)
    chunks = write_tensor_chunks(root, (features, labels), chunk_size=chunk_size)
    manifest = FeatureCacheManifest(
        kind="pooled_embedding_dataset",
        cache_id="unit",
        dataset="unit",
        dataset_variant="unit",
        input_version="unit:v1",
        producer="unit",
        output="pooled_embedding",
        params={},
        row_count=rows,
        content_hash="rows",
        label_schema_hash="labels",
        chunks=chunks,
    )
    write_feature_cache_manifest(root, manifest)
    return load_feature_cache(root)


class DatasetCapturingTrainer(FakeTrainer):
    def fit(
        self, *, train, val, train_batch_sampler_factory=None, checkpoint_dir=None
    ) -> FakeHistory:
        self.train_dataset = train
        return super().fit(
            train=train,
            val=val,
            train_batch_sampler_factory=train_batch_sampler_factory,
            checkpoint_dir=checkpoint_dir,
        )


def test_chunked_train_slice_is_materialized_for_the_trainer(tmp_path: Path) -> None:
    dataset = _chunked_dataset(tmp_path / "cache")
    trainer = DatasetCapturingTrainer()

    train_models_on_time_slices(
        dataset,
        _multi_split(),
        {"2002": _cumulative_slices()["2002"]},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "artifacts",
        experiment_config=_config(),
    )

    assert isinstance(trainer.train_dataset, TensorDataset)
    assert len(trainer.train_dataset) == 3


def test_oversized_chunked_slice_falls_back_to_lazy_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_module, "_MATERIALIZE_SLICE_MAX_BYTES", 0)
    dataset = _chunked_dataset(tmp_path / "cache")
    trainer = DatasetCapturingTrainer()

    train_models_on_time_slices(
        dataset,
        _multi_split(),
        {"2002": _cumulative_slices()["2002"]},
        "fake",
        TinyTrainerConfig(),
        trainer,  # type: ignore[arg-type]
        artifacts_dir=tmp_path / "artifacts",
        experiment_config=_config(),
    )

    assert isinstance(trainer.train_dataset, Subset)


def test_lazy_chunked_slice_trains_chunk_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch import nn

    from drift_happens.model.trainer.pytorch import (
        PytorchTrainer,
        PytorchTrainerConfig,
    )

    # Force the lazy fallback so training takes the chunk-blocked sampler path.
    monkeypatch.setattr(training_module, "_MATERIALIZE_SLICE_MAX_BYTES", 0)
    rows, chunk_size = 9, 3
    cache = tmp_path / "cache"
    features = torch.randn(rows, 2, generator=torch.manual_seed(0))
    labels = torch.arange(rows) % 2
    chunks = write_tensor_chunks(cache, (features, labels), chunk_size=chunk_size)
    write_feature_cache_manifest(
        cache,
        FeatureCacheManifest(
            kind="sequence_embedding_dataset",
            cache_id="unit",
            dataset="arxiv",
            dataset_variant="unit",
            input_version="unit:v1",
            producer="roberta-base",
            output="last_hidden_state",
            params={},
            row_count=rows,
            content_hash="rows",
            label_schema_hash="labels",
            chunks=chunks,
        ),
    )
    dataset = load_feature_cache(cache)

    # Count chunk cache MISSES (actual disk deserializations) to confirm batches do
    # not thrash across chunks. The forced-lazy budget here is zero, so the shuffle
    # window is bounded to a single chunk and each chunk is deserialized once per epoch,
    # never per sample.
    misses: list[int] = []
    original = dataset._load_chunk

    def counting_load(chunk_index: int):
        if chunk_index not in dataset._chunk_cache:
            misses.append(chunk_index)
        return original(chunk_index)

    dataset._load_chunk = counting_load  # type: ignore[method-assign]

    num_epochs = 2

    def factory() -> nn.Module:
        torch.manual_seed(0)
        return nn.Linear(2, 2)

    trainer = PytorchTrainer(
        model_factory=factory,
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        config=PytorchTrainerConfig(num_epochs=num_epochs, batch_size=2, device="cpu"),
    )

    split = DatasetSplit(
        train_df=pd.DataFrame(
            {"year": [2000] * rows, "label": labels.tolist()}, index=list(range(rows))
        ),
        val_df=pd.DataFrame({"year": [], "label": []}),
        test_df=pd.DataFrame({"year": [], "label": []}),
    )
    train_models_on_time_slices(
        dataset,
        split,
        {"2000": _slice()},
        "fake",
        TinyTrainerConfig(),
        trainer,
        artifacts_dir=tmp_path / "artifacts",
        experiment_config=_config(),
    )

    # A zero materialize budget caps the shuffle window at one chunk, so each of the
    # three chunks is deserialized once per epoch -> no per-sample reloads.
    assert sorted(misses) == sorted(list(range(3)) * num_epochs)


def test_shuffle_window_chunks_bounded_by_memory_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The lazy shuffle window is sized by the materialize budget, not a fixed chunk
    # count, so the chunks held resident at once never exceed what an in-core slice
    # would hold, and the window adapts to the per-chunk byte size.
    dataset = _chunked_dataset(tmp_path / "cache", rows=20, chunk_size=2)
    chunk_bytes = (dataset.root / dataset.manifest.chunks[0].path).stat().st_size
    num_chunks = len(dataset.manifest.chunks)

    # A budget of three chunks pools three chunks per window.
    monkeypatch.setattr(
        training_module, "_MATERIALIZE_SLICE_MAX_BYTES", chunk_bytes * 3
    )
    assert training_module._shuffle_window_chunks(dataset) == 3

    # Below one chunk it still pools one, so the loader always makes progress.
    monkeypatch.setattr(
        training_module, "_MATERIALIZE_SLICE_MAX_BYTES", chunk_bytes - 1
    )
    assert training_module._shuffle_window_chunks(dataset) == 1

    # Far above the cache it is capped at the number of chunks present.
    monkeypatch.setattr(
        training_module, "_MATERIALIZE_SLICE_MAX_BYTES", chunk_bytes * 1000
    )
    assert training_module._shuffle_window_chunks(dataset) == num_chunks


def test_materialized_slice_matches_lazy_subset_batch_for_batch(tmp_path: Path) -> None:
    dataset = _chunked_dataset(tmp_path / "cache", rows=6, chunk_size=2)
    indices = [5, 0, 3, 1]

    def batches(source) -> list[list[torch.Tensor]]:
        loader = DataLoader(
            source,
            batch_size=3,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        )
        return [list(batch) for batch in loader]

    lazy = batches(Subset(dataset, indices))
    materialized_source = training_module._slice_dataset(dataset, indices)
    assert isinstance(materialized_source, TensorDataset)
    materialized = batches(materialized_source)

    assert len(lazy) == len(materialized)
    for lazy_batch, materialized_batch in zip(lazy, materialized, strict=True):
        for lazy_tensor, materialized_tensor in zip(
            lazy_batch, materialized_batch, strict=True
        ):
            assert lazy_tensor.dtype == materialized_tensor.dtype
            assert torch.equal(lazy_tensor, materialized_tensor)


def test_train_models_chains_the_failure_cause(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"fake \(slice 2000\)") as excinfo:
        train_models_on_time_slices(
            _tensor_dataset(),
            _split(),
            {"2000": _slice()},
            "fake",
            TinyTrainerConfig(),
            FakeTrainer(fail_fit=True),  # type: ignore[arg-type]
            artifacts_dir=tmp_path,
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "boom" in str(excinfo.value.__cause__)
