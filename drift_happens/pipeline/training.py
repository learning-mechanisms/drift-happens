"""Train model grids across temporal slices and persist slice artifacts."""

import gc
import hashlib
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from torch.utils.data import Dataset, Subset, TensorDataset
from tqdm import tqdm

from drift_happens.configs import ExperimentConfig, RunIdentity
from drift_happens.dataset.cache import ChunkedTensorDataset
from drift_happens.dataset.chunk_sampler import ChunkBlockedBatchSampler
from drift_happens.model.trainer.pytorch import PytorchTrainer
from drift_happens.runtime.metrics import MetricRecord, MetricSink
from drift_happens.runtime.progress import (
    sweep_progress_requested,
    write_sweep_progress_event,
)
from drift_happens.runtime.stages import (
    WorkUnitCompletion,
    read_json_object,
    work_unit_completion_matches,
    write_json_atomic,
)
from drift_happens.sample.splits import (
    DatasetSplit,
    DatasetTimeSplitConfig,
)
from drift_happens.sample.time.filter import filter_dataset_by_time_slice
from drift_happens.utils.env import resume_checkpoints_enabled
from drift_happens.utils.log import get_logger
from drift_happens.utils.pytorch import seed_everything

logger = get_logger()

# A direct module-CLI run carries no experiment config; seed its per-slice initialization with this
# base seed (matching ExperimentConfig.seed's default) so the unmanaged entry point is deterministic
# and reproduces an orchestrator run at the default seed instead of drifting with the global RNG.
_DEFAULT_DIRECT_CLI_SEED = 0

# A shuffled loader over a lazy Subset of a chunked cache re-deserializes a whole
# chunk per sample, so slices within this budget are materialized once. The
# sequence-embedding caches exceed node RAM at production scale and stay lazy.
_MATERIALIZE_SLICE_MAX_BYTES = 8 * 1024**3


def _slice_dataset(tensor_dataset: Any, indices: list[int]) -> Dataset:
    """
    Per-slice dataset view, materialized into memory when that fits.

    The materialized ``TensorDataset`` is row-for-row identical to
    ``Subset(tensor_dataset, indices)`` under any sampler.
    """
    if indices and isinstance(tensor_dataset, ChunkedTensorDataset):
        nbytes = _estimated_rows_nbytes(tensor_dataset, len(indices))
        if nbytes <= _MATERIALIZE_SLICE_MAX_BYTES:
            return TensorDataset(*tensor_dataset.gather(indices))
        logger.info(
            f"Slice of {len(indices)} cached rows exceeds the in-memory budget; "
            "falling back to lazy chunked reads."
        )
    return Subset(tensor_dataset, indices)


def _chunk_blocked_sampler_factory(
    train_dataset: Subset,
    *,
    seed: int,
    batch_size: int,
) -> Callable[[int], ChunkBlockedBatchSampler]:
    """
    Per-epoch chunk-blocked batch sampler factory for a lazy chunked slice.

    Each epoch's sampler is seeded by ``seed + epoch`` to mirror the global shuffle
    loader's generator, so the chunk-blocked shuffle is deterministic and a resumed
    slice reproduces an uninterrupted run.
    """
    underlying = train_dataset.dataset
    assert isinstance(underlying, ChunkedTensorDataset)
    shuffle_window = _shuffle_window_chunks(underlying)
    underlying.ensure_chunk_cache_size(shuffle_window)
    row_indices = list(train_dataset.indices)

    def factory(epoch: int) -> ChunkBlockedBatchSampler:
        return ChunkBlockedBatchSampler(
            underlying,
            row_indices,
            batch_size=batch_size,
            drop_last=False,
            seed=seed + epoch,
            shuffle_window=shuffle_window,
        )

    return factory


def _estimated_rows_nbytes(dataset: ChunkedTensorDataset, row_count: int) -> int:
    """
    Estimate the bytes needed to materialize ``row_count`` cache rows.

    Uses the on-disk size of the first (full-sized) chunk as the per-row rate;
    serialization framing makes this a slight over-estimate, which only errs toward the
    lazy fallback.
    """
    chunk = dataset.manifest.chunks[0]
    file_nbytes = (dataset.root / chunk.path).stat().st_size
    return (file_nbytes * row_count) // chunk.length


def _shuffle_window_chunks(dataset: ChunkedTensorDataset) -> int:
    """
    How many cache chunks the lazy path pools (and keeps resident) per shuffle window.

    A window's chunks are all held in memory at once, so bound the window by the same
    in-memory budget the materialize guard uses: the lazy path never keeps more than
    ``_MATERIALIZE_SLICE_MAX_BYTES`` of chunks resident, the footprint we already accept
    for an in-core slice. Deriving the count from the per-chunk size (rather than a fixed
    number) keeps that footprint bounded as the per-chunk bytes vary with sequence length
    and hidden size. At least one chunk is always pooled, and never more than the cache
    can hold.
    """
    per_chunk_bytes = max(
        1, _estimated_rows_nbytes(dataset, dataset.manifest.chunks[0].length)
    )
    window = _MATERIALIZE_SLICE_MAX_BYTES // per_chunk_bytes
    return max(1, min(len(dataset.manifest.chunks), window))


def train_models_on_time_slices(
    tensor_dataset: Any,
    dataset_splits: DatasetSplit,
    training_time_slices: dict[Any, DatasetTimeSplitConfig],
    trainer_key: str,
    trainer_config: BaseModel,
    trainer: PytorchTrainer,
    time_col: str = "year",
    *,
    artifacts_dir: Path,
    tqdm_start_pos: int = 1,
    resume: bool = True,
    run_identity: RunIdentity | None = None,
    experiment_config: ExperimentConfig | None = None,
    metric_sink: MetricSink | None = None,
) -> None:
    """Train a subset of models on multiple time slices."""
    logger.info(f"Starting training on time slices for model '{trainer_key}'")

    config_dir = artifacts_dir / trainer_key
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(_trainer_config_json(trainer_config))

    # List of model keys that failed at any time slice
    error_keys: list[str] = []
    error_causes: list[BaseException] = []

    # ------------------- Inner loop: train on all time slices ------------------- #
    slice_iter = training_time_slices.items()
    total_slices = len(training_time_slices)
    write_sweep_progress_event(
        "train_slices_started",
        trainer_key=trainer_key,
        total_slices=total_slices,
    )
    for train_slice_key, train_slice_config in tqdm(
        slice_iter,
        desc=f"Time slices for model '{trainer_key}'",
        unit="slice",
        leave=False,
        position=tqdm_start_pos + 1,
        colour="blue",
        disable=sweep_progress_requested(),
    ):
        logger.info(
            f"\nTraining model '{trainer_key}' on time slice '{train_slice_key}'..."
        )
        write_sweep_progress_event(
            "train_slice_started",
            trainer_key=trainer_key,
            train_slice=str(train_slice_key),
            total_slices=total_slices,
        )
        try:
            # ------------------------- Save slice config ------------------------ #
            slice_dir = config_dir / f"train_slice_{train_slice_key}"
            slice_dir.mkdir(parents=True, exist_ok=True)
            (slice_dir / "time_slice_config.json").write_text(
                train_slice_config.model_dump_json()
            )

            if resume and _train_slice_complete(
                slice_dir,
                trainer_key=trainer_key,
                train_slice=train_slice_key,
                identity=run_identity,
            ):
                logger.info(
                    f"\n[SKIP] Model '{trainer_key}', time slice '{train_slice_key}' already trained. Skipping...\n"
                )
                write_sweep_progress_event(
                    "train_slice_skipped",
                    trainer_key=trainer_key,
                    train_slice=str(train_slice_key),
                    total_slices=total_slices,
                )
                # The append-only ledger already holds this slice's row from the
                # attempt that trained it; logging again would duplicate it.
                continue
            _clear_partial_train_slice(slice_dir)

            base_seed = (
                experiment_config.seed
                if experiment_config is not None
                else _DEFAULT_DIRECT_CLI_SEED
            )
            slice_seed = _slice_seed(base_seed, trainer_key, train_slice_key)

            # ------------------------- Prepare datasets ------------------------- #

            df_train = filter_dataset_by_time_slice(
                df=dataset_splits.train_df,
                time_col=time_col,
                split_config=train_slice_config,
            )
            df_val = filter_dataset_by_time_slice(
                df=dataset_splits.val_df,
                time_col=time_col,
                split_config=train_slice_config,
            )
            df_test = filter_dataset_by_time_slice(
                df=dataset_splits.test_df,
                time_col=time_col,
                split_config=train_slice_config,
            )
            if not df_train.index.intersection(df_val.index.union(df_test.index)).empty:
                raise RuntimeError("Train and held-out datasets overlap!")

            train_dataset = _slice_dataset(tensor_dataset, df_train.index.tolist())
            val_dataset = (
                _slice_dataset(tensor_dataset, df_val.index.tolist())
                if len(df_val) > 0
                else None
            )

            # ------------------------------- Train ------------------------------ #
            gc.collect()
            # Reseed per slice so a slice's initialization is independent of how much
            # RNG the earlier slices consumed. A resumed run skips completed slices
            # without drawing RNG, so without this it would reach this slice at a
            # different global-RNG position than an uninterrupted run and produce
            # different weights at the same configured seed. This governs the model
            # init only while the trainer config leaves its own seed unset (every
            # production builder does); a config seed would make reset_model reseed
            # to that constant and override the per-slice seed.
            seed_everything(slice_seed)
            trainer.reset_model()
            # Only the lazy chunked path needs (and supports) the chunk-blocked
            # sampler; the materialized/pooled path trains with plain global
            # shuffling, so leave fit's default arguments untouched there.
            fit_kwargs: dict[str, Any] = {}
            if isinstance(train_dataset, Subset) and isinstance(
                train_dataset.dataset, ChunkedTensorDataset
            ):
                fit_kwargs["train_batch_sampler_factory"] = (
                    _chunk_blocked_sampler_factory(
                        train_dataset,
                        seed=slice_seed,
                        batch_size=trainer.batch_size,
                    )
                )
            checkpoint_dir = slice_dir / "checkpoints"
            _prepare_resumable_checkpoint(checkpoint_dir, identity=run_identity)
            training_history = trainer.fit(
                train=train_dataset,
                val=val_dataset,
                checkpoint_dir=checkpoint_dir,
                **fit_kwargs,
            )
            gc.collect()

            # --------------------------- Save history --------------------------- #
            history_path = slice_dir / "training_history.json"
            history_path.write_text(training_history.model_dump_json())

            trainer.save_model(slice_dir / "trained_model.pt")
            _write_train_slice_completion(
                slice_dir,
                trainer_key=trainer_key,
                train_slice=train_slice_key,
                identity=run_identity,
                seed=slice_seed,
            )
            _log_slice_completed(
                metric_sink,
                experiment_config,
                train_slice=train_slice_key,
            )
            write_sweep_progress_event(
                "train_slice_finished",
                trainer_key=trainer_key,
                train_slice=str(train_slice_key),
                total_slices=total_slices,
            )
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        except Exception as e:
            write_sweep_progress_event(
                "train_slice_failed",
                trainer_key=trainer_key,
                train_slice=str(train_slice_key),
                total_slices=total_slices,
                error=f"{type(e).__name__}: {e}",
            )
            logger.error(
                f"failure in model '{trainer_key}', time slice '{train_slice_key}'",
                exc_info=e,
            )
            error_keys.append(f"{trainer_key} (slice {train_slice_key})")
            error_causes.append(e)

    # ------------------------------------ Summary ----------------------------------- #
    if len(error_keys) > 0:
        logger.info("=" * 23)
        logger.info("  FAILED CONFIGURATIONS")
        logger.info("=" * 23 + "\n")
        for k in error_keys:
            logger.info(f" - {k}")
        logger.info(f"\nTotal failed: {len(error_keys)}")
        raise RuntimeError(
            "training failed for "
            + ", ".join(error_keys[:10])
            + ("..." if len(error_keys) > 10 else "")
        ) from error_causes[0]
    else:
        logger.info("\nAll configurations completed successfully!\n")


def _slice_seed(base_seed: int, trainer_key: str, train_slice_key: object) -> int:
    """
    Derive a stable per-slice seed in numpy's valid range from the run identity.

    Distinct ``(base_seed, trainer_key, train_slice_key)`` triples yield distinct seeds,
    so each slice initializes independently of the others; the same triple always yields
    the same seed, so a resumed slice matches its uninterrupted run.
    """
    digest = hashlib.sha256(
        f"{base_seed}:{trainer_key}:{train_slice_key}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _train_slice_complete(
    slice_dir: Path,
    *,
    trainer_key: str,
    train_slice: object,
    identity: RunIdentity | None,
) -> bool:
    if not (slice_dir / "training_history.json").exists():
        return False
    if not _trained_model_artifact_exists(slice_dir):
        return False
    completion = read_json_object(slice_dir / "completion.json")
    if not completion:
        return identity is None
    return work_unit_completion_matches(
        completion,
        identity=identity,
        trainer_key=trainer_key,
        train_slice=train_slice,
    )


def _trainer_config_json(trainer_config: BaseModel) -> str:
    """Return a JSON artifact for trainer configs with runtime tensor fields."""
    return trainer_config.model_dump_json(fallback=_trainer_config_json_fallback)


def _trainer_config_json_fallback(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"trainer config field is not JSON serializable: {type(value)}")


def _clear_partial_train_slice(slice_dir: Path) -> None:
    for filename in (
        "training_history.json",
        "trained_model.pt",
        "trained_model.pth",
        "trained_model.model",
        "completion.json",
    ):
        (slice_dir / filename).unlink(missing_ok=True)


def _prepare_resumable_checkpoint(
    checkpoint_dir: Path, *, identity: RunIdentity | None
) -> None:
    """
    Prepare the per-epoch checkpoint dir for a slice that is about to train.

    Epoch-level resume is opt-in (``--resume-checkpoints`` /
    ``DRIFT_RESUME_CHECKPOINTS``). When disabled (the default) the slice always starts
    fresh, so any leftover checkpoint is cleared. When enabled, a checkpoint is kept
    only if it was left by this run identity, so a resumed slice continues from the
    matching attempt rather than a stale one.
    """
    token = _checkpoint_identity_token(identity)
    keep = (
        resume_checkpoints_enabled()
        and read_json_object(checkpoint_dir / "identity.json") == token
    )
    if not keep:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(checkpoint_dir / "identity.json", token)


def _checkpoint_identity_token(identity: RunIdentity | None) -> dict[str, str | None]:
    return {
        "source_identity": identity.source_identity if identity else None,
        "config_hash": identity.config_hash if identity else None,
        "snapshot_sha256": identity.snapshot_sha256 if identity else None,
    }


def _trained_model_artifact_exists(slice_dir: Path) -> bool:
    return any(
        (slice_dir / f"trained_model{suffix}").exists()
        for suffix in (".pt", ".pth", ".model")
    )


def _write_train_slice_completion(
    slice_dir: Path,
    *,
    trainer_key: str,
    train_slice: object,
    identity: RunIdentity | None,
    seed: int | None = None,
) -> None:
    completion = WorkUnitCompletion(
        kind="train_slice",
        stage="train",
        exit_status="ok",
        seed=seed,
        source_identity=identity.source_identity if identity else None,
        config_hash=identity.config_hash if identity else None,
        snapshot_sha256=identity.snapshot_sha256 if identity else None,
        trainer_key=trainer_key,
        train_slice=str(train_slice),
        ended_at=datetime.now(UTC),
    )
    write_json_atomic(slice_dir / "completion.json", completion.model_dump(mode="json"))


def _log_slice_completed(
    metric_sink: MetricSink | None,
    cfg: ExperimentConfig | None,
    *,
    train_slice: object,
) -> None:
    if metric_sink is None or cfg is None:
        return
    metric_sink.log(
        MetricRecord.from_config(
            cfg,
            phase="train",
            metric="train/slice_completed",
            value=1.0,
            train_slice=str(train_slice),
        )
    )
