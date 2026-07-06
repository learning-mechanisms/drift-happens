"""Per-model training/evaluation coordination for dataset pipeline CLIs."""

import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from drift_happens.utils.log import configure_logging, get_logger

logger = get_logger()


@dataclass(frozen=True)
class ModelFailure:
    """A single model key that failed during a per-model run."""

    key: str
    error: str


def run_per_model(
    ctx,
    keys: list[str],
    fn_single: Callable[..., None],
    n_workers: int,
    extra_args: tuple = (),
    *,
    fail_fast: bool = False,
    exit_on_failure: bool = True,
) -> list[ModelFailure]:
    """
    Coordinate per-model training or evaluation, isolating per-model failures.

    A failure in one model key is logged once its result is collected and recorded;
    the remaining models still run. When any model failed, the run is reported at the end and,
    by default, the process exits with code 2 so callers (and CI) can distinguish
    "completed with failures" from a clean run.

    Args:
        ctx: Pipeline context passed through to ``fn_single``.
        keys: Model keys to run.
        fn_single: Callable invoked as ``fn_single(ctx, key, *extra_args)``.
        n_workers: Run sequentially when < 2, otherwise across a process pool.
        extra_args: Extra positional arguments forwarded to ``fn_single``.
        fail_fast: Re-raise on the first failure instead of continuing.
        exit_on_failure: Exit with code 2 when any model failed. Set ``False`` to
            return the failures instead (used by tests and programmatic callers).

    Returns:
        The list of model failures (empty on success). Does not return when
        ``exit_on_failure`` is set and at least one model failed.
    """
    failures: list[ModelFailure] = []
    _tqdm_kwargs: dict[str, Any] = {"desc": "Models", "unit": "model", "colour": "blue"}

    def _record_failure(key: str, exc: Exception) -> None:
        # isolate per-model failures
        if fail_fast:
            raise exc
        logger.error(f"model '{key}' failed: {type(exc).__name__}: {exc}", exc_info=exc)
        failures.append(ModelFailure(key=key, error=str(exc)))

    if n_workers < 2:
        for key in tqdm(keys, **_tqdm_kwargs):
            try:
                fn_single(ctx, key, *extra_args)
            except Exception as exc:  # noqa: BLE001
                _record_failure(key, exc)
    else:
        # Spawn workers do not run the CLI's __main__ logging setup, so configure
        # logging in each at the parent's level.
        with mp.get_context("spawn").Pool(
            processes=n_workers,
            initializer=configure_logging,
            initargs=(logging.getLogger().getEffectiveLevel(),),
        ) as pool:
            async_results = [
                (key, pool.apply_async(fn_single, (ctx, key, *extra_args)))
                for key in keys
            ]
            for key, result in tqdm(
                async_results,
                total=len(keys),
                position=0,
                **_tqdm_kwargs,
            ):
                try:
                    result.get()
                except Exception as exc:  # noqa: BLE001
                    _record_failure(key, exc)

    if failures:
        logger.error(
            f"{len(failures)} of {len(keys)} model(s) failed: "
            + ", ".join(failure.key for failure in failures)
        )
        if exit_on_failure:
            raise SystemExit(2)

    return failures
