from pathlib import Path

import polars as pl

from drift_happens.dataset.imdb_faces.const import (
    IMDB_PREPROCESSED_DIR,
)
from drift_happens.utils.log import get_logger

logger = get_logger()


def get_cache_path(target_size: int = 32) -> Path:
    return IMDB_PREPROCESSED_DIR / f"imdb_processed{target_size}x{target_size}.parquet"


def write_preprocessed_df_to_cache(
    df: pl.DataFrame,
    target_size: int = 32,
) -> None:
    cache_path = get_cache_path(target_size=target_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so an interrupted save never leaves a truncated file at a
    # path a later run would happily load.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(cache_path)
    logger.info(f"Wrote preprocessed IMDB dataset to {cache_path}")


def load_preprocessed_df(target_size: int = 32) -> pl.DataFrame:
    cache_path = get_cache_path(target_size=target_size)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Preprocessed cache not found: {cache_path}. "
            "Run `imdb-faces full` (or `prepare`) to build it first."
        )
    df = pl.read_parquet(cache_path)
    logger.info(f"Loaded preprocessed IMDB dataset from {cache_path}")
    return df
