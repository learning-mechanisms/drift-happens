"""The IMDB preprocessed-cache path must encode its target size so different sizes do
not clobber one another (mirrors the yearbook downscaled-cache naming)."""

from pathlib import Path

import polars as pl
import pytest

from drift_happens.dataset.imdb_faces import load
from drift_happens.dataset.imdb_faces.load import (
    get_cache_path,
    write_preprocessed_df_to_cache,
)


def test_cache_path_encodes_the_target_size() -> None:
    assert get_cache_path(32).name == "imdb_processed32x32.parquet"
    assert get_cache_path(64).name == "imdb_processed64x64.parquet"


def test_cache_path_defaults_to_32() -> None:
    assert get_cache_path().name == "imdb_processed32x32.parquet"


def test_distinct_sizes_use_distinct_files() -> None:
    assert get_cache_path(32) != get_cache_path(64)


def test_cache_write_leaves_no_tmp_files_behind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(load, "IMDB_PREPROCESSED_DIR", tmp_path / "processed")

    write_preprocessed_df_to_cache(pl.DataFrame({"age": [30]}))

    cache_path = load.get_cache_path()
    assert pl.read_parquet(cache_path).height == 1
    assert not list(cache_path.parent.glob("*.tmp"))


def test_interrupted_cache_write_leaves_no_cache_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(load, "IMDB_PREPROCESSED_DIR", tmp_path / "processed")

    def torn_parquet(self, file, *args, **kwargs):
        Path(file).write_bytes(b"partial")
        raise OSError("interrupted mid-write")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", torn_parquet)
    with pytest.raises(OSError, match="interrupted mid-write"):
        write_preprocessed_df_to_cache(pl.DataFrame({"age": [30]}))

    # The torn write stayed at the .tmp path; the cache path a later run
    # would load from was never created.
    assert not load.get_cache_path().exists()
