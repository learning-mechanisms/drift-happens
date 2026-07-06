from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from drift_happens.dataset.amazon_reviews_23 import preprocess


def _write_group_parquet(reviews_dir: Path, group: str, text: str) -> None:
    reviews_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "title": ["t"],
            "text": [text],
            "rating": [5],
        }
    ).write_parquet(reviews_dir / f"{group}.parquet")


class _EmptyDataset:
    def iter(self, batch_size: int):
        return iter(())


def test_prepare_reviews_df_creates_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pyarrow's ParquetWriter does not create parent directories, so a fresh
    # checkout (where the reviews directory was never materialized by the cache
    # path) must not crash with FileNotFoundError.
    reviews_dir = tmp_path / "processed" / "reviews"
    assert not reviews_dir.exists()
    monkeypatch.setattr(preprocess, "AR23_PREPROCESSED_REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(
        preprocess, "load_dataset", lambda *args, **kwargs: _EmptyDataset()
    )

    # The empty stream reaches the "no reviews streamed" guard, which proves the
    # writer-open path was reached without a FileNotFoundError from the missing
    # directory.
    with pytest.raises(ValueError, match="No reviews streamed"):
        preprocess.prepare_reviews_df(["foo"], skip_existing=False)

    assert reviews_dir.is_dir()


def test_merge_review_dfs_writes_merged_file_and_leaves_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews_dir = tmp_path / "reviews"
    merged_path = tmp_path / "_merged.parquet"
    _write_group_parquet(reviews_dir, "foo", "happy")
    monkeypatch.setattr(preprocess, "AR23_PREPROCESSED_REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(
        preprocess, "AR23_PREPROCESSED_REVIEWS_MERGED_PATH", merged_path
    )

    preprocess.merge_review_dfs(["foo"], assume_yes=True)

    assert merged_path.exists()
    assert pl.read_parquet(merged_path)["text"].to_list() == ["happy"]
    # The atomic swap must not leave the staging file behind.
    assert not merged_path.with_suffix(merged_path.suffix + ".tmp").exists()


def test_merge_review_dfs_failed_sink_keeps_existing_merged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews_dir = tmp_path / "reviews"
    merged_path = tmp_path / "_merged.parquet"
    _write_group_parquet(reviews_dir, "foo", "new")
    # A previously-merged, valid file that a failed re-merge must not corrupt.
    pl.DataFrame({"keep": [1]}).write_parquet(merged_path)
    monkeypatch.setattr(preprocess, "AR23_PREPROCESSED_REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(
        preprocess, "AR23_PREPROCESSED_REVIEWS_MERGED_PATH", merged_path
    )

    def _boom(
        self: pl.LazyFrame, path: object, *args: object, **kwargs: object
    ) -> None:
        # Simulate an interrupted stream that partially writes its target before
        # failing, so a sink straight to the final path would corrupt it.
        Path(str(path)).write_bytes(b"truncated garbage")
        raise RuntimeError("sink interrupted")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", _boom)

    with pytest.raises(RuntimeError, match="sink interrupted"):
        preprocess.merge_review_dfs(["foo"], assume_yes=True)

    # The existing merged file survives untouched and no partial tmp remains.
    assert pl.read_parquet(merged_path)["keep"].to_list() == [1]
    assert not merged_path.with_suffix(merged_path.suffix + ".tmp").exists()
