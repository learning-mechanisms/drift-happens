from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from drift_happens.dataset.yearbook import transform
from drift_happens.dataset.yearbook.transform import (
    convert_to_tensor_dataset,
    convert_to_trainable_tensor,
    load_preprocessed_images_into_df,
)


def test_convert_to_trainable_tensor_converts_nhwc_to_nchw() -> None:
    imgs = torch.arange(2 * 3 * 4 * 1).reshape(2, 3, 4, 1)

    out = convert_to_trainable_tensor(imgs)

    assert out.shape == (2, 1, 3, 4)
    torch.testing.assert_close(out[:, 0], imgs[..., 0])


def test_convert_to_tensor_dataset_maps_gender_and_handles_empty_frame() -> None:
    df = pd.DataFrame(
        {
            "img": [np.zeros((32, 32, 3)), np.ones((32, 32, 3))],
            "gender": ["F", "M"],
        }
    )

    dataset = convert_to_tensor_dataset(df)
    empty = convert_to_tensor_dataset(pd.DataFrame({"img": [], "gender": []}))

    assert dataset.tensors[0].shape == (2, 3, 32, 32)
    torch.testing.assert_close(
        dataset.tensors[1], torch.tensor([0, 1], dtype=torch.uint8)
    )
    assert empty.tensors[0].shape == (0, 3, 32, 32)


def test_load_preprocessed_images_uses_cache_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    cached = pd.DataFrame({"gender": ["F"], "year": [1905]})
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached.to_pickle(cache_dir / "yearbook_images.pkl")
    monkeypatch.setattr(transform, "YB_PREPROCESSED_DIR", cache_dir)

    out = load_preprocessed_images_into_df(tmp_path / "raw")

    pd.testing.assert_frame_equal(out, cached)


def test_cache_writes_leave_no_tmp_files_behind(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(transform, "YB_PREPROCESSED_DIR", cache_dir)
    raw = pd.DataFrame(
        {
            "img": [np.zeros((8, 8, 3), dtype=np.float32)],
            "gender": ["F"],
            "year": [1905],
        }
    )
    monkeypatch.setattr(transform, "load_raw_images_into_df", lambda *a, **k: raw)

    transform.load_downscaled_images_into_df(tmp_path / "raw", downscale_size=(4, 4))

    assert (cache_dir / "yearbook_images.pkl").exists()
    assert (cache_dir / "yearbook_images_downscaled_4x4.pkl").exists()
    assert not list(cache_dir.glob("*.tmp"))


def test_interrupted_cache_write_leaves_no_cache_file(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(transform, "YB_PREPROCESSED_DIR", cache_dir)
    raw = pd.DataFrame({"gender": ["F"], "year": [1905]})
    monkeypatch.setattr(transform, "load_raw_images_into_df", lambda *a, **k: raw)

    def torn_pickle(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_pickle", torn_pickle)
    with pytest.raises(OSError, match="interrupted mid-write"):
        load_preprocessed_images_into_df(tmp_path / "raw")

    # The torn write stayed at the .tmp path; the cache path a later run
    # would load from was never created.
    assert not (cache_dir / "yearbook_images.pkl").exists()
    assert (cache_dir / "yearbook_images.pkl.tmp").read_bytes() == b"partial"


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))


def test_load_raw_images_skips_non_png_entries(tmp_path: Path) -> None:
    # A stray non-image file (e.g. .DS_Store, a .txt) in F/ or M/ must be
    # ignored rather than fed to cv2.imread and crashing the whole load.
    _write_png(tmp_path / "F" / "1905_a.png")
    _write_png(tmp_path / "M" / "1910_b.png")
    (tmp_path / "F" / "junk.txt").write_text("not an image")
    (tmp_path / "F" / ".DS_Store").write_bytes(b"\x00\x01")

    df = transform.load_raw_images_into_df(tmp_path)

    assert sorted(df["year"].tolist()) == [1905, 1910]
    assert sorted(df["gender"].tolist()) == ["F", "M"]


def test_load_raw_images_reports_unreadable_png(tmp_path: Path) -> None:
    (tmp_path / "M").mkdir(parents=True)
    bad = tmp_path / "F" / "1905_bad.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a real png")

    with pytest.raises(ValueError, match="unreadable image"):
        transform.load_raw_images_into_df(tmp_path)
