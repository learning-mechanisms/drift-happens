from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from drift_happens.dataset.imdb_faces import transform
from drift_happens.dataset.imdb_faces.transform import (
    _center_crop_square,
    _is_square,
    _load_images_threaded,
    _matlab_datenum_to_datetime,
    _parse_dob_from_filename,
    _parse_imdb_metadata,
    _read_image,
)


def test_is_square_respects_tolerance() -> None:
    assert _is_square(np.zeros((100, 105, 3)), 0.1)
    assert _is_square(np.zeros((200, 190, 3)), 0.1)
    assert not _is_square(np.zeros((100, 150, 3)), 0.1)


def test_matlab_datenum_to_datetime_handles_missing_and_old_dates() -> None:
    datenum = date(2000, 1, 1).toordinal() + 366

    assert _matlab_datenum_to_datetime(float("nan")) is None
    assert _matlab_datenum_to_datetime(None) is None
    assert _matlab_datenum_to_datetime(1) is None
    assert _matlab_datenum_to_datetime(datenum) == date(2000, 1, 1)


def test_parse_imdb_metadata_treats_empty_cells_as_missing(
    tmp_path: Path, monkeypatch
) -> None:
    # Empty MATLAB cells unwrap to None; they must take the missing-value path
    # instead of crashing np.isnan.
    meta = {
        "imdb": {
            "dob": [np.array([])],
            "photo_taken": [np.array(1964)],
            "full_path": [np.array("30/nm0000030_rm1_1929-5-4_1964.jpg")],
            "gender": [np.array([])],
            "name": [np.array("Someone")],
            "face_location": [np.array([1, 2, 3, 4])],
            "face_score": [np.array(2.5)],
            "second_face_score": [np.array([])],
        }
    }
    monkeypatch.setattr(
        "drift_happens.dataset.imdb_faces.transform.loadmat", lambda *a, **k: meta
    )

    records = _parse_imdb_metadata(tmp_path / "imdb_meta.mat")

    assert len(records) == 1
    assert records[0].dob is None
    assert records[0].gender is None
    assert records[0].second_face_score is None


def test_parse_dob_from_filename_returns_none_for_invalid_names() -> None:
    assert _parse_dob_from_filename("missing_parts.jpg") is None
    assert _parse_dob_from_filename("a_b_1942-0-0_c.jpg") is None
    assert _parse_dob_from_filename(
        "30/nm0000030_rm3412315136_1929-5-4_1964.jpg"
    ) == date(1929, 5, 4)


def test_center_crop_square_crops_long_side() -> None:
    image = np.arange(2 * 4 * 3).reshape(2, 4, 3)

    cropped = _center_crop_square(image)

    assert cropped.shape == (2, 2, 3)
    np.testing.assert_array_equal(cropped, image[:, 1:3])


def _make_test_images(tmp_path: Path) -> tuple[Path, Path]:
    """Return (square 8x8, wide 8x16) JPEG paths written under tmp_path."""
    square = tmp_path / "square.jpg"
    wide = tmp_path / "wide.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(square)
    Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(wide)
    return square, wide


def test_read_image_filters_non_square_and_resizes(tmp_path: Path) -> None:
    square, wide = _make_test_images(tmp_path)

    assert _read_image(wide, require_square=True) is None
    resized = _read_image(square, require_square=True, target_size=4)
    assert resized is not None
    assert resized.shape == (4, 4, 3)


def test_load_images_threaded_returns_aligned_mask(tmp_path: Path) -> None:
    square, wide = _make_test_images(tmp_path)

    images, mask = _load_images_threaded(
        [square, wide],
        require_square=True,
        target_size=4,
        max_workers=1,
    )

    assert len(images) == 1
    assert mask.tolist() == [True, False]


class _StopBeforeLoadingError(Exception):
    pass


def test_image_paths_use_the_given_raw_dataset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The image paths must be rooted at the raw_dataset_path argument, not the
    # IMDB_UNPACK_DIR module constant, so a non-default dataset root is honored.
    captured: dict[str, list[Path]] = {}

    monkeypatch.setattr(
        transform,
        "load_full_metadata",
        lambda raw_dataset_path: pl.DataFrame({"full_path": ["12/img.jpg"]}),
    )

    def _capture(image_paths, **kwargs):
        captured["paths"] = list(image_paths)
        raise _StopBeforeLoadingError

    monkeypatch.setattr(transform, "_load_images_threaded", _capture)

    custom_root = tmp_path / "custom_root"
    with pytest.raises(_StopBeforeLoadingError):
        transform.load_and_preprocess_images_into_df(raw_dataset_path=custom_root)

    assert captured["paths"] == [custom_root / "imdb_crop" / "12/img.jpg"]
