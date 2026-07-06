"""Transform IMDB face metadata and images into trainable datasets."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from scipy.io import loadmat

from drift_happens.dataset.imdb_faces.const import (
    IMDB_UNPACK_DIR,
)
from drift_happens.dataset.yearbook.transform import convert_to_trainable_tensor
from drift_happens.utils.log import get_logger

logger = get_logger()

# record count of the canonical imdb_meta.mat shipped with the IMDB-WIKI crop
IMDB_METADATA_EXPECTED_RECORDS = 460723


def load_full_metadata(
    raw_dataset_path: Path = IMDB_UNPACK_DIR,
) -> pl.DataFrame:
    """Load the full unfiltered IMDB metadata from the .mat file."""
    logger.info(f"Loading full IMDB metadata from {raw_dataset_path / 'imdb_meta.mat'}")

    metadata = _parse_imdb_metadata(raw_dataset_path / "imdb_meta.mat")
    if len(metadata) != IMDB_METADATA_EXPECTED_RECORDS:
        raise ValueError(
            f"IMDB metadata record count mismatch: "
            f"expected {IMDB_METADATA_EXPECTED_RECORDS}, got {len(metadata)}"
        )

    metadata_df = pl.DataFrame([vars(record) for record in metadata])
    if metadata_df.filter(pl.col("dob").is_null()).height / metadata_df.height >= 0.01:
        raise ValueError("More than 1% of dob are missing!")

    metadata_df = metadata_df.with_columns(
        sample_idx=pl.arange(0, metadata_df.height),
        age=(pl.col("photo_taken") - pl.col("dob").dt.year()),
    )

    logger.info(f"Loaded {len(metadata_df)} metadata records")
    return metadata_df.select(
        "sample_idx",
        "photo_taken",
        "gender",
        "dob",
        "age",
        "name",
        "celeb_id",
        "full_path",
        "face_location",
        "face_score",
        "second_face_score",
    )


def load_and_preprocess_images_into_df(
    raw_dataset_path: Path = IMDB_UNPACK_DIR,
    target_size: int = 32,
) -> pl.DataFrame:
    """Load and filter IMDB crop images plus metadata into a polars DataFrame."""
    df_metadata = load_full_metadata(raw_dataset_path=raw_dataset_path)

    image_paths = [
        raw_dataset_path / "imdb_crop" / full_path
        for full_path in df_metadata["full_path"]
    ]

    filter_imgs, filter_mask = _load_images_threaded(
        image_paths,
        require_square=True,
        square_tol=0.04,
        target_size=target_size,
    )

    metadata_df_filtered = (
        df_metadata.select("sample_idx", "photo_taken", "celeb_id", "age", "gender")
        .filter(filter_mask)
        .with_columns(
            img=pl.Series(np.array(filter_imgs)),
        )
    )
    metadata_df_filtered = metadata_df_filtered.filter(pl.col("age").is_not_null())
    filtered_ratio = metadata_df_filtered.height / df_metadata.height

    logger.info(
        f"Loaded {len(metadata_df_filtered)} images "
        f"({filtered_ratio:.2%} of total) "
        f"from {raw_dataset_path}"
    )
    return metadata_df_filtered


def convert_to_tensor_dataset(df: pl.DataFrame) -> torch.utils.data.TensorDataset:
    X_train = torch.tensor(df["img"].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(df["gender"].to_numpy())

    return torch.utils.data.TensorDataset(convert_to_trainable_tensor(X_train), y_train)


# ------------------------------------------------------------------------------------ #
#                                   INTERNAL HELPERS                                   #
# ------------------------------------------------------------------------------------ #


@dataclass
class _FaceMetadata:
    dob: date | None
    photo_taken: int
    full_path: str
    gender: int | None  # 0=female, 1=male
    name: str
    face_location: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    face_score: float
    second_face_score: float | None
    celeb_id: int | None = None  # IMDb only


def _is_square(image: np.ndarray, tol: float = 0.04) -> bool:
    h, w = image.shape[:2]
    return abs(h - w) / max(h, w) <= tol


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)

    top = (h - side) // 2
    left = (w - side) // 2

    return img[top : top + side, left : left + side]


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)  # type: ignore
    return np.asarray(pil)


def _matlab_datenum_to_datetime(datenum: float | None) -> date | None:
    # missing date: empty metadata cell or NaN datenum
    if datenum is None or np.isnan(datenum):
        return None
    d = date.fromordinal(int(datenum))
    # implausibly early date signals a conversion issue
    if d < date(1800, 1, 1):
        return None
    return d - timedelta(days=366)


def _unwrap_scalar(x):
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if x.size == 0:
            return None
        return x.item()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _parse_dob_from_filename(filename: str) -> date | None:
    # Example filename: '30/nm0000030_rm3412315136_1929-5-4_1964.jpg'
    try:
        parts = filename.split("_")
        dob_str = parts[2]  # '1929-5-4'
        dob_parts = list(map(int, dob_str.split("-")))
        return date(dob_parts[0], dob_parts[1], dob_parts[2])
    except (IndexError, ValueError):
        return None


def _parse_imdb_metadata(mat_path: Path) -> list[_FaceMetadata]:
    data = loadmat(mat_path, simplify_cells=True)

    # IMDb/Wiki metadata lives under "imdb" or "wiki"
    meta = data["imdb"] if "imdb" in data else data["wiki"]
    records = []

    n = len(meta["photo_taken"])

    for i in range(n):
        dob_raw = _unwrap_scalar(meta["dob"][i])
        gender_raw = _unwrap_scalar(meta["gender"][i])
        second_score_raw = _unwrap_scalar(meta["second_face_score"][i])

        record = _FaceMetadata(
            # Conversion issues (off by 1 year and sometimes 1 day, matlab vs python)
            # filenames sometimes contain dates like "1942-0-0" so be use
            # our conversion as source of truth
            dob=_matlab_datenum_to_datetime(dob_raw),
            photo_taken=int(_unwrap_scalar(meta["photo_taken"][i])),
            full_path=str(_unwrap_scalar(meta["full_path"][i])),
            # _unwrap_scalar returns None for empty metadata cells; treat that
            # like NaN so missing values all take the None path.
            gender=(
                None if gender_raw is None or np.isnan(gender_raw) else int(gender_raw)
            ),
            name=str(_unwrap_scalar(meta["name"][i])),
            face_location=tuple(map(int, meta["face_location"][i])),  # type: ignore
            face_score=float(_unwrap_scalar(meta["face_score"][i])),
            second_face_score=(
                None
                if second_score_raw is None or np.isnan(second_score_raw)
                else float(second_score_raw)
            ),
            celeb_id=(
                int(_unwrap_scalar(meta["celeb_id"][i])) if "celeb_id" in meta else None
            ),
        )

        records.append(record)

    return records


def _read_image(
    path: Path,
    *,
    require_square: bool = False,
    square_tol: float = 0.04,
    target_size: int | None = None,
) -> np.ndarray | None:
    """
    Load one image with optional square filtering and resizing.

    Returns None if filtered out.
    """
    with Image.open(path) as pil_image:
        image = np.asarray(pil_image.convert("RGB"))

    if require_square and not _is_square(image, square_tol):
        return None

    if target_size is not None:
        image = _center_crop_square(image)
        image = resize_square(image, target_size)

    return image


def _load_images_threaded(
    image_paths: list[Path],
    *,
    start: int | None = None,
    stop: int | None = None,
    max_workers: int = 16,
    require_square: bool = False,
    square_tol: float = 0.04,
    target_size: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Returns:
      - images: The successfully loaded images
      - mask: A boolean mask aligned with image_paths
    """
    n = len(image_paths)
    logger.info(f"Loading {n} images with {max_workers} threads...")

    mask = np.zeros(n, dtype=bool)

    indices = list(range(n))[slice(start, stop)]
    paths_subset = [image_paths[i] for i in indices]

    def _read(idx: int, path: Path) -> tuple[int, np.ndarray | None]:
        img = _read_image(
            path,
            require_square=require_square,
            square_tol=square_tol,
            target_size=target_size,
        )
        return idx, img

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_read, indices, paths_subset))

    images: list[np.ndarray] = []

    for idx, img in results:
        if img is not None:
            mask[idx] = True
            images.append(img)

    logger.info(f"Loaded {len(images)} / {n} images successfully.")
    return images, mask
