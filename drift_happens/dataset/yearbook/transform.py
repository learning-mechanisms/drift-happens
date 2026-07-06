from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from drift_happens.dataset.yearbook.const import YB_PREPROCESSED_DIR, YB_UNPACK_DIR


def load_raw_images_into_df(
    raw_dataset_path: Path = YB_UNPACK_DIR
    / "faces_aligned_small_mirrored_co_aligned_cropped_cleaned",
) -> pd.DataFrame:
    """Load all images from a dataset root directory into a DataFrame."""
    women = sorted((raw_dataset_path / "F").glob("*.png"))
    men = sorted((raw_dataset_path / "M").glob("*.png"))

    samples: list[dict] = []
    for gender, zipped_sample_list in [("F", women), ("M", men)]:
        for abs_path in zipped_sample_list:
            relative_path = str(abs_path).removeprefix(str(raw_dataset_path) + "/")
            img = cv2.imread(str(abs_path))
            if img is None:
                raise ValueError(f"unreadable image: {abs_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples.append(
                {
                    "relative_path": relative_path,
                    "gender": gender,
                    "year": int(relative_path.split("/")[-1].split("_")[0]),
                    "img": img.astype(np.float32),
                }
            )

    df = pd.DataFrame(samples).sample(frac=1, random_state=42)  # random shuffle

    # merge sort to keep random order stable
    df = df.sort_values(by="year", kind="mergesort")

    # create `sample_id` column
    df = df.reset_index(drop=True)
    df["sample_id"] = df.index

    return df


def load_preprocessed_images_into_df(
    raw_dataset_path: Path = YB_UNPACK_DIR
    / "faces_aligned_small_mirrored_co_aligned_cropped_cleaned",
) -> pd.DataFrame:
    """Return a cached full-resolution image DataFrame, building the cache if absent."""
    cache_file = YB_PREPROCESSED_DIR / "yearbook_images.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    df = load_raw_images_into_df(raw_dataset_path)
    _write_pickle_atomic(df, cache_file)

    return df


def load_downscaled_images_into_df(
    raw_dataset_path: Path = YB_UNPACK_DIR
    / "faces_aligned_small_mirrored_co_aligned_cropped_cleaned",
    downscale_size: tuple[int, int] = (32, 32),
) -> pd.DataFrame:
    """Return a cached downscaled image DataFrame, building the cache if absent."""
    cache_file = (
        YB_PREPROCESSED_DIR
        / f"yearbook_images_downscaled_{downscale_size[0]}x{downscale_size[1]}.pkl"
    )
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    df = load_preprocessed_images_into_df(raw_dataset_path)

    # downscale images
    df["img"] = df["img"].apply(
        lambda img: cv2.resize(  # type: ignore
            img, downscale_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    )

    _write_pickle_atomic(df, cache_file)

    return df


def _write_pickle_atomic(df: pd.DataFrame, cache_file: Path) -> None:
    """Write-then-rename so an interrupted save never leaves a truncated file at a path
    a later run would happily load."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_name(cache_file.name + ".tmp")
    df.to_pickle(tmp_path)
    tmp_path.replace(cache_file)


def convert_to_trainable_tensor(imgs: torch.Tensor) -> torch.Tensor:
    """Convert an N×H×W×C image batch to N×C×H×W."""
    return torch.permute(imgs, (0, 3, 1, 2))


def convert_to_tensor_dataset(df: pd.DataFrame) -> torch.utils.data.TensorDataset:
    X_train = (
        torch.tensor(np.stack(df["img"].to_list()), dtype=torch.float32)
        if len(df) > 0
        else torch.empty((0, 32, 32, 3), dtype=torch.float32)
    )
    y_train = torch.tensor((df["gender"] == "M").to_numpy())

    return torch.utils.data.TensorDataset(
        convert_to_trainable_tensor(X_train),
        y_train.to(torch.uint8),
    )
