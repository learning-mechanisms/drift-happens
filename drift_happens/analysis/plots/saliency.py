"""Gradient-saliency grid for trained Yearbook image models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, get_args

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from torch import nn
from torch.utils.data import Dataset

from drift_happens.analysis.plots import style
from drift_happens.evaluation.interpretability.saliency import compute_saliency_map

DEFAULT_EVAL_YEARS = (1960, 1980, 2000)
_CELL_INCHES = 2.4
_CUTOFF_BLOCK_SPACER = 0.24
_CUTOFF_HEADER_HEIGHTS = (0.22, 0.18)
_CUTOFF_HEADER_ROWS = len(_CUTOFF_HEADER_HEIGHTS)
_CUTOFF_TRAIN_LABEL_FONT_SIZE = 15
_EVAL_LABEL_FONT_SIZE = 12
_ROW_LABEL_FONT_SIZE = 12
_COLORBAR_LABEL_FONT_SIZE = 11
_COLORBAR_TICK_FONT_SIZE = 10
_OVERLAY_ALPHA = 0.5
_SALIENCY_CMAP = "hot"
_SALIENCY_NORM = Normalize(vmin=0.0, vmax=1.0)

ToDataset = Callable[[pd.DataFrame], Dataset]


@dataclass(frozen=True, slots=True)
class SaliencyRow:
    """A trained model and the label shown on the left of its grid row."""

    label: str
    model: nn.Module


@dataclass(frozen=True, slots=True)
class SaliencyCutoffRow:
    """One model family rendered across multiple training cutoffs."""

    label: str
    models: Sequence[nn.Module]


def saliency_grid(
    rows: Sequence[SaliencyRow],
    frame: pd.DataFrame,
    to_dataset: ToDataset,
    path: Path,
    *,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    sample_per_year: int | None = None,
    seed: int = 0,
    sample_seeds: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> Path:
    """Render a (model x cutoff) by eval-year grid of mean saliency over mean faces."""
    if not rows:
        raise ValueError("saliency grid needs at least one model row")
    if not eval_years:
        raise ValueError("saliency grid needs at least one evaluation year")
    sample_seeds = _sample_seeds(eval_years, sample_seeds)
    device = device or torch.device("cpu")
    fig, axes = plt.subplots(
        len(rows),
        len(eval_years),
        figsize=(len(eval_years) * _CELL_INCHES, len(rows) * _CELL_INCHES),
        squeeze=False,
        layout="constrained",
    )
    saliency_mappable = ScalarMappable(norm=_SALIENCY_NORM, cmap=_SALIENCY_CMAP)
    for row_index, row in enumerate(rows):
        model = row.model.to(device).eval()
        for col_index, year in enumerate(eval_years):
            window = _window(
                frame,
                year,
                sample_per_year,
                _eval_year_seed(seed, col_index, sample_seeds),
            )
            saliency, face = _mean_saliency(model, window, to_dataset, device)
            ax = axes[row_index][col_index]
            ax.imshow(_compose_saliency_image(face, saliency))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(f"Eval {year}", fontsize=_EVAL_LABEL_FONT_SIZE)
            if col_index == 0:
                ax.set_ylabel(row.label, fontsize=_ROW_LABEL_FONT_SIZE)
    bar = fig.colorbar(saliency_mappable, ax=axes, fraction=0.02, pad=0.02)
    _label_saliency_colorbar(bar)
    return style.save(fig, path, tight=False)


def saliency_cutoff_grid(
    rows: Sequence[SaliencyCutoffRow],
    cutoffs: Sequence[int],
    frame: pd.DataFrame,
    to_dataset: ToDataset,
    path: Path,
    *,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    sample_per_year: int | None = None,
    seed: int = 0,
    sample_seeds: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> Path:
    """Render trainer rows with cutoff blocks laid out horizontally."""
    if not rows:
        raise ValueError("saliency cutoff grid needs at least one model row")
    if not cutoffs:
        raise ValueError("saliency cutoff grid needs at least one train cutoff")
    if not eval_years:
        raise ValueError("saliency cutoff grid needs at least one evaluation year")
    sample_seeds = _sample_seeds(eval_years, sample_seeds)
    for row in rows:
        if len(row.models) != len(cutoffs):
            raise ValueError(
                "each saliency cutoff row must provide one model per cutoff"
            )

    device = device or torch.device("cpu")
    width_ratios = _cutoff_grid_width_ratios(len(cutoffs), len(eval_years))
    height_ratios = [*_CUTOFF_HEADER_HEIGHTS, *([1.0] * len(rows))]
    ncols = len(width_ratios)
    fig, axes = plt.subplots(
        len(rows) + _CUTOFF_HEADER_ROWS,
        ncols,
        figsize=(
            sum(width_ratios) * _CELL_INCHES,
            sum(height_ratios) * _CELL_INCHES,
        ),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )
    saliency_mappable = ScalarMappable(norm=_SALIENCY_NORM, cmap=_SALIENCY_CMAP)
    cutoff_title_col = len(eval_years) // 2
    data_axes = []
    for row_axes in axes[:_CUTOFF_HEADER_ROWS]:
        for ax in row_axes:
            ax.set_axis_off()
    for spacer_col in _cutoff_grid_spacer_columns(len(cutoffs), len(eval_years)):
        for row_axes in axes:
            row_axes[spacer_col].set_axis_off()
    for cutoff_index, cutoff in enumerate(cutoffs):
        label_col = _cutoff_grid_data_column(
            cutoff_index, cutoff_title_col, len(eval_years)
        )
        axes[0][label_col].text(
            0.5,
            0.5,
            f"Train ≤{cutoff}",
            ha="center",
            va="center",
            fontsize=_CUTOFF_TRAIN_LABEL_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )
        for eval_index, year in enumerate(eval_years):
            col_index = _cutoff_grid_data_column(
                cutoff_index, eval_index, len(eval_years)
            )
            axes[1][col_index].text(
                0.5,
                0.5,
                f"Eval {year}",
                ha="center",
                va="center",
                fontsize=_EVAL_LABEL_FONT_SIZE,
                clip_on=False,
            )
    for row_index, row in enumerate(rows):
        for cutoff_index, model in enumerate(row.models):
            model = model.to(device).eval()
            for eval_index, year in enumerate(eval_years):
                col_index = _cutoff_grid_data_column(
                    cutoff_index, eval_index, len(eval_years)
                )
                saliency, face = _mean_saliency(
                    model,
                    _window(
                        frame,
                        year,
                        sample_per_year,
                        _eval_year_seed(seed, eval_index, sample_seeds),
                    ),
                    to_dataset,
                    device,
                )
                ax = axes[row_index + _CUTOFF_HEADER_ROWS][col_index]
                data_axes.append(ax)
                ax.imshow(_compose_saliency_image(face, saliency))
                ax.set_xticks([])
                ax.set_yticks([])
                if col_index == 0:
                    ax.set_ylabel(row.label, fontsize=_ROW_LABEL_FONT_SIZE)
    bar = fig.colorbar(saliency_mappable, ax=data_axes, fraction=0.02, pad=0.02)
    _label_saliency_colorbar(bar)
    return style.save(fig, path, tight=False)


def build_yearbook_saliency(
    runs_root: Path,
    trainers: Sequence[str],
    cutoffs: Sequence[int],
    path: Path,
    *,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    sample_per_year: int | None = None,
    seed: int = 0,
    sample_seeds: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> Path:
    """Load the per-cutoff Yearbook checkpoints and render the saliency grid."""
    from drift_happens.dataset.yearbook.transform import (
        convert_to_tensor_dataset,
        load_downscaled_images_into_df,
    )

    device = device or torch.device("cpu")
    rows = [
        SaliencyCutoffRow(
            label=trainer,
            models=[
                _load_model(trainer, _checkpoint(runs_root, trainer, cutoff), device)
                for cutoff in cutoffs
            ],
        )
        for trainer in trainers
    ]
    return saliency_cutoff_grid(
        rows,
        cutoffs,
        load_downscaled_images_into_df(),
        convert_to_tensor_dataset,
        path,
        eval_years=eval_years,
        sample_per_year=sample_per_year,
        seed=seed,
        sample_seeds=sample_seeds,
        device=device,
    )


def save_yearbook_saliency_panels(
    runs_root: Path,
    trainers: Sequence[str],
    cutoffs: Sequence[int],
    out_dir: Path,
    *,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    sample_per_year: int | None = None,
    seed: int = 0,
    sample_seeds: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> list[Path]:
    """Save one saliency strip across eval years per trainer and cutoff."""
    from drift_happens.dataset.yearbook.transform import (
        convert_to_tensor_dataset,
        load_downscaled_images_into_df,
    )

    device = device or torch.device("cpu")
    frame = load_downscaled_images_into_df()
    sample_seeds = _sample_seeds(eval_years, sample_seeds)
    paths = []
    for trainer in trainers:
        for cutoff in cutoffs:
            model = _load_model(
                trainer, _checkpoint(runs_root, trainer, cutoff), device
            )
            paths.append(
                _save_strip(
                    model,
                    frame,
                    convert_to_tensor_dataset,
                    eval_years,
                    out_dir / f"{trainer}_{cutoff}.png",
                    sample_per_year=sample_per_year,
                    seed=seed,
                    sample_seeds=sample_seeds,
                    device=device,
                )
            )
    return paths


def _save_strip(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    to_dataset: ToDataset,
    eval_years: Sequence[int],
    path: Path,
    *,
    sample_per_year: int | None,
    seed: int,
    sample_seeds: Sequence[int],
    device: torch.device,
) -> Path:
    model = model.to(device).eval()
    fig, axes = plt.subplots(
        1,
        len(eval_years),
        figsize=(len(eval_years) * _CELL_INCHES, _CELL_INCHES),
        squeeze=False,
        layout="constrained",
    )
    for col, year in enumerate(eval_years):
        saliency, face = _mean_saliency(
            model,
            _window(
                frame,
                year,
                sample_per_year,
                _eval_year_seed(seed, col, sample_seeds),
            ),
            to_dataset,
            device,
        )
        ax = axes[0][col]
        ax.imshow(_compose_saliency_image(face, saliency))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Eval {year}")
    return style.save(fig, path)


def _label_saliency_colorbar(bar: Colorbar) -> None:
    bar.set_ticks([0.0, 1.0])
    bar.set_ticklabels(["low", "high"])
    bar.ax.tick_params(labelsize=_COLORBAR_TICK_FONT_SIZE)
    bar.set_label("Saliency", fontsize=_COLORBAR_LABEL_FONT_SIZE)


def _sample_seeds(
    eval_years: Sequence[int], sample_seeds: Sequence[int] | None
) -> tuple[int, ...]:
    if sample_seeds is None:
        return ()
    values = tuple(sample_seeds)
    if values and len(values) != len(eval_years):
        raise ValueError(
            "sample_seeds must match the number of evaluation years "
            f"({len(eval_years)}), got {len(values)}"
        )
    return values


def _eval_year_seed(
    seed: int, eval_year_index: int, sample_seeds: Sequence[int]
) -> int:
    if sample_seeds:
        return sample_seeds[eval_year_index]
    return seed + eval_year_index


def _cutoff_grid_width_ratios(cutoff_count: int, eval_year_count: int) -> list[float]:
    ratios = []
    for cutoff_index in range(cutoff_count):
        if cutoff_index > 0:
            ratios.append(_CUTOFF_BLOCK_SPACER)
        ratios.extend([1.0] * eval_year_count)
    return ratios


def _cutoff_grid_spacer_columns(cutoff_count: int, eval_year_count: int) -> list[int]:
    return [
        cutoff_index * (eval_year_count + 1) - 1
        for cutoff_index in range(1, cutoff_count)
    ]


def _cutoff_grid_data_column(
    cutoff_index: int, eval_year_index: int, eval_year_count: int
) -> int:
    return cutoff_index * eval_year_count + eval_year_index + cutoff_index


def _window(
    frame: pd.DataFrame, year: int, sample_per_year: int | None, seed: int
) -> pd.DataFrame:
    window = frame[frame["year"] == year]
    if sample_per_year is not None and len(window) > sample_per_year:
        window = window.sample(n=sample_per_year, random_state=seed)
    window = window.reset_index(drop=True)
    if window.empty:
        raise ValueError(f"no Yearbook samples for evaluation year {year}")
    return window


def _mean_saliency(
    model: nn.Module, frame: pd.DataFrame, to_dataset: ToDataset, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    dataset = to_dataset(frame)
    saliencies, images = [], []
    for index in range(len(dataset)):  # type: ignore[arg-type]
        image, _ = dataset[index]
        images.append(image.numpy())
        saliencies.append(compute_saliency_map(model, image, device=device))
    return np.mean(saliencies, axis=0), np.mean(images, axis=0)


def _as_image(image_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(image_chw, (1, 2, 0))
    low, high = float(image.min()), float(image.max())
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)


def _compose_saliency_image(face_chw: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    face = _as_image(face_chw)
    heatmap = plt.get_cmap(_SALIENCY_CMAP)(_peak_normalized(saliency))[..., :3]
    return (1.0 - _OVERLAY_ALPHA) * face + _OVERLAY_ALPHA * heatmap


def _peak_normalized(saliency: np.ndarray) -> np.ndarray:
    peak = float(saliency.max())
    return saliency / peak if peak > 0 else saliency


def _checkpoint(runs_root: Path, trainer: str, cutoff: int) -> Path:
    path = runs_root / trainer / f"train_slice_{cutoff}" / "trained_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing trained checkpoint: {path}")
    return path


def _load_model(preset: str, checkpoint: Path, device: torch.device) -> nn.Module:
    from drift_happens.model.dataset.image.architectures import (
        CNNPreset,
        ImageModelFactory,
        MLPPreset,
        ResNetPreset,
    )

    if preset in get_args(MLPPreset):
        model: nn.Module = ImageModelFactory.create_mlp(
            num_input_channels=3,
            height=32,
            width=32,
            num_classes=2,
            preset=cast(MLPPreset, preset),
        )
    elif preset in get_args(ResNetPreset):
        model = ImageModelFactory.create_resnet(
            num_input_channels=3, num_classes=2, preset=cast(ResNetPreset, preset)
        )
    elif preset in get_args(CNNPreset):
        model = ImageModelFactory.create_cnn(
            num_input_channels=3, num_classes=2, preset=cast(CNNPreset, preset)
        )
    else:
        raise ValueError(f"unsupported saliency model preset: {preset!r}")
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model
