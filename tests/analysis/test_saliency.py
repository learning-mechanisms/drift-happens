"""Saliency grid renders from tiny in-memory models and a synthetic dataframe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from drift_happens.analysis.plots.saliency import (
    SaliencyCutoffRow,
    SaliencyRow,
    saliency_cutoff_grid,
    saliency_grid,
)
from drift_happens.dataset.yearbook.transform import convert_to_tensor_dataset


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def _synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "year": year,
            "gender": "M" if i % 2 == 0 else "F",
            "img": rng.random((32, 32, 3), dtype=np.float32) * 255.0,
        }
        for year in (1960, 1980, 2000)
        for i in range(3)
    ]
    return pd.DataFrame(rows)


def _sampled_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {
            "year": year,
            "gender": "M" if index % 2 == 0 else "F",
            "img": rng.random((32, 32, 3), dtype=np.float32) * 255.0,
            "sample_id": index,
        }
        for year in (1960, 1980)
        for index in range(20)
    ]
    return pd.DataFrame(rows)


def test_saliency_grid_writes_pdf(tmp_path) -> None:
    torch.manual_seed(0)
    rows = [
        SaliencyRow("mlp_l\nTrain ≤1950", _TinyNet()),
        SaliencyRow("resnet_s\nTrain ≤1970", _TinyNet()),
    ]
    path = saliency_grid(
        rows, _synthetic_df(), convert_to_tensor_dataset, tmp_path / "saliency.pdf"
    )
    assert path.exists()
    assert path.stat().st_size > 6000


def test_saliency_grid_matches_sampled_images_by_eval_year(tmp_path) -> None:
    torch.manual_seed(0)
    chosen_samples: list[tuple[int, int]] = []

    def to_dataset(frame: pd.DataFrame):
        chosen_samples.append(
            (int(frame["year"].iloc[0]), int(frame["sample_id"].iloc[0]))
        )
        return convert_to_tensor_dataset(frame)

    saliency_grid(
        [
            SaliencyRow("cnn_l\nTrain ≤1950", _TinyNet()),
            SaliencyRow("resnet_s\nTrain ≤1950", _TinyNet()),
        ],
        _sampled_df(),
        to_dataset,
        tmp_path / "sampled.pdf",
        eval_years=(1960, 1980),
        sample_per_year=1,
    )

    assert len(chosen_samples) == 4
    by_year = {
        year: [sample for chosen_year, sample in chosen_samples if chosen_year == year]
        for year in (1960, 1980)
    }
    assert by_year[1960][0] == by_year[1960][1]
    assert by_year[1980][0] == by_year[1980][1]
    assert by_year[1960][0] != by_year[1980][0]


def test_saliency_grid_uses_explicit_sample_seeds(tmp_path) -> None:
    torch.manual_seed(0)
    frame = _sampled_df()
    chosen_samples: list[tuple[int, int]] = []

    def to_dataset(sampled: pd.DataFrame):
        chosen_samples.append(
            (int(sampled["year"].iloc[0]), int(sampled["sample_id"].iloc[0]))
        )
        return convert_to_tensor_dataset(sampled)

    saliency_grid(
        [SaliencyRow("cnn_l\nTrain <=1950", _TinyNet())],
        frame,
        to_dataset,
        tmp_path / "seeded.pdf",
        eval_years=(1960, 1980),
        sample_per_year=1,
        seed=99,
        sample_seeds=(8, 9),
    )

    expected = [
        (
            year,
            int(
                frame[frame["year"] == year]
                .sample(n=1, random_state=sample_seed)["sample_id"]
                .iloc[0]
            ),
        )
        for year, sample_seed in ((1960, 8), (1980, 9))
    ]
    assert chosen_samples == expected


def test_saliency_cutoff_grid_matches_sampled_images_by_eval_year(tmp_path) -> None:
    torch.manual_seed(0)
    chosen_samples: list[tuple[int, int]] = []

    def to_dataset(frame: pd.DataFrame):
        chosen_samples.append(
            (int(frame["year"].iloc[0]), int(frame["sample_id"].iloc[0]))
        )
        return convert_to_tensor_dataset(frame)

    saliency_cutoff_grid(
        [
            SaliencyCutoffRow("cnn_l", [_TinyNet(), _TinyNet()]),
            SaliencyCutoffRow("resnet_s", [_TinyNet(), _TinyNet()]),
        ],
        (1950, 1970),
        _sampled_df(),
        to_dataset,
        tmp_path / "cutoffs.pdf",
        eval_years=(1960, 1980),
        sample_per_year=1,
    )

    assert len(chosen_samples) == 8
    by_year = {
        year: [sample for chosen_year, sample in chosen_samples if chosen_year == year]
        for year in (1960, 1980)
    }
    assert len(set(by_year[1960])) == 1
    assert len(set(by_year[1980])) == 1
    assert by_year[1960][0] != by_year[1980][0]


def test_saliency_grid_rejects_mismatched_sample_seeds(tmp_path) -> None:
    with pytest.raises(ValueError, match="sample_seeds"):
        saliency_grid(
            [SaliencyRow("mlp_l", _TinyNet())],
            _sampled_df(),
            convert_to_tensor_dataset,
            tmp_path / "x.pdf",
            eval_years=(1960, 1980),
            sample_per_year=1,
            sample_seeds=(8,),
        )


def test_saliency_grid_rejects_missing_year(tmp_path) -> None:
    with pytest.raises(ValueError, match="no Yearbook samples"):
        saliency_grid(
            [SaliencyRow("mlp_l", _TinyNet())],
            _synthetic_df(),
            convert_to_tensor_dataset,
            tmp_path / "x.pdf",
            eval_years=(1900,),
        )
