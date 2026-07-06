"""Tests for DINOv2 model loading."""

import pytest
import torch
from torch import nn

from drift_happens.model.dataset.image.transfer_learning.dino_v2 import (
    _load_dinov2_backbone,
)


def test_load_dinov2_backbone_retries_corrupt_hub_cache(monkeypatch) -> None:
    model = nn.Identity()
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_load(repo_or_dir: str, model_name: str, **kwargs: object) -> nn.Module:
        calls.append((repo_or_dir, model_name, kwargs))
        if len(calls) == 1:
            raise ModuleNotFoundError(
                "No module named 'dinov2'",
                name="dinov2",
            )
        return model

    monkeypatch.setattr(torch.hub, "load", fake_load)

    loaded = _load_dinov2_backbone("dinov2_vits14")

    assert loaded is model
    assert calls == [
        ("facebookresearch/dinov2", "dinov2_vits14", {}),
        ("facebookresearch/dinov2", "dinov2_vits14", {"force_reload": True}),
    ]


def test_load_dinov2_backbone_raises_unrelated_missing_module(monkeypatch) -> None:
    def fake_load(repo_or_dir: str, model_name: str, **kwargs: object) -> nn.Module:
        raise ModuleNotFoundError(
            "No module named 'other_package'", name="other_package"
        )

    monkeypatch.setattr(torch.hub, "load", fake_load)

    with pytest.raises(ModuleNotFoundError, match="other_package"):
        _load_dinov2_backbone("dinov2_vits14")


def test_load_dinov2_backbone_retries_missing_submodule(monkeypatch) -> None:
    # A partially corrupt cache fails on a dinov2 submodule (dotted name), which
    # must also trigger the force-reload, not just the top-level package.
    model = nn.Identity()
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_load(repo_or_dir: str, model_name: str, **kwargs: object) -> nn.Module:
        calls.append((repo_or_dir, model_name, kwargs))
        if len(calls) == 1:
            raise ModuleNotFoundError(
                "No module named 'dinov2.hub.backbones'",
                name="dinov2.hub.backbones",
            )
        return model

    monkeypatch.setattr(torch.hub, "load", fake_load)

    loaded = _load_dinov2_backbone("dinov2_vits14")

    assert loaded is model
    assert calls == [
        ("facebookresearch/dinov2", "dinov2_vits14", {}),
        ("facebookresearch/dinov2", "dinov2_vits14", {"force_reload": True}),
    ]
