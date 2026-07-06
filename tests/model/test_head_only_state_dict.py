"""Tests for the head-only checkpoint save/load contract of HeadOnlyStateDictMixin."""

import pytest
import torch
import torch.nn as nn

from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningConfig,
)
from drift_happens.model.dataset.image.transfer_learning.head_only import (
    HeadOnlyStateDictMixin,
)


class _ToyTransfer(HeadOnlyStateDictMixin):
    """A frozen-backbone + trainable-head stand-in that exercises the mixin directly."""

    def __init__(self, *, fine_tune: bool) -> None:
        super().__init__()
        self.config = TransferLearningConfig(fine_tune=fine_tune)
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad_(False)


def test_head_only_state_dict_keeps_only_the_trainable_head() -> None:
    model = _ToyTransfer(fine_tune=False)
    assert set(model.state_dict()) == {"head.weight", "head.bias"}


def test_head_only_round_trip_restores_the_head() -> None:
    source = _ToyTransfer(fine_tune=False)
    with torch.no_grad():
        source.head.weight.fill_(1.5)
        source.head.bias.fill_(-0.25)

    target = _ToyTransfer(fine_tune=False)
    target.load_state_dict(source.state_dict())

    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.head.bias, source.head.bias)


def test_head_only_accepts_a_payload_wrapper() -> None:
    source = _ToyTransfer(fine_tune=False)
    target = _ToyTransfer(fine_tune=False)
    target.load_state_dict({"trainable_state_dict": source.state_dict()})
    assert torch.equal(target.head.weight, source.head.weight)


def test_head_only_rejects_a_checkpoint_missing_the_head() -> None:
    target = _ToyTransfer(fine_tune=False)
    # A stale/renamed checkpoint whose keys do not include the trainable head would
    # otherwise load nothing and leave the head at its random initialization.
    with pytest.raises(ValueError, match="trainable parameters"):
        target.load_state_dict({"backbone.weight": torch.zeros(4, 4)})


def test_head_only_rejects_an_empty_checkpoint() -> None:
    target = _ToyTransfer(fine_tune=False)
    with pytest.raises(ValueError, match="trainable parameters"):
        target.load_state_dict({})


def test_fine_tune_load_stays_strict() -> None:
    # In fine-tune mode a checkpoint with missing keys raises via PyTorch's strict check.
    source = _ToyTransfer(fine_tune=True)
    target = _ToyTransfer(fine_tune=True)
    target.load_state_dict(source.state_dict())

    with pytest.raises(RuntimeError):
        target.load_state_dict({"head.weight": torch.zeros(2, 4)})


def test_fine_tune_load_rejects_unexpected_keys() -> None:
    # The pre-filter must not run in fine-tune mode, or an unexpected key would be
    # dropped before PyTorch's strict check could reject it.
    target = _ToyTransfer(fine_tune=True)
    payload = dict(target.state_dict())
    payload["not_a_real_param"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="[Uu]nexpected"):
        target.load_state_dict(payload)


def test_load_state_dict_forwards_assign() -> None:
    # assign=True must reach nn.Module so the loaded tensor replaces the parameter.
    source = _ToyTransfer(fine_tune=True)
    target = _ToyTransfer(fine_tune=True)
    replacement = source.state_dict()
    target.load_state_dict(replacement, assign=True)
    assert target.head.weight.data_ptr() == replacement["head.weight"].data_ptr()


class _ParentWithTransfer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = _ToyTransfer(fine_tune=False)


def test_nested_state_dict_drops_the_backbone() -> None:
    # Under a parent, torch recurses with a prefix; the backbone must still be dropped.
    parent = _ParentWithTransfer()
    assert set(parent.state_dict()) == {"enc.head.weight", "enc.head.bias"}
