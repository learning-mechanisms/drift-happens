"""
Lock the behaviour of the autouse ``_no_cuda_torch`` guard in conftest.py.

By default the guard pins ``CUDA_VISIBLE_DEVICES`` empty so tests stay off CUDA; tests
marked ``cuda_allowed`` (the opt-in GPU model-matrix smoke) are exempt so they can run
on CUDA when requested.
"""

from __future__ import annotations

import os

import pytest
from tests.conftest import CUDA_ALLOWED_MARKER

# Captured at module import, before any fixture monkeypatching, to distinguish the
# guard's own pin from a pre-existing outer value inside the marked-test exemption check.
_OUTER_CVD: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")


def test_cuda_is_blocked_for_unmarked_tests() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


@pytest.mark.cuda_allowed
def test_cuda_allowed_marker_is_visible_to_the_guard(
    request: pytest.FixtureRequest,
) -> None:
    # Verify the marker constant matches the literal everyone applies.
    assert request.node.get_closest_marker(CUDA_ALLOWED_MARKER) is not None
    # Verify the guard's exemption branch: the env must not have been pinned to "".
    # Only meaningful when the outer environment did not already export CUDA_VISIBLE_DEVICES="".
    if _OUTER_CVD != "":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == _OUTER_CVD
