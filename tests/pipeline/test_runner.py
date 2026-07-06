from __future__ import annotations

import pytest

from drift_happens.pipeline._shared.runner import ModelFailure, run_per_model


def _make_fn(calls: list[str], *, fail_on: set[str]):
    def fn_single(ctx, key) -> None:
        calls.append(key)
        if key in fail_on:
            raise RuntimeError(f"boom {key}")

    return fn_single


def test_run_per_model_isolates_failures_and_runs_remaining() -> None:
    calls: list[str] = []
    failures = run_per_model(
        None,
        ["a", "b", "c"],
        _make_fn(calls, fail_on={"b"}),
        n_workers=1,
        exit_on_failure=False,
    )

    assert calls == ["a", "b", "c"]
    assert failures == [ModelFailure(key="b", error="boom b")]


def test_run_per_model_reports_success_with_empty_failures() -> None:
    calls: list[str] = []
    failures = run_per_model(
        None,
        ["a", "b"],
        _make_fn(calls, fail_on=set()),
        n_workers=1,
        exit_on_failure=False,
    )

    assert calls == ["a", "b"]
    assert failures == []


def test_run_per_model_exits_with_code_two_on_failure() -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_per_model(
            None,
            ["a", "b"],
            _make_fn([], fail_on={"a"}),
            n_workers=1,
        )

    assert excinfo.value.code == 2


def test_run_per_model_fail_fast_reraises_first_error() -> None:
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="boom a"):
        run_per_model(
            None,
            ["a", "b"],
            _make_fn(calls, fail_on={"a"}),
            n_workers=1,
            fail_fast=True,
        )

    assert calls == ["a"]


def test_failures_are_logged_at_error_with_traceback(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner

    mock_logger = MagicMock()
    monkeypatch.setattr(runner, "logger", mock_logger)

    run_per_model(
        None, ["x"], _make_fn([], fail_on={"x"}), n_workers=1, exit_on_failure=False
    )

    first = mock_logger.error.call_args_list[0]
    assert "RuntimeError" in first.args[0]
    assert first.kwargs.get("exc_info") is not None
    assert not mock_logger.info.called


def test_bare_exception_keeps_its_type_in_the_log(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner

    mock_logger = MagicMock()
    monkeypatch.setattr(runner, "logger", mock_logger)

    def fn(ctx, key) -> None:
        raise ValueError()

    run_per_model(None, ["x"], fn, n_workers=1, exit_on_failure=False)

    assert "ValueError" in mock_logger.error.call_args_list[0].args[0]


def test_extra_args_forwarded_to_fn_single() -> None:
    received: list = []

    def fn(ctx, key, extra) -> None:
        received.append((key, extra))

    sentinel = object()
    run_per_model(
        None, ["a", "b"], fn, n_workers=1, extra_args=(sentinel,), exit_on_failure=False
    )

    assert received == [("a", sentinel), ("b", sentinel)]


def test_parallel_pool_configures_worker_logging(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner
    from drift_happens.utils.log import configure_logging

    captured: dict = {}

    class _FakePool:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def apply_async(self, fn, args):
            result = MagicMock()
            result.get.return_value = None
            return result

    fake_ctx = MagicMock()
    fake_ctx.Pool = _FakePool
    monkeypatch.setattr(runner.mp, "get_context", lambda method: fake_ctx)

    run_per_model(None, ["a", "b"], lambda *a: None, n_workers=2, exit_on_failure=False)

    assert captured["initializer"] is configure_logging
    assert isinstance(captured["initargs"][0], int)


def test_pool_branch_isolates_failures_and_returns_model_failures(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner

    class _FakePool:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def apply_async(self, fn, args):
            result = MagicMock()
            key = args[1]
            if key == "b":
                result.get.side_effect = RuntimeError("boom b")
            else:
                result.get.return_value = None
            return result

    fake_ctx = MagicMock()
    fake_ctx.Pool = _FakePool
    monkeypatch.setattr(runner.mp, "get_context", lambda method: fake_ctx)

    failures = run_per_model(
        None, ["a", "b", "c"], lambda *a: None, n_workers=2, exit_on_failure=False
    )

    assert failures == [ModelFailure(key="b", error="boom b")]


def test_pool_branch_exits_with_code_two_on_failure(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner

    class _FakePool:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def apply_async(self, fn, args):
            result = MagicMock()
            result.get.side_effect = RuntimeError("boom")
            return result

    fake_ctx = MagicMock()
    fake_ctx.Pool = _FakePool
    monkeypatch.setattr(runner.mp, "get_context", lambda method: fake_ctx)

    with pytest.raises(SystemExit) as excinfo:
        run_per_model(None, ["a"], lambda *a: None, n_workers=2)

    assert excinfo.value.code == 2


def test_pool_branch_forwards_extra_args(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from drift_happens.pipeline._shared import runner

    received: list = []

    class _FakePool:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def apply_async(self, fn, args):
            # Execute in-process to capture the forwarded args.
            received.append(args)
            result = MagicMock()
            result.get.return_value = None
            return result

    fake_ctx = MagicMock()
    fake_ctx.Pool = _FakePool
    monkeypatch.setattr(runner.mp, "get_context", lambda method: fake_ctx)

    sentinel = object()
    run_per_model(
        None,
        ["a"],
        lambda *a: None,
        n_workers=2,
        extra_args=(sentinel,),
        exit_on_failure=False,
    )

    # args tuple passed to apply_async must include extra_args at the end.
    assert received[0][-1] is sentinel
