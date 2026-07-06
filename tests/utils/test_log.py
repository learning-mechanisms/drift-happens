from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from tqdm import tqdm

from drift_happens.utils.log import configure_logging, get_logger, shutdown_logging


@pytest.fixture(autouse=True)
def _teardown_logging():
    yield
    shutdown_logging()


def test_configure_logging_writes_plain_and_json_logs(tmp_path: Path) -> None:
    plain = tmp_path / "logs" / "console.log"
    events = tmp_path / "logs" / "events.jsonl"

    log = configure_logging(
        console=False,
        plain_log_file=plain,
        json_log_file=events,
    ).bind(run="smoke")
    log.info("hello", answer=42)

    assert "hello" in plain.read_text()
    row = json.loads(events.read_text().splitlines()[0])
    assert row["event"] == "hello"
    assert row["answer"] == 42
    assert row["run"] == "smoke"
    assert row["level"] == "info"
    assert row["logger"] == "drift_happens"
    assert "timestamp" in row
    assert "module" in row
    assert "func_name" in row
    assert "lineno" in row
    assert "process" in row
    assert "thread" in row


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    configure_logging(console=False, json_log_file=first)
    configure_logging(console=False, json_log_file=second)
    get_logger("test").info("once")

    assert first.exists() and first.read_text() == ""
    assert len(second.read_text().splitlines()) == 1

    shutdown_logging()
    assert logging.getLogger().handlers == []


def test_console_logging_is_tqdm_compatible(capsys) -> None:
    configure_logging(console=True, console_colors="never")
    for _ in tqdm(range(1), desc="progress", file=sys.stdout):
        get_logger("progress").info("while_progress", item=1)

    captured = capsys.readouterr()
    assert "while_progress" in captured.out
