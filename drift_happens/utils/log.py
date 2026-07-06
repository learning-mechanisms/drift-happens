"""Application logging setup built on structlog and stdlib logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO, Literal

import structlog
from tqdm import tqdm

ColorMode = Literal["auto", "always", "never"]


class _TqdmCompatStream:
    """
    Stream wrapper that routes writes through ``tqdm.write``.

    ``tqdm`` redraws progress bars with carriage returns. Direct log writes to the same
    stream can leave stale bar snapshots behind; ``tqdm.write`` coordinates the write
    with active bars.
    """

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream

    def write(self, msg: str) -> int:
        # StreamHandler always writes msg + terminator ('\n'), so strip it before handing off.
        tqdm.write(msg[:-1], file=self._stream)
        return len(msg)

    def flush(self) -> None:
        self._stream.flush()


def configure_logging(
    level: int = logging.INFO,
    *,
    console: bool = True,
    console_colors: ColorMode = "auto",
    plain_log_file: Path | None = None,
    json_log_file: Path | None = None,
) -> structlog.stdlib.BoundLogger:
    """Configure process-wide application logging."""
    logging.disable(logging.NOTSET)
    root = logging.getLogger()
    _clear_handlers(root)
    root.setLevel(level)

    if console:
        console_handler = logging.StreamHandler(_TqdmCompatStream(sys.stdout))
        console_handler.setFormatter(
            _console_formatter(colors=_use_colors(console_colors))
        )
        root.addHandler(console_handler)

    if plain_log_file is not None:
        plain_log_file.parent.mkdir(parents=True, exist_ok=True)
        plain_handler = logging.FileHandler(plain_log_file)
        plain_handler.setFormatter(_console_formatter(colors=False))
        root.addHandler(plain_handler)

    if json_log_file is not None:
        json_log_file.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(_json_formatter())
        root.addHandler(json_handler)

    logging.captureWarnings(True)
    _configure_structlog()
    return get_logger("drift_happens")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger."""
    return structlog.get_logger(name) if name else structlog.get_logger()


def shutdown_logging() -> None:
    """Flush and close configured stdlib handlers."""
    _clear_handlers(logging.getLogger())
    logging.captureWarnings(False)


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def _shared_processors() -> list[structlog.typing.Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.PROCESS,
                structlog.processors.CallsiteParameter.THREAD,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _formatter(
    renderer: structlog.typing.Processor,
) -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def _console_formatter(*, colors: bool) -> structlog.stdlib.ProcessorFormatter:
    return _formatter(structlog.dev.ConsoleRenderer(colors=colors))


def _json_formatter() -> structlog.stdlib.ProcessorFormatter:
    return _formatter(structlog.processors.JSONRenderer())


def _use_colors(mode: ColorMode) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()
