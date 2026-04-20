"""Project-wide logging configuration.

Two sinks are installed on the root logger:

* **Console** (stderr) in a human-readable format — what you see when
  the server runs in the foreground.
* **File** (``logs/neosmart.jsonl``) in one-JSON-object-per-line format,
  enabled when ``settings.logging.json_to_file`` is true. Suitable for
  post-hoc analysis and for forwarding to a log collector.

Downstream modules do not call :func:`configure_logging` themselves —
they just ``logger = logging.getLogger(__name__)``. The web entry point
(FastAPI lifespan) and training/evaluation CLIs call the configurator
once at startup.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from neosmart.config import AppSettings, get_settings


class JsonLineFormatter(logging.Formatter):
    """Emit each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any explicit `extra={...}` fields the caller passed in.
        reserved = set(logging.LogRecord.__dict__) | {"message", "asctime"}
        for key, value in record.__dict__.items():
            if key in reserved or key in payload:
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_NEOSMART_CONFIGURED = "_neosmart_configured"


def configure_logging(settings: AppSettings | None = None) -> None:
    """Install console + (optional) JSON-file handlers on the root logger.

    Idempotent: safe to call multiple times (uvicorn reload, tests).
    """
    s = settings if settings is not None else get_settings()
    root = logging.getLogger()
    if getattr(root, _NEOSMART_CONFIGURED, False):
        return

    root.setLevel(s.logging.level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(s.logging.level)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    if s.logging.json_to_file:
        logs_dir = s.paths.resolve(s.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            logs_dir / "neosmart.jsonl", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonLineFormatter())
        root.addHandler(file_handler)

    # Quiet common noisy third-party loggers.
    for noisy in ("urllib3", "PIL", "matplotlib", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setattr(root, _NEOSMART_CONFIGURED, True)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with logging configured lazily on first call."""
    configure_logging()
    return logging.getLogger(name)


_BANNER_LINES = (
    "",
    " ███╗   ██╗███████╗ ██████╗ ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗",
    " ████╗  ██║██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝",
    " ██╔██╗ ██║█████╗  ██║   ██║███████╗██╔████╔██║███████║██████╔╝   ██║   ",
    " ██║╚██╗██║██╔══╝  ██║   ██║╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║   ",
    " ██║ ╚████║███████╗╚██████╔╝███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║   ",
    " ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ",
)

_SECTION_WIDTH = 74


def print_banner(subtitle: str | None = None) -> None:
    """Write the NEOSMART banner to stderr (bypasses logger, no timestamp)."""
    out = sys.stderr
    for line in _BANNER_LINES:
        print(line, file=out)
    if subtitle:
        print(f" {subtitle}", file=out)
    print("", file=out)
    out.flush()


def print_section(title: str) -> None:
    """Write a console section divider to stderr (bypasses logger)."""
    out = sys.stderr
    tag = f"──  {title}  "
    fill = "─" * max(0, _SECTION_WIDTH - len(tag))
    print("", file=out)
    print(f"{tag}{fill}", file=out)
    out.flush()
