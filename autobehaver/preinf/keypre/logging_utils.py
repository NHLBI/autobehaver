"""Logging helpers for keypre."""

from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging.

    Safe to call multiple times:
      - In the main process, it sets up handlers/format.
      - In spawned multiprocessing workers (Windows), it sets up logging there too.
      - In forked multiprocessing workers (Linux), it will usually detect existing handlers
        and only update the log level.
    """

    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
