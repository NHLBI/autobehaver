"""Configure process-local logging for KeyPre."""

from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging, preserving existing handlers when present.

    Spawned workers configure their own handlers; forked workers can reuse them.
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
