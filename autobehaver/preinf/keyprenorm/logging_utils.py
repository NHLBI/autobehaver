"""Route KeyPreNorm worker logs through a parent-process queue listener."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from typing import List, Optional, Tuple


def _parse_log_level(level_str: str) -> int:
    """Resolve a logging level name or raise for an invalid name."""
    level = getattr(logging, str(level_str).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid --log_level '{level_str}'. Use DEBUG, INFO, WARNING, ERROR.")
    return level


def setup_logging_queue(*, log_level: str, log_file: Optional[str]) -> Tuple[mp.Queue, QueueListener]:
    """Start a logging queue listener and return (queue, listener).

    The caller must stop the listener after workers and final logging finish.
    """
    level = _parse_log_level(log_level)

    log_queue: mp.Queue = mp.get_context("spawn").Queue(-1)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: List[logging.Handler] = []

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    handlers.append(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        handlers.append(fh)

    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(QueueHandler(log_queue))

    keyprenorm_logger = logging.getLogger("keyprenorm")
    keyprenorm_logger.propagate = True
    keyprenorm_logger.setLevel(level)

    return log_queue, listener


def setup_worker_logging(log_queue: mp.Queue, log_level: str) -> None:
    """Route worker log records to the parent-process queue."""
    level = _parse_log_level(log_level)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(QueueHandler(log_queue))

    keyprenorm_logger = logging.getLogger("keyprenorm")
    keyprenorm_logger.propagate = True
    keyprenorm_logger.setLevel(level)
