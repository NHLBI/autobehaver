"""Batching helpers for mmgen.

mmgen processes many recordings; to keep memory usage bounded, the pipeline
processes inference files in batches.

This module contains small helpers for creating ordered batches.
"""

from __future__ import annotations


def make_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into consecutive batches.

    Notes
    -----
    - Preserves order.
    - Returns an empty list if items is empty.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
