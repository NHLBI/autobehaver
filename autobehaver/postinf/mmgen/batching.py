"""Batch MMGen recordings and consume bounded worker results in deterministic input
order.
"""

from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import chain, islice


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


def ordered_worker_results(function, items, *, num_workers, initializer=None,
                           initargs=(), tasks_per_worker=100):
    """Keep a bounded queue and preserve scientific input/reduction order.

    ProcessPoolExecutor reports abrupt worker exits rather than leaving a lost
    Pool task waiting forever. Recycling whole generations also supports Python
    3.10, whose executor lacks individual-worker recycling.
    """
    if num_workers < 1 or tasks_per_worker < 1:
        raise ValueError("Worker counts and recycling intervals must be positive.")
    size = len(items) if hasattr(items, '__len__') else None
    if size == 0:
        return
    workers = min(int(num_workers), size) if size is not None else int(num_workers)
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        for item in items:
            yield function(item)
        return
    generation_size = workers * int(tasks_per_worker)
    items = iter(items)
    sentinel = object()
    while True:
        first = next(items, sentinel)
        if first is sentinel:
            return
        generation = chain([first], islice(items, generation_size-1))
        with ProcessPoolExecutor(max_workers=workers, initializer=initializer,
                                 initargs=initargs, mp_context=mp.get_context("spawn")) as executor:
            pending = deque()
            for _ in range(workers):
                item = next(generation, sentinel)
                if item is not sentinel:
                    pending.append(executor.submit(function, item))
            while pending:
                result = pending.popleft().result()
                # Refill only when the caller has consumed the current result.
                yield result
                item = next(generation, sentinel)
                if item is not sentinel:
                    pending.append(executor.submit(function, item))
