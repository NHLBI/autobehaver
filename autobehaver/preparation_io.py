"""Bounded file-I/O concurrency and task-scoped preparation progress."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextvars import ContextVar
from contextlib import contextmanager
import time

_callback = ContextVar('preparation_progress', default=None)

@contextmanager
def preparation_progress(callback):
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)

def report(message):
    callback = _callback.get()
    if callback is not None:
        callback(message)

def io_results(function, items, *, label, workers=4):
    """Bound submissions to twice the worker count; yield in original order.

    Errors propagate; completion messages are emitted only on the consuming
    task thread. No GUI calls or cross-task callback sharing in I/O workers.
    """
    items = list(items)
    total = len(items)
    started = last = time.monotonic()
    report(f'{label}: 0 / {total}')
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        pending = deque()
        for _ in range(min(total, workers * 2)):
            pending.append(pool.submit(function, next(iterator)))
        completed = 0
        while pending:
            result = pending.popleft().result()
            completed += 1
            now = time.monotonic()
            if now-last >= .25 or completed == total:
                report(f'{label}: {completed} / {total} ({now-started:.1f}s)')
                last = now
            yield result
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(function, item))
    report(f'{label}: complete ({time.monotonic()-started:.1f}s)')
