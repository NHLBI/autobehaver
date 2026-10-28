"""Run bounded FeatureImp process tasks and surface worker failures.

Prefer forkserver or spawn to avoid inherited native-library state, and preserve
explicit user thread limits.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Sequence

from autobehaver.progress import progress as tqdm

try:  
    from concurrent.futures.process import BrokenProcessPool
except Exception:  # pragma: no cover
    BrokenProcessPool = RuntimeError  # type: ignore[assignment]


_THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def choose_safe_start_method(preferred: str | None = None) -> str:
    """Return a multiprocessing start method that avoids ``fork`` when possible."""

    available = tuple(mp.get_all_start_methods())

    if preferred:
        preferred = str(preferred).strip().lower()
        if preferred in available and preferred != "fork":
            return preferred

    for method in ("spawn", "forkserver"):
        if method in available:
            return method

    current = mp.get_start_method(allow_none=True)
    if current and current != "fork":
        return current

    if available:
        # Last resort: if a platform exposes only ``fork``, we still return it
        # so the caller can proceed with a warning instead of crashing here.
        return available[0]

    return "spawn"


def cap_worker_thread_env() -> list[str]:
    """Default common BLAS/OpenMP env vars to 1 thread per worker process.

    Existing user-provided settings are preserved.
    """

    changed: list[str] = []
    for key in _THREAD_LIMIT_ENV_KEYS:
        if str(os.environ.get(key, "")).strip() == "":
            os.environ[key] = "1"
            changed.append(key)
    return changed


def _main_module_is_importable() -> bool:
    """Return whether multiprocessing spawn/forkserver can re-import ``__main__``."""

    try:
        import __main__ as main_mod
    except Exception:
        return False

    main_file = getattr(main_mod, "__file__", None)
    if not main_file:
        return False
    main_file = str(main_file).strip()
    if main_file in {"", "<stdin>", "-c"}:
        return False
    return os.path.exists(main_file)


def _shutdown_executor(executor: ProcessPoolExecutor, *, wait: bool, cancel_futures: bool) -> None:
    """Call executor.shutdown with the arguments supported by the active Python runtime."""

    try:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    except TypeError:  # Allow executor implementations without cancel_futures.
        executor.shutdown(wait=wait)


def iter_parallel_results(
    worker_fn: Callable[[Any], Any], tasks: Sequence[Any] | Iterable[Any], *,
    num_workers: int, initializer: Callable[..., Any] | None = None,
    initargs: Sequence[Any] = (), desc: str, stage_name: str,
    preferred_start_method: str | None = None, show_progress: bool = True,
) -> Iterator[Any]:
    """Yield results without retaining an unbounded set of submitted Futures."""
    from .bounded_pool import BoundedProcessRunner
    with BoundedProcessRunner(num_workers=num_workers, initializer=initializer,
                              initargs=initargs, preferred_start_method=preferred_start_method) as runner:
        yield from runner.run(worker_fn, tasks, desc=desc, stage_name=stage_name,
                              show_progress=show_progress)
