"""Parallel execution helpers for feature importance analysis.

Why this exists
---------------
FeatureImp trains many XGBoost + SHAP worker tasks. On Linux/HPC systems, the
``multiprocessing.Pool`` default start method (``fork``) can deadlock or
crash when child processes inherit OpenMP / native-library state from the
parent. XGBoost explicitly recommends using ``spawn`` or ``forkserver`` for
multiprocessing in those cases.

This module centralizes a safer execution pattern:
  - prefer ``forkserver`` when available, else ``spawn``;
  - cap BLAS/OpenMP thread counts to 1 per worker process unless the user has
    already set them explicitly; and
  - surface abrupt worker death as a clear ``RuntimeError`` instead of hanging
    forever inside an ordered ``Pool.imap`` loop.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Sequence

from tqdm import tqdm

try:  # Python 3.7+
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
    except TypeError:  # Python < 3.9 does not support cancel_futures
        executor.shutdown(wait=wait)


def iter_parallel_results(
    worker_fn: Callable[[Any], Any],
    tasks: Sequence[Any] | Iterable[Any],
    *,
    num_workers: int,
    initializer: Callable[..., Any] | None = None,
    initargs: Sequence[Any] = (),
    desc: str,
    stage_name: str,
    preferred_start_method: str | None = None,
    show_progress: bool = True,
) -> Iterator[Any]:
    """Yield task results using a safer multiprocessing backend.

    Parameters
    ----------
    worker_fn:
        Top-level callable executed for each task.
    tasks:
        Iterable of picklable task payloads.
    num_workers:
        Requested worker-process count. Values < 1 are coerced to 1.
    initializer, initargs:
        Optional worker initializer executed once per worker process.
    desc:
        Progress label shown to the user.
    stage_name:
        Human-readable stage name used in error messages.
    preferred_start_method:
        Optional override. If invalid or unavailable, a safe default is chosen.
    show_progress:
        When False, suppress tqdm progress bars while preserving stage prints.
    """

    task_list = list(tasks)
    total = len(task_list)
    workers = max(1, int(num_workers or 1))

    if workers == 1:
        print(f"{stage_name}: running serially with 1 worker.")
        if initializer is not None:
            initializer(*tuple(initargs))
        if show_progress:
            with tqdm(
                total=total,
                desc=desc,
                file=sys.stdout,
                mininterval=5.0,
                disable=(total == 0),
                leave=True,
            ) as pbar:
                for task in task_list:
                    yield worker_fn(task)
                    pbar.update(1)
        else:
            for task in task_list:
                yield worker_fn(task)
        return

    start_method = choose_safe_start_method(preferred_start_method)
    if start_method in {"spawn", "forkserver"} and not _main_module_is_importable():
        print(
            f"{stage_name}: falling back to serial execution because multiprocessing start method "
            f"'{start_method}' requires an importable __main__ module."
        )
        if initializer is not None:
            initializer(*tuple(initargs))
        if show_progress:
            with tqdm(
                total=total,
                desc=desc,
                file=sys.stdout,
                mininterval=5.0,
                disable=(total == 0),
                leave=True,
            ) as pbar:
                for task in task_list:
                    yield worker_fn(task)
                    pbar.update(1)
        else:
            for task in task_list:
                yield worker_fn(task)
        return

    ctx = mp.get_context(start_method)
    changed_env = cap_worker_thread_env()

    print(
        f"{stage_name}: running with {workers} workers using multiprocessing start method "
        f"'{start_method}'."
    )
    if changed_env:
        print(
            f"{stage_name}: defaulted per-worker BLAS/OpenMP thread caps to 1 for: "
            f"{', '.join(changed_env)}"
        )

    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=initializer,
            initargs=tuple(initargs),
        )

        future_to_idx = {
            executor.submit(worker_fn, task): idx for idx, task in enumerate(task_list)
        }

        if show_progress:
            with tqdm(
                total=total,
                desc=desc,
                file=sys.stdout,
                mininterval=5.0,
                disable=(total == 0),
                leave=True,
            ) as pbar:
                for future in as_completed(future_to_idx):
                    task_idx = future_to_idx[future]
                    try:
                        result = future.result()
                    except BrokenProcessPool as exc:
                        _shutdown_executor(executor, wait=False, cancel_futures=True)
                        raise RuntimeError(
                            f"{stage_name} worker process crashed or terminated unexpectedly while using "
                            f"multiprocessing start method '{start_method}'. This usually indicates a "
                            "native-library failure inside a worker (for example XGBoost/SHAP/OpenMP) "
                            "or a hard out-of-memory kill."
                        ) from exc
                    except BaseException as exc:
                        _shutdown_executor(executor, wait=False, cancel_futures=True)
                        raise RuntimeError(
                            f"{stage_name} task {task_idx + 1}/{total} raised "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    pbar.update(1)
                    yield result
        else:
            for future in as_completed(future_to_idx):
                task_idx = future_to_idx[future]
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    _shutdown_executor(executor, wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"{stage_name} worker process crashed or terminated unexpectedly while using "
                        f"multiprocessing start method '{start_method}'. This usually indicates a "
                        "native-library failure inside a worker (for example XGBoost/SHAP/OpenMP) "
                        "or a hard out-of-memory kill."
                    ) from exc
                except BaseException as exc:
                    _shutdown_executor(executor, wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"{stage_name} task {task_idx + 1}/{total} raised "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                yield result

    finally:
        if executor is not None:
            try:
                _shutdown_executor(executor, wait=True, cancel_futures=False)
            except Exception:
                pass
