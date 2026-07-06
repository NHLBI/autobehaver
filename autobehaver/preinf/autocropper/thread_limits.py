"""Thread-limit helpers for autocropper.

The autocropper stage fans out over Python processes, chamber-crop threads,
OpenCV decoders, ffmpeg subprocesses, and native math libraries. On shared HPC
nodes that can accidentally create far more runnable threads than Slurm allocated.
This module centralizes conservative thread caps before heavy libraries are used.
"""

from __future__ import annotations

import os
from typing import Iterable


_NATIVE_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def _positive_int(value: object, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(default))


def configure_native_threading(
    native_threads: int | str | None = 1,
    *,
    env_vars: Iterable[str] = _NATIVE_THREAD_ENV_VARS,
    overwrite: bool = False,
    configure_cv2: bool = True,
) -> int:
    """Set conservative native-thread limits and return the effective value.

    Parameters
    ----------
    native_threads:
        Number of native threads to allow for libraries such as OpenMP, MKL,
        OpenBLAS, and OpenCV. Values below 1 are coerced to 1.
    overwrite:
        If False, existing environment variables are left alone. This lets users
        override the cap from the shell. If True, this function enforces the cap.
    configure_cv2:
        If True, also call ``cv2.setNumThreads`` when OpenCV is importable.
    """

    n_threads = _positive_int(native_threads, default=1)
    n_text = str(n_threads)

    for name in env_vars:
        if overwrite or not os.environ.get(name):
            os.environ[name] = n_text

    if configure_cv2:
        try:
            import cv2  # type: ignore

            try:
                cv2.setNumThreads(n_threads)
            except Exception:
                pass
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            # OpenCV may not be importable in tests that only exercise parsing.
            pass

    return n_threads
