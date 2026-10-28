"""Limit native-library and decoder threads within Autocropper workers."""

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
    """Set native-library thread limits and return the requested positive cap.

    Preserve existing environment settings unless overwrite is true. Apply the
    OpenCV cap separately when configure_cv2 is true and OpenCV is available.
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


def open_video_capture(video_path):
    """Open a movie with a bounded FFmpeg decoder thread pool.

    OpenCV's image-operation thread limit does not cap VideoCapture decoding;
    CAP_PROP_N_THREADS must be supplied when opening the capture.
    """
    import cv2

    threads = _positive_int(os.environ.get("OPENCV_FOR_THREADS_NUM"), default=1)
    return cv2.VideoCapture(
        str(video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads]
    )
