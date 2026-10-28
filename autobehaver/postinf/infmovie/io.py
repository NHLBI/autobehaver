"""Match inference CSVs to source movies and bound InfMovie decoder and worker counts.

Matching checks original and timestamp-stripped stems.
"""

from __future__ import annotations

import os
import math
from autobehaver.portable_paths import original_name
import re


_TIMESTAMP_TOKEN_RE = re.compile(r"(?:^|_)(\d{14})(AM|PM)(?:$|_)")


def resolve_movie_fps(reported, fallback=40.0):
    """Use one finite positive source timebase for planning and rendering."""
    fallback = float(fallback)
    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError('default_movie_fps must be finite and positive')
    reported = float(reported or 0.0)
    return reported if math.isfinite(reported) and reported > 0 else fallback


def open_video_capture(path, threads=1):
    """Bound decoder threads separately from OpenCV's image-operation threads."""
    import cv2
    if hasattr(cv2, 'CAP_PROP_N_THREADS') and hasattr(cv2, 'CAP_FFMPEG'):
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_N_THREADS, max(1, int(threads))])
    return cv2.VideoCapture(str(path))


def worker_count(requested, tasks):
    """Respect runnable tasks and the allocation/affinity CPU limit."""
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    try:
        limit = max(1, int(allocated)) if allocated else len(os.sched_getaffinity(0))
    except (ValueError, AttributeError):
        limit = os.cpu_count() or 1
    return min(max(1, int(requested)), max(1, int(tasks)), max(1, limit))


def _strip_timestamp_tokens(name: str) -> str:
    """Remove timestamp tokens (14 digits + AM/PM) from an underscore-delimited name."""

    # Example: "..._20250819015233PM_chamber6" -> "..._chamber6"
    out = _TIMESTAMP_TOKEN_RE.sub("_", name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def find_movie_file(movie_dir: str, filename_part: str):
    """Find a matching movie file in a folder given a filename substring.

    This keeps inference CSV stems aligned with source movie filenames.
    """

    filename_part = (filename_part or "").strip()
    filename_part_no_ts = _strip_timestamp_tokens(filename_part) if filename_part else ""

    exact, fallback = [], []
    def matches(needle, haystack):
        return bool(needle) and bool(re.search(r"(?:^|_)" + re.escape(needle) + r"(?:_|$)", haystack, re.IGNORECASE))
    for file in sorted(os.listdir(movie_dir)):
        if not file.lower().endswith(".mp4"):
            continue

        raw_base = os.path.splitext(original_name(os.path.join(movie_dir,file)))[0]
        raw_base_no_ts = _strip_timestamp_tokens(raw_base)

        # Prefer strong matches that preserve the timestamp when present.
        if matches(filename_part, raw_base):
            exact.append(os.path.join(movie_dir, file))

        # Remove timestamp tokens from both sides.
        if matches(filename_part_no_ts, raw_base_no_ts):
            fallback.append(os.path.join(movie_dir, file))

    equality = [p for p in exact if os.path.splitext(original_name(p))[0].lower() == filename_part.lower()]
    candidates = equality or exact or fallback
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous movie identity for {filename_part!r}: {candidates}")
    if candidates:
        return candidates[0]

    return None
