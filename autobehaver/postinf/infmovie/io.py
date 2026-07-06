"""I/O and file matching helpers for infmovie.

The inference-movie overlay workflow needs to match inference CSVs to the
corresponding source .mp4 movies.

Inference CSV filenames may include timestamp tokens that also appear in the
source movie filenames. Matching therefore checks both the raw basenames and
timestamp-stripped versions of both names.
"""

from __future__ import annotations

import os
import re


_TIMESTAMP_TOKEN_RE = re.compile(r"(?:^|_)(\d{14})(AM|PM)(?:$|_)")


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

    # Deterministic iteration order makes debugging easier.
    for file in sorted(os.listdir(movie_dir)):
        if not file.endswith(".mp4"):
            continue

        raw_base = os.path.splitext(file)[0]
        raw_base_no_ts = _strip_timestamp_tokens(raw_base)

        # Prefer strong matches that preserve the timestamp when present.
        if filename_part and filename_part in raw_base:
            return os.path.join(movie_dir, file)

        # Remove timestamp tokens from both sides.
        if filename_part_no_ts and filename_part_no_ts in raw_base_no_ts:
            return os.path.join(movie_dir, file)

    return None
