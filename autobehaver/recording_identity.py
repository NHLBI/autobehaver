"""Movie-pair identities shared by review and desktop job preparation."""
from __future__ import annotations

import os
import re
from .portable_paths import original_name


def derive_pair_key(movie_path: str) -> str:
    """Remove timestamps and collapse movie0/movie1 using the established convention."""
    base = original_name(movie_path)
    stem, _ = os.path.splitext(base)
    stem = re.sub(r"(?<!\d)\d{8}[_-]?\d{6}(?:AM|PM)?(?![A-Za-z0-9])", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?<![A-Za-z0-9])movie[01](?![A-Za-z0-9])", "movie", stem, flags=re.IGNORECASE)
    stem = re.sub(r"__+", "_", stem)
    stem = re.sub(r"--+", "-", stem)
    stem = re.sub(r"[_-]{2,}", "_", stem)
    return stem.strip("_- ")
