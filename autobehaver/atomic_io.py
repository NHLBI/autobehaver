"""Publish pipeline CSV and JSON files through temporary sibling files."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    from .completion import file_digest
    return file_digest(path, chunk_size=chunk_size)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".ab-{hashlib.sha256(destination.name.encode('utf-8')).hexdigest()[:10]}-",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    return Path(name)


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    """Retry short Windows sharing conflicts without hiding persistent failures."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def atomic_write_dataframe_csv(frame: pd.DataFrame, destination: str | Path, **to_csv_kwargs: Any) -> None:
    """Write a CSV beside its destination and publish it with one replace."""

    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        frame.to_csv(temporary, **to_csv_kwargs)
        from .completion import invalidate_digest
        invalidate_digest(destination)
        atomic_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Any, destination: str | Path) -> None:
    """Write deterministic JSON and publish it with one replace."""

    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary, destination)
        from .completion import invalidate_digest
        invalidate_digest(destination)
    finally:
        temporary.unlink(missing_ok=True)
