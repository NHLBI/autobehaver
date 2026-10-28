"""Carry current chamber inclusion and labels through filesystem discovery.

Existing outputs cannot override an explicit Autocropper exclusion.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from autobehaver.portable_paths import original_name

SELECTION_FILENAME = "_crop_selection.json"


def read_selection(directory):
    """Read validated chamber selections, returning an empty mapping when absent."""
    path = Path(directory) / SELECTION_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["recordings"]
        if payload.get("version") != 1 or not isinstance(records, dict):
            raise ValueError("unsupported selection format")
        for key, record in records.items():
            if not key or not isinstance(record, dict) or not isinstance(record.get("included"), bool):
                raise ValueError(f"invalid chamber selection: {key!r}")
        return records
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid chamber selection file {path}: {exc}") from exc


def write_selection(directory, records):
    """Merge and atomically publish chamber selections for a stage directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from autobehaver.file_locks import FileLock
    lock_path = directory / ".crop-selection.lock"
    if lock_path.is_dir():
        raise RuntimeError(f"Legacy crop-selection lock needs verified recovery: {lock_path}")
    lock = FileLock(lock_path, timeout_sec=120)
    lock.acquire()
    temporary = None
    try:
        combined = read_selection(directory)
        combined.update(records)
        path = directory / SELECTION_FILENAME
        temporary = directory / (".crop-selection-" + uuid.uuid4().hex[:12] + ".tmp")
        temporary.write_text(json.dumps({"version": 1, "recordings": combined}, indent=2), encoding="utf-8")
        # Windows readers and file scanners can briefly prevent atomic replacement.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        finally:
            lock.release()



def selection_for_path(path, records):
    """Resolve the longest matching recording identity through the filename map."""
    name = re.sub(r"_+", "_", original_name(path))
    matches = [(key, record) for key, record in records.items()
               if re.search(r"(?:^|_)" + re.escape(re.sub(r"_+", "_", key)) + r"(?=DLC|_|\.|$)", name)]
    if not matches:
        return None
    # A full recording/chamber name takes precedence over any shorter legacy key.
    key, record = max(matches, key=lambda item: len(item[0]))
    return {**record, "recording_id": key}


def filter_included_paths(paths):
    """Keep currently included recordings with their current label prefixes.

    Retain files without a matching selection record for compatibility.
    """
    selections = {}
    kept = []
    for path in paths:
        parent = Path(path).parent
        if parent not in selections:
            selections[parent] = read_selection(parent)
        record = selection_for_path(path, selections[parent])
        current_label = record.get("label_prefix") if record is not None else None
        current_name = True
        if current_label:
            prefix = re.sub(r"_+", "_", str(current_label).strip()).rstrip("_")
            stem = prefix + "_" + re.sub(r"_+", "_", record["recording_id"])
            current_name = bool(re.match(re.escape(stem) + r"(?=DLC|_|\.|$)",
                                         re.sub(r"_+", "_", original_name(path))))
        if record is None or (record["included"] and current_name):
            kept.append(path)
    return kept


def propagate_selection(input_directories, output_directory):
    """Merge input selections into the output directory, rejecting conflicting records."""
    records = {}
    for directory in sorted(set(map(Path, input_directories))):
        for key, record in read_selection(directory).items():
            if key in records and records[key] != record:
                raise ValueError(f"Conflicting chamber selections for {key!r} across input folders")
            records[key] = record
    if records:
        write_selection(output_directory, records)
