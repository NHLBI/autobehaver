"""Assign MMGen files deterministically to independent processing shards.

Animal-row assignment keeps all recordings of an animal together for matrix merging and
append workflows.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, List

from .naming import collapse_individual_id, parse_group_and_individual
from autobehaver.portable_paths import original_name


def _stable_hash_int(text: str) -> int:
    """Return a stable non-negative integer hash for a string.

    We deliberately do *not* use Python's built-in `hash()` because it is salted
    per-process (PYTHONHASHSEED) and therefore not stable across runs.
    """

    b = text.encode("utf-8", errors="ignore")
    # md5 is sufficient here: we only need deterministic bucketing, not crypto.
    d = hashlib.md5(b).digest()
    # Use 8 bytes => 64-bit int
    return int.from_bytes(d[:8], byteorder="big", signed=False)


def assign_shard(filename: str, *, num_shards: int, strategy: str = "hash") -> int:
    """Assign a filename to a shard index in [0, num_shards-1].

    Parameters
    ----------
    filename:
        Inference CSV basename (preferred) or any stable identifier string.
    num_shards:
        Total number of shards.
    strategy:
        One of:
          - "hash" (default): md5(filename.lower()) % num_shards
          - "trial": extract trial number from *_trial<NUM>*; fallback to hash.
          - "rpi": extract rpi number from *rpi-<NUM>*; fallback to hash.
    """

    if num_shards <= 1:
        return 0

    s = str(strategy or "hash").strip().lower()
    name = str(filename)

    if s == "trial":
        m = re.search(r"(?:^|_)trial(\d+)", name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)) % int(num_shards)
            except Exception:
                pass

    if s == "rpi":
        m = re.search(r"rpi-(\d+)", name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)) % int(num_shards)
            except Exception:
                pass

    # Default / fallback: hash
    return _stable_hash_int(name.lower()) % int(num_shards)


def select_files_for_shard(
    files: Iterable[str],
    *,
    num_shards: int,
    shard_index: int,
    strategy: str = "hash",
    inference_csv_dir: str = "",
) -> List[str]:
    """Select the files assigned to one shard.

    For row_identity, distribute sorted animal identities round-robin and keep all of an
    animal's recordings together; other strategies use the original logical filename.
    """

    if num_shards <= 1:
        return list(files)

    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards-1}], got {shard_index}")

    files_list = [str(f) for f in files]

    # Row-identity-aware sharding is especially useful for append/update mode.
    # We deterministically map *unique collapsed row identities* to shards in a
    # dense round-robin assignment so that, when num_shards <= n_rows, every
    # shard receives at least one row and all files for a row stay together.
    if str(strategy).strip().lower() == "row_identity":
        row_by_file: dict[str, str] = {}
        for f in files_list:
            try:
                _group, individual_id, _sex = parse_group_and_individual(os.path.join(inference_csv_dir, str(f)))
                row_identity = collapse_individual_id(str(individual_id))
            except Exception:
                # Fall back to the raw basename if parsing fails.
                row_identity = str(f)
            row_by_file[str(f)] = str(row_identity)
        assignments = assign_rows_to_shards(row_by_file, num_shards=int(num_shards))
        return [f for f in files_list if assignments[f] == int(shard_index)]

    out: List[str] = []
    for f in files_list:
        resolved = original_name(os.path.join(inference_csv_dir, str(f)))
        if assign_shard(resolved, num_shards=num_shards, strategy=strategy) == int(shard_index):
            out.append(str(f))
    return out


def assign_rows_to_shards(row_by_file: dict[str, str], *, num_shards: int) -> dict[str, int]:
    """Assign the complete sorted animal cohort once, keeping recordings together."""
    rows = sorted(set(row_by_file.values()))
    row_to_shard = {row: index % int(num_shards) for index, row in enumerate(rows)}
    return {name: row_to_shard[row] for name, row in row_by_file.items()}
