"""Deterministic, communication-free sharding utilities for mmgen.

The intent is to support *blind sharding* across multiple HPC jobs/nodes:

  - Each shard independently discovers the same universe of input files.
  - Each shard deterministically selects its subset based only on filenames.
  - No coordination / shared state is required.

This allows you to run (for example) 10 parallel jobs:

    python -m autobehaver.postinf.mmgen --process_only --num_shards 10 --shard_index 0
    ...
    python -m autobehaver.postinf.mmgen --process_only --num_shards 10 --shard_index 9

and later merge those shard stores.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from .naming import collapse_individual_id, parse_group_and_individual


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
) -> List[str]:
    """Filter a list of filenames down to those assigned to this shard."""

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
        unique_rows: list[str] = []
        seen_rows: set[str] = set()
        for f in files_list:
            try:
                _group, individual_id, _sex = parse_group_and_individual(str(f))
                row_identity = collapse_individual_id(str(individual_id))
            except Exception:
                # Fall back to the raw basename if parsing fails.
                row_identity = str(f)
            row_by_file[str(f)] = str(row_identity)
            if str(row_identity) not in seen_rows:
                seen_rows.add(str(row_identity))
                unique_rows.append(str(row_identity))

        unique_rows = sorted(unique_rows)
        row_to_shard = {rid: (i % int(num_shards)) for i, rid in enumerate(unique_rows)}
        return [f for f in files_list if row_to_shard.get(row_by_file.get(f, f), -1) == int(shard_index)]

    out: List[str] = []
    for f in files_list:
        if assign_shard(str(f), num_shards=num_shards, strategy=strategy) == int(shard_index):
            out.append(str(f))
    return out
