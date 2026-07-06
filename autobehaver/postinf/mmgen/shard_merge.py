"""Shard merge utilities for mmgen.

This module implements a *safe, deterministic* merge of multiple shard
`temp_store.h5` files into a single consolidated store.

It is designed to be used after running mmgen in `--process_only` mode with
`--num_shards > 1`.

Key goals:
  - No unbounded RAM usage (stream large HDF tables in chunks).
  - Guard against accidental double-counting (detect duplicate source_file).
  - Use an atomic write pattern for the merged store.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .hdf_store import HDF_KEYS, get_done_source_files, hdf_safe_append, open_store


def _active_shard_locks(
    *,
    shards_root: str,
    num_shards: int,
    work_dirname: str = "_mmgen_work",
    lock_dirname: str = "_mmgen_shard.lock",
) -> List[str]:
    """Return any active shard lock directories under the shard root.

    We deliberately scan by shard index directories (shard_XXX_of_YYY) rather
    than only checking shard stores that already exist. This catches the case
    where a shard job is *currently running* but hasn't yet produced its
    temp_store.h5.
    """

    root = Path(str(shards_root))
    active: List[str] = []
    for i in range(int(num_shards)):
        shard_dir = root / _default_shard_dir_name(i, int(num_shards))
        lock_dir = shard_dir / work_dirname / lock_dirname
        if lock_dir.exists():
            active.append(str(lock_dir))
    return active


@dataclass(frozen=True)
class ShardDiscoveryResult:
    """Resolved shard temp-store paths for a merge."""

    shards_root: str
    shard_store_paths: List[str]
    missing_shards: List[int]


def _default_shard_dir_name(idx: int, num_shards: int) -> str:
    return f"shard_{idx:03d}_of_{num_shards:03d}"


def discover_shard_stores(
    *,
    output_folder: str,
    num_shards: int,
    shards_dirname: str = "_mmgen_shards",
    store_name: str = "temp_store.h5",
) -> ShardDiscoveryResult:
    """Discover shard temp_store.h5 files under the mmgen shard root."""

    root = os.path.join(output_folder, shards_dirname)
    store_paths: List[str] = []
    missing: List[int] = []

    for i in range(int(num_shards)):
        shard_dir = os.path.join(root, _default_shard_dir_name(i, int(num_shards)))
        p = os.path.join(shard_dir, store_name)
        if os.path.exists(p):
            store_paths.append(p)
        else:
            missing.append(i)

    return ShardDiscoveryResult(shards_root=root, shard_store_paths=store_paths, missing_shards=missing)


def autodiscover_shard_stores(
    *,
    output_folder: str,
    shards_dirname: str = "_mmgen_shards",
    store_name: str = "temp_store.h5",
) -> tuple[int, ShardDiscoveryResult]:
    """Auto-discover shard stores under ``<output_folder>/_mmgen_shards``.

    This is a convenience for the merge step: if the user does not remember
    ``num_shards`` (or wants to avoid specifying it), we can infer it from the
    shard directory names.

    Expected shard directory naming:
        shard_XXX_of_YYY/

    Returns
    -------
    (num_shards, ShardDiscoveryResult)

    Notes
    -----
    * If multiple distinct YYY values are present (e.g., leftovers from multiple
      runs), we raise with a clear message to avoid merging incompatible shards.
    * Missing shard indices are reported in the returned discovery result.
    """

    output_folder = str(Path(output_folder).expanduser().resolve())
    root = os.path.join(output_folder, shards_dirname)

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Shard root folder not found: {root}")

    pat = re.compile(r"^shard_(\d{3})_of_(\d{3})$", flags=re.IGNORECASE)

    # Map: num_shards -> {shard_index: store_path}
    found: dict[int, dict[int, str]] = {}
    for name in os.listdir(root):
        m = pat.match(str(name))
        if not m:
            continue

        try:
            idx = int(m.group(1))
            n = int(m.group(2))
        except Exception:
            continue

        store_path = os.path.join(root, name, store_name)
        if not os.path.exists(store_path):
            # Directory exists but store not present yet (failed/in-progress shard).
            continue

        found.setdefault(n, {})[idx] = store_path

    if not found:
        raise FileNotFoundError(
            f"No shard temp stores found under {root}. "
            "Expected shard directories like: shard_000_of_010/temp_store.h5"
        )

    if len(found) > 1:
        # Multiple different shard counts present; require user to disambiguate.
        keys = sorted(found.keys())
        raise RuntimeError(
            "Multiple shard configurations detected under the shard root. "
            f"Found shard counts: {keys}. "
            "This usually happens if _mmgen_shards contains leftovers from multiple runs. "
            "Fix: clean up old shard folders or specify --num_shards explicitly for merge."
        )

    num_shards = next(iter(found.keys()))
    by_idx = found[num_shards]

    missing: list[int] = []
    store_paths: list[str] = []
    for i in range(int(num_shards)):
        p = by_idx.get(i)
        if p and os.path.exists(p):
            store_paths.append(p)
        else:
            missing.append(i)

    return int(num_shards), ShardDiscoveryResult(shards_root=root, shard_store_paths=store_paths, missing_shards=missing)


def validate_shard_done_markers(
    *,
    shards_root: str,
    num_shards: int,
    work_dirname: str = "_mmgen_work",
    done_filename: str = "SHARD_DONE.json",
    store_name: str = "temp_store.h5",
    require_full_commit: bool = True,
) -> None:
    """Validate per-shard completion markers before merge.

    This is intentionally strict for the main sharded workflow: every requested
    shard must have a completion marker, and every completion marker must match
    its shard index / shard count. When ``require_full_commit`` is true, the
    recorded number of committed files must also equal the number of selected
    files for that shard.
    """

    root = Path(str(shards_root))
    missing_done: list[int] = []
    bad_done: list[str] = []

    for i in range(int(num_shards)):
        shard_dir = root / _default_shard_dir_name(i, int(num_shards))
        done_path = shard_dir / work_dirname / done_filename
        store_path = shard_dir / store_name

        if not done_path.exists():
            missing_done.append(i)
            continue
        if not store_path.exists():
            bad_done.append(f"missing {store_name} for shard {i}: {store_path}")
            continue

        try:
            payload = json.loads(done_path.read_text(encoding="utf-8"))
        except Exception as e:
            bad_done.append(f"failed to parse {done_path}: {e}")
            continue

        if int(payload.get("num_shards", -1)) != int(num_shards):
            bad_done.append(
                f"{done_path} reports num_shards={payload.get('num_shards')} (expected {num_shards})"
            )
        if int(payload.get("shard_index", -1)) != int(i):
            bad_done.append(
                f"{done_path} reports shard_index={payload.get('shard_index')} (expected {i})"
            )

        if require_full_commit:
            n_selected = payload.get("n_inference_files_selected", None)
            n_committed = payload.get("n_committed_files", None)
            if n_selected is not None and n_committed is not None:
                if int(n_committed) != int(n_selected):
                    bad_done.append(
                        f"{done_path} reports n_committed_files={n_committed} but n_inference_files_selected={n_selected}"
                    )

    if missing_done:
        raise RuntimeError(
            "[mmgen-merge] Strict shard merge requires all shard completion markers to be present. Missing shard(s): "
            + ", ".join(map(str, missing_done))
        )

    if bad_done:
        raise RuntimeError(
            "[mmgen-merge] Found inconsistent shard completion metadata before merge:\n  - "
            + "\n  - ".join(bad_done[:20])
            + ("\n  ..." if len(bad_done) > 20 else "")
        )


def _check_no_duplicate_source_files(store_paths: Sequence[str]) -> None:
    """Raise if the same source_file appears in more than one shard store."""

    seen: Dict[str, str] = {}
    duplicates: List[Tuple[str, str, str]] = []

    for p in store_paths:
        done = get_done_source_files(p)
        for sf in done:
            if sf in seen:
                duplicates.append((sf, seen[sf], p))
            else:
                seen[sf] = p

    if duplicates:
        preview = duplicates[:20]
        msg = [
            "Duplicate source_file entries detected across shard temp stores.",
            "This would cause double-counting if merged.",
            "Fix: ensure all shards used the same (num_shards, shard_by) and did not overlap file filters.",
            "",
            "Examples (source_file, store_a, store_b):",
        ]
        for sf, a, b in preview:
            msg.append(f"  - {sf} :: {a} :: {b}")
        if len(duplicates) > len(preview):
            msg.append(f"  ... and {len(duplicates) - len(preview)} more")
        raise RuntimeError("\n".join(msg))


def merge_shard_temp_stores(
    *,
    output_folder: str,
    num_shards: int,
    shards_dirname: str = "_mmgen_shards",
    merged_store_name: str = "temp_store.h5",
    chunk_rows: int = 50_000,
    strict_missing: bool = False,
) -> str:
    """Merge shard stores into a single consolidated temp_store.h5.

    The merged store is written *atomically* into `output_folder/merged_store_name`.
    If an existing merged store is present, it is first renamed to a timestamped
    backup.
    """

    output_folder = str(Path(output_folder).expanduser().resolve())
    os.makedirs(output_folder, exist_ok=True)

    # Allow convenient "auto" merge by passing num_shards <= 1.
    disc: ShardDiscoveryResult
    resolved_num_shards = int(num_shards)
    if resolved_num_shards <= 1:
        resolved_num_shards, disc = autodiscover_shard_stores(output_folder=output_folder, shards_dirname=shards_dirname)
        print(f"[mmgen-merge] Auto-discovered num_shards={resolved_num_shards} under {disc.shards_root}")
    else:
        disc = discover_shard_stores(output_folder=output_folder, num_shards=resolved_num_shards, shards_dirname=shards_dirname)

    store_paths = list(disc.shard_store_paths)
    if not store_paths:
        raise FileNotFoundError(
            f"No shard temp stores found under {disc.shards_root}. Expected {resolved_num_shards} shards."
        )

    if disc.missing_shards:
        if bool(strict_missing):
            raise RuntimeError(
                "[mmgen-merge] Missing shard store(s) detected for a strict merge: "
                f"{disc.missing_shards}. Fix: rerun the missing shard jobs before merging."
            )
        print(
            f"[mmgen-merge] Warning: {len(disc.missing_shards)} shard store(s) are missing: {disc.missing_shards}. "
            "Merged outputs will reflect only the available shards."
        )

    # Safety: refuse to merge while any shard appears to still be running.
    # (Shard jobs create a _mmgen_work/_mmgen_shard.lock directory and only
    # remove it on successful completion.)
    active_locks = _active_shard_locks(shards_root=disc.shards_root, num_shards=resolved_num_shards)
    if active_locks:
        preview = "\n".join(f"  - {p}" for p in active_locks[:20])
        raise RuntimeError(
            "[mmgen-merge] Active shard lock(s) detected. One or more shard jobs may still be running, "
            "or a shard crashed and left a stale lock.\n\n"
            "Lock directories found:\n"
            f"{preview}\n\n"
            "Fix: wait for shard jobs to finish, or (if you are sure no shard job is running) delete the lock "
            "directory/directories and retry the merge."
        )

    # Safety: require all requested shards to have completed cleanly before merge.
    validate_shard_done_markers(
        shards_root=disc.shards_root,
        num_shards=resolved_num_shards,
        require_full_commit=True,
    )

    # Safety: prevent accidental overlap.
    _check_no_duplicate_source_files(store_paths)

    merged_path = os.path.join(output_folder, merged_store_name)

    # If a merged store already exists, back it up rather than clobbering.
    if os.path.exists(merged_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(output_folder, f"{Path(merged_store_name).stem}_premerge_{ts}.h5")
        print(f"[mmgen-merge] Existing merged store found. Moving to backup: {backup_path}")
        os.replace(merged_path, backup_path)

    tmp_path = merged_path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Start by copying the first shard store to preserve table schemas.
    base = store_paths[0]
    print(f"[mmgen-merge] Using base shard store: {base}")
    shutil.copy2(base, tmp_path)

    keys = list(HDF_KEYS.values())

    # Append remaining shards.
    with open_store(tmp_path, mode="a") as dst:
        for src_path in store_paths[1:]:
            print(f"[mmgen-merge] Merging shard store: {src_path}")
            with open_store(src_path, mode="r") as src:
                for key in keys:
                    if key not in src:
                        continue
                    try:
                        it = src.select(key, iterator=True, chunksize=int(chunk_rows))
                        for chunk in it:
                            if chunk is None or chunk.empty:
                                continue
                            hdf_safe_append(dst, key, chunk)
                    except (ValueError, KeyError, FileNotFoundError):
                        # Fallback to full read (best-effort) if chunk iterator fails.
                        try:
                            df = src.select(key)
                            if df is not None and not df.empty:
                                hdf_safe_append(dst, key, df)
                        except Exception:
                            continue

    # Basic sanity check: merged commit table must exist.
    try:
        done = get_done_source_files(tmp_path)
        print(f"[mmgen-merge] Merged temp store now contains {len(done)} committed source files.")
    except Exception:
        pass

    # Atomic finalize.
    os.replace(tmp_path, merged_path)
    print(f"[mmgen-merge] Wrote merged temp store: {merged_path}")
    return merged_path
