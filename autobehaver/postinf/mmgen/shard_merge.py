"""Verify and stream MMGen shard stores into a consolidated HDF5 store.

Reject active or overlapping shards and publish the verified store with its calculation
identity.
"""

from __future__ import annotations

from autobehaver.progress import progress

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from autobehaver.atomic_io import atomic_write_json, sha256_file
from .integrity import STORE_IDENTITY_FILENAME, content_hash, validate_commits, read_store_identity
from .hdf_store import HDF_KEYS, get_done_source_files, hdf_safe_append, open_store
from .recording_commits import append_recording_commits, read_recording_commits
from .locks import shard_lock_active
from .cohort import read_shard_cohort, verify_shard_membership


def _active_shard_locks(
    *,
    shards_root: str,
    num_shards: int,
    work_dirname: str = "_mmgen_work",
    lock_dirname: str = "_mmgen_shard.lock",
) -> List[str]:
    """Return shard locks held by live writers under the shard root.

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
        if shard_lock_active(lock_dir):
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
    cohort = read_shard_cohort(root, int(num_shards))
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

        identity = read_store_identity(shard_dir / work_dirname)
        manifest_path = shard_dir / work_dirname / 'SHARD_MANIFEST.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None
        verify_shard_membership(cohort, i, markers=[manifest, payload], sources=identity['sources'])
        if payload.get('temp_store_sha256') != sha256_file(store_path):
            raise RuntimeError(f'Changed shard store: {store_path}')
        if payload.get('calculation_sha256') != identity['calculation_sha256'] or payload.get('sources_sha256') != identity['sources_sha256']:
            raise RuntimeError(f'Changed shard identity: {shard_dir}')
        done = validate_commits(store_path, identity)
        if done != set(identity['sources']) or len(done) != int(payload.get('n_committed_files', -1)):
            raise RuntimeError(f'Incomplete shard source inventory: {shard_dir}')

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
    verified_store_identities: list[tuple[str, str]] | None = None,
) -> str:
    """Verify and merge completed shard stores using bounded table reads.

    Copy and verify an existing destination as a timestamped backup before atomically
    replacing it. Publish the merged calculation identity for finalization checks.
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
    # Probe actual ownership; a persistent OS lock file is harmless after exit.
    active_locks = _active_shard_locks(shards_root=disc.shards_root, num_shards=resolved_num_shards)
    if active_locks:
        preview = "\n".join(f"  - {p}" for p in active_locks[:20])
        raise RuntimeError(
            "[mmgen-merge] Active shard lock(s) detected. One or more shard jobs may still be running, "
            "or an older directory lock has an unverified owner.\n\n"
            "Held locks found:\n"
            f"{preview}\n\n"
            "Wait for live shard jobs to finish. Completed-shard records are still required before merge."
        )

    # Safety: require all requested shards to have completed cleanly before merge.
    validate_shard_done_markers(
        shards_root=disc.shards_root,
        num_shards=resolved_num_shards,
        require_full_commit=True,
    )

    verified_shards = [(str(path), sha256_file(path)) for path in progress(store_paths, desc='MMGen checking shard hashes')]

    # Safety: prevent accidental overlap.
    _check_no_duplicate_source_files(store_paths)

    merged_path = os.path.join(output_folder, merged_store_name)

    # Validate all shard calculation identities and collect exact source content.
    calculation = None
    sources = {}
    for path in store_paths:
        identity = read_store_identity(Path(path).parent/'_mmgen_work')
        if calculation is None:
            calculation = identity['calculation']
        elif calculation != identity['calculation']:
            raise RuntimeError('Shard scientific settings or source code differ')
        overlap = set(sources).intersection(identity['sources'])
        if overlap:
            raise RuntimeError(f'Duplicate shard sources: {sorted(overlap)[:10]}')
        sources.update(identity['sources'])
    cohort = read_shard_cohort(disc.shards_root, resolved_num_shards)
    if set(sources) != {record['source_file'] for record in cohort['files']}:
        raise RuntimeError('Shard stores do not cover the complete frozen cohort.')
    merged_identity = {'calculation':calculation, 'calculation_sha256':content_hash(calculation),
                       'sources':sources, 'sources_sha256':content_hash(sources)}
    tmp_path = merged_path + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        shutil.copy2(store_paths[0], tmp_path)
        with open_store(tmp_path, mode='a') as dst:
            for src_path in progress(store_paths[1:], desc='MMGen merging shard stores'):
                with open_store(src_path, mode='r') as src:
                    for key in HDF_KEYS.values():
                        if key == HDF_KEYS['video_lengths_tmp'] or key not in src:
                            continue
                        # Never append a full-read fallback after a partial stream.
                        for chunk in src.select(key, iterator=True, chunksize=int(chunk_rows)):
                            if chunk is not None and not chunk.empty:
                                hdf_safe_append(dst, key, chunk)
                    append_recording_commits(dst, read_recording_commits(src))
            dst.flush(fsync=True)
        if validate_commits(tmp_path, merged_identity) != set(sources):
            raise RuntimeError('Merged store does not contain exact verified sources')
        if os.path.exists(merged_path):
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            backup = Path(output_folder)/f'{Path(merged_store_name).stem}_premerge_{stamp}.h5'
            shutil.copy2(merged_path, backup)
            if sha256_file(backup) != sha256_file(merged_path):
                raise RuntimeError('Merged store backup verification failed')
        # An interruption between store and identity leaves an identity mismatch;
        # finalize refuses reuse rather than accepting a mixed generation.
        os.replace(tmp_path, merged_path)
        identity_path = Path(output_folder)/'_mmgen_work'/STORE_IDENTITY_FILENAME
        atomic_write_json(merged_identity, identity_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if verified_store_identities is not None:
        verified_store_identities.extend(verified_shards)
        verified_store_identities.append((merged_path, sha256_file(merged_path)))
    return merged_path
