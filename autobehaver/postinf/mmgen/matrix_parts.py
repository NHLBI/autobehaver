"""Matrix-first shard helpers for mmgen.

This module implements the first milestone of the matrix-first redesign:
- finalize one existing shard-local temp store into a shard-local master matrix
- merge shard-local master matrices into the final MASTER_FEATURE_MATRIX.csv

The implementation deliberately reuses the existing mmgen finalization/math so the
resulting matrix stays aligned with the global finalize path while avoiding
one giant merged temp-store bottleneck for the matrix-only workflow.
"""

from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from autobehaver import __version__ as AUTOBEHAVER_VERSION

from .discovery import list_inference_csv_files
from .feature_matrix import reorder_master_matrix_columns, write_feature_units_manifest
from .hdf_store import cleanup_temp_store
from .locks import DirLock
from .runtime import MMGenRuntimeSettings
from .sharding import select_files_for_shard


SHARDS_DIRNAME_DEFAULT = "_mmgen_shards"
SHARD_MATRIX_DONE_FILENAME = "SHARD_MATRIX_DONE.json"
SHARD_MANIFEST_FILENAME = "SHARD_MANIFEST.json"
FINAL_MATRIX_MERGE_MANIFEST = "MATRIX_PARTS_MERGED.json"


def get_shard_output_dir(output_dir: str, num_shards: int, shard_index: int, *, shards_dirname: str = SHARDS_DIRNAME_DEFAULT) -> str:
    return os.path.join(output_dir, str(shards_dirname), f"shard_{int(shard_index):03d}_of_{int(num_shards):03d}")


def discover_shard_output_dirs(output_dir: str, num_shards: int | None = None, *, shards_dirname: str = SHARDS_DIRNAME_DEFAULT) -> list[Path]:
    shards_root = Path(output_dir) / str(shards_dirname)
    if not shards_root.exists():
        raise FileNotFoundError(f"Shard root not found: {shards_root}")
    if num_shards is not None and int(num_shards) > 1:
        shard_dirs = [Path(get_shard_output_dir(output_dir, int(num_shards), i, shards_dirname=shards_dirname)) for i in range(int(num_shards))]
        missing = [str(p) for p in shard_dirs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Expected shard directories are missing: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            )
        return shard_dirs
    pat = re.compile(r"^shard_(\d{3})_of_(\d{3})$")
    found: list[tuple[int, int, Path]] = []
    for child in shards_root.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if not m:
            continue
        found.append((int(m.group(1)), int(m.group(2)), child))
    if not found:
        raise FileNotFoundError(f"No shard directories found under {shards_root}")
    totals = sorted({total for _, total, _ in found})
    if len(totals) != 1:
        raise RuntimeError(f"Shard directories disagree on total shard count under {shards_root}: {totals}")
    total = totals[0]
    paths_by_idx = {idx: path for idx, _, path in found}
    missing = [i for i in range(total) if i not in paths_by_idx]
    if missing:
        raise RuntimeError("Missing shard directories for indices: " + ", ".join(str(i) for i in missing))
    return [paths_by_idx[i] for i in range(total)]


def _read_done_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_selected_files_from_manifest(
    manifest_path: Path,
    *,
    num_shards: int,
    shard_index: int,
    shard_by: str,
) -> list[str] | None:
    payload = _read_done_payload(manifest_path)
    if not isinstance(payload, dict):
        return None
    manifest_num_shards = payload.get("num_shards")
    manifest_shard_index = payload.get("shard_index")
    manifest_shard_by = payload.get("shard_by")
    if manifest_num_shards is not None and int(manifest_num_shards) != int(num_shards):
        raise RuntimeError(
            f"Shard manifest {manifest_path} was created for num_shards={manifest_num_shards}, but finalize requested num_shards={int(num_shards)}"
        )
    if manifest_shard_index is not None and int(manifest_shard_index) != int(shard_index):
        raise RuntimeError(
            f"Shard manifest {manifest_path} was created for shard_index={manifest_shard_index}, but finalize requested shard_index={int(shard_index)}"
        )
    if manifest_shard_by is not None and str(manifest_shard_by) != str(shard_by):
        raise RuntimeError(
            f"Shard manifest {manifest_path} was created for shard_by={manifest_shard_by}, but finalize requested shard_by={str(shard_by)}"
        )
    selected = payload.get("selected_files")
    if selected is None:
        return None
    if not isinstance(selected, list):
        raise RuntimeError(f"Shard manifest selected_files is not a list: {manifest_path}")
    return [str(f) for f in selected]




MATRIX_SCHEMA_VERSION = 1


def _canonical_matrix_runtime_settings(settings: dict | None) -> dict:
    if not isinstance(settings, dict):
        return {}
    excluded = {
        "num_workers",
        "input_batch_size",
        "keep_temp_files",
        "emit_master_matrix",
        "emit_full_outputs",
        "generate_quiver_plots",
        "run_transition_statistics",
        "random_seed",
    }
    return {k: settings[k] for k in sorted(settings.keys()) if k not in excluded}


def _matrix_merge_consistency_signature(*, matrix_done: dict, process_done: dict | None = None) -> dict:
    if not isinstance(matrix_done, dict):
        raise RuntimeError("Shard matrix completion marker is missing or invalid; cannot validate merge consistency")
    signature = {
        "matrix_schema_version": int(matrix_done.get("matrix_schema_version", -1)),
        "package_version": matrix_done.get("package_version"),
        "min_bout": matrix_done.get("min_bout"),
        "edge_skip": matrix_done.get("edge_skip"),
        "inf_tag": matrix_done.get("inf_tag"),
        "file_filter": matrix_done.get("file_filter"),
        "runtime_settings": _canonical_matrix_runtime_settings(matrix_done.get("runtime_settings")),
    }
    if isinstance(process_done, dict):
        signature["num_shards"] = process_done.get("num_shards")
        signature["shard_by"] = process_done.get("shard_by")
    return signature


def _acquire_or_fail(lock: DirLock, *, lock_dir: Path, error_header: str, metadata: dict) -> bool:
    try:
        lock.acquire(metadata=metadata)
        return True
    except Exception as e:
        raise RuntimeError(
            f"{error_header}\n"
            f"  lock: {lock_dir}\n\n"
            "If you are sure no relevant job is running, delete the lock directory and retry. "
            f"(Original error: {e})"
        )


def _validate_unique_individuals(df: pd.DataFrame, *, source_label: str) -> None:
    if "individual" not in df.columns:
        raise ValueError(f"{source_label} is missing required 'individual' column")
    dup_mask = df["individual"].astype(str).duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(df.loc[dup_mask, "individual"].astype(str).unique().tolist())
        raise ValueError(f"Duplicate row identities found in {source_label}: " + ", ".join(dupes[:10]) + (" ..." if len(dupes) > 10 else ""))


def _validate_required_matrix_columns(df: pd.DataFrame, *, source_label: str, required: tuple[str, ...] = ("individual", "group", "sex")) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Shard matrix part is missing required identifier columns in {source_label}: " + ", ".join(missing)
        )


def _require_completed_shard_processing(*, work_dir: Path, shard_dir: Path, expected_selected_files: int | None) -> dict:
    process_done_path = work_dir / "SHARD_DONE.json"
    process_done = _read_done_payload(process_done_path)
    if not isinstance(process_done, dict):
        raise RuntimeError(
            f"Shard-local matrix finalize requires a completed shard-processing marker before it can proceed: {process_done_path}"
        )

    selected_n = process_done.get("n_inference_files_selected")
    committed_n = process_done.get("n_committed_files")
    if selected_n is None or committed_n is None:
        raise RuntimeError(
            f"Shard processing completion marker is missing required counts: {process_done_path}"
        )

    selected_n = int(selected_n)
    committed_n = int(committed_n)
    if committed_n != selected_n:
        raise RuntimeError(
            "Shard processing did not complete all selected files before shard-local matrix finalize. "
            f"Shard {shard_dir} has n_inference_files_selected={selected_n} but n_committed_files={committed_n}."
        )

    if expected_selected_files is not None and int(expected_selected_files) != selected_n:
        raise RuntimeError(
            "Shard manifest / finalize request disagrees with completed shard-processing counts. "
            f"Shard {shard_dir} expected {int(expected_selected_files)} selected files, but SHARD_DONE.json reports {selected_n}."
        )
    return process_done


def finalize_shard_temp_store(*, inference_csv_dir: str, poscalc_dir: str, output_dir: str, min_bout: int, edge_skip: int, inf_tag: str, runtime_settings: MMGenRuntimeSettings, file_filter: str | None, num_shards: int, shard_index: int, shard_by: str, shards_dirname: str = SHARDS_DIRNAME_DEFAULT, caller_holds_shard_lock: bool = False) -> str | None:
    if int(num_shards) <= 1:
        raise ValueError("finalize_shard_temp_store requires num_shards > 1")
    shard_dir = Path(get_shard_output_dir(output_dir, int(num_shards), int(shard_index), shards_dirname=shards_dirname))
    temp_store_path = shard_dir / "temp_store.h5"
    if not temp_store_path.exists():
        raise FileNotFoundError(f"Shard temp store not found: {temp_store_path}")

    work_dir = shard_dir / "_mmgen_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    done_marker_path = work_dir / SHARD_MATRIX_DONE_FILENAME
    shard_manifest_path = work_dir / SHARD_MANIFEST_FILENAME

    shard_files = _load_selected_files_from_manifest(
        shard_manifest_path,
        num_shards=int(num_shards),
        shard_index=int(shard_index),
        shard_by=str(shard_by),
    )
    if shard_files is None:
        inf_files = list_inference_csv_files(inference_csv_dir)
        if file_filter is not None and str(file_filter).strip() != "":
            needle = str(file_filter).strip().lower()
            inf_files = [f for f in inf_files if needle in str(f).lower()]
        shard_files = select_files_for_shard(inf_files, num_shards=int(num_shards), shard_index=int(shard_index), strategy=str(shard_by))

    _require_completed_shard_processing(
        work_dir=work_dir,
        shard_dir=shard_dir,
        expected_selected_files=(None if shard_files is None else int(len(shard_files))),
    )

    if not shard_files:
        payload = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "shard_by": str(shard_by),
            "n_selected_files": 0,
            "n_matrix_rows": 0,
            "matrix_path": None,
        }
        done_marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if not bool(runtime_settings.keep_temp_files):
            cleanup_temp_store(str(temp_store_path), keep_temp_files=False)
        return None

    shard_lock_dir = work_dir / "_mmgen_shard.lock"
    shard_lock: DirLock | None = None
    shard_lock_acquired = False
    if not bool(caller_holds_shard_lock):
        shard_lock = DirLock(shard_lock_dir, timeout_sec=0.0)
        _acquire_or_fail(
            shard_lock,
            lock_dir=shard_lock_dir,
            error_header=(
                "[mmgen] Shard lifecycle lock already exists. Another shard processing/finalize job may already be running for this shard.\n"
                f"  shard: {int(shard_index)}/{int(num_shards)-1}"
            ),
            metadata={
                "kind": "mmgen_shard_lifecycle_finalize",
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "shard_by": str(shard_by),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": socket.gethostname(),
            },
        )
        shard_lock_acquired = True

    lock_dir = work_dir / "_mmgen_shard_matrix.lock"
    lock = DirLock(lock_dir, timeout_sec=0.0)
    _acquire_or_fail(
        lock,
        lock_dir=lock_dir,
        error_header=(
            "[mmgen] Shard-matrix finalize lock already exists. Another finalize job may already be running for this shard.\n"
            f"  shard: {int(shard_index)}/{int(num_shards)-1}"
        ),
        metadata={
            "kind": "mmgen_shard_matrix_finalize",
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "shard_by": str(shard_by),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
        },
    )

    try:
        from . import pipeline

        shard_runtime_settings = replace(runtime_settings, emit_master_matrix=True, emit_full_outputs=False)
        pipeline.run_post_inference_analysis(
            inference_csv_dir,
            poscalc_dir,
            str(shard_dir),
            int(min_bout),
            int(edge_skip),
            str(inf_tag),
            runtime_settings=shard_runtime_settings,
            resume=False,
            finalize_only=True,
            process_only=False,
            file_filter=None,
            num_shards=1,
            shard_index=0,
            shard_by="hash",
            selected_files=shard_files,
            drop_all_na_feature_columns=False,
            write_provenance_artifacts=False,
            spatial_in_memory_for_matrix=True,
            caller_holds_shard_lock=bool(caller_holds_shard_lock),
        )
        matrix_path = shard_dir / "MASTER_FEATURE_MATRIX.csv"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Shard finalization completed but shard matrix was not written: {matrix_path}")
        try:
            n_rows = int(len(pd.read_csv(matrix_path, usecols=["individual"])))
        except Exception:
            n_rows = -1
        payload = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "package_version": str(AUTOBEHAVER_VERSION),
            "matrix_schema_version": int(MATRIX_SCHEMA_VERSION),
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "shard_by": str(shard_by),
            "n_selected_files": int(len(shard_files)),
            "n_matrix_rows": int(n_rows),
            "matrix_path": str(matrix_path.resolve()),
            "runtime_settings": asdict(shard_runtime_settings),
            "min_bout": int(min_bout),
            "edge_skip": int(edge_skip),
            "inf_tag": str(inf_tag),
            "file_filter": (None if file_filter is None else str(file_filter)),
        }
        done_marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if not bool(runtime_settings.keep_temp_files):
            cleanup_temp_store(str(temp_store_path), keep_temp_files=False)
        return str(matrix_path)
    finally:
        try:
            lock.release()
        except Exception:
            pass
        if shard_lock_acquired and shard_lock is not None:
            try:
                shard_lock.release()
            except Exception:
                pass


def merge_matrix_part_files(*, output_dir: str, num_shards: int | None = None, mm_per_px: float = 0.0169896744624207, video_fps: float = 40.0, shards_dirname: str = SHARDS_DIRNAME_DEFAULT) -> str:
    shard_dirs = discover_shard_output_dirs(output_dir, num_shards=num_shards, shards_dirname=shards_dirname)
    matrix_parts: list[pd.DataFrame] = []
    seen_ids: set[str] = set()
    merged_from: list[str] = []
    union_columns: list[str] = ["individual", "group", "sex"]
    expected_signature: dict | None = None
    expected_signature_source: str | None = None
    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir)
        work_dir = shard_dir / "_mmgen_work"
        if not work_dir.exists():
            raise FileNotFoundError(f"Missing shard work directory: {work_dir}")

        matrix_done = _read_done_payload(work_dir / SHARD_MATRIX_DONE_FILENAME)
        process_done = _read_done_payload(work_dir / "SHARD_DONE.json")
        shard_manifest = _read_done_payload(work_dir / SHARD_MANIFEST_FILENAME)
        matrix_path = shard_dir / "MASTER_FEATURE_MATRIX.csv"

        selected_n = None
        if isinstance(matrix_done, dict):
            selected_n = matrix_done.get("n_selected_files")
        if selected_n is None and isinstance(shard_manifest, dict):
            selected_n = shard_manifest.get("n_selected_files")
        if selected_n is None and isinstance(process_done, dict):
            selected_n = process_done.get("n_inference_files_selected")
        if selected_n is None:
            raise RuntimeError(
                f"Cannot determine selected-file count for shard {shard_dir}. Expected {SHARD_MATRIX_DONE_FILENAME}, {SHARD_MANIFEST_FILENAME}, or SHARD_DONE.json."
            )

        if int(selected_n) == 0:
            if not isinstance(matrix_done, dict):
                raise RuntimeError(
                    f"Shard {shard_dir} appears to have zero selected files, but {SHARD_MATRIX_DONE_FILENAME} is missing. Run --finalize_shard_matrix for this shard before merging matrix parts."
                )
            continue

        signature = _matrix_merge_consistency_signature(matrix_done=matrix_done, process_done=process_done)
        if expected_signature is None:
            expected_signature = signature
            expected_signature_source = str(work_dir / SHARD_MATRIX_DONE_FILENAME)
        elif signature != expected_signature:
            raise RuntimeError(
                "Shard matrix parts were produced with inconsistent matrix-generation settings and cannot be merged safely. "
                f"Reference: {expected_signature_source}; mismatch: {work_dir / SHARD_MATRIX_DONE_FILENAME}."
            )

        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing shard matrix part: {matrix_path}")
        if not isinstance(matrix_done, dict):
            raise RuntimeError(f"Shard matrix part exists but completion marker is missing: {work_dir / SHARD_MATRIX_DONE_FILENAME}")

        df = pd.read_csv(matrix_path)
        _validate_required_matrix_columns(df, source_label=str(matrix_path))
        _validate_unique_individuals(df, source_label=str(matrix_path))
        for col in df.columns:
            if col not in union_columns:
                union_columns.append(col)
        overlap = seen_ids.intersection(df["individual"].astype(str).tolist())
        if overlap:
            raise ValueError("Duplicate animal identities across shard matrix parts: " + ", ".join(sorted(list(overlap))[:10]) + (" ..." if len(overlap) > 10 else ""))
        seen_ids.update(df["individual"].astype(str).tolist())
        matrix_parts.append(df)
        merged_from.append(str(matrix_path.resolve()))
    if not matrix_parts:
        raise RuntimeError(f"No shard matrix parts were found under {Path(output_dir) / shards_dirname}")
    aligned_parts = [df.reindex(columns=union_columns) for df in matrix_parts]
    merged = pd.concat(aligned_parts, axis=0, ignore_index=True, sort=False)
    merged.dropna(axis=1, how="all", inplace=True)
    _validate_unique_individuals(merged, source_label="merged matrix parts")
    merged = merged.sort_values(by=["group", "individual"], kind="stable").reset_index(drop=True)
    merged = reorder_master_matrix_columns(merged)
    output_path = Path(output_dir) / "MASTER_FEATURE_MATRIX.csv"
    merged.to_csv(output_path, index=False)
    write_feature_units_manifest(master_matrix=merged, output_dir=str(output_path.parent), mm_per_px=float(mm_per_px), fps=float(video_fps))
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_shard_parts": int(len(matrix_parts)),
        "n_rows": int(len(merged)),
        "n_columns": int(len(merged.columns)),
        "schema_union_columns": list(merged.columns),
        "output_path": str(output_path.resolve()),
        "merged_from": merged_from,
    }
    (Path(output_dir) / FINAL_MATRIX_MERGE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return str(output_path)
