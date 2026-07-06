"""Lightweight provenance artifacts for mmgen.

This module writes small, durable metadata files that make later incremental
updates possible without requiring users to keep large temp_store.h5 files
forever.

Artifacts written here are intentionally lightweight:
- MMGEN_SOURCE_LEDGER.csv: one row per committed inference CSV/source file
- MMGEN_ROW_IDENTITY_SUMMARY.csv: one row per master-matrix row identity
- MMGEN_BATCH_MANIFEST.json: compact run/config/schema manifest

The canonical row identity for mmgen is the final master-matrix ``individual``
value, which corresponds to the collapsed per-animal ID (movie removed).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from autobehaver import __version__ as AUTOBEHAVER_VERSION

from .naming import collapse_individual_id, parse_group_and_individual


SOURCE_LEDGER_FILENAME = "MMGEN_SOURCE_LEDGER.csv"
ROW_IDENTITY_SUMMARY_FILENAME = "MMGEN_ROW_IDENTITY_SUMMARY.csv"
BATCH_MANIFEST_FILENAME = "MMGEN_BATCH_MANIFEST.json"
ROW_IDENTITY_COLUMN = "individual"
ROW_IDENTITY_DEFINITION = (
    "collapsed animal_id / master-matrix individual value "
    "(movie token removed; group/sex/rpi/trial/chamber retained)"
)


_LEDGER_COLUMNS = [
    "source_file",
    "source_path",
    "source_exists",
    "source_file_size_bytes",
    "source_file_mtime_utc",
    "poscalc_file",
    "poscalc_path",
    "poscalc_exists",
    "poscalc_file_size_bytes",
    "poscalc_file_mtime_utc",
    "n_frames",
    "group",
    "sex",
    "individual_id",
    "animal_id",
    "row_identity",
    "rpi",
    "chamber",
    "trial",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_file_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {"exists": False, "size_bytes": pd.NA, "mtime_utc": pd.NA}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "size_bytes": pd.NA, "mtime_utc": pd.NA}
    try:
        stat = p.stat()
        mtime_utc = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
        return {"exists": True, "size_bytes": int(stat.st_size), "mtime_utc": mtime_utc}
    except Exception:
        return {"exists": True, "size_bytes": pd.NA, "mtime_utc": pd.NA}


def _derive_poscalc_filename(source_file: str, inf_tag: str) -> str:
    sf = str(source_file)
    if inf_tag and str(inf_tag) in sf:
        return sf.replace(str(inf_tag), "")
    return sf


def _safe_int_from_regex(pattern: str, text: str | None) -> int | pd.NA:
    if not text:
        return pd.NA
    m = re.search(pattern, str(text))
    if not m:
        return pd.NA
    try:
        return int(m.group(1))
    except Exception:
        return pd.NA


def _safe_str_from_regex(pattern: str, text: str | None) -> str | pd.NA:
    if not text:
        return pd.NA
    m = re.search(pattern, str(text))
    if not m:
        return pd.NA
    return str(m.group(1))


def build_source_ledger_df(
    *,
    inference_csv_dir: str,
    poscalc_dir: str,
    lengths_df: pd.DataFrame | None,
    animal_mapping: dict[str, Any] | None,
    file_to_animal: dict[str, str] | None,
    inf_tag: str,
) -> pd.DataFrame:
    """Build a lightweight ledger of committed source files.

    The authoritative input is ``video_lengths_tmp`` (represented here as
    ``lengths_df``). That table tells us which inference CSVs were actually
    committed into the finalized dataset.
    """
    animal_mapping = animal_mapping or {}
    file_to_animal = file_to_animal or {}

    if lengths_df is None or lengths_df.empty or "source_file" not in lengths_df.columns:
        return pd.DataFrame(columns=_LEDGER_COLUMNS)

    df = lengths_df.copy().drop_duplicates(subset=["source_file"], keep="last").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        source_file = str(row.get("source_file", "")).strip()
        if not source_file:
            continue

        source_path = os.path.join(inference_csv_dir, source_file)
        source_stats = _safe_file_stats(source_path)

        poscalc_file = _derive_poscalc_filename(source_file, inf_tag)
        poscalc_path = os.path.join(poscalc_dir, poscalc_file)
        poscalc_stats = _safe_file_stats(poscalc_path)

        group = row.get("group", pd.NA)
        individual_id = row.get("individual", pd.NA)
        sex = row.get("sex", pd.NA)

        if pd.isna(group) or pd.isna(individual_id) or pd.isna(sex):
            try:
                parsed_group, parsed_individual_id, parsed_sex = parse_group_and_individual(source_file)
                if pd.isna(group):
                    group = parsed_group
                if pd.isna(individual_id):
                    individual_id = parsed_individual_id
                if pd.isna(sex):
                    sex = parsed_sex
            except Exception:
                pass

        row_identity = (
            file_to_animal.get(source_file)
            or file_to_animal.get(source_path)
            or collapse_individual_id(str(individual_id))
        )
        animal_info = animal_mapping.get(row_identity, {}) if row_identity is not None else {}

        rpi = animal_info.get("rpi") or _safe_str_from_regex(r"(rpi-\d+)", str(individual_id))
        chamber = animal_info.get("chamber") or _safe_str_from_regex(r"(chamber\d+)", str(individual_id))
        trial = animal_info.get("trial")
        if trial is None or pd.isna(trial):
            trial = _safe_int_from_regex(r"_trial(\d+)", str(individual_id))

        try:
            n_frames = int(row.get("n_frames"))
        except Exception:
            n_frames = pd.NA

        rows.append(
            {
                "source_file": source_file,
                "source_path": source_path,
                "source_exists": bool(source_stats["exists"]),
                "source_file_size_bytes": source_stats["size_bytes"],
                "source_file_mtime_utc": source_stats["mtime_utc"],
                "poscalc_file": poscalc_file,
                "poscalc_path": poscalc_path,
                "poscalc_exists": bool(poscalc_stats["exists"]),
                "poscalc_file_size_bytes": poscalc_stats["size_bytes"],
                "poscalc_file_mtime_utc": poscalc_stats["mtime_utc"],
                "n_frames": n_frames,
                "group": group,
                "sex": sex,
                "individual_id": individual_id,
                "animal_id": row_identity,
                "row_identity": row_identity,
                "rpi": rpi,
                "chamber": chamber,
                "trial": trial,
            }
        )

    out = pd.DataFrame(rows, columns=_LEDGER_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(["row_identity", "source_file"], kind="stable").reset_index(drop=True)


def build_row_identity_summary_df(
    *,
    source_ledger_df: pd.DataFrame | None,
    master_matrix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize committed source files by stable mmgen row identity."""
    ledger_df = source_ledger_df.copy() if source_ledger_df is not None else pd.DataFrame(columns=_LEDGER_COLUMNS)
    matrix_df = master_matrix_df.copy() if master_matrix_df is not None else None

    if matrix_df is not None and not matrix_df.empty:
        if ROW_IDENTITY_COLUMN not in matrix_df.columns:
            raise ValueError(f"master_matrix_df is missing required row identity column: {ROW_IDENTITY_COLUMN}")
        dup_mask = matrix_df[ROW_IDENTITY_COLUMN].astype(str).duplicated(keep=False)
        if dup_mask.any():
            dupes = sorted(matrix_df.loc[dup_mask, ROW_IDENTITY_COLUMN].astype(str).unique().tolist())
            raise ValueError(
                "Duplicate master-matrix row identities detected in 'individual': "
                + ", ".join(dupes[:10])
                + (" ..." if len(dupes) > 10 else "")
            )
        matrix_meta = (
            matrix_df[[ROW_IDENTITY_COLUMN, "group", "sex"]]
            .drop_duplicates(subset=[ROW_IDENTITY_COLUMN])
            .rename(columns={ROW_IDENTITY_COLUMN: "row_identity", "group": "matrix_group", "sex": "matrix_sex"})
        )
        matrix_meta["in_master_matrix"] = True
    else:
        matrix_meta = pd.DataFrame(columns=["row_identity", "matrix_group", "matrix_sex", "in_master_matrix"])

    if ledger_df.empty:
        if matrix_meta.empty:
            return pd.DataFrame(columns=[
                "row_identity",
                "master_matrix_individual",
                "group",
                "sex",
                "rpi",
                "chamber",
                "trial",
                "n_source_files",
                "total_n_frames",
                "source_files",
                "in_master_matrix",
            ])
        out = matrix_meta.copy()
        out["master_matrix_individual"] = out["row_identity"]
        out["group"] = out["matrix_group"]
        out["sex"] = out["matrix_sex"]
        out["rpi"] = pd.NA
        out["chamber"] = pd.NA
        out["trial"] = pd.NA
        out["n_source_files"] = 0
        out["total_n_frames"] = 0
        out["source_files"] = ""
        out = out[[
            "row_identity",
            "master_matrix_individual",
            "group",
            "sex",
            "rpi",
            "chamber",
            "trial",
            "n_source_files",
            "total_n_frames",
            "source_files",
            "in_master_matrix",
        ]]
        return out.sort_values(["row_identity"], kind="stable").reset_index(drop=True)

    grouped = (
        ledger_df.groupby("row_identity", observed=True)
        .agg(
            group=("group", "first"),
            sex=("sex", "first"),
            rpi=("rpi", "first"),
            chamber=("chamber", "first"),
            trial=("trial", "first"),
            n_source_files=("source_file", "count"),
            total_n_frames=("n_frames", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            source_files=("source_file", lambda s: ", ".join(sorted(map(str, s)))),
        )
        .reset_index()
    )

    out = grouped.merge(matrix_meta, on="row_identity", how="left")
    out["in_master_matrix"] = out["in_master_matrix"].fillna(False).astype(bool)
    out["master_matrix_individual"] = out["row_identity"].where(out["in_master_matrix"], pd.NA)
    if "matrix_group" in out.columns:
        out["group"] = out["group"].where(out["group"].notna(), out["matrix_group"])
    if "matrix_sex" in out.columns:
        out["sex"] = out["sex"].where(out["sex"].notna(), out["matrix_sex"])

    out = out[[
        "row_identity",
        "master_matrix_individual",
        "group",
        "sex",
        "rpi",
        "chamber",
        "trial",
        "n_source_files",
        "total_n_frames",
        "source_files",
        "in_master_matrix",
    ]]
    return out.sort_values(["row_identity"], kind="stable").reset_index(drop=True)


def _schema_sha256_from_columns(columns: list[str]) -> str:
    payload = "\n".join(map(str, columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_source_ledger_csv(output_dir: str, ledger_df: pd.DataFrame) -> str:
    path = os.path.join(output_dir, SOURCE_LEDGER_FILENAME)
    ledger_df.to_csv(path, index=False)
    return path


def write_row_identity_summary_csv(output_dir: str, row_identity_df: pd.DataFrame) -> str:
    path = os.path.join(output_dir, ROW_IDENTITY_SUMMARY_FILENAME)
    row_identity_df.to_csv(path, index=False)
    return path


def write_batch_manifest_json(
    *,
    output_dir: str,
    inference_csv_dir: str,
    poscalc_dir: str,
    inf_tag: str,
    min_bout: int,
    edge_skip: int,
    resume: bool,
    finalize_only: bool,
    process_only: bool,
    file_filter: str | None,
    num_shards: int,
    shard_index: int,
    shard_by: str,
    emit_master_matrix: bool,
    emit_full_outputs: bool,
    lengths_df: pd.DataFrame | None,
    source_ledger_df: pd.DataFrame | None,
    row_identity_df: pd.DataFrame | None,
    master_matrix_df: pd.DataFrame | None,
    animal_mapping: dict[str, Any] | None,
    append_new_data: bool = False,
) -> str:
    """Write a lightweight run/config/schema manifest for mmgen finalization."""
    matrix_df = master_matrix_df.copy() if master_matrix_df is not None else None
    ledger_df = source_ledger_df if source_ledger_df is not None else pd.DataFrame(columns=_LEDGER_COLUMNS)
    row_df = row_identity_df if row_identity_df is not None else pd.DataFrame()
    lengths = lengths_df if lengths_df is not None else pd.DataFrame()
    animal_mapping = animal_mapping or {}

    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "written_at_utc": _utc_now_iso(),
        "autobehaver_version": AUTOBEHAVER_VERSION,
        "row_identity": {
            "column": ROW_IDENTITY_COLUMN,
            "definition": ROW_IDENTITY_DEFINITION,
            "row_identity_summary_file": ROW_IDENTITY_SUMMARY_FILENAME,
        },
        "source_ledger": {
            "file": SOURCE_LEDGER_FILENAME,
            "n_rows": int(len(ledger_df)),
            "n_unique_row_identities": int(ledger_df["row_identity"].nunique()) if not ledger_df.empty and "row_identity" in ledger_df.columns else 0,
        },
        "master_matrix": {
            "enabled": bool(emit_master_matrix),
            "path": "MASTER_FEATURE_MATRIX.csv" if bool(emit_master_matrix) else None,
            "n_rows": int(len(matrix_df)) if matrix_df is not None and not matrix_df.empty else 0,
            "n_columns": int(len(matrix_df.columns)) if matrix_df is not None and not matrix_df.empty else 0,
            "schema_sha256": _schema_sha256_from_columns(list(matrix_df.columns)) if matrix_df is not None and not matrix_df.empty else None,
        },
        "input_folders": {
            "inference_csv_dir": str(inference_csv_dir),
            "poscalc_dir": str(poscalc_dir),
        },
        "processing": {
            "inf_tag": str(inf_tag),
            "min_bout": int(min_bout),
            "edge_skip": int(edge_skip),
            "resume": bool(resume),
            "finalize_only": bool(finalize_only),
            "process_only": bool(process_only),
            "file_filter": None if file_filter in {None, ""} else str(file_filter),
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "shard_by": str(shard_by),
            "emit_master_matrix": bool(emit_master_matrix),
            "emit_full_outputs": bool(emit_full_outputs),
            "append_new_data": bool(append_new_data),
        },
        "counts": {
            "committed_source_files": int(len(lengths.drop_duplicates(subset=["source_file"], keep="last"))) if not lengths.empty and "source_file" in lengths.columns else 0,
            "row_identity_count": int(len(row_df)) if row_df is not None and not row_df.empty else 0,
            "animal_mapping_count": int(len(animal_mapping)),
        },
    }

    path = os.path.join(output_dir, BATCH_MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path
