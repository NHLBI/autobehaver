"""Append new MMGen sources by recomputing affected animal rows from a frozen batch
plan.

Publish the matrix, ledger, and generation receipt together, preserving prior numbered
reports. Changed historical inputs or missing sources needed for recomputation are
rejected.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from .discovery import list_inference_csv_files
from .feature_matrix import validate_feature_matrix
from .ledger import (
    BATCH_MANIFEST_FILENAME,
    ROW_IDENTITY_SUMMARY_FILENAME,
    SOURCE_LEDGER_FILENAME,
    build_row_identity_summary_df,
    write_batch_manifest_json,
    write_row_identity_summary_csv,
)
from .mapping import create_animal_mapping
from .naming import collapse_individual_id, parse_group_and_individual
from .io import find_matching_poscalc_file, read_metadata_csv
from autobehaver.batch_inputs import select_batch_inputs
from autobehaver.portable_paths import original_name
from .feature_schema import complete_master_schema, write_schema_metadata, schema_metadata
from .shard_merge import merge_shard_temp_stores
from .hdf_store import cleanup_verified_temp_stores
from .locks import DirLock, canonical_writer
from . import pipeline
from autobehaver.atomic_io import atomic_write_json, sha256_file
from .integrity import calculation_identity, content_hash, source_identities

APPEND_BATCHES_DIRNAME = "_mmgen_append_batches"
APPEND_LATEST_REPORT_FILENAME = "MMGEN_APPEND_LATEST.json"
APPEND_PLAN_FILENAME = "APPEND_PLAN.json"
APPEND_SELECTED_FILES_FILENAME = "APPEND_SELECTED_FILES.csv"
APPEND_AFFECTED_ROWS_FILENAME = "APPEND_AFFECTED_ROWS.csv"
from .publication import (OUTPUT_RECEIPT_FILENAME, IDENTITY_RELATIVE_PATH, PUBLICATION_JOURNAL,
    CORE_ARTIFACTS, _publication_copy, _recover_publication, _verified_generation, _publish_generation)


def _cfg_calculation(cfg):
    return calculation_identity(cfg.runtime_settings,min_bout=cfg.min_bout,
                                edge_skip=cfg.edge_skip,inf_tag=cfg.inf_tag)


def _verify_existing_source_continuity(cfg, generation, inventory):
    """Check all available prior inputs, including metadata-derived alignment."""
    previous = generation['identity']['sources']
    available = set(inventory['source_file'].astype(str)) if not inventory.empty else set()
    selected = sorted(set(previous).intersection(available))
    current = source_identities(selected, inference_csv_dir=cfg.inference_csv_dir,
        poscalc_dir=cfg.poscalc_dir, inf_tag=cfg.inf_tag,
        source_frame_offset=cfg.runtime_settings.source_frame_offset)
    changed = [name for name in selected if previous[name] != current[name]]
    if changed:
        raise RuntimeError('Previously processed source content or original-frame metadata changed; fresh processing required: ' + ', '.join(changed[:10]))


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schema_sha256(columns: list[str]) -> str:
    payload = "\n".join(map(str, columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_file_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {"exists": False, "size_bytes": pd.NA, "mtime_utc": pd.NA, 'sha256':pd.NA}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "size_bytes": pd.NA, "mtime_utc": pd.NA, 'sha256':pd.NA}
    try:
        stat = p.stat()
        mtime_utc = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
        return {"exists": True, "size_bytes": int(stat.st_size), "mtime_utc": mtime_utc, 'sha256':sha256_file(p)}
    except Exception as exc:
        raise RuntimeError(f'Cannot verify source contents: {p}') from exc


def _derive_poscalc_filename(source_file: str, inf_tag: str) -> str:
    sf = str(source_file)
    if inf_tag and str(inf_tag) in sf:
        return sf.replace(str(inf_tag), "")
    return sf


def _build_current_inventory(*, inference_csv_dir: str, poscalc_dir: str, inf_tag: str) -> pd.DataFrame:
    """Index inference/positional pairs with their animal identities and current content
    hashes.
    """
    rows: list[dict[str, Any]] = []
    for source_file in list_inference_csv_files(inference_csv_dir):
        source_path = os.path.join(inference_csv_dir, source_file)
        poscalc_path = find_matching_poscalc_file(source_path, poscalc_dir, inf_tag)
        # New unrelated inputs may arrive before their positional partner. Keep
        # them visible to selection; selected/affected and prior sources are
        # checked before a plan is frozen or consumed.
        poscalc_file = (os.path.basename(poscalc_path) if poscalc_path else
                        _derive_poscalc_filename(original_name(source_path), inf_tag))

        group, individual_id, sex = parse_group_and_individual(source_path)
        row_identity = collapse_individual_id(individual_id)

        source_stats = _safe_file_stats(source_path)
        poscalc_stats = _safe_file_stats(poscalc_path)

        rows.append(
            {
                "source_file": source_file,
                "source_path": source_path,
                "source_exists": bool(source_stats["exists"]),
                "source_file_size_bytes": source_stats["size_bytes"],
                "source_file_mtime_utc": source_stats["mtime_utc"],
                "source_file_sha256": source_stats['sha256'],
                "poscalc_file": poscalc_file,
                "poscalc_path": poscalc_path,
                "poscalc_exists": bool(poscalc_stats["exists"]),
                "poscalc_file_size_bytes": poscalc_stats["size_bytes"],
                "poscalc_file_mtime_utc": poscalc_stats["mtime_utc"],
                "poscalc_file_sha256": poscalc_stats['sha256'],
                "group": group,
                "sex": sex,
                "individual_id": individual_id,
                "row_identity": row_identity,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
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
                "group",
                "sex",
                "individual_id",
                "row_identity",
            ]
        )
    return out.sort_values(["row_identity", "source_file"], kind="stable").reset_index(drop=True)


def _filter_inventory_by_substring(inv_df: pd.DataFrame, needle: str | None) -> pd.DataFrame:
    if inv_df.empty or needle is None or str(needle).strip() == "":
        return inv_df.copy()
    q = str(needle).strip().lower()
    return inv_df[inv_df["source_path"].map(original_name).str.lower().str.contains(q, na=False, regex=False)].copy().reset_index(drop=True)


COMPARE_COLUMNS = [
    "row_identity",
    "poscalc_file",
    "source_file_sha256",
    "poscalc_file_sha256",
]
_NUMERIC_COMPARE_COLUMNS = {"source_file_size_bytes", "poscalc_file_size_bytes"}


def _classify_current_vs_ledger(*, current_inventory_df: pd.DataFrame, existing_ledger_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (new_df, changed_df, unchanged_df, missing_in_current_df)."""
    current = current_inventory_df.copy()
    ledger = existing_ledger_df.copy()

    if current.empty:
        empty = pd.DataFrame(columns=current.columns)
        missing = pd.DataFrame(columns=ledger.columns)
        return empty.copy(), empty.copy(), empty.copy(), missing

    if ledger.empty or "source_file" not in ledger.columns:
        empty = pd.DataFrame(columns=current.columns)
        return current.copy(), empty.copy(), empty.copy(), pd.DataFrame(columns=ledger.columns)

    if ledger['source_file'].astype(str).duplicated().any():
        raise RuntimeError('Duplicate ledger sources cannot be certified.')
    if not {'source_file_sha256','poscalc_file_sha256'}.issubset(ledger):
        raise RuntimeError('Legacy ledger has no content hashes; fresh processing is required.')
    merged = current.merge(
        ledger[["source_file"] + [c for c in COMPARE_COLUMNS if c in ledger.columns]],
        on="source_file",
        how="left",
        suffixes=("", "__ledger"),
        indicator=True,
    )

    new_mask = merged["_merge"].eq("left_only")
    both_mask = merged["_merge"].eq("both")

    changed_mask = pd.Series(False, index=merged.index)
    for col in COMPARE_COLUMNS:
        ledger_col = f"{col}__ledger"
        if col not in merged.columns or ledger_col not in merged.columns:
            continue
        if col in _NUMERIC_COMPARE_COLUMNS:
            lhs = pd.to_numeric(merged[col], errors="coerce")
            rhs = pd.to_numeric(merged[ledger_col], errors="coerce")
            mismatch = lhs.ne(rhs) & ~(lhs.isna() & rhs.isna())
        else:
            lhs = merged[col].fillna("").astype(str)
            rhs = merged[ledger_col].fillna("").astype(str)
            mismatch = lhs.ne(rhs)
        changed_mask = changed_mask | (both_mask & mismatch)

    unchanged_mask = both_mask & (~changed_mask)

    new_df = merged.loc[new_mask, current.columns].copy().reset_index(drop=True)
    changed_df = merged.loc[changed_mask, current.columns].copy().reset_index(drop=True)
    unchanged_df = merged.loc[unchanged_mask, current.columns].copy().reset_index(drop=True)

    missing_in_current_df = ledger.loc[~ledger["source_file"].astype(str).isin(current["source_file"].astype(str))].copy().reset_index(drop=True)
    return new_df, changed_df, unchanged_df, missing_in_current_df


def _atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    path = str(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_df_csv(path: str | os.PathLike[str], df: pd.DataFrame) -> None:
    path = str(path)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _copy_if_missing(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    if os.path.exists(dst):
        return
    if os.path.exists(src):
        shutil.copy2(src, dst)


def _write_append_report(output_folder: str, report: dict[str, Any], *, latest: bool) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True)
    if latest:
        path = os.path.join(output_folder, APPEND_LATEST_REPORT_FILENAME)
        _atomic_write_text(path, payload)
        return path
    path = os.path.join(output_folder, "MMGEN_APPEND_REPORT.json")
    _atomic_write_text(path, payload)
    return path


def _cleanup_append_batch_temp_artifacts(batch_dir: Path, identities: list[tuple[str, str]]) -> None:
    """Remove verified large stores after canonical append publication succeeds."""
    cleanup_verified_temp_stores(str(batch_dir), identities)


def _load_required_canonical_state_owned(output_folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require a verified canonical matrix and ledger with unique animal rows."""
    _verified_generation(output_folder)
    matrix_path = os.path.join(output_folder, "MASTER_FEATURE_MATRIX.csv")
    ledger_path = os.path.join(output_folder, SOURCE_LEDGER_FILENAME)
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            "append_new_data requires an existing MASTER_FEATURE_MATRIX.csv in the output folder"
        )
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(
            "append_new_data requires an existing MMGEN_SOURCE_LEDGER.csv in the output folder"
        )
    matrix_df = read_metadata_csv(matrix_path)
    ledger_df = read_metadata_csv(ledger_path)
    if "individual" not in matrix_df.columns:
        raise ValueError("Existing MASTER_FEATURE_MATRIX.csv is missing required 'individual' column")
    if "row_identity" not in ledger_df.columns or "source_file" not in ledger_df.columns:
        raise ValueError("Existing MMGEN_SOURCE_LEDGER.csv is missing required columns")
    dup_mask = matrix_df["individual"].astype(str).duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(matrix_df.loc[dup_mask, "individual"].astype(str).unique().tolist())
        raise ValueError(
            "Existing MASTER_FEATURE_MATRIX.csv contains duplicate 'individual' row identities: "
            + ", ".join(dupes[:10])
            + (" ..." if len(dupes) > 10 else "")
        )
    return matrix_df, ledger_df


def _append_batch_dir(root_out: str, append_batch_id: str) -> Path:
    return Path(root_out) / APPEND_BATCHES_DIRNAME / str(append_batch_id)


def _build_affected_rows_df(*, affected_all_files_df: pd.DataFrame, existing_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the recordings needed for each affected animal and mark existing rows."""
    if affected_all_files_df.empty:
        return pd.DataFrame(columns=["row_identity", "already_in_canonical", "n_selected_files", "selected_files"])

    existing_ids = set(existing_matrix_df["individual"].astype(str).tolist()) if not existing_matrix_df.empty else set()
    grouped = (
        affected_all_files_df.groupby("row_identity", observed=True)
        .agg(
            n_selected_files=("source_file", "count"),
            selected_files=("source_file", lambda s: ", ".join(sorted(map(str, s)))),
        )
        .reset_index()
    )
    grouped["already_in_canonical"] = grouped["row_identity"].astype(str).isin(existing_ids)
    return grouped.sort_values(["row_identity"], kind="stable").reset_index(drop=True)


def _write_append_plan_files(*, batch_dir: Path, plan: dict[str, Any], selected_files_df: pd.DataFrame, affected_rows_df: pd.DataFrame) -> None:
    _atomic_write_df_csv(batch_dir / APPEND_SELECTED_FILES_FILENAME, selected_files_df)
    _atomic_write_df_csv(batch_dir / APPEND_AFFECTED_ROWS_FILENAME, affected_rows_df)
    plan['plan_artifacts']={name:sha256_file(batch_dir/name) for name in (APPEND_SELECTED_FILES_FILENAME,APPEND_AFFECTED_ROWS_FILENAME)}
    plan['plan_sha256']=content_hash({k:v for k,v in plan.items() if k!='plan_sha256'})
    _atomic_write_text(batch_dir / APPEND_PLAN_FILENAME, json.dumps(plan, indent=2, sort_keys=True))


def _load_append_plan(root_out: str, append_batch_id: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]:
    """Load an append plan and verify the identities of its selected-source artifacts."""
    batch_dir = _append_batch_dir(root_out, append_batch_id)
    plan_path = batch_dir / APPEND_PLAN_FILENAME
    selected_path = batch_dir / APPEND_SELECTED_FILES_FILENAME
    affected_path = batch_dir / APPEND_AFFECTED_ROWS_FILENAME
    if not plan_path.exists():
        raise FileNotFoundError(f"Append batch plan not found: {plan_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Append batch selected-files table not found: {selected_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get('plan_sha256') != content_hash({k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise RuntimeError('Append plan content identity is missing or changed.')
    if set(plan.get('plan_artifacts',{})) != {APPEND_SELECTED_FILES_FILENAME,APPEND_AFFECTED_ROWS_FILENAME}:
        raise RuntimeError('Append plan must certify both frozen selection and affected-row tables.')
    for name,digest in plan.get('plan_artifacts',{}).items():
        if name not in (APPEND_SELECTED_FILES_FILENAME,APPEND_AFFECTED_ROWS_FILENAME) or sha256_file(batch_dir/name)!=digest:
            raise RuntimeError('Frozen append-plan table changed.')
    selected_df = read_metadata_csv(selected_path)
    affected_df = read_metadata_csv(affected_path) if affected_path.exists() else pd.DataFrame()
    return plan, selected_df, affected_df, batch_dir


def _create_append_plan_owned(cfg) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None, Path | None]:
    """Freeze new sources and all recordings needed to recompute affected animals."""
    root_out = str(cfg.output_dir)
    os.makedirs(root_out, exist_ok=True)

    existing_matrix_df, existing_ledger_df = _load_required_canonical_state(root_out)
    generation = _verified_generation(root_out)
    current_calculation = _cfg_calculation(cfg)
    if content_hash(current_calculation) != generation['identity']['calculation_sha256']:
        raise RuntimeError('Append calculation/settings/code differ from canonical data; reprocess the whole cohort into fresh outputs.')

    current_inventory_df = _build_current_inventory(
        inference_csv_dir=str(cfg.inference_csv_dir),
        poscalc_dir=str(cfg.poscalc_dir),
        inf_tag=str(cfg.inf_tag),
    )
    selected_inventory_df = current_inventory_df
    if cfg.file_list:
        selected_paths = set(select_batch_inputs(current_inventory_df['source_path'].tolist(), cfg.file_list))
        selected_inventory_df = current_inventory_df[current_inventory_df['source_path'].isin(selected_paths)].copy()
    selected_inventory_df = _filter_inventory_by_substring(selected_inventory_df, cfg.file_filter)
    _verify_existing_source_continuity(cfg, generation, current_inventory_df)
    _, changed_any_existing, _, _ = _classify_current_vs_ledger(
        current_inventory_df=current_inventory_df,existing_ledger_df=existing_ledger_df)
    if not changed_any_existing.empty:
        raise RuntimeError('Previously processed source content changed, including outside the selected append filter; fresh processing required.')

    if selected_inventory_df.empty:
        print("[mmgen-append] No inference files matched the file_list/file_filter selection. Nothing to do.")
        return None, None, None, None

    new_df, changed_df, unchanged_df, missing_ledger_df = _classify_current_vs_ledger(
        current_inventory_df=selected_inventory_df,
        existing_ledger_df=existing_ledger_df,
    )

    if not changed_df.empty:
        preview = changed_df["source_file"].astype(str).tolist()[:10]
        raise RuntimeError(
            "append_new_data detected previously processed files whose size/mtime/row identity changed. "
            "This mode currently supports only new files. Changed files:\n  - "
            + "\n  - ".join(preview)
            + ("\n  ..." if len(changed_df) > 10 else "")
        )

    if new_df.empty:
        print("[mmgen-append] No new source files were found relative to MMGEN_SOURCE_LEDGER.csv. Nothing to append.")
        if not missing_ledger_df.empty:
            print(
                f"[mmgen-append] Warning: {len(missing_ledger_df)} previously committed source files are not present in the current input folders. "
                "This does not block the current no-op append, but future updates affecting those rows may fail until the files are restored."
            )
        return None, None, None, None

    affected_row_ids = sorted(set(new_df["row_identity"].astype(str).tolist()))
    affected_all_files_df = current_inventory_df[current_inventory_df["row_identity"].astype(str).isin(affected_row_ids)].copy()

    existing_affected_df = existing_ledger_df[existing_ledger_df["row_identity"].astype(str).isin(affected_row_ids)].copy()
    _new_affected_df, changed_existing_affected_df, _unchanged_affected_df, _missing_affected_df = _classify_current_vs_ledger(
        current_inventory_df=affected_all_files_df,
        existing_ledger_df=existing_affected_df,
    )
    if not changed_existing_affected_df.empty:
        preview = changed_existing_affected_df["source_file"].astype(str).tolist()[:10]
        raise RuntimeError(
            "append_new_data detected previously processed files within affected rows whose size/mtime/row identity changed. "
            "This mode currently supports only genuinely new files. Changed files:\n  - "
            + "\n  - ".join(preview)
            + ("\n  ..." if len(changed_existing_affected_df) > 10 else "")
        )

    missing_prior_sources_df = existing_affected_df[
        ~existing_affected_df["source_file"].astype(str).isin(affected_all_files_df["source_file"].astype(str))
    ].copy()
    if not missing_prior_sources_df.empty:
        preview = missing_prior_sources_df["source_file"].astype(str).tolist()[:10]
        raise RuntimeError(
            "append_new_data needs to recompute affected rows from all current source files, but some previously committed files for those rows are missing from the input folders. Missing files:\n  - "
            + "\n  - ".join(preview)
            + ("\n  ..." if len(missing_prior_sources_df) > 10 else "")
        )

    missing_poscalc_df = affected_all_files_df[~affected_all_files_df["poscalc_exists"].astype(bool)].copy()
    if not missing_poscalc_df.empty:
        preview = missing_poscalc_df["poscalc_file"].astype(str).tolist()[:10]
        raise FileNotFoundError(
            "append_new_data found affected source files whose paired positional_calcs CSV is missing:\n  - "
            + "\n  - ".join(preview)
            + ("\n  ..." if len(missing_poscalc_df) > 10 else "")
        )

    requested_num_shards = max(1, int(getattr(cfg, "num_shards", 1)))
    if requested_num_shards > 1 and str(getattr(cfg, "shard_by", "row_identity")) != "row_identity":
        raise ValueError("append_new_data sharded mode currently requires shard_by='row_identity'")
    active_num_shards = int(min(requested_num_shards, max(1, len(affected_row_ids))))

    append_batch_id = str(getattr(cfg, "append_batch_id", None) or f"append_{_utc_now_compact()}")
    batch_dir = _append_batch_dir(root_out, append_batch_id)
    if batch_dir.exists():
        raise FileExistsError(
            f"Append batch folder already exists: {batch_dir}. Choose a new --append_batch_id or remove the old batch folder first."
        )
    batch_dir.mkdir(parents=True, exist_ok=False)

    selected_files_df = (
        affected_all_files_df[
            [
                "source_file",
                "source_path",
                "poscalc_file",
                "poscalc_path",
                "source_file_sha256",
                "poscalc_file_sha256",
                "group",
                "sex",
                "individual_id",
                "row_identity",
            ]
        ]
        .sort_values(["row_identity", "source_file"], kind="stable")
        .reset_index(drop=True)
    )
    affected_rows_df = _build_affected_rows_df(affected_all_files_df=affected_all_files_df, existing_matrix_df=existing_matrix_df)

    existing_row_ids = set(existing_matrix_df["individual"].astype(str).tolist()) if not existing_matrix_df.empty else set()
    replaced_row_ids = sorted(existing_row_ids.intersection(set(affected_row_ids)))
    appended_row_ids = sorted(set(affected_row_ids) - set(replaced_row_ids))

    plan: dict[str, Any] = {
        "append_batch_id": append_batch_id,
        "planned_at_utc": _utc_now_iso(),
        "root_output_dir": str(Path(root_out).resolve()),
        "input_folders": {
            "inference_csv_dir": str(cfg.inference_csv_dir),
            "poscalc_dir": str(cfg.poscalc_dir),
        },
        "processing": {
            "inf_tag": str(cfg.inf_tag),
            "file_filter": None if cfg.file_filter in {None, ""} else str(cfg.file_filter),
            "emit_master_matrix": True,
            "emit_full_outputs": bool(cfg.emit_full_outputs),
            "requested_num_shards": int(requested_num_shards),
            "active_num_shards": int(active_num_shards),
            "shard_by": "row_identity" if requested_num_shards > 1 else "hash",
            "min_bout": int(cfg.min_bout),
            "edge_skip": int(cfg.edge_skip),
        },
        "canonical_schema_sha256": _schema_sha256(list(existing_matrix_df.columns)),
        'canonical_generation':{'receipt_sha256':generation['receipt_sha256'],'identity_sha256':generation['identity_sha256']},
        'calculation_sha256':content_hash(current_calculation),
        'selected_sources':source_identities(selected_files_df['source_file'],inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag,source_frame_offset=cfg.runtime_settings.source_frame_offset),
        "expected_row_identities": list(map(str, affected_row_ids)),
        "counts": {
            "selected_inventory_files": int(len(selected_inventory_df)),
            "new_source_files": int(len(new_df)),
            "unchanged_selected_files": int(len(unchanged_df)),
            "affected_row_identities": int(len(affected_row_ids)),
            "delta_processing_files": int(len(selected_files_df)),
            "replaced_existing_rows": int(len(replaced_row_ids)),
            "brand_new_rows": int(len(appended_row_ids)),
        },
        "new_source_files": sorted(new_df["source_file"].astype(str).tolist()),
        "selected_files_csv": APPEND_SELECTED_FILES_FILENAME,
        "affected_rows_csv": APPEND_AFFECTED_ROWS_FILENAME,
    }

    _write_append_plan_files(batch_dir=batch_dir, plan=plan, selected_files_df=selected_files_df, affected_rows_df=affected_rows_df)

    print(
        f"[mmgen-append] Planned append batch {append_batch_id}: {len(new_df)} new source files, "
        f"{len(affected_row_ids)} affected row identities, {len(selected_files_df)} total files in the delta batch."
    )
    if requested_num_shards > 1 and active_num_shards != requested_num_shards:
        print(
            f"[mmgen-append] Requested {requested_num_shards} shard(s), but only {active_num_shards} active shard(s) are needed "
            "for the affected row identities. Launch shard_index values in [0, active_num_shards-1]."
        )
    print(f"[mmgen-append] Append batch folder: {batch_dir}")
    return plan, selected_files_df, affected_rows_df, batch_dir


def _validate_plan_against_cfg_owned(plan: dict[str, Any], cfg) -> None:
    """Require append paths, calculation settings, and sources to match the frozen plan."""
    if plan.get('plan_sha256') != content_hash({k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise RuntimeError('Append plan content identity missing or changed.')
    if set(plan.get('plan_artifacts',{})) != {APPEND_SELECTED_FILES_FILENAME,APPEND_AFFECTED_ROWS_FILENAME}:
        raise RuntimeError('Append plan must certify both frozen tables.')
    plan_inputs = plan.get("input_folders", {}) or {}
    if str(Path(plan_inputs.get("inference_csv_dir", "")).resolve()) != str(Path(str(cfg.inference_csv_dir)).resolve()):
        raise RuntimeError("append_new_data batch plan was created for a different inference_csv_dir than the current command")
    if str(Path(plan_inputs.get("poscalc_dir", "")).resolve()) != str(Path(str(cfg.poscalc_dir)).resolve()):
        raise RuntimeError("append_new_data batch plan was created for a different poscalc_dir than the current command")
    if str(Path(plan.get('root_output_dir','')).resolve()) != str(Path(cfg.output_dir).resolve()):
        raise RuntimeError('Append plan targets a different canonical output folder.')
    if plan.get('calculation_sha256') != content_hash(_cfg_calculation(cfg)):
        raise RuntimeError('Scientific settings/source code changed after append planning.')
    generation=_verified_generation(cfg.output_dir)
    if plan.get('canonical_generation') != {'receipt_sha256':generation['receipt_sha256'],'identity_sha256':generation['identity_sha256']}:
        raise RuntimeError('Canonical generation changed after append planning; create a fresh plan.')
    if plan.get('selected_sources') != source_identities(plan.get('selected_sources',{}),
            inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag,source_frame_offset=cfg.runtime_settings.source_frame_offset):
        raise RuntimeError('Selected source content changed after append planning.')
    current=_build_current_inventory(inference_csv_dir=cfg.inference_csv_dir,
                                     poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag)
    _verify_existing_source_continuity(cfg, generation, current)
    ledger=read_metadata_csv(Path(cfg.output_dir)/SOURCE_LEDGER_FILENAME)
    _, changed, _, _ = _classify_current_vs_ledger(current_inventory_df=current,existing_ledger_df=ledger)
    if not changed.empty:
        raise RuntimeError('Previously processed input content changed after append planning.')
    batch_dir=_append_batch_dir(cfg.output_dir,plan['append_batch_id'])
    for name,digest in plan.get('plan_artifacts',{}).items():
        if name not in (APPEND_SELECTED_FILES_FILENAME,APPEND_AFFECTED_ROWS_FILENAME) or sha256_file(batch_dir/name)!=digest:
            raise RuntimeError('Frozen append-plan table changed.')


def _apply_delta_batch_to_canonical_owned(*, cfg, plan: dict[str, Any], delta_output_dir: Path) -> None:
    """Replace affected rows and publish the matrix, ledger, and manifest as one
    generation.
    """
    root_out = str(cfg.output_dir)

    # Canonical append/update lock: prevents two append merges from racing.
    append_lock_dir = Path(root_out) / "_mmgen_work" / "_mmgen_append.lock"
    os.makedirs(str(append_lock_dir.parent), exist_ok=True)
    append_lock = DirLock(append_lock_dir, timeout_sec=0.0)
    try:
        append_lock.acquire(metadata={"kind": "mmgen_append_update", "batch_id": str(plan.get("append_batch_id"))})
    except Exception as e:
        raise RuntimeError(
            "[mmgen-append] Another append/finalize job may already be updating the canonical output folder.\n"
            f"  lock: {append_lock_dir}\n\n"
            "Native locks release automatically when their writer exits. "
            f"(Original error: {e})"
        )

    try:
        _validate_plan_against_cfg(plan,cfg)
        existing_matrix_df, existing_ledger_df = _load_required_canonical_state(root_out)
        current_schema_sha = _schema_sha256(list(existing_matrix_df.columns))
        planned_schema_sha = str(plan.get("canonical_schema_sha256") or "")
        if planned_schema_sha and current_schema_sha != planned_schema_sha:
            raise RuntimeError(
                "append_new_data cannot apply this delta batch because the canonical master-matrix schema changed since the batch was planned."
            )

        delta_matrix_path = delta_output_dir / "MASTER_FEATURE_MATRIX.csv"
        delta_ledger_path = delta_output_dir / SOURCE_LEDGER_FILENAME
        if not delta_matrix_path.exists():
            raise FileNotFoundError(f"append_new_data delta batch did not produce {delta_matrix_path}")
        if not delta_ledger_path.exists():
            raise FileNotFoundError(f"append_new_data delta batch did not produce {delta_ledger_path}")

        delta_matrix_df = read_metadata_csv(delta_matrix_path)
        delta_ledger_df = read_metadata_csv(delta_ledger_path)
        delta_generation=_verified_generation(delta_output_dir)
        if delta_generation['identity']['calculation_sha256'] != plan['calculation_sha256'] or delta_generation['identity']['sources'] != plan['selected_sources']:
            raise RuntimeError('Delta output generation does not match the frozen append source/calculation plan.')
        is_valid_matrix, matrix_error, _ = validate_feature_matrix(str(delta_matrix_path))
        if not is_valid_matrix:
            raise RuntimeError(f"append_new_data delta batch produced an invalid master matrix: {matrix_error}")

        expected_row_ids = set(map(str, plan.get("expected_row_identities", [])))
        delta_row_ids_set = set(delta_matrix_df["individual"].astype(str).tolist()) if not delta_matrix_df.empty else set()
        if delta_row_ids_set != expected_row_ids:
            raise RuntimeError(
                "append_new_data delta batch did not produce the expected set of row identities. "
                f"Expected: {sorted(expected_row_ids)}; Produced: {sorted(delta_row_ids_set)}"
            )
        if "row_identity" not in delta_ledger_df.columns:
            raise RuntimeError("append_new_data delta ledger is missing required 'row_identity' column")
        delta_ledger_ids = set(delta_ledger_df["row_identity"].astype(str).tolist()) if not delta_ledger_df.empty else set()
        if expected_row_ids != delta_ledger_ids:
            raise RuntimeError(
                "append_new_data delta ledger does not cover all expected affected row identities. "
                f"Expected subset: {sorted(expected_row_ids)}; Ledger rows: {sorted(delta_ledger_ids)}"
            )
        if set(delta_ledger_df['source_file'].astype(str)) != set(plan['selected_sources']):
            raise RuntimeError('Delta ledger must cover exactly the planned source files.')

        # Identity/receipt checks above still enforce the scientific generation.
        # Valid historical censored masters gain only recognized NaN columns.
        existing_matrix_df = complete_master_schema(existing_matrix_df)
        delta_matrix_df = complete_master_schema(delta_matrix_df)
        existing_cols = list(existing_matrix_df.columns)
        optional_cols = [c for c in delta_matrix_df if c not in existing_cols]
        aligned_cols = existing_cols + optional_cols
        existing_matrix_df = existing_matrix_df.reindex(columns=aligned_cols)
        delta_matrix_df = delta_matrix_df.reindex(columns=aligned_cols)

        affected_delta_ids = delta_matrix_df["individual"].astype(str).tolist()
        replaced_row_ids = sorted(set(existing_matrix_df["individual"].astype(str)).intersection(affected_delta_ids))
        appended_row_ids = sorted(set(affected_delta_ids) - set(replaced_row_ids))

        merged_matrix_df = pd.concat(
            [
                existing_matrix_df.loc[~existing_matrix_df["individual"].astype(str).isin(affected_delta_ids)].copy(),
                delta_matrix_df.copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        dup_mask = merged_matrix_df["individual"].astype(str).duplicated(keep=False)
        if dup_mask.any():
            dupes = sorted(merged_matrix_df.loc[dup_mask, "individual"].astype(str).unique().tolist())
            raise RuntimeError(
                "append_new_data produced duplicate canonical row identities after merge: "
                + ", ".join(dupes[:10])
                + (" ..." if len(dupes) > 10 else "")
            )

        affected_row_ids = list(map(str, plan.get("expected_row_identities", [])))
        merged_ledger_df = pd.concat(
            [
                existing_ledger_df.loc[~existing_ledger_df["row_identity"].astype(str).isin(affected_row_ids)].copy(),
                delta_ledger_df.copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        merged_ledger_df = merged_ledger_df.sort_values(["row_identity", "source_file"], kind="stable").reset_index(drop=True)

        # Stage the entire canonical generation; never publish matrix before ledger.
        stage_dir=Path(root_out)/'_mmgen_work/append_stage'
        stage_dir.mkdir(parents=True,exist_ok=True)
        _atomic_write_df_csv(stage_dir/'MASTER_FEATURE_MATRIX.csv',merged_matrix_df)
        write_schema_metadata(stage_dir)
        _atomic_write_df_csv(stage_dir/SOURCE_LEDGER_FILENAME,merged_ledger_df)
        old_generation=_verified_generation(root_out)
        combined_sources = {**old_generation['identity']['sources'],
                            **delta_generation['identity']['sources']}
        if set(combined_sources) != set(merged_ledger_df['source_file'].astype(str)):
            raise RuntimeError('Combined source identities and canonical ledger disagree.')
        new_identity=dict(calculation=old_generation['identity']['calculation'],
                          calculation_sha256=plan['calculation_sha256'],sources=combined_sources,
                          sources_sha256=content_hash(combined_sources))
        atomic_write_json(new_identity,stage_dir/IDENTITY_RELATIVE_PATH)

        animal_mapping, _file_to_animal = create_animal_mapping(str(cfg.inference_csv_dir), str(cfg.poscalc_dir), root_out)
        row_identity_df = build_row_identity_summary_df(
            source_ledger_df=merged_ledger_df,
            master_matrix_df=merged_matrix_df,
        )
        write_row_identity_summary_csv(str(stage_dir), row_identity_df)

        lengths_like_df = merged_ledger_df[["source_file"]].drop_duplicates().copy() if not merged_ledger_df.empty else pd.DataFrame(columns=["source_file"])
        processing = plan.get("processing", {}) or {}
        write_batch_manifest_json(
            output_dir=str(stage_dir),
            inference_csv_dir=str(cfg.inference_csv_dir),
            poscalc_dir=str(cfg.poscalc_dir),
            inf_tag=str(processing.get("inf_tag", cfg.inf_tag)),
            min_bout=int(processing.get("min_bout", cfg.min_bout)),
            edge_skip=int(processing.get("edge_skip", cfg.edge_skip)),
            resume=False,
            finalize_only=False,
            process_only=False,
            file_filter=processing.get("file_filter"),
            num_shards=int(processing.get("active_num_shards", 1)),
            shard_index=0,
            shard_by=str(processing.get("shard_by", "hash")),
            emit_master_matrix=True,
            emit_full_outputs=bool(processing.get("emit_full_outputs", False)),
            lengths_df=lengths_like_df,
            source_ledger_df=merged_ledger_df,
            row_identity_df=row_identity_df,
            master_matrix_df=merged_matrix_df,
            animal_mapping=animal_mapping,
            append_new_data=True,
            source_frame_offset=int(cfg.runtime_settings.source_frame_offset),
        )

        if not validate_feature_matrix(str(stage_dir/'MASTER_FEATURE_MATRIX.csv'))[0]:
            raise RuntimeError('Staged canonical matrix failed validation.')
        from .feature_matrix import write_feature_units_manifest
        write_feature_units_manifest(merged_matrix_df, str(stage_dir),
            mm_per_px=(0.0169896744624207 if cfg.mm_per_px is None else float(cfg.mm_per_px)),
            fps=(40.0 if cfg.video_fps is None else float(cfg.video_fps)))
        artifacts={}
        for name in CORE_ARTIFACTS:
            artifacts[name]=sha256_file(stage_dir/name)
        artifacts['MASTER_FEATURE_MATRIX_SCHEMA.json'] = sha256_file(stage_dir/'MASTER_FEATURE_MATRIX_SCHEMA.json')
        artifacts['MASTER_FEATURE_MATRIX_UNITS.csv'] = sha256_file(stage_dir/'MASTER_FEATURE_MATRIX_UNITS.csv')
        historical = {name: record for name, record in
                      old_generation['receipt'].get('historical_artifacts', {}).items()
                      if name not in artifacts}
        for name, digest in old_generation['receipt']['artifacts'].items():
            if name not in artifacts:
                historical[name] = {'sha256': digest,
                    'calculation_sha256': old_generation['identity']['calculation_sha256'],
                    'sources_sha256': old_generation['identity']['sources_sha256']}
        receipt=dict(calculation_sha256=new_identity['calculation_sha256'],
                     sources_sha256=new_identity['sources_sha256'],artifacts=artifacts,
                     historical_artifacts=historical,
                     output_families={'canonical_matrix': 'completed',
                                      'canonical_full_outputs': 'not_refreshed_by_append'})
        atomic_write_json(receipt,stage_dir/OUTPUT_RECEIPT_FILENAME)
        # Re-verify sources and old canonical generation immediately before publication.
        _validate_plan_against_cfg(plan,cfg)
        _publish_generation(root_out,stage_dir,
            (*CORE_ARTIFACTS,'MASTER_FEATURE_MATRIX_SCHEMA.json','MASTER_FEATURE_MATRIX_UNITS.csv',IDENTITY_RELATIVE_PATH,OUTPUT_RECEIPT_FILENAME))

        counts = plan.get("counts", {}) or {}
        report = {
            "append_batch_id": str(plan.get("append_batch_id")),
            "written_at_utc": _utc_now_iso(),
            "delta_output_dir": str(delta_output_dir.resolve()),
            "file_filter": processing.get("file_filter"),
            "counts": {
                "selected_inventory_files": int(counts.get("selected_inventory_files", 0)),
                "new_source_files": int(counts.get("new_source_files", 0)),
                "unchanged_selected_files": int(counts.get("unchanged_selected_files", 0)),
                "affected_row_identities": int(counts.get("affected_row_identities", len(affected_row_ids))),
                "delta_processing_files": int(counts.get("delta_processing_files", 0)),
                "replaced_rows": int(len(replaced_row_ids)),
                "appended_rows": int(len(appended_row_ids)),
                "canonical_matrix_rows_after_merge": int(len(merged_matrix_df)),
                "canonical_ledger_rows_after_merge": int(len(merged_ledger_df)),
                "active_num_shards": int(processing.get("active_num_shards", 1)),
            },
            "affected_row_identities": affected_row_ids,
            "replaced_row_identities": replaced_row_ids,
            "appended_row_identities": appended_row_ids,
            "new_source_files": list(map(str, plan.get("new_source_files", []))),
            "notes": {
                "canonical_full_outputs_updated_in_place": False,
                "delta_full_outputs_location": str(delta_output_dir.resolve()) if bool(processing.get("emit_full_outputs", False)) else None,
                "canonical_master_matrix_updated": True,
                "canonical_ledger_updated": True,
                "canonical_row_identity_summary_updated": True,
                "batch_mode": "sharded" if int(processing.get("active_num_shards", 1)) > 1 else "single_job",
            },
        }
        _write_append_report(str(delta_output_dir), report, latest=False)
        _write_append_report(root_out, report, latest=True)

        print(
            f"[mmgen-append] Canonical MASTER_FEATURE_MATRIX.csv updated successfully. "
            f"Replaced {len(replaced_row_ids)} rows and appended {len(appended_row_ids)} rows."
        )
        if bool(processing.get("emit_full_outputs", False)):
            print(
                "[mmgen-append] Full/user-facing outputs were written to the append batch subfolder only; "
                "the canonical root 1_/2_/... outputs were not modified in place."
            )
    finally:
        try:
            append_lock.release()
        except Exception:
            pass


def _run_single_job_append_from_plan(*, cfg, plan: dict[str, Any], selected_files_df: pd.DataFrame, batch_dir: Path) -> None:
    """Execute or resume one frozen batch; reuse a verified delta after publication failure."""
    if int(cfg.shard_index) != 0:
        raise ValueError('A one-job append plan requires shard_index=0')
    with canonical_writer(batch_dir):
        _validate_plan_against_cfg(plan, cfg)
        _recover_publication(batch_dir)
        processing = plan.get('processing', {}) or {}
        selected_files = selected_files_df['source_file'].astype(str).tolist()
        if not selected_files:
            raise RuntimeError('Frozen append plan contains no selected source files.')
        process_only = bool(cfg.process_only)
        finalize_only = bool(cfg.finalize_only or cfg.merge_shards)
        temp_path = batch_dir / 'temp_store.h5'

        if (batch_dir / OUTPUT_RECEIPT_FILENAME).is_file():
            delta = _verified_generation(batch_dir)
            if (delta['identity']['calculation_sha256'] != plan['calculation_sha256']
                    or delta['identity']['sources'] != plan['selected_sources']):
                raise RuntimeError('Completed delta differs from the frozen append plan.')
            print('[mmgen-append] Reusing the verified completed delta generation.')
        else:
            if finalize_only and not temp_path.is_file():
                raise FileNotFoundError('One-job append finalization needs its retained store; run the planned processing step first.')
            print(f"[mmgen-append] Running one-job batch {plan.get('append_batch_id')} on {len(selected_files)} frozen sources.")
            pipeline.run_post_inference_analysis(
                str(cfg.inference_csv_dir), str(cfg.poscalc_dir), str(batch_dir),
                int(processing.get('min_bout', cfg.min_bout)),
                int(processing.get('edge_skip', cfg.edge_skip)),
                str(processing.get('inf_tag', cfg.inf_tag)),
                runtime_settings=replace(cfg.runtime_settings, keep_temp_files=True,
                    emit_master_matrix=not process_only,
                    emit_full_outputs=bool(processing.get('emit_full_outputs', False)) and not process_only),
                resume=bool(cfg.resume), finalize_only=finalize_only, process_only=process_only,
                file_filter=None, num_shards=1, shard_index=0, shard_by='hash',
                selected_files=selected_files,
            )
        if process_only:
            return
        verified_stores = [(str(temp_path), sha256_file(temp_path))] if temp_path.is_file() else []
        _apply_delta_batch_to_canonical(cfg=cfg, plan=plan, delta_output_dir=batch_dir)
        if not cfg.keep_temp_files:
            _cleanup_append_batch_temp_artifacts(batch_dir, verified_stores)


def _run_append_shard_from_plan(*, cfg, plan: dict[str, Any], selected_files_df: pd.DataFrame, batch_dir: Path) -> None:
    """Process one animal-row partition of the frozen append batch."""
    _validate_plan_against_cfg(plan,cfg)
    processing = plan.get("processing", {}) or {}
    active_num_shards = int(processing.get("active_num_shards", 1))
    shard_by = str(processing.get("shard_by", "row_identity"))
    if active_num_shards == 1:
        _run_single_job_append_from_plan(cfg=cfg, plan=plan, selected_files_df=selected_files_df, batch_dir=batch_dir)
        return
    if int(cfg.shard_index) < 0 or int(cfg.shard_index) >= active_num_shards:
        raise ValueError(f"append shard_index must be in [0, {active_num_shards-1}], got {cfg.shard_index}")

    print(
        f"[mmgen-append] Processing append batch {plan.get('append_batch_id')} shard {int(cfg.shard_index)}/{active_num_shards-1} "
        f"using shard_by={shard_by}."
    )
    pipeline.run_post_inference_analysis(
        str(cfg.inference_csv_dir),
        str(cfg.poscalc_dir),
        str(batch_dir),
        int(processing.get("min_bout", cfg.min_bout)),
        int(processing.get("edge_skip", cfg.edge_skip)),
        str(processing.get("inf_tag", cfg.inf_tag)),
        runtime_settings=replace(
            cfg.runtime_settings,
            emit_master_matrix=False,
            emit_full_outputs=False,
        ),
        resume=bool(cfg.resume),
        finalize_only=False,
        process_only=True,
        file_filter=None,
        num_shards=active_num_shards,
        shard_index=int(cfg.shard_index),
        shard_by=shard_by,
        selected_files=selected_files_df["source_file"].astype(str).tolist() if not selected_files_df.empty else [],
    )


def _validate_append_shard_done_markers(*, batch_dir: Path, active_num_shards: int, shard_by: str) -> None:
    """Require matching completion markers and stores for every planned append shard."""
    missing_done: list[int] = []
    bad_done: list[str] = []
    for i in range(int(active_num_shards)):
        shard_dir = batch_dir / "_mmgen_shards" / f"shard_{i:03d}_of_{int(active_num_shards):03d}"
        done_path = shard_dir / "_mmgen_work" / "SHARD_DONE.json"
        store_path = shard_dir / "temp_store.h5"
        if not done_path.exists():
            missing_done.append(i)
            continue
        if not store_path.exists():
            bad_done.append(f"missing temp_store.h5 for shard {i}: {store_path}")
            continue
        try:
            payload = json.loads(done_path.read_text(encoding="utf-8"))
        except Exception as e:
            bad_done.append(f"failed to parse {done_path}: {e}")
            continue
        if int(payload.get("num_shards", -1)) != int(active_num_shards):
            bad_done.append(f"{done_path} reports num_shards={payload.get('num_shards')} (expected {active_num_shards})")
        if int(payload.get("shard_index", -1)) != int(i):
            bad_done.append(f"{done_path} reports shard_index={payload.get('shard_index')} (expected {i})")
        if str(payload.get("shard_by", "")) != str(shard_by):
            bad_done.append(f"{done_path} reports shard_by={payload.get('shard_by')} (expected {shard_by})")

    if missing_done:
        raise RuntimeError(
            "append_new_data strict shard merge requires all shard completion markers to be present. Missing shard(s): "
            + ", ".join(map(str, missing_done))
        )
    if bad_done:
        raise RuntimeError(
            "append_new_data found inconsistent shard completion metadata before merge:\n  - "
            + "\n  - ".join(bad_done[:20])
            + ("\n  ..." if len(bad_done) > 20 else "")
        )


def _merge_finalize_append_batch(*, cfg, plan: dict[str, Any], batch_dir: Path) -> None:
    """Merge completed append shards and publish affected animal rows under a canonical
    lock.
    """
    _validate_plan_against_cfg(plan,cfg)
    processing = plan.get("processing", {}) or {}
    active_num_shards = int(processing.get("active_num_shards", 1))
    shard_by = str(processing.get("shard_by", "row_identity"))
    if active_num_shards == 1:
        selected_df = read_metadata_csv(batch_dir / APPEND_SELECTED_FILES_FILENAME)
        _run_single_job_append_from_plan(cfg=cfg, plan=plan, selected_files_df=selected_df, batch_dir=batch_dir)
        return

    merge_lock_dir = batch_dir / "_mmgen_work" / "_mmgen_merge.lock"
    os.makedirs(str(merge_lock_dir.parent), exist_ok=True)
    merge_lock = DirLock(merge_lock_dir, timeout_sec=0.0)
    try:
        merge_lock.acquire(metadata={"kind": "mmgen_append_merge", "batch_id": str(plan.get("append_batch_id"))})
    except Exception as e:
        raise RuntimeError(
            "[mmgen-append] Append-batch merge lock already exists. Another merge/finalize job may already be running for this batch.\n"
            f"  lock: {merge_lock_dir}\n\n"
            "Native locks release automatically when their writer exits. "
            f"(Original error: {e})"
        )

    try:
        _validate_append_shard_done_markers(batch_dir=batch_dir, active_num_shards=active_num_shards, shard_by=shard_by)

        verified_stores: list[tuple[str, str]] = []
        merge_shard_temp_stores(
            output_folder=str(batch_dir),
            num_shards=active_num_shards,
            strict_missing=True,
            verified_store_identities=verified_stores,
        )

        pipeline.run_post_inference_analysis(
            str(cfg.inference_csv_dir),
            str(cfg.poscalc_dir),
            str(batch_dir),
            int(processing.get("min_bout", cfg.min_bout)),
            int(processing.get("edge_skip", cfg.edge_skip)),
            str(processing.get("inf_tag", cfg.inf_tag)),
            runtime_settings=replace(
                cfg.runtime_settings,
                emit_master_matrix=True,
                emit_full_outputs=bool(processing.get("emit_full_outputs", False)),
            ),
            resume=False,
            finalize_only=True,
            process_only=False,
            file_filter=None,
            num_shards=1,
            shard_index=0,
            shard_by="hash",
            selected_files=list(plan.get("selected_sources", {})),
        )
        _apply_delta_batch_to_canonical(cfg=cfg, plan=plan, delta_output_dir=batch_dir)

        if not cfg.keep_temp_files:
            _cleanup_append_batch_temp_artifacts(batch_dir, verified_stores)
    finally:
        try:
            merge_lock.release()
        except Exception:
            pass


def run_mmgen_append_new_data(cfg) -> None:
    """Plan, execute, resume or finalize a frozen append batch of any active size.

    An existing explicit append_batch_id always reuses its saved selection.
    One-job plans, including larger plans reduced to one affected animal, support
    process_only followed by merge_shards/finalize_only, or a direct execution.
    """
    root_out = str(cfg.output_dir)
    os.makedirs(root_out, exist_ok=True)
    if bool(cfg.plan_only):
        _create_append_plan(cfg)
        return
    if bool(cfg.process_only) and bool(cfg.merge_shards or cfg.finalize_only):
        raise ValueError('Append processing and finalization must be separate invocations.')

    batch_id = getattr(cfg, 'append_batch_id', None)
    explicit_plan = bool(batch_id and (_append_batch_dir(root_out, str(batch_id)) / APPEND_PLAN_FILENAME).is_file())
    if explicit_plan or bool(cfg.process_only or cfg.merge_shards or cfg.finalize_only):
        if not batch_id:
            raise ValueError('Planned append processing/finalization requires append_batch_id')
        plan, selected_df, _affected_df, batch_dir = _load_append_plan(root_out, str(batch_id))
    else:
        if int(cfg.num_shards) > 1:
            raise ValueError('append_new_data with num_shards > 1 requires a saved --plan_only batch first')
        plan, selected_df, _affected_df, batch_dir = _create_append_plan(cfg)
        if plan is None:
            return
    _validate_plan_against_cfg(plan, cfg)
    active = int(plan['processing']['active_num_shards'])
    if active == 1:
        _run_single_job_append_from_plan(cfg=cfg, plan=plan, selected_files_df=selected_df, batch_dir=batch_dir)
    elif bool(cfg.merge_shards or cfg.finalize_only):
        _merge_finalize_append_batch(cfg=cfg, plan=plan, batch_dir=batch_dir)
    elif bool(cfg.process_only):
        _run_append_shard_from_plan(cfg=cfg, plan=plan, selected_files_df=selected_df, batch_dir=batch_dir)
    else:
        raise ValueError('A multi-job append plan requires per-shard --process_only, then --merge_shards.')


def _load_required_canonical_state(output_folder: str):
    with canonical_writer(output_folder):
        return _load_required_canonical_state_owned(output_folder)


def _create_append_plan(cfg):
    with canonical_writer(cfg.output_dir):
        return _create_append_plan_owned(cfg)


def _validate_plan_against_cfg(plan: dict[str, Any], cfg):
    with canonical_writer(cfg.output_dir):
        return _validate_plan_against_cfg_owned(plan, cfg)


def _apply_delta_batch_to_canonical(*, cfg, plan: dict[str, Any], delta_output_dir: Path):
    with canonical_writer(cfg.output_dir):
        return _apply_delta_batch_to_canonical_owned(cfg=cfg, plan=plan, delta_output_dir=delta_output_dir)
