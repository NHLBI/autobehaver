"""Temporary HDF5 store utilities for mmgen.

mmgen uses a single on-disk HDF5 file (via pandas.HDFStore/PyTables) as a
memory-safe aggregation buffer while processing many recordings in parallel.

This module centralizes:
  - HDFStore compression/options
  - canonical HDF keys for each logical table
  - schema-safe append logic (including schema expansion when new columns
    appear later in the run)
  - helper utilities for initializing and cleaning up the temp store

The store utilities centralize the on-disk HDF5 aggregation behavior used by mmgen
without changing behavior.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical temp-store configuration
# ---------------------------------------------------------------------------

# HDF5 options (compression helps IO without changing logic)
HDF_OPTS: Dict[str, object] = dict(complevel=5, complib="blosc")


# HDF5 keys for each logical table
HDF_KEYS: Dict[str, str] = {
    "bouts_inst": "/bouts_inst",
    "bouts_5": "/bouts_5",
    "bouts_9": "/bouts_9",
    "positional": "/positional",
    "postural": "/postural",
    # Per-frame kinematic signals for TW kinematics
    "kin_frames": "/kin_frames",
    # Compact aggregate tables written per file, read at the end
    "agg_inst_tmp": "/agg_inst_tmp",
    "agg_5_tmp": "/agg_5_tmp",
    "agg_9_tmp": "/agg_9_tmp",
    "transitions_tmp": "/transitions_tmp",
    # Compact per-recording standardized measurement summaries
    "measurements_tmp": "/measurements_tmp",
    # Per-recording bout and sleep summary tables
    "bout_summary_tmp": "/bout_summary_tmp",
    "sleep_summary_tmp": "/sleep_summary_tmp",

    # Resume / progress tracking (one row per processed inference CSV)
    "video_lengths_tmp": "/video_lengths_tmp",
}


# Conservative string sizes to avoid PyTables resizing overhead
HDF_MIN_ITEMSIZE: Dict[str, int] = {
    "group": 32,
    "individual": 128,
    "bout_type": 96,
    "category": 16,
    "source_file": 512,
    "sex": 8,
    "from_behavior": 96,
    "to_behavior": 96,
    "kin_context_valid": 128,
    # Standardized measurements
    "group_label": 32,
    "individual_id": 128,
}


# Columns you might want indexed/filterable in HDF
HDF_DATA_COLUMN_CANDIDATES: List[str] = [
    "group",
    "individual",
    "bout_type",
    "category",
    "source_file",
    "sex",
    "frame",
    "frame_idx",
    # for TW per-frame tables
    "kin_context_valid",
    # include transition columns so we can filter/query if needed
    "from_behavior",
    "to_behavior",
    # standardized measurements
    "group_label",
    "individual_id",
]


# Define the complete, static schema for postural and positional features.
# These represent ALL possible columns that can be created.

# POSTURAL FEATURES - Complete list from _calculate_postural_features
PREDEFINED_POSTURAL_COLS: List[str] = [
    "group",
    "individual",
    "masked_orient",
    # Stance and extension features (for Floor and Ceiling)
    "stance_area",
    "extension_raw_FrontLegs_avg",
    "extension_raw_MidLegs_avg",
    "extension_raw_BackLegs_avg",
    # Left-Right asymmetry features (for Floor orientation only)
    "asymmetry_magnitude_Eyes_LR",
    "asymmetry_magnitude_Thorax_LR",
    "asymmetry_magnitude_Abdomen_LR",
    "asymmetry_magnitude_Wings_LR",
    "asymmetry_magnitude_FrontLegs_LR",
    "asymmetry_magnitude_MidLegs_LR",
    "asymmetry_magnitude_BackLegs_LR",
    "overall_asymmetry_LR",
    # Top-Bottom asymmetry features (for Wall-Side orientation only)
    "asymmetry_magnitude_Head_TB",
    "asymmetry_magnitude_Thorax_TB",
    "asymmetry_magnitude_Abdomen_TB",
    "overall_asymmetry_TB",
]


# POSITIONAL FEATURES - Complete list from process_video_data
PREDEFINED_POSITIONAL_COLS: List[str] = [
    "group",
    "individual",
    "source_file",
    "frame_idx",
    # Raw (smoothed) labels
    "masked_pose",
    "masked_orient",
    "masked_pose_orient",
    # Bout-filtered labels
    "masked_pose_valid",
    "masked_orient_valid",
    "masked_pose_orient_valid",
    # Position + orientation
    "centroid_x",
    "centroid_y",
    "angle",
    # Per-frame movement + angle features used by additional metrics
    "speed_mm_s",
    "anglefeature_Wings_L_R",
    "anglefeature_Legs_FL_FR",
    "anglefeature_Legs_ML_MR",
    "anglefeature_Legs_BL_BR",
]


def open_store(path: str, mode: str) -> pd.HDFStore:
    """Open a pandas HDFStore using the canonical temp-store options."""
    return pd.HDFStore(path, mode=mode, **HDF_OPTS)


def hdf_safe_append(
    store: pd.HDFStore,
    key: str,
    df: Optional[pd.DataFrame],
    *,
    data_column_candidates: Optional[List[str]] = None,
    min_itemsize_map: Optional[Dict[str, int]] = None,
) -> None:
    """Append a DataFrame to HDF5 with schema alignment and optional schema expansion.

    The helper mirrors the current mmgen HDF5 row-existence check.
    """

    if df is None or df.empty:
        return

    # ------------------------------------------------------------------
    # IMPORTANT: PyTables table appends require *stable dtypes*.
    #
    # In mmgen we append per-file aggregate tables ("/agg_*_tmp") where some
    # columns may be absent for certain files (e.g., missing kinematic columns).
    # When we later align schemas via reindex(), those absent columns become
    # NaN-filled. If the table schema was created with integer dtypes (common for
    # count-like columns), the NaNs will upcast incoming data to float and
    # PyTables will raise:
    #   ValueError: cannot match existing table structure ... on appending data
    #
    # To make the aggregation tables robust, we proactively promote integer
    # columns to float for agg tmp keys so NaNs are representable.
    # This does not change downstream calculations (counts are still numeric).
    # ------------------------------------------------------------------
    if key.startswith("/agg_") and key.endswith("_tmp"):
        # Copy only for these small aggregate tables (avoid copying huge per-frame tables).
        df = df.copy()
        int_cols = df.select_dtypes(include=["int", "int64", "uint", "uint64"]).columns
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype(float)

    _candidates = data_column_candidates or HDF_DATA_COLUMN_CANDIDATES
    _min_itemsize = min_itemsize_map or HDF_MIN_ITEMSIZE

    # ------------------------------------------------------------------
    # 1) If the key exists, align schemas (and expand schema if needed)
    # ------------------------------------------------------------------
    if key in store:
        existing_schema = store.select(key, start=0, stop=0)  # 0-row DF, preserves columns+dtypes
        existing_cols = list(existing_schema.columns)

        # Columns present in incoming df but missing from the stored table schema
        cols_to_add_to_existing = [c for c in df.columns if c not in existing_cols]

        if cols_to_add_to_existing:
            # IMPORTANT: Without this, Pandas will silently drop these new columns by reordering.
            print(f"\n[HDF SCHEMA EXPAND] Key={key}  Adding {len(cols_to_add_to_existing)} new columns")
            preview = cols_to_add_to_existing[:20]
            print(
                f"  New cols (first {len(preview)}): {preview}{' ...' if len(cols_to_add_to_existing) > 20 else ''}"
            )

            # If table already has rows, rewrite it into a temp key with the expanded schema
            nrows = getattr(store.get_storer(key), "nrows", 0)
            if nrows > 0:
                expanded_cols = existing_cols + cols_to_add_to_existing
                tmp_key = f"{key}__schema_expand_tmp"

                # Remove a leftover tmp key from a previous crash/run
                if tmp_key in store:
                    store.remove(tmp_key)

                # Build append options for expanded schema
                data_cols_expanded = [c for c in _candidates if c in expanded_cols]
                str_cols_expanded = sorted(
                    set(existing_schema.select_dtypes(include="object").columns.tolist())
                    | set(df.select_dtypes(include="object").columns.tolist())
                )
                min_item_expanded = {c: _min_itemsize.get(c, 64) for c in str_cols_expanded}

                # Rewrite existing data in chunks (memory-safe)
                for chunk in store.select(key, chunksize=50000, iterator=True):
                    # Reindex adds missing columns (filled with NaN) and enforces column order in one step.
                    chunk = chunk.reindex(columns=expanded_cols)

                    store.append(
                        tmp_key,
                        chunk,
                        format="table",
                        data_columns=data_cols_expanded,
                        index=False,
                        # Keep all-NA columns: schema expansion often introduces
                        # new columns that are initially all NaN for earlier chunks.
                        dropna=False,
                        min_itemsize=min_item_expanded if min_item_expanded else None,
                    )

                # Replace the stored key with the expanded key
                store.remove(key)
                store._handle.rename_node("/", key.lstrip("/"), name=tmp_key.lstrip("/"))

                # Refresh schema after expansion
                existing_schema = store.select(key, start=0, stop=0)
                existing_cols = list(existing_schema.columns)
            else:
                # Table exists but has no rows: easiest is to drop it and let the new append recreate it.
                store.remove(key)
                existing_schema = None
                existing_cols = list(df.columns)

        # Align to stored schema: add missing columns (NaN), drop extras, and match column order.
        # This avoids repeatedly inserting columns (which can fragment the DataFrame and slow down).
        df = df.reindex(columns=existing_cols)

        # Coerce dtypes where it matters (especially for new columns which become float due to NaNs)
        if existing_schema is not None:
            for c in existing_cols:
                try:
                    target = existing_schema[c].dtype

                    # If stored dtype is float, force incoming to float
                    if pd.api.types.is_float_dtype(target):
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

                    # If stored dtype is object, ensure object
                    elif pd.api.types.is_object_dtype(target):
                        df[c] = df[c].astype(object)

                    # If stored dtype is int, try to coerce only when safe (no NaNs)
                    elif pd.api.types.is_integer_dtype(target):
                        if pd.api.types.is_float_dtype(df[c].dtype) and df[c].notna().all():
                            ser = df[c]
                            if ((ser % 1) == 0).all():
                                df[c] = ser.astype(target)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 2) Append (key may or may not exist at this point)
    # ------------------------------------------------------------------
    # Consolidate block layout for very wide aggregate tables.
    #
    # Some pandas versions can leave DataFrames fragmented (multiple blocks of
    # the same dtype) after dtype conversions / reindexing. PyTables encodes the
    # block partitioning in the table schema, so a different fragmentation
    # pattern on a later append can raise:
    #   ValueError: cannot match existing table structure ... on appending data
    #
    # A deep copy consolidates blocks and makes appends deterministic.
    if key.startswith("/agg_") and key.endswith("_tmp"):
        df = df.copy()

    data_cols = [c for c in _candidates if c in df.columns]

    # Preset min_itemsize for all object (string) columns present
    str_cols = df.select_dtypes(include="object").columns.tolist()
    min_item = {c: _min_itemsize.get(c, 64) for c in str_cols}

    store.append(
        key,
        df,
        format="table",
        data_columns=data_cols,
        index=False,
        # Critical for schema-stable appends: some aligned/expanded columns may
        # be all-NA for a given file/batch. If pandas drops them, PyTables
        # raises "cannot match existing table structure".
        dropna=False,
        min_itemsize=min_item if min_item else None,
    )


def initialize_temp_store(
    temp_store_path: str,
    *,
    bout_dict: Optional[Dict[str, pd.DataFrame]],
    pos_df: Optional[pd.DataFrame],
    post_df: Optional[pd.DataFrame],
    kin_df: Optional[pd.DataFrame],
    predefined_kin_cols: Optional[Iterable[str]],
) -> None:
    """Initialize temp HDF5 tables.

    This helper initializes the canonical temp-store tables for a run.
    """

    predefined_kin_cols_list = list(predefined_kin_cols) if predefined_kin_cols is not None else []

    with open_store(temp_store_path, mode="w") as store:
        if bout_dict:
            for suffix, df in bout_dict.items():
                if df is not None and not df.empty:
                    key = HDF_KEYS[f"bouts{suffix}"]
                    hdf_safe_append(store, key, df.head(0))

        if pos_df is not None and not pos_df.empty:
            all_pos_cols = sorted(set(PREDEFINED_POSITIONAL_COLS) | set(pos_df.columns))
            empty_pos_df = pd.DataFrame(columns=all_pos_cols)
            hdf_safe_append(store, HDF_KEYS["positional"], empty_pos_df)
        elif pos_df is None or pos_df.empty:
            empty_pos_df = pd.DataFrame(columns=PREDEFINED_POSITIONAL_COLS)
            hdf_safe_append(store, HDF_KEYS["positional"], empty_pos_df)

        if post_df is not None and not post_df.empty:
            all_post_cols = sorted(set(PREDEFINED_POSTURAL_COLS) | set(post_df.columns))
            empty_post_df = pd.DataFrame(columns=all_post_cols)
            hdf_safe_append(store, HDF_KEYS["postural"], empty_post_df)
        elif post_df is None or post_df.empty:
            empty_post_df = pd.DataFrame(columns=PREDEFINED_POSTURAL_COLS)
            hdf_safe_append(store, HDF_KEYS["postural"], empty_post_df)

        if kin_df is not None and not kin_df.empty:
            all_kin_cols = sorted(set(predefined_kin_cols_list) | set(kin_df.columns))
            empty_kin_df = pd.DataFrame(columns=all_kin_cols)
            hdf_safe_append(store, HDF_KEYS["kin_frames"], empty_kin_df)
        else:
            empty_kin_df = pd.DataFrame(columns=predefined_kin_cols_list)
            hdf_safe_append(store, HDF_KEYS["kin_frames"], empty_kin_df)


def cleanup_temp_store(temp_store_path: str, *, keep_temp_files: bool) -> None:
    """Remove the temp store file unless `keep_temp_files` is True."""

    print("\nCleaning up temporary files...")
    try:
        import os

        if os.path.exists(temp_store_path):
            if keep_temp_files:
                print(f"Keeping temp file for debugging: {temp_store_path}")
            else:
                os.remove(temp_store_path)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Resume helpers
# -----------------------------------------------------------------------------


def get_done_source_files(temp_store_path: str) -> set[str]:
    """Return the set of inference CSV basenames that are fully processed.

    In mmgen "resume" mode, we treat the presence of a row in
    ``video_lengths_tmp`` as a *commit marker* that all tables for that file
    were appended successfully.

    If the key is missing (older temp stores) or unreadable, returns an empty
    set.
    """

    try:
        df = pd.read_hdf(temp_store_path, key=HDF_KEYS["video_lengths_tmp"])
        if df is None or df.empty or "source_file" not in df.columns:
            return set()
        return set(df["source_file"].astype(str).tolist())
    except Exception:
        return set()


def _store_select_any(store: pd.HDFStore, key: str, *, where: str) -> bool:
    """Fast existence check for a where-clause query."""

    if key not in store:
        return False
    try:
        out = store.select(key, where=where, start=0, stop=1)
        return out is not None and not out.empty
    except Exception:
        return False


def file_has_any_rows(
    temp_store_path: str,
    *,
    source_file: str,
    individual_id: str | None = None,
) -> bool:
    """Return True if the temp store already contains any rows for this file."""

    sf = str(source_file)
    ind = str(individual_id) if individual_id is not None else None

    with open_store(temp_store_path, mode="r") as store:
        # Prefer lightweight summary tables first (fewer rows).
        if _store_select_any(store, HDF_KEYS["bout_summary_tmp"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["sleep_summary_tmp"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["measurements_tmp"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["transitions_tmp"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["kin_frames"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["positional"], where=f'source_file == "{sf}"'):
            return True

        # Postural table doesn't carry source_file; fall back to individual id.
        if ind is not None and _store_select_any(store, HDF_KEYS["postural"], where=f'individual == "{ind}"'):
            return True

        # Bouts/agg tables can be large; only check the instantaneous one.
        if _store_select_any(store, HDF_KEYS["bouts_inst"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["agg_inst_tmp"], where=f'source_file == "{sf}"'):
            return True

        if _store_select_any(store, HDF_KEYS["video_lengths_tmp"], where=f'source_file == "{sf}"'):
            return True

    return False


def _rewrite_key_excluding_rows(
    store: pd.HDFStore,
    *,
    key: str,
    tmp_key: str,
    drop_mask_fn,
    chunksize: int = 50_000,
) -> None:
    """Rewrite a table-format HDF5 key, excluding rows where drop_mask_fn is True."""

    if key not in store:
        return

    # Remove any stale tmp key.
    if tmp_key in store:
        try:
            store.remove(tmp_key)
        except Exception:
            pass

    wrote_any = False
    try:
        it = store.select(key, iterator=True, chunksize=int(chunksize))
        for chunk in it:
            if chunk is None or chunk.empty:
                continue
            try:
                mask = drop_mask_fn(chunk)
                filtered = chunk.loc[~mask] if mask is not None else chunk
            except Exception:
                # If masking fails, keep the chunk (defensive).
                filtered = chunk
            if filtered is None or filtered.empty:
                continue
            hdf_safe_append(store, tmp_key, filtered)
            wrote_any = True
    except Exception:
        # If streaming fails, fall back to full read (best-effort).
        try:
            df = store.select(key)
            mask = drop_mask_fn(df)
            filtered = df.loc[~mask] if mask is not None else df
            if filtered is not None and not filtered.empty:
                hdf_safe_append(store, tmp_key, filtered)
                wrote_any = True
        except Exception:
            wrote_any = False

    # If all rows were dropped, we still need to leave behind an empty table
    # with the same schema (otherwise downstream reads can fail).
    if not wrote_any:
        try:
            template = store.select(key, start=0, stop=1)
            empty_df = template.head(0) if template is not None else pd.DataFrame()
            present_data_cols = [col for col in HDF_STORE_BASE_DATA_COLUMNS if col in empty_df.columns]
            empty_df.to_hdf(
                store.filename,
                key=tmp_key,
                mode='a',
                format='table',
                append=False,
                index=False,
                data_columns=present_data_cols,
                min_itemsize=HDF_MIN_ITEMSIZE,
            )
            wrote_any = True
        except Exception:
            # Best-effort: if we can't create an empty replacement, keep the
            # existing key intact rather than deleting it.
            return

    # Replace the existing key.
    try:
        store.remove(key)
    except Exception:
        pass

    try:
        store.rename(tmp_key, key)
    except Exception:
        # If rename fails, leave tmp_key (defensive).
        pass


def purge_file_from_store(
    temp_store_path: str,
    *,
    source_file: str,
    individual_id: str | None = None,
    verbose: bool = True,
) -> None:
    """Remove rows associated with a specific inference CSV from the temp store.

    This is a best-effort recovery helper for resume mode.

    It is only used when the file is *not* present in ``video_lengths_tmp``
    (i.e., not committed), but appears to have written some rows (partial
    write). Purging allows safe re-processing without double-counting.
    """

    sf = str(source_file)
    ind = str(individual_id) if individual_id is not None else None

    if verbose:
        print(f"[mmgen-resume] Purging partial rows for: {sf}")

    with open_store(temp_store_path, mode="a") as store:
        # Keys that include 'source_file'
        sf_keys = [
            HDF_KEYS["bouts_inst"],
            HDF_KEYS["bouts_5"],
            HDF_KEYS["bouts_9"],
            HDF_KEYS["agg_inst_tmp"],
            HDF_KEYS["agg_5_tmp"],
            HDF_KEYS["agg_9_tmp"],
            HDF_KEYS["transitions_tmp"],
            HDF_KEYS["measurements_tmp"],
            HDF_KEYS["bout_summary_tmp"],
            HDF_KEYS["sleep_summary_tmp"],
            HDF_KEYS["positional"],
            HDF_KEYS["kin_frames"],
            HDF_KEYS["video_lengths_tmp"],
        ]

        for key in sf_keys:
            if key not in store:
                continue
            # Quick existence check to avoid rewriting tables unnecessarily.
            if not _store_select_any(store, key, where=f'source_file == "{sf}"'):
                continue
            tmp_key = f"{key}__purge_tmp"
            _rewrite_key_excluding_rows(
                store,
                key=key,
                tmp_key=tmp_key,
                drop_mask_fn=lambda df, sf=sf: df.get("source_file", pd.Series([None] * len(df))).astype(str)
                == sf,
            )

        # Postural table: no source_file; purge by per-file individual id.
        if ind is not None and HDF_KEYS["postural"] in store:
            key = HDF_KEYS["postural"]
            if _store_select_any(store, key, where=f'individual == "{ind}"'):
                tmp_key = f"{key}__purge_tmp"
                _rewrite_key_excluding_rows(
                    store,
                    key=key,
                    tmp_key=tmp_key,
                    drop_mask_fn=lambda df, ind=ind: df.get("individual", pd.Series([None] * len(df))).astype(str)
                    == ind,
                )

