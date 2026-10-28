"""Maintain MMGen temporary HDF5 tables for bounded-memory aggregation.

Align schemas, widen fixed string fields when needed, and retain recording commit
metadata for verified reuse and cleanup.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def select_individual_rows(store: pd.HDFStore, key: str, individual: str, *, columns=None):
    """Select an exact identity through PyTables variable binding, including quotes."""
    selected_id = str(individual)
    return store.select(key, where="individual == selected_id", columns=columns)


# Canonical temp-store configuration

# HDF5 options (compression helps IO without changing logic)
HDF_OPTS: Dict[str, object] = dict(complevel=5, complib="blosc")


# HDF5 keys for each logical table
HDF_KEYS: Dict[str, str] = {
    "bouts_inst": "/bouts_inst",
    "bouts_5": "/bouts_5",
    "bouts_9": "/bouts_9",
    "positional": "/positional",
    "postural": "/postural",
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
    # Standardized measurements
    "group_label": 32,
    "individual_id": 128,
}


# Indexed HDF query columns.
HDF_DATA_COLUMN_CANDIDATES: List[str] = [
    "group",
    "individual",
    "bout_type",
    "category",
    "source_file",
    "sex",
    "frame",
    "frame_idx",
    # include transition columns so we can filter/query if needed
    "from_behavior",
    "to_behavior",
    # standardized measurements
    "group_label",
    "individual_id",
]


# Extra space added above the observed maximum string length when creating or
# expanding HDF5 string columns.  HDF5/PyTables table string columns are
# fixed-width once created; this small pad avoids immediate re-expansion when a
# later batch has a value that is only a few characters longer.
HDF_STRING_WIDTH_PAD: int = 8
HDF_STRING_WIDTH_ROUND_TO: int = 8


def _safe_string_len(value: object, *, encoding: str = "UTF-8") -> int:
    """Return encoded string bytes, treating missing values as length 0."""

    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    return len(str(value).encode(encoding))


def _round_itemsize(width: int, *, round_to: int = HDF_STRING_WIDTH_ROUND_TO) -> int:
    """Round a positive HDF string itemsize up to a small boundary."""

    width = max(1, int(width))
    round_to = max(1, int(round_to))
    return int(((width + round_to - 1) // round_to) * round_to)


def build_hdf_min_itemsize_from_observed_strings(
    observed_values: Dict[str, Iterable[object]],
    *,
    base: Optional[Dict[str, int]] = None,
    pad: int = HDF_STRING_WIDTH_PAD,
    round_to: int = HDF_STRING_WIDTH_ROUND_TO,
) -> Dict[str, int]:
    """Return an HDF ``min_itemsize`` map sized from observed string values.

    Pandas/PyTables table-format HDF5 string columns are fixed-width after a
    table is created.  mmgen therefore needs to choose widths before the first
    append.  This helper keeps the historical defaults as floors, then expands
    any listed column to ``max_observed_length + pad`` rounded up to
    ``round_to``.
    """

    out: Dict[str, int] = dict(HDF_MIN_ITEMSIZE if base is None else base)
    for col, values in (observed_values or {}).items():
        max_len = 0
        for value in values or []:
            max_len = max(max_len, _safe_string_len(value))
        if max_len <= 0:
            continue
        floor = int(out.get(str(col), 64))
        out[str(col)] = max(floor, _round_itemsize(max_len + int(pad), round_to=int(round_to)))
    return out


def _object_string_columns(df: pd.DataFrame) -> List[str]:
    """Identify object columns stored as HDF string fields."""

    cols: List[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_object_dtype(df[c].dtype) or pd.api.types.is_string_dtype(df[c].dtype):
                cols.append(c)
        except Exception:
            pass
    return cols


def _observed_string_max_lens(df: pd.DataFrame, *, columns: Optional[Iterable[str]] = None,
                              encoding: str = "UTF-8") -> Dict[str, int]:
    """Return observed maximum string lengths for selected columns."""

    if df is None or df.empty:
        return {}
    cols = list(columns) if columns is not None else _object_string_columns(df)
    out: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            continue
        max_len = 0
        try:
            # Use a Python loop rather than Series.str.len so mixed object
            # columns and NA values behave consistently.
            for value in df[c].to_numpy(dtype=object, copy=False):
                max_len = max(max_len, _safe_string_len(value, encoding=encoding))
        except Exception:
            try:
                max_len = max(len(str(value).encode(encoding)) for value in df[c])
            except Exception:
                max_len = 0
        if max_len > 0:
            out[str(c)] = int(max_len)
    return out


def _min_itemsize_for_df(df: pd.DataFrame, min_itemsize_map: Dict[str, int], *,
                         encoding: str = "UTF-8") -> Dict[str, int]:
    """Build a min_itemsize dict that also honors values in this DataFrame."""

    str_cols = _object_string_columns(df)
    observed = _observed_string_max_lens(df, columns=str_cols, encoding=encoding)
    out: Dict[str, int] = {}
    for c in str_cols:
        base_width = int(min_itemsize_map.get(c, 64))
        observed_width = observed.get(c, 0)
        if observed_width > 0:
            base_width = max(base_width, _round_itemsize(observed_width + HDF_STRING_WIDTH_PAD))
        out[c] = int(base_width)
    return out


def _merge_itemsize_maps(*maps: Dict[str, int]) -> Dict[str, int]:
    """Retain the largest requested or stored byte width for every column."""
    merged: Dict[str, int] = {}
    for widths in maps:
        for column, width in widths.items():
            merged[column] = max(merged.get(column, 0), int(width))
    return merged


def _stored_string_fields(store: pd.HDFStore, key: str) -> Dict[str, tuple[List[str], int]]:
    """Map physical string fields, including legacy blocks, to logical columns."""
    if key not in store:
        return {}
    storer = store.get_storer(key)
    coldtypes = storer.table.coldtypes
    fields = {}
    for axis in storer.values_axes:
        physical = axis.cname
        dtype = coldtypes.get(physical)
        base = dtype.subdtype[0] if dtype is not None and dtype.subdtype else dtype
        if base is not None and base.kind == "S":
            fields[physical] = (list(axis.values), int(base.itemsize))
    return fields


def _stored_string_itemsize_by_column(store: pd.HDFStore, key: str) -> Dict[str, int]:
    """Return current fixed string widths for a table-format HDF key."""

    if key not in store:
        return {}
    try:
        fields = _stored_string_fields(store, key)
    except Exception:
        return {}

    out: Dict[str, int] = {}
    for columns, width in fields.values():
        for column in columns:
            out[column] = width
    return out


def _required_string_width_updates(
    store: pd.HDFStore,
    key: str,
    df: pd.DataFrame,
    min_itemsize_map: Dict[str, int],
) -> Dict[str, int]:
    """Return HDF string columns that must be widened before appending df."""

    if key not in store or df is None or df.empty:
        return {}
    existing_widths = _stored_string_itemsize_by_column(store, key)
    if not existing_widths:
        return {}
    encoding = store.get_storer(key).encoding
    required = _min_itemsize_for_df(df, min_itemsize_map, encoding=encoding)
    updates: Dict[str, int] = {}
    for col, required_width in required.items():
        if col not in existing_widths:
            continue
        old_width = int(existing_widths[col])
        # Pandas validates min_itemsize itself, including padding and caller floors.
        if int(required_width) > old_width:
            requested = max(
                int(min_itemsize_map.get(col, 64)),
                int(required_width),
                _round_itemsize(old_width * 2),
            )
            updates[col] = int(requested)
    return updates


def _long_string_examples(df: pd.DataFrame, col: str, *, limit: int, n: int = 5) -> List[str]:
    """Return example values from df[col] whose string length exceeds limit."""

    if df is None or df.empty or col not in df.columns:
        return []
    examples: List[str] = []
    seen: set[str] = set()
    try:
        values = df[col].to_numpy(dtype=object, copy=False)
    except Exception:
        values = df[col].tolist()
    for value in values:
        if _safe_string_len(value) <= int(limit):
            continue
        s = str(value)
        if s in seen:
            continue
        seen.add(s)
        examples.append(s)
        if len(examples) >= int(n):
            break
    return examples


def _parse_hdf_string_width_error(exc: BaseException) -> tuple[str, int, int] | None:
    """Parse Pandas/PyTables fixed-string-width append errors."""

    msg = str(exc)
    m = re.search(
        r"Trying to store a string with len \[(\d+)\] in \[(.+?)\] column.*?limit of \[(\d+)\]",
        msg,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    return str(m.group(2)), int(m.group(1)), int(m.group(3))


def _replace_hdf_key_with_tmp(store: pd.HDFStore, *, key: str, tmp_key: str) -> None:
    """Publish a rebuilt table, preserving custom attributes and rollback data."""
    if tmp_key not in store:
        raise RuntimeError(f"Replacement table is missing: {tmp_key}")
    key_path, tmp_path = "/" + key.strip("/"), "/" + tmp_key.strip("/")
    backup = None
    if key in store:
        previous = store.get_storer(key).attrs
        replacement = store.get_storer(tmp_key).attrs
        # Pandas creates fresh schema attributes; retain additional caller metadata.
        for name in previous._v_attrnamesuser:
            if name not in replacement._v_attrnamesuser:
                setattr(replacement, name, getattr(previous, name))
        backup = key_path + "__replace_backup_" + uuid.uuid4().hex[:12]
        store._handle.rename_node(key_path, backup.rsplit("/", 1)[1])
    try:
        store._handle.rename_node(tmp_path, key_path.rsplit("/", 1)[1])
    except Exception:
        if backup is not None:
            store._handle.rename_node(backup, key_path.rsplit("/", 1)[1])
            if tmp_key in store:
                store.remove(tmp_key)
        raise
    if backup is not None:
        store.remove(backup)


def _rewrite_key_with_string_itemsize(
    store: pd.HDFStore,
    key: str,
    *,
    min_itemsize_map: Dict[str, int],
    data_column_candidates: List[str],
    columns: Optional[List[str]] = None,
    chunksize: int = 50_000,
    reason: str = "string width expansion",
) -> None:
    """Rewrite an existing HDF table with wider fixed-width string columns."""

    if key not in store:
        return

    storer = store.get_storer(key)
    nrows = int(getattr(storer, "nrows", 0) or 0)
    encoding, errors, nan_rep = storer.encoding, storer.errors, storer.nan_rep
    min_itemsize_map = _merge_itemsize_maps(min_itemsize_map, _stored_string_itemsize_by_column(store, key))
    existing_schema = store.select(key, start=0, stop=0)
    existing_cols = list(columns) if columns is not None else list(existing_schema.columns)

    if nrows <= 0:
        store.remove(key)
        return

    tmp_key = f"{key}__string_expand_tmp"
    if tmp_key in store:
        store.remove(tmp_key)

    data_cols = [c for c in dict.fromkeys(storer.data_columns + data_column_candidates) if c in existing_cols]
    print(f"\n[HDF STRING WIDTH EXPAND] Key={key}  Reason={reason}")
    preview = {k: int(v) for k, v in sorted(min_itemsize_map.items()) if k in existing_cols}
    if preview:
        shown = dict(list(preview.items())[:12])
        print(f"  Using string widths: {shown}{' ...' if len(preview) > 12 else ''}")

    for chunk in store.select(key, chunksize=int(chunksize), iterator=True):
        if chunk is None:
            continue
        chunk = chunk.reindex(columns=existing_cols)
        min_item = _min_itemsize_for_df(chunk, min_itemsize_map, encoding=encoding)
        store.append(
            tmp_key,
            chunk,
            format="table",
            data_columns=data_cols,
            index=False,
            dropna=False,
            min_itemsize=min_item if min_item else None,
            encoding=encoding, errors=errors, nan_rep=nan_rep,
        )

    if (int(store.get_storer(tmp_key).nrows) != nrows
            or list(store.select(tmp_key, start=0, stop=0).columns) != existing_cols):
        raise RuntimeError(f"String-width rebuild changed rows or columns for {key}")
    _replace_hdf_key_with_tmp(store, key=key, tmp_key=tmp_key)


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
    """Append aligned data, expanding columns and fixed string widths when required.

    Keep unavailable columns and promote aggregate integers when NaNs are needed for
    schema alignment.
    """

    if df is None or df.empty:
        return

    # PyTables appends require stable column dtypes.
    #
    # In mmgen we append per-file aggregate tables ("/agg_*_tmp") where some
    # columns may be absent for certain files (e.g., missing kinematic columns).
    # When we later align schemas via reindex(), those absent columns become
    # NaN-filled. If the table schema was created with integer dtypes (common for
    # count-like columns), the NaNs will upcast incoming data to float and
    # PyTables will raise:
    #   ValueError: cannot match existing table structure ... on appending data
    #
    # Promote aggregate integer fields to floating point when unavailable
    # values require NaN representation.
    # This does not change downstream calculations (counts are still numeric).
    if key.startswith("/agg_") and key.endswith("_tmp"):
        # Copy only for these small aggregate tables (avoid copying huge per-frame tables).
        df = df.copy()
        int_cols = df.select_dtypes(include=["int", "int64", "uint", "uint64"]).columns
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype(float)

    _candidates = HDF_DATA_COLUMN_CANDIDATES if data_column_candidates is None else data_column_candidates
    _min_itemsize = dict(HDF_MIN_ITEMSIZE if min_itemsize_map is None else min_itemsize_map)

    # 1) If the key exists, align schemas (and expand schema if needed)
    if key in store:
        existing_schema = store.select(key, start=0, stop=0)  # 0-row DF, preserves columns+dtypes
        existing_cols = list(existing_schema.columns)

        # Columns present in incoming df but missing from the stored table schema
        cols_to_add_to_existing = [c for c in df.columns if c not in existing_cols]

        if cols_to_add_to_existing:
            # Preserve incoming columns before reordering to the stored schema.
            print(f"\n[HDF SCHEMA EXPAND] Key={key}  Adding {len(cols_to_add_to_existing)} new columns")
            preview = cols_to_add_to_existing[:20]
            print(
                f"  New cols (first {len(preview)}): {preview}{' ...' if len(cols_to_add_to_existing) > 20 else ''}"
            )

            # If table already has rows, rewrite it into a temp key with the expanded schema
            nrows = getattr(store.get_storer(key), "nrows", 0)
            if nrows > 0:
                storer = store.get_storer(key)
                encoding, errors, nan_rep = storer.encoding, storer.errors, storer.nan_rep
                expanded_cols = existing_cols + cols_to_add_to_existing
                tmp_key = f"{key}__schema_expand_tmp"

                # Remove a leftover tmp key from a previous crash/run
                if tmp_key in store:
                    store.remove(tmp_key)

                # Build append options for expanded schema
                data_cols_expanded = [c for c in dict.fromkeys(storer.data_columns + _candidates) if c in expanded_cols]
                str_cols_expanded = sorted(
                    set(_object_string_columns(existing_schema)) | set(_object_string_columns(df))
                )
                _min_itemsize = _merge_itemsize_maps(
                    _min_itemsize, _stored_string_itemsize_by_column(store, key),
                    _min_itemsize_for_df(df, _min_itemsize, encoding=encoding),
                )
                min_item_expanded = {c: _min_itemsize.get(c, 64) for c in str_cols_expanded}

                # Rewrite existing data in chunks (memory-safe)
                for chunk in store.select(key, chunksize=50000, iterator=True):
                    # Reindex adds missing columns (filled with NaN) and enforces column order in one step.
                    chunk = chunk.reindex(columns=expanded_cols)
                    # If a newly added column is string/object in the incoming
                    # batch, keep the all-NA historical rows as object too;
                    # otherwise pandas may create the expanded table with a
                    # float dtype for that column and coerce later strings to NaN.
                    for c in str_cols_expanded:
                        if c in chunk.columns:
                            try:
                                chunk[c] = chunk[c].astype(object)
                            except Exception:
                                pass

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
                        encoding=encoding, errors=errors, nan_rep=nan_rep,
                    )

                # Replace the stored key with the expanded key
                if (int(store.get_storer(tmp_key).nrows) != int(nrows)
                        or list(store.select(tmp_key, start=0, stop=0).columns) != expanded_cols):
                    raise RuntimeError(f"Schema rebuild changed rows or columns for {key}")
                _replace_hdf_key_with_tmp(store, key=key, tmp_key=tmp_key)

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

    # 1b) If the table already exists, widen fixed-width string columns
    #     before appending values that would otherwise exceed the table's
    #     current capacity. This complements the existing column-addition
    #     schema expansion above.
    if key in store:
        width_updates = _required_string_width_updates(store, key, df, _min_itemsize)
        existing_widths = _stored_string_itemsize_by_column(store, key)
        _min_itemsize = _merge_itemsize_maps(_min_itemsize, existing_widths, width_updates)
        shared_strings = any(columns != [physical] for physical, (columns, _) in _stored_string_fields(store, key).items())
        if width_updates or shared_strings:
            for col, new_width in sorted(width_updates.items()):
                old_width = int(existing_widths.get(col, 0))
                examples = _long_string_examples(df, col, limit=old_width)
                print(
                    f"[HDF STRING WIDTH EXPAND] Key={key} column={col}: "
                    f"old_width={old_width}, new_width={int(new_width)}"
                )
                if examples:
                    print(f"  Example over-limit value(s): {examples[:3]}")
            _rewrite_key_with_string_itemsize(
                store,
                key,
                min_itemsize_map=_min_itemsize,
                data_column_candidates=_candidates,
                reason="incoming string requirements or legacy shared string layout",
            )

    # 2) Append (key may or may not exist at this point)
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

    storer = store.get_storer(key) if key in store else None
    encoding = storer.encoding if storer is not None else "UTF-8"
    data_cols = [c for c in dict.fromkeys((storer.data_columns if storer is not None else []) + _candidates) if c in df.columns]

    # Preset min_itemsize for all object/string columns present. Include the
    # observed incoming values so a first append can create the table wide enough
    # even when the global/preflight map did not know about that column.
    min_item = _min_itemsize_for_df(df, _min_itemsize, encoding=encoding)

    def _append_once() -> None:
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
            encoding=encoding,
            errors=storer.errors if storer is not None else "strict",
            nan_rep=storer.nan_rep if storer is not None else "nan",
        )

    try:
        _append_once()
    except ValueError as e:
        parsed = _parse_hdf_string_width_error(e)
        if parsed is None:
            raise
        col, incoming_len, old_limit = parsed
        new_width = max(
            int(_min_itemsize.get(col, 64)),
            _round_itemsize(int(incoming_len) + HDF_STRING_WIDTH_PAD),
            _round_itemsize(int(old_limit) * 2),
        )
        examples = _long_string_examples(df, col, limit=int(old_limit))
        print(
            f"\n[HDF STRING WIDTH ERROR RECOVERY] Key={key} column={col}: "
            f"incoming_len={incoming_len}, old_width={old_limit}, retry_width={new_width}"
        )
        if examples:
            print(f"  Example over-limit value(s): {examples[:5]}")

        # A legacy error may name a shared physical block rather than a column.
        logical_columns = [col] if col in df.columns else _stored_string_fields(store, key).get(col, ([], 0))[0]
        if not logical_columns:
            raise
        retry_itemsize = _merge_itemsize_maps(
            _min_itemsize, _stored_string_itemsize_by_column(store, key),
            _min_itemsize_for_df(df, _min_itemsize, encoding=encoding),
            {column: int(new_width) for column in logical_columns},
        )

        if key in store:
            _rewrite_key_with_string_itemsize(
                store,
                key,
                min_itemsize_map=retry_itemsize,
                data_column_candidates=_candidates,
                reason="caught Pandas/PyTables fixed string-width error",
            )
        min_item = _min_itemsize_for_df(df, retry_itemsize, encoding=encoding)
        _append_once()


def initialize_temp_store(
    temp_store_path: str,
    *,
    bout_dict: Optional[Dict[str, pd.DataFrame]],
    pos_df: Optional[pd.DataFrame],
    post_df: Optional[pd.DataFrame],
    min_itemsize_map: Optional[Dict[str, int]] = None,
) -> None:
    """Initialize temp HDF5 tables.

    This helper initializes the canonical temp-store tables for a run.
    """


    with open_store(temp_store_path, mode="w") as store:
        if bout_dict:
            for suffix, df in bout_dict.items():
                if str(suffix).startswith('__'):
                    continue
                if df is not None and not df.empty:
                    key = HDF_KEYS[f"bouts{suffix}"]
                    hdf_safe_append(store, key, df.head(0), min_itemsize_map=min_itemsize_map)

        if pos_df is not None and not pos_df.empty:
            all_pos_cols = sorted(set(PREDEFINED_POSITIONAL_COLS) | set(pos_df.columns))
            empty_pos_df = pd.DataFrame(columns=all_pos_cols)
            hdf_safe_append(store, HDF_KEYS["positional"], empty_pos_df, min_itemsize_map=min_itemsize_map)
        elif pos_df is None or pos_df.empty:
            empty_pos_df = pd.DataFrame(columns=PREDEFINED_POSITIONAL_COLS)
            hdf_safe_append(store, HDF_KEYS["positional"], empty_pos_df, min_itemsize_map=min_itemsize_map)

        if post_df is not None and not post_df.empty:
            all_post_cols = sorted(set(PREDEFINED_POSTURAL_COLS) | set(post_df.columns))
            empty_post_df = pd.DataFrame(columns=all_post_cols)
            hdf_safe_append(store, HDF_KEYS["postural"], empty_post_df, min_itemsize_map=min_itemsize_map)
        elif post_df is None or post_df.empty:
            empty_post_df = pd.DataFrame(columns=PREDEFINED_POSTURAL_COLS)
            hdf_safe_append(store, HDF_KEYS["postural"], empty_post_df, min_itemsize_map=min_itemsize_map)


def cleanup_temp_store(temp_store_path: str, *, keep_temp_files: bool) -> None:
    """Remove the temp store file unless `keep_temp_files` is True."""

    print("\nCleaning up temporary files...")
    try:
        import os

        if os.path.exists(temp_store_path):
            if keep_temp_files:
                print(f"Keeping temporary store: {temp_store_path}")
            else:
                os.remove(temp_store_path)
    except OSError as exc:
        print(f"[mmgen] Could not remove temporary store {temp_store_path}: {exc}")


def cleanup_verified_temp_stores(owned_directory: str, identities: Iterable[tuple[str, str]]) -> None:
    """Remove only unchanged stores verified by the completed run.

    All candidates are checked before any deletion. Run manifests and published
    scientific outputs are not candidates; foreign, linked, or changed files are
    retained for inspection.
    """
    from pathlib import Path
    from autobehaver.atomic_io import sha256_file

    candidates = []
    try:
        owner = Path(owned_directory).resolve()
        for filename, expected_hash in identities:
            path = Path(filename)
            if not path.exists():
                continue
            resolved = path.resolve()
            if (path.is_symlink() or owner not in resolved.parents
                    or path.name != 'temp_store.h5' or not path.is_file()
                    or sha256_file(path) != expected_hash):
                print(f"[mmgen] Retaining temporary stores: cleanup identity/ownership check failed for {path}")
                return
            candidates.append(path)
    except OSError as exc:
        print(f"[mmgen] Retaining completed temporary stores: cleanup preflight could not read {owned_directory}: {exc}")
        return
    for path in dict.fromkeys(candidates):
        try:
            path.unlink()
        except OSError as exc:
            print(f"[mmgen] Could not remove completed temporary store {path}: {exc}")


# Resume helpers


def get_done_source_files(temp_store_path: str) -> set[str]:
    """Read source basenames from modern or legacy completion records.

    Return an empty set when no records exist. Malformed records are rejected;
    callers separately verify identities and fingerprints.
    """

    from .recording_commits import has_recording_commits, read_recording_commits
    with open_store(temp_store_path, mode='r') as store:
        if not has_recording_commits(store):
            return set()
        df = read_recording_commits(store)
        if df.empty:
            return set()
        if 'source_file' not in df:
            raise RuntimeError('Invalid Mmgen commit table: source_file missing')
        return set(df['source_file'].astype(str))


def _store_select_any(store: pd.HDFStore, key: str, *, where: str) -> bool:
    """Check whether a table has any row matching the supplied query."""

    if key not in store:
        return False
    out = store.select(key, where=where, start=0, stop=1)
    return out is not None and not out.empty


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
        if _store_select_any(store, HDF_KEYS["positional"], where=f'source_file == "{sf}"'):
            return True

        # Check recording identity for postural rows, including legacy tables.
        if ind is not None and _store_select_any(store, HDF_KEYS["postural"], where=f'individual == "{ind}"'):
            return True

        # Bouts/agg tables can be large; only check the instantaneous one.
        if _store_select_any(store, HDF_KEYS["bouts_inst"], where=f'source_file == "{sf}"'):
            return True
        if _store_select_any(store, HDF_KEYS["agg_inst_tmp"], where=f'source_file == "{sf}"'):
            return True

        from .recording_commits import read_recording_commits
        records = read_recording_commits(store)
        if not records.empty and records['source_file'].astype(str).eq(sf).any():
            return True

    return False


def _rewrite_key_excluding_rows(store, *, key, tmp_key, drop_mask_fn, chunksize=50_000):
    """Stream a filtered replacement table inside the caller's staged store transaction."""
    if key not in store:
        return
    if tmp_key in store:
        store.remove(tmp_key)
    wrote = False
    for chunk in store.select(key, iterator=True, chunksize=int(chunksize)):
        mask = drop_mask_fn(chunk)
        if mask is None or len(mask) != len(chunk):
            raise RuntimeError(f'Invalid purge mask for {key}')
        kept = chunk.loc[~mask]
        if not kept.empty:
            hdf_safe_append(store, tmp_key, kept)
            wrote = True
    store.remove(key)
    if wrote:
        _replace_hdf_key_with_tmp(store, key=key, tmp_key=tmp_key)


def purge_file_from_store(temp_store_path, *, source_file, individual_id=None, verbose=True):
    """Publish all-key purge only after staging and verifying every table.

    Failed reads, filters, writes or replacement leave the original store intact.
    New stores carry recording provenance even for postural data; legacy
    postural tables may use the established per-recording individual fallback.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path
    from .recording_commits import purge_native_recording_commit, read_recording_commits
    target = Path(temp_store_path)
    fd, name = tempfile.mkstemp(prefix=target.name+'.purge.', suffix='.h5', dir=target.parent)
    os.close(fd)
    staged = Path(name)
    try:
        shutil.copy2(target, staged)
        with open_store(str(staged), mode='a') as store:
            for key in list(store.keys()):
                schema = store.select(key, start=0, stop=0)
                if 'source_file' in schema:
                    column, value = 'source_file', str(source_file)
                elif key == HDF_KEYS['postural'] and individual_id is not None:
                    column, value = 'individual', str(individual_id)
                else:
                    raise RuntimeError(f'Cannot safely purge table without source provenance: {key}')
                _rewrite_key_excluding_rows(store, key=key, tmp_key=key+'__purge_tmp',
                    drop_mask_fn=lambda df, c=column, v=value: df[c].astype(str).eq(v))
            purge_native_recording_commit(store, source_file)
            store.flush(fsync=True)
        with open_store(str(staged), mode='r') as store:
            records = read_recording_commits(store)
            if not records.empty and records['source_file'].astype(str).eq(str(source_file)).any():
                raise RuntimeError('Purge verification failed: recording completion records')
            for key in store.keys():
                for chunk in store.select(key, iterator=True, chunksize=50_000):
                    if 'source_file' in chunk and chunk['source_file'].astype(str).eq(str(source_file)).any():
                        raise RuntimeError(f'Purge verification failed: {key}')
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
