"""Validate normalized features, paired labels, frame metadata, and completion receipts."""

from __future__ import annotations

import logging
import json
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
from autobehaver.completion import receipt_matches
from ..frame_identity import retained_frame_indices

logger = logging.getLogger("keyprenorm")


def _published_integrity(h5_path, hf, rows, expected_identity, expected_source_rows, expected_edge_skip):
    """Check the published identity, frame timeline, source length, and completion receipt."""
    identity = hf.attrs.get("content_identity")
    if identity is None and expected_identity is None:
        return None
    if isinstance(identity, bytes):
        identity = identity.decode("utf-8")
    if expected_identity is not None and identity != expected_identity:
        return "content_identity_mismatch_or_missing"
    try:
        metadata = json.loads(identity)
        if "feature_identity" in metadata:
            metadata = json.loads(metadata["feature_identity"])
        edge_skip = int(metadata["edge_skip"])
        declared = hf.attrs["original_frame_count"]
        if not np.isfinite(declared) or declared != int(declared) or int(declared) < 0:
            return "invalid_original_frame_count"
        source_rows = int(declared)
        if expected_source_rows is not None and source_rows != int(expected_source_rows):
            return "original_frame_count_mismatch"
        if expected_edge_skip is not None and edge_skip != int(expected_edge_skip):
            return "normalization_edge_skip_mismatch"
        raw = np.asarray(hf["source_frame_indices"][:])
        expected = retained_frame_indices(source_rows, edge_skip)
        if raw.ndim != 1 or len(raw) != rows or not np.array_equal(raw, expected):
            return "original_frame_indices_mismatch"
        if "normalization_edge_skip" in hf.attrs and int(hf.attrs["normalization_edge_skip"]) != edge_skip:
            return "normalization_edge_skip_mismatch"
        if "source_positional_sha256" in hf.attrs and hf.attrs["source_positional_sha256"] != metadata.get("source"):
            return "positional_source_identity_mismatch"
    except (ValueError, KeyError, TypeError, OverflowError):
        return "missing_or_invalid_original_frame_metadata"
    if not receipt_matches(h5_path, identity):
        return "published_content_receipt_mismatch_or_missing"
    return None


def inspect_normalized_h5(
    h5_path: str,
    expected_n_cols: int,
    *,
    expected_rows: Optional[int] = None,
    expected_columns: Optional[Sequence[str]] = None,
    expected_identity: Optional[str] = None,
    expected_source_rows: Optional[int] = None,
    expected_edge_skip: Optional[int] = None,
) -> Tuple[bool, Optional[int], str]:
    """Validate normalized HDF5 shape, schema, finite values, and optional provenance.

    Scan values in chunks and return (ok, row_count, reason).
    """
    try:
        with h5py.File(h5_path, "r") as hf:
            if expected_identity is not None and hf.attrs.get('content_identity') != expected_identity:
                return False, None, 'content_identity_mismatch_or_missing'
            if "data" not in hf:
                return False, None, "missing_dataset:data"
            dset = hf["data"]
            if getattr(dset, "ndim", None) != 2:
                return False, None, "data_not_2d"
            n_rows, n_cols = dset.shape
            n_rows_i = int(n_rows)
            n_cols_i = int(n_cols)

            if n_cols_i != int(expected_n_cols):
                return False, n_rows_i, f"wrong_n_cols:{n_cols_i}"

            if expected_rows is not None and n_rows_i != int(expected_rows):
                return False, n_rows_i, f"row_count_mismatch:{n_rows_i}!={int(expected_rows)}"

            cols_attr = hf.attrs.get("columns", None)
            if cols_attr is None:
                return False, n_rows_i, "missing_attr:columns"
            try:
                if len(cols_attr) != int(expected_n_cols):
                    return False, n_rows_i, f"columns_attr_len:{len(cols_attr)}"
            except Exception:
                return False, n_rows_i, "bad_attr:columns"

            decoded_columns = [
                c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                for c in cols_attr
            ]
            if any(not column.strip() for column in decoded_columns):
                return False, n_rows_i, "blank_column_name"
            if len(set(decoded_columns)) != len(decoded_columns):
                return False, n_rows_i, "duplicate_column_names"
            if expected_columns is not None and decoded_columns != [str(c) for c in expected_columns]:
                return False, n_rows_i, "column_schema_mismatch"

            chunk_rows = max(1, min(4096, n_rows_i or 1))
            for start in range(0, n_rows_i, chunk_rows):
                values = np.asarray(dset[start : start + chunk_rows])
                if not np.isfinite(values).all():
                    return False, n_rows_i, "nonfinite_values"

            integrity_reason = _published_integrity(h5_path, hf, n_rows_i, expected_identity,
                                                   expected_source_rows, expected_edge_skip)
            if integrity_reason is not None:
                return False, n_rows_i, integrity_reason

            return True, n_rows_i, "ok"
    except Exception as e:
        return False, None, f"open_error:{type(e).__name__}"


def inspect_label_h5(label_h5_path: str, expected_len: Optional[int], *, expected_identity: Optional[str] = None,
                      expected_source_rows: Optional[int] = None, expected_edge_skip: Optional[int] = None,
                      expected_feature_identity: Optional[str] = None) -> Tuple[bool, str]:
    """Validate label datasets, frame metadata, and optional provenance.

    Return (ok, reason).
    """
    try:
        with h5py.File(label_h5_path, "r") as hf:
            if expected_identity is not None and hf.attrs.get('content_identity') != expected_identity:
                return False, 'content_identity_mismatch_or_missing'
            if expected_feature_identity is not None:
                binding = json.loads(hf.attrs.get('content_identity', '{}'))
                if binding.get('feature_identity') != expected_feature_identity:
                    return False, 'label_feature_identity_mismatch_or_missing'
            cols_attr = hf.attrs.get("columns", None)
            if cols_attr is not None:
                cols = [
                    c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in cols_attr
                ]
            else:
                cols = list(hf.keys())

            if not cols:
                return False, "no_columns"

            lengths = []
            for col in cols:
                if col not in hf:
                    return False, f"missing_dataset:{col}"
                ds = hf[col]
                if getattr(ds, "ndim", None) != 1:
                    return False, f"dataset_not_1d:{col}"
                lengths.append(int(ds.shape[0]))

            if len(set(lengths)) != 1:
                return False, "inconsistent_lengths"

            if expected_len is not None and lengths[0] != int(expected_len):
                return False, f"length_mismatch:{lengths[0]}"

            integrity_reason = _published_integrity(label_h5_path, hf, lengths[0], expected_identity,
                                                   expected_source_rows, expected_edge_skip)
            if integrity_reason is not None:
                return False, integrity_reason

            return True, "ok"
    except Exception as e:
        return False, f"open_error:{type(e).__name__}"
