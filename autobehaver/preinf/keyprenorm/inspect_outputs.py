"""Output inspection helpers for keyprenorm."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import h5py

logger = logging.getLogger("keyprenorm")


def inspect_normalized_h5(
    h5_path: str,
    expected_n_cols: int,
    max_rows: int,
    require_full_length: bool,
) -> Tuple[bool, Optional[int], str]:
    """Quickly validate a normalized HDF5 output.

    Returns:
        (ok, n_rows, reason)
    """
    try:
        with h5py.File(h5_path, "r") as hf:
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

            if max_rows > 0 and n_rows_i > int(max_rows):
                return False, n_rows_i, f"too_many_rows:{n_rows_i}"

            if require_full_length and max_rows > 0 and n_rows_i != int(max_rows):
                return False, n_rows_i, f"not_full_length:{n_rows_i}"

            cols_attr = hf.attrs.get("columns", None)
            if cols_attr is None:
                return False, n_rows_i, "missing_attr:columns"
            try:
                if len(cols_attr) != int(expected_n_cols):
                    return False, n_rows_i, f"columns_attr_len:{len(cols_attr)}"
            except Exception:
                return False, n_rows_i, "bad_attr:columns"

            return True, n_rows_i, "ok"
    except Exception as e:
        return False, None, f"open_error:{type(e).__name__}"


def inspect_label_h5(label_h5_path: str, expected_len: Optional[int]) -> Tuple[bool, str]:
    """Validate label HDF5 output.

    Returns:
        (ok, reason)
    """
    try:
        with h5py.File(label_h5_path, "r") as hf:
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

            return True, "ok"
    except Exception as e:
        return False, f"open_error:{type(e).__name__}"
