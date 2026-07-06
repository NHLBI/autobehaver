"""Normalization application for keyprenorm.

This module takes preprocessed positional_calcs CSVs and writes normalized HDF5 outputs.
"""

from __future__ import annotations

import logging
import os
import multiprocessing as mp
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from . import __stage_version__
from .logging_utils import setup_worker_logging

logger = logging.getLogger("keyprenorm")

# -----------------------------
# Normalization worker
# -----------------------------

_NORM_CANONICAL_COLUMNS: Optional[list[str]] = None
_NORM_MEANS: Optional[np.ndarray] = None
_NORM_STDS: Optional[np.ndarray] = None
_NORM_EDGE_SKIP: int = 0
_NORM_MAX_ROWS: int = 0
_NORM_REQUIRE_FULL_LENGTH: bool = False
_NORM_OUTPUT_DIR: Optional[str] = None
_NORM_LABELS_INPUT_DIR: Optional[str] = None
_NORM_LABELS_OUTPUT_DIR: Optional[str] = None


def init_normalize_worker(
    canonical_columns: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    edge_skip: int,
    max_rows: int,
    require_full_length: bool,
    output_dir: str,
    labels_input_dir: Optional[str],
    labels_output_dir: Optional[str],
    log_queue: mp.Queue,
    log_level: str,
) -> None:
    global _NORM_CANONICAL_COLUMNS, _NORM_MEANS, _NORM_STDS
    global _NORM_EDGE_SKIP, _NORM_MAX_ROWS, _NORM_REQUIRE_FULL_LENGTH
    global _NORM_OUTPUT_DIR, _NORM_LABELS_INPUT_DIR, _NORM_LABELS_OUTPUT_DIR

    _NORM_CANONICAL_COLUMNS = list(canonical_columns)
    _NORM_MEANS = means.astype(np.float32, copy=False)
    _NORM_STDS = stds.astype(np.float32, copy=False)
    _NORM_EDGE_SKIP = int(edge_skip)
    _NORM_MAX_ROWS = int(max_rows)
    _NORM_REQUIRE_FULL_LENGTH = bool(require_full_length)
    _NORM_OUTPUT_DIR = output_dir
    _NORM_LABELS_INPUT_DIR = labels_input_dir
    _NORM_LABELS_OUTPUT_DIR = labels_output_dir

    setup_worker_logging(log_queue, log_level)


def normalize_one_csv(csv_file: str) -> Dict[str, object]:
    """Normalize one CSV file and write HDF5 outputs."""
    try:
        if _NORM_CANONICAL_COLUMNS is None or _NORM_MEANS is None or _NORM_STDS is None or _NORM_OUTPUT_DIR is None:
            raise RuntimeError("Normalize worker not initialized.")

        cols = _NORM_CANONICAL_COLUMNS
        nrows = _NORM_MAX_ROWS if _NORM_MAX_ROWS > 0 else None

        df = (
            pd.read_csv(
                csv_file,
                header=0,
                skiprows=range(1, _NORM_EDGE_SKIP + 1),
                nrows=nrows,
                dtype=np.float32,
            )
            .reindex(columns=cols)
        )

        if _NORM_REQUIRE_FULL_LENGTH and _NORM_MAX_ROWS > 0 and len(df) < _NORM_MAX_ROWS:
            return {
                "status": "skipped_short",
                "file": csv_file,
                "rows": int(len(df)),
                "labels_saved": False,
                "error": None,
            }

        x = df.to_numpy(dtype=np.float32, copy=False)

        means = _NORM_MEANS
        stds = _NORM_STDS

        x_norm = np.empty_like(x, dtype=np.float32)

        # Compute (x - mean) / std with broadcasting, avoiding large boolean-indexed copies.
        np.subtract(x, means, out=x_norm)
        nonzero = stds > 0
        np.divide(x_norm, stds, out=x_norm, where=nonzero)

        # Columns with zero stddev become 0
        if np.any(~nonzero):
            x_norm[:, ~nonzero] = 0.0

        np.nan_to_num(x_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        base_name = os.path.basename(csv_file)
        new_file_name = base_name.replace(".csv", "_normalized.h5")
        out_path = os.path.join(_NORM_OUTPUT_DIR, new_file_name)
        tmp_path = out_path + ".tmp"

        with h5py.File(tmp_path, "w") as hf:
            hf.create_dataset("data", data=x_norm, compression="gzip", compression_opts=9)
            hf.attrs["columns"] = [c.encode("utf-8") for c in cols]
            hf.attrs["keyprenorm_version"] = __stage_version__.encode("utf-8")

        os.replace(tmp_path, out_path)

        labels_saved = False
        if _NORM_LABELS_INPUT_DIR and _NORM_LABELS_OUTPUT_DIR:
            base_filename = base_name.replace("_positional_calcs.csv", "")
            label_file_name = f"{base_filename}_labels.csv"
            label_file_path = os.path.join(_NORM_LABELS_INPUT_DIR, label_file_name)

            if os.path.exists(label_file_path):
                label_df = pd.read_csv(
                    label_file_path,
                    header=0,
                    skiprows=range(1, _NORM_EDGE_SKIP + 1),
                    nrows=nrows,
                    dtype={"Frame": np.int64, "Behavior": str, "Orientation": str},
                )

                label_out_path = os.path.join(
                    _NORM_LABELS_OUTPUT_DIR, label_file_name.replace(".csv", ".h5")
                )
                tmp_label_path = label_out_path + ".tmp"

                with h5py.File(tmp_label_path, "w") as hf:
                    for col in label_df.columns:
                        series = label_df[col]
                        if series.dtype == "object":
                            hf.create_dataset(
                                col,
                                data=series.astype(str).to_numpy(),
                                dtype=h5py.string_dtype(encoding="utf-8"),
                                compression="gzip",
                                compression_opts=9,
                            )
                        else:
                            if pd.api.types.is_integer_dtype(series.dtype):
                                hf.create_dataset(
                                    col,
                                    data=series.to_numpy(dtype=np.int64, copy=False),
                                    compression="gzip",
                                    compression_opts=9,
                                )
                            else:
                                hf.create_dataset(
                                    col,
                                    data=series.to_numpy(dtype=np.float32, copy=False),
                                    compression="gzip",
                                    compression_opts=9,
                                )

                    hf.attrs["columns"] = [c.encode("utf-8") for c in label_df.columns]
                    hf.attrs["keyprenorm_version"] = __stage_version__.encode("utf-8")

                os.replace(tmp_label_path, label_out_path)
                labels_saved = True

                if len(label_df) != x_norm.shape[0]:
                    logger.warning(
                        "Label rows != data rows for %s (labels=%d, data=%d)",
                        label_file_name,
                        len(label_df),
                        x_norm.shape[0],
                    )

        return {
            "status": "success",
            "file": csv_file,
            "rows": int(x_norm.shape[0]),
            "labels_saved": labels_saved,
            "error": None,
        }

    except Exception as e:
        logger.exception("Error processing %s", csv_file)
        return {"status": "failed", "file": csv_file, "labels_saved": False, "error": str(e)}
