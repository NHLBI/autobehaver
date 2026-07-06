"""Normalization parameter calculation (mean/std) for keyprenorm."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from multiprocessing import Pool
import multiprocessing as mp
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import setup_worker_logging

logger = logging.getLogger("keyprenorm")

# -----------------------------
# Normalization parameter calculation
# -----------------------------

_STATS_CANONICAL_COLUMNS: List[str] | None = None
_STATS_EDGE_SKIP: int = 0
_STATS_MAX_ROWS: int = 0
_STATS_REQUIRE_FULL_LENGTH: bool = False
_STATS_LOG_LEVEL: str = "INFO"


def init_stats_worker(
    canonical_columns: Sequence[str],
    edge_skip: int,
    max_rows: int,
    require_full_length: bool,
    log_queue: mp.Queue,
    log_level: str,
) -> None:
    global _STATS_CANONICAL_COLUMNS, _STATS_EDGE_SKIP, _STATS_MAX_ROWS, _STATS_REQUIRE_FULL_LENGTH
    global _STATS_LOG_LEVEL

    _STATS_CANONICAL_COLUMNS = list(canonical_columns)
    _STATS_EDGE_SKIP = int(edge_skip)
    _STATS_MAX_ROWS = int(max_rows)
    _STATS_REQUIRE_FULL_LENGTH = bool(require_full_length)
    _STATS_LOG_LEVEL = str(log_level)

    setup_worker_logging(log_queue, log_level)


def process_stats_batch(batch: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Compute partial sum/sumsq/count for a batch of CSV files (vectorized)."""
    if _STATS_CANONICAL_COLUMNS is None:
        raise RuntimeError("Stats worker not initialized with canonical columns.")

    cols = _STATS_CANONICAL_COLUMNS
    n_cols = len(cols)

    local_sum = np.zeros(n_cols, dtype=np.float64)
    local_sumsq = np.zeros(n_cols, dtype=np.float64)
    local_count = np.zeros(n_cols, dtype=np.int64)
    skipped_short: List[str] = []

    nrows = _STATS_MAX_ROWS if _STATS_MAX_ROWS > 0 else None

    for csv_file in batch:
        try:
            df = (
                pd.read_csv(
                    csv_file,
                    header=0,
                    skiprows=range(1, _STATS_EDGE_SKIP + 1),
                    nrows=nrows,
                    dtype=np.float32,
                )
                .reindex(columns=cols)
            )

            if _STATS_REQUIRE_FULL_LENGTH and _STATS_MAX_ROWS > 0 and len(df) < _STATS_MAX_ROWS:
                skipped_short.append(csv_file)
                continue

            x = df.to_numpy(dtype=np.float32, copy=False)
            local_sum += np.nansum(x, axis=0, dtype=np.float64)
            local_sumsq += np.nansum(x * x, axis=0, dtype=np.float64)
            local_count += np.sum(~np.isnan(x), axis=0, dtype=np.int64)

        except Exception:
            logger.exception("Could not process %s during norm-param calculation", os.path.basename(csv_file))

    return local_sum, local_sumsq, local_count, skipped_short


def calculate_normalization_params(
    *,
    csv_files: Sequence[str],
    canonical_columns: Sequence[str],
    norm_params_path: str,
    norm_params_dir: str,
    num_workers: int,
    edge_skip: int,
    max_rows: int,
    require_full_length: bool,
    batch_size: int,
    log_queue: mp.Queue,
    log_level: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate per-column mean and stddev using the canonical schema.

    Returns:
        (norm_df, skipped_short_files)
    """
    if not csv_files:
        raise FileNotFoundError("No CSV files provided for normalization parameter calculation.")

    n_cols = len(canonical_columns)
    cumulative_sum = np.zeros(n_cols, dtype=np.float64)
    cumulative_sumsq = np.zeros(n_cols, dtype=np.float64)
    cumulative_count = np.zeros(n_cols, dtype=np.int64)
    skipped_short_all: List[str] = []

    batches = [list(csv_files[i : i + batch_size]) for i in range(0, len(csv_files), batch_size)]

    logger.info(
        "Calculating normalization parameters from %d files (batch_size=%d, workers=%d)...",
        len(csv_files),
        batch_size,
        num_workers,
    )

    if num_workers <= 1:
        init_stats_worker(canonical_columns, edge_skip, max_rows, require_full_length, log_queue, log_level)
        results = [process_stats_batch(b) for b in batches]
    else:
        with Pool(
            processes=num_workers,
            initializer=init_stats_worker,
            initargs=(canonical_columns, edge_skip, max_rows, require_full_length, log_queue, log_level),
        ) as pool:
            results = pool.map(process_stats_batch, batches)

    for local_sum, local_sumsq, local_count, skipped_short in results:
        cumulative_sum += local_sum
        cumulative_sumsq += local_sumsq
        cumulative_count += local_count
        skipped_short_all.extend(skipped_short)

    if skipped_short_all:
        logger.warning(
            "Skipped %d short files during norm-param calculation (require_full_length=True).",
            len(skipped_short_all),
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        means = cumulative_sum / cumulative_count
        variances = (cumulative_sumsq / cumulative_count) - (means**2)
        variances = np.maximum(variances, 0.0)
        stddevs = np.sqrt(variances)

    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    stddevs = np.nan_to_num(stddevs, nan=0.0, posinf=0.0, neginf=0.0)

    norm_df = pd.DataFrame({"Header": list(canonical_columns), "Mean": means, "StdDev": stddevs})

    # Save params + timestamped backup
    import os

    os.makedirs(os.path.dirname(norm_params_path) or ".", exist_ok=True)
    os.makedirs(norm_params_dir, exist_ok=True)

    norm_df.to_csv(norm_params_path, index=False, float_format="%.17g")
    logger.info("Normalization parameters saved to: %s", norm_params_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(norm_params_dir, f"norm_params_{timestamp}.csv")
    norm_df.to_csv(backup_path, index=False, float_format="%.17g")
    logger.info("Timestamped backup saved to: %s", backup_path)

    return norm_df, skipped_short_all
