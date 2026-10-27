"""Calculate per-feature population means and standard deviations for KeyPreNorm."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from ..execution import worker_results
import multiprocessing as mp
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import setup_worker_logging
from .csv_chunks import RetainedCsvChunks

logger = logging.getLogger("keyprenorm")

# Normalization parameter calculation

_STATS_CANONICAL_COLUMNS: List[str] | None = None
_STATS_EDGE_SKIP: int = 0
_STATS_CHUNK_ROWS: int = 4096
_STATS_LOG_LEVEL: str = "INFO"


def init_stats_worker(
    canonical_columns: Sequence[str],
    edge_skip: int,
    log_queue: mp.Queue,
    log_level: str,
    chunk_rows: int = 4096,
) -> None:
    """Initialize canonical columns, edge trimming, and logging for statistics workers."""
    global _STATS_CANONICAL_COLUMNS, _STATS_EDGE_SKIP, _STATS_CHUNK_ROWS
    global _STATS_LOG_LEVEL

    _STATS_CANONICAL_COLUMNS = list(canonical_columns)
    _STATS_EDGE_SKIP = int(edge_skip)
    _STATS_CHUNK_ROWS = int(chunk_rows)
    _STATS_LOG_LEVEL = str(log_level)

    setup_worker_logging(log_queue, log_level)


def merge_centered_moments(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
    """Merge finite populations with the Chan centered-moment formula.

    Return mean, M2 (sum of squared centered deviations), and count per feature.
    """
    count = count_a + count_b
    delta = mean_b - mean_a
    fraction = np.divide(count_b, count, out=np.zeros_like(mean_a), where=count > 0)
    mean = mean_a + delta * fraction
    factor = np.divide(count_a.astype(float) * count_b, count,
                       out=np.zeros_like(mean_a), where=count > 0)
    m2 = m2_a + m2_b + delta * delta * factor
    return mean, m2, count


def process_stats_batch(batch: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, M2, and count per feature from finite retained values.

    Reject input infinities and count each feature independently.
    """
    if _STATS_CANONICAL_COLUMNS is None:
        raise RuntimeError("Stats worker not initialized with canonical columns.")

    cols = _STATS_CANONICAL_COLUMNS
    n_cols = len(cols)

    local_mean = np.zeros(n_cols, dtype=np.float64)
    local_m2 = np.zeros(n_cols, dtype=np.float64)
    local_count = np.zeros(n_cols, dtype=np.int64)


    for csv_file in batch:
        try:
            for df in RetainedCsvChunks(csv_file, cols, _STATS_EDGE_SKIP, chunk_rows=_STATS_CHUNK_ROWS):
                x = df.to_numpy(dtype=np.float32, copy=False)
                if np.isinf(x).any():
                    raise ValueError(f"Infinite feature value(s) in {os.path.basename(csv_file)}")
                x64 = x.astype(np.float64)
                count = np.sum(np.isfinite(x64), axis=0, dtype=np.int64)
                mean = np.divide(np.nansum(x64, axis=0), count,
                                 out=np.zeros(n_cols), where=count > 0)
                np.subtract(x64, mean, out=x64)
                np.square(x64, out=x64)
                m2 = np.nansum(x64, axis=0)
                local_mean, local_m2, local_count = merge_centered_moments(
                    local_mean, local_m2, local_count, mean, m2, count)

        except Exception:
            logger.exception("Could not process %s during norm-param calculation", os.path.basename(csv_file))
            raise

    return local_mean, local_m2, local_count


def calculate_normalization_params(
    *,
    csv_files: Sequence[str],
    canonical_columns: Sequence[str],
    norm_params_path: str,
    norm_params_dir: str,
    num_workers: int,
    edge_skip: int,
    batch_size: int,
    log_queue: mp.Queue,
    log_level: str,
    chunk_rows: int = 4096,
) -> pd.DataFrame:
    """Calculate and write means and population standard deviations for retained rows.

    Use each feature's finite-value count and variance M2/count (ddof=0).
    Return the canonical Header/Mean/StdDev table.
    """
    if not csv_files:
        raise FileNotFoundError("No CSV files provided for normalization parameter calculation.")

    n_cols = len(canonical_columns)
    cumulative_mean = np.zeros(n_cols, dtype=np.float64)
    cumulative_m2 = np.zeros(n_cols, dtype=np.float64)
    cumulative_count = np.zeros(n_cols, dtype=np.int64)

    batches = [list(csv_files[i : i + batch_size]) for i in range(0, len(csv_files), batch_size)]

    logger.info(
        "Calculating normalization parameters from %d files (batch_size=%d, workers=%d)...",
        len(csv_files),
        batch_size,
        num_workers,
    )

    results = worker_results(process_stats_batch, batches, num_workers=num_workers,
        initializer=init_stats_worker,
        initargs=(canonical_columns, edge_skip, log_queue, log_level, chunk_rows))

    for local_mean, local_m2, local_count in results:
        cumulative_mean, cumulative_m2, cumulative_count = merge_centered_moments(
            cumulative_mean, cumulative_m2, cumulative_count, local_mean, local_m2, local_count)

    missing_observations = [
        str(canonical_columns[index])
        for index, count in enumerate(cumulative_count)
        if int(count) == 0
    ]
    if missing_observations:
        raise ValueError(
            "Cannot calculate normalization parameters because feature column(s) have no "
            f"finite observations: {missing_observations[:25]}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        means = cumulative_mean
        variances = cumulative_m2 / cumulative_count
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

    return norm_df
