"""Size-check helpers for keyprenorm."""

from __future__ import annotations

import logging
import os
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("keyprenorm")

_SIZECHECK_BPR: int | None = None  # bytes-per-row estimate for size check


def get_accurate_bytes_per_row(csv_files: Sequence[str], num_samples: int = 10) -> int:
    """Estimate bytes-per-row for a rough CSV row-count sanity check."""
    logger.info("Calibrating file size check for this dataset...")

    if not csv_files:
        return 100

    if len(csv_files) > num_samples:
        sample_files = np.random.choice(list(csv_files), num_samples, replace=False)
    else:
        sample_files = list(csv_files)

    bpr_values: List[float] = []
    for file_path in sample_files:
        try:
            file_size = os.path.getsize(file_path)
            # Count lines quickly (includes header)
            with open(file_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f)
            if row_count > 1:
                bpr_values.append(file_size / row_count)
        except Exception as e:
            logger.warning("Could not analyze %s for calibration: %s", os.path.basename(file_path), e)

    if bpr_values:
        return int(round(float(np.mean(bpr_values))))
    logger.warning("Could not calibrate bytes-per-row. Falling back to 100.")
    return 100


def init_sizecheck_worker(bytes_per_row: int) -> None:
    global _SIZECHECK_BPR
    _SIZECHECK_BPR = int(bytes_per_row)


def estimate_rows_from_size_worker(csv_file: str) -> Tuple[str, int, int]:
    """Return (path, size_bytes, estimated_rows) for a CSV file."""
    try:
        file_size = os.path.getsize(csv_file)
        bpr = _SIZECHECK_BPR or 100
        est_rows = (file_size // bpr) - 1  # subtract header
        return csv_file, file_size, int(est_rows)
    except Exception:
        return csv_file, -1, -1
