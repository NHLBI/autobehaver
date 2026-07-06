"""Output writer utilities for keypre."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from . import settings

logger = logging.getLogger(__name__)

# Mirror settings globals (used by finalize_save_and_summarize)
nan_edge_effects = settings.nan_edge_effects
edge_skip = settings.edge_skip


def finalize_save_and_summarize(
    poscorr_dict: dict,
    csv_file: str,
    poscalc_file_path: str,
    *,
    apply_edge_nans: bool = True,
) -> dict:
    """Build output df, apply dtype + edge NaNs, write to disk, and return metadata dict."""
    poscorr_df = pd.DataFrame(poscorr_dict)

    # Downcast float columns to float32 to save disk space.
    float_cols = poscorr_df.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        poscorr_df = poscorr_df.astype({col: np.float32 for col in float_cols})

    # In raw-only mode we intentionally do not apply edge NaNs.
    if apply_edge_nans and nan_edge_effects and len(poscorr_df) > edge_skip * 2:
        poscorr_df.iloc[:edge_skip, :] = np.nan
        poscorr_df.iloc[-edge_skip:, :] = np.nan

    tmp_path = poscalc_file_path + ".tmp"
    poscorr_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, poscalc_file_path)
    logger.info(f"Processed and saved raw positional calculations: {poscalc_file_path}")

    return {
        "status": "success",
        "file": csv_file,
        "error": None,
        "out": poscalc_file_path,
        "n_rows": int(poscorr_df.shape[0]),
        "n_cols": int(poscorr_df.shape[1]),
    }
