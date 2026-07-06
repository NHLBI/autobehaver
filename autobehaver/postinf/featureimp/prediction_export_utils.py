"""Helpers for assembling user-facing prediction export tables."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def insert_group_separator_rows(df: pd.DataFrame, *, group_col: str = "group") -> pd.DataFrame:
    """Insert blank separator rows between contiguous group blocks.

    This avoids concatenating an all-NA one-row DataFrame into the result,
    which triggers a noisy pandas FutureWarning in newer versions.
    """
    if df.empty or group_col not in df.columns:
        return df

    group_series = df[group_col].reset_index(drop=True)
    insert_positions = [
        idx
        for idx in range(1, len(group_series))
        if group_series.iat[idx] != group_series.iat[idx - 1]
    ]
    if not insert_positions:
        return df

    out = df.copy()
    original_index = np.arange(len(out), dtype=int)
    insert_positions_arr = np.asarray(insert_positions, dtype=int)
    index_shift = np.searchsorted(insert_positions_arr, original_index, side="right")
    out.index = original_index + index_shift
    out = out.reindex(range(len(df) + len(insert_positions))).reset_index(drop=True)
    return out
