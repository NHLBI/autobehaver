"""DLC I/O helpers for keypre."""

from __future__ import annotations

import pandas as pd


def load_dlc_csv_flat(csv_file: str) -> pd.DataFrame:
    """Read a DLC CSV (3-row multi-index header) and flatten to one-level columns.

    Output columns are like ``Ocelli_x``, ``Ocelli_y``, ``Ocelli_likelihood``.
    Assumes the first column is a frame index and should be used as ``index_col=0``.
    """

    df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    df.columns = [f"{col[1]}_{col[2]}" for col in df.columns]
    if "bodyparts_coords" in df.columns:
        df = df.drop(columns=["bodyparts_coords"])
    return df
