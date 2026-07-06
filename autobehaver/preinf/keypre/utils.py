"""Small helpers for keypre."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def add_existing_columns(poscorr_dict: dict, source_df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Add columns to poscorr_dict if they exist in source_df, keeping requested order."""
    for col in columns:
        if col in source_df.columns:
            poscorr_dict[col] = source_df[col]


def shoelace_area(df: pd.DataFrame, vertices: list[str]) -> pd.Series:
    """Calculate polygon area for each row using the Shoelace formula."""

    # Extract x and y coordinates for all vertices into numpy arrays for performance
    x_coords = df[[f"{v}_x" for v in vertices]].values
    y_coords = df[[f"{v}_y" for v in vertices]].values

    # Vectorized Shoelace formula
    term1 = np.sum(x_coords * np.roll(y_coords, -1, axis=1), axis=1)
    term2 = np.sum(y_coords * np.roll(x_coords, -1, axis=1), axis=1)

    return 0.5 * np.abs(term1 - term2)


def find_processing_log(input_dir: str) -> str | None:
    """Find the processing log file in the input directory."""
    for file in os.listdir(input_dir):
        if file.endswith("_processing_log.txt"):
            return os.path.join(input_dir, file)
    return None
