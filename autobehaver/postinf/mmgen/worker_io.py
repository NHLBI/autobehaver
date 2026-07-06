"""I/O helpers for the mmgen per-recording worker.

This module centralizes the low-level file loading and alignment steps used by
`worker.process_video_data`.

These helpers keep file alignment and substrate loading behavior explicit and centralized.
"""

from __future__ import annotations

import os

import pandas as pd

from .io import find_matching_poscalc_file


def _align_poscalc_to_inference(
    df: pd.DataFrame,
    poscalc_df: pd.DataFrame,
    *,
    edge_skip: int,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inference and positional_calcs dataframes with matching row counts.

    PosePred inference writes a full timeline with a ``Frame Number`` column.
    Prefer that explicit 1-based frame index whenever available; otherwise use
    the ``edge_skip : edge_skip + len(df)`` convention with conservative fallbacks.
    """

    if "Frame Number" in df.columns:
        frame_numbers = pd.to_numeric(df["Frame Number"], errors="coerce")
        if frame_numbers.notna().all():
            idx = frame_numbers.astype(int).to_numpy() - 1
            if len(idx) == len(df) and len(idx) > 0 and int(idx.min()) >= 0 and int(idx.max()) < len(poscalc_df):
                return df.reset_index(drop=True), poscalc_df.iloc[idx].reset_index(drop=True)

    start = max(int(edge_skip or 0), 0)
    stop = start + len(df)
    aligned = poscalc_df.iloc[start:stop].reset_index(drop=True)
    if len(aligned) == len(df):
        return df.reset_index(drop=True), aligned

    if len(poscalc_df) >= len(df):
        print(
            f"WARNING: Edge-skip alignment produced {len(aligned)} rows for {source_name}; "
            f"falling back to first {len(df)} positional_calcs rows."
        )
        return df.reset_index(drop=True), poscalc_df.iloc[: len(df)].reset_index(drop=True)

    n = min(len(df), len(poscalc_df))
    print(
        f"WARNING: Inference/poscalc row-count mismatch for {source_name}; "
        f"truncating to common prefix length {n} (inference={len(df)}, poscalc={len(poscalc_df)})."
    )
    return df.iloc[:n].reset_index(drop=True), poscalc_df.iloc[:n].reset_index(drop=True)


def load_inference_and_poscalc(
    inf_csv_path: str,
    poscalc_dir: str,
    *,
    edge_skip: int,
    inf_tag: str,
):
    """Load inference CSV and matching positional_calcs CSV and align them.

    Returns (df, poscalc_df, aligned_poscalc_df) on success.
    Returns (None, None, None) if the matching positional_calcs file is missing.
    """

    df = pd.read_csv(inf_csv_path)

    poscalc_csv_path = find_matching_poscalc_file(
        os.path.basename(inf_csv_path), poscalc_dir, inf_tag
    )
    if poscalc_csv_path is None:
        print(
            f"WARNING: No matching positional_calcs file found for {os.path.basename(inf_csv_path)}. "
            "Skipping this recording."
        )
        return None, None, None

    poscalc_df = pd.read_csv(poscalc_csv_path)
    df_aligned, aligned_poscalc_df = _align_poscalc_to_inference(
        df,
        poscalc_df,
        edge_skip=edge_skip,
        source_name=os.path.basename(inf_csv_path),
    )

    return df_aligned, poscalc_df, aligned_poscalc_df


def extract_smoothed_labels(df: pd.DataFrame):
    """Extract smoothed pose/orientation labels from an inference dataframe."""

    masked_poses = df["Smoothed Pose Label"].to_numpy()

    has_orient = "Smoothed Orient Label" in df.columns
    masked_orients = None
    if has_orient:
        masked_orients = df["Smoothed Orient Label"].to_numpy()

    return masked_poses, masked_orients, has_orient
