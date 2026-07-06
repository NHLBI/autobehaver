"""Egocentric transforms (center + rotate) and asymmetry features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .spec import asymmetry_lr_pairs, asymmetry_tb_pairs

logger = logging.getLogger(__name__)


def add_centered_and_rotated_keypoints(input_df, parts):
    """Create *_centered_(x/y) and *_rotated_(x/y) columns in input_df.

    Output convention (egocentric Euclidean frame):
      - centroid is (0, 0)
      - +y is forward/anterior ("up")
      - +x is fly-right

    NOTE: DLC image coords are +y down, so we rotate in image coords and then flip the
    rotated y-axis to produce a standard Cartesian (+y up) frame.
    """

    centered_columns = {}
    for part in parts:
        x_col = f"{part}_x"
        y_col = f"{part}_y"

        if x_col in input_df.columns and y_col in input_df.columns:
            centered_columns[f"{part}_centered_x"] = input_df[x_col] - input_df["centroid_x"]
            centered_columns[f"{part}_centered_y"] = input_df[y_col] - input_df["centroid_y"]
        else:
            logger.warning(f"Warning: Columns for {part} not found in DataFrame.")

    input_df = pd.concat([input_df, pd.DataFrame(centered_columns, index=input_df.index)], axis=1)

    # Rotate by -angle (bearing), which aligns the body axis to image-north (negative y in image coords).
    rotation_angles = -np.radians(input_df["angle"])
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)

    rotated_columns = {}
    for part in parts:
        cx = f"{part}_centered_x"
        cy = f"{part}_centered_y"
        if cx in input_df.columns and cy in input_df.columns:
            x_rot = input_df[cx] * cos_a - input_df[cy] * sin_a
            y_rot = input_df[cx] * sin_a + input_df[cy] * cos_a

            rotated_columns[f"{part}_rotated_x"] = x_rot
            # Flip y to convert from image-up (negative) to Euclidean-up (positive)
            rotated_columns[f"{part}_rotated_y"] = -y_rot
        else:
            logger.warning(f"Warning: Centered columns for {part} not found in DataFrame.")

    input_df = pd.concat([input_df, pd.DataFrame(rotated_columns, index=input_df.index)], axis=1)
    return input_df


def calculate_asymmetry_features(df, asymmetry_lr_pairs, asymmetry_tb_pairs):
    """Compute asymmetry features using rotated keypoints (egocentric frame)."""

    asym_data_cols = {}

    # --- 1) Left/Right asymmetry ---
    for part1, part2, name in asymmetry_lr_pairs:
        p1x, p1y = f"{part1}_rotated_x", f"{part1}_rotated_y"
        p2x, p2y = f"{part2}_rotated_x", f"{part2}_rotated_y"
        if not all(c in df.columns for c in (p1x, p1y, p2x, p2y)):
            continue

        # symmetric about x=0 => compare distances from body axis
        asym_data_cols[f"asym_lr_x_{name}"] = np.abs(np.abs(df[p1x]) - np.abs(df[p2x]))
        # along-body mismatch
        asym_data_cols[f"asym_lr_y_{name}"] = np.abs(df[p1y] - df[p2y])

    # --- 2) Top/Bottom asymmetry (same logic for consistency) ---
    for part1, part2, name in asymmetry_tb_pairs:
        p1x, p1y = f"{part1}_rotated_x", f"{part1}_rotated_y"
        p2x, p2y = f"{part2}_rotated_x", f"{part2}_rotated_y"
        if not all(c in df.columns for c in (p1x, p1y, p2x, p2y)):
            continue

        asym_data_cols[f"asym_tb_x_{name}"] = np.abs(np.abs(df[p1x]) - np.abs(df[p2x]))
        asym_data_cols[f"asym_tb_y_{name}"] = np.abs(df[p1y] - df[p2y])

    if not asym_data_cols:
        return df, []

    created_cols = list(asym_data_cols.keys())
    df = pd.concat([df, pd.DataFrame(asym_data_cols, index=df.index)], axis=1)
    return df, created_cols


def add_asymmetry_features(input_df, poscorr_dict):
    """Compute asymmetry features from rotated coords and store them in poscorr_dict."""

    input_df, created_cols = calculate_asymmetry_features(
        input_df,
        asymmetry_lr_pairs=asymmetry_lr_pairs,
        asymmetry_tb_pairs=asymmetry_tb_pairs,
    )

    for col in created_cols:
        poscorr_dict[col] = input_df[col]

    return input_df
