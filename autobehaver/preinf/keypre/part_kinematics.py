"""Per-keypoint kinematics features (velocity/accel/jerk) for keypre."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import settings
from .spec import keypoints_velocity_parts

logger = logging.getLogger(__name__)

rolling_windows = settings.rolling_windows


def calculate_keypoints_velocity_series(dataframe, parts, intervals):
    """Vectorized calculation of velocity for specified body parts using rotated keypoint coordinates."""

    # Identify all relevant rotated coordinate columns present in the dataframe
    x_cols = [f"{part}_rotated_x" for part in parts if f"{part}_rotated_x" in dataframe.columns]
    y_cols = [f"{part}_rotated_y" for part in parts if f"{part}_rotated_y" in dataframe.columns]
    base_parts = [col.replace("_rotated_x", "") for col in x_cols]

    if not x_cols or not y_cols:
        logger.warning("Warning: No rotated keypoint columns found for velocity calculation.")
        return dataframe

    # Create sub-dataframes for x and y coordinates
    df_x = dataframe[x_cols]
    df_y = dataframe[y_cols]

    # --- Calculate Instantaneous Velocity ---
    # Central difference method for displacement (t+1 - t-1), averaged for one frame interval
    dx = (df_x.shift(-1) - df_x.shift(1)) / 2.0
    dy = (df_y.shift(-1) - df_y.shift(1)) / 2.0

    #
    # FIX: Align dy column names to dx to prevent pandas from creating a 56-column union.
    # This ensures the addition is element-wise and produces a 28-column result.
    dy.columns = dx.columns

    # Calculate velocity magnitude for all parts at once
    inst_velocity_df = np.sqrt(dx**2 + dy**2)
    inst_velocity_df.columns = [f"partvelocity_{part}_inst" for part in base_parts]

    # --- Calculate Rolling Averages ---
    rolling_dfs = []
    for interval in intervals:
        rolling_df = inst_velocity_df.rolling(window=interval, min_periods=1, center=True).mean()
        rolling_df.columns = [f"partvelocity_{part}_{interval}" for part in base_parts]
        rolling_dfs.append(rolling_df)

    # --- Calculate Rolling Standard Deviations ---
    rolling_std_dfs = []
    for interval in intervals:
        rolling_std_df = inst_velocity_df.rolling(window=interval, min_periods=1, center=True).std()
        rolling_std_df.columns = [f"partvelocitystd_{part}_{interval}" for part in base_parts]
        rolling_std_dfs.append(rolling_std_df)

    # Combine all new data columns and add to the main dataframe
    new_data = pd.concat([inst_velocity_df] + rolling_dfs + rolling_std_dfs, axis=1)
    dataframe = pd.concat([dataframe, new_data], axis=1)

    return dataframe


def calculate_keypoints_acceleration_series(dataframe, parts, intervals):
    """Vectorized calculation of acceleration for specified body parts."""
    inst_vel_cols = [
        f"partvelocity_{part}_inst" for part in parts if f"partvelocity_{part}_inst" in dataframe.columns
    ]
    base_parts = [col.replace("partvelocity_", "").replace("_inst", "") for col in inst_vel_cols]

    if not inst_vel_cols:
        logger.warning("Warning: No instantaneous velocity columns found for acceleration calculation.")
        return dataframe

    df_vel = dataframe[inst_vel_cols]

    # --- Calculate Instantaneous Acceleration ---
    inst_accel_df = (df_vel.shift(-1) - df_vel.shift(1)) / 2.0
    inst_accel_df.columns = [f"partaccel_{part}_inst" for part in base_parts]

    # --- Calculate Rolling Averages ---
    rolling_dfs = []
    for interval in intervals:
        rolling_df = inst_accel_df.rolling(window=interval, min_periods=1, center=True).mean()
        rolling_df.columns = [f"partaccel_{part}_{interval}" for part in base_parts]
        rolling_dfs.append(rolling_df)

    new_data = pd.concat([inst_accel_df] + rolling_dfs, axis=1)
    dataframe = pd.concat([dataframe, new_data], axis=1)

    return dataframe


def calculate_keypoints_jerk_series(dataframe, parts, intervals):
    """Vectorized calculation of jerk for specified body parts."""
    inst_accel_cols = [
        f"partaccel_{part}_inst" for part in parts if f"partaccel_{part}_inst" in dataframe.columns
    ]
    base_parts = [col.replace("partaccel_", "").replace("_inst", "") for col in inst_accel_cols]

    if not inst_accel_cols:
        logger.warning("Warning: No instantaneous acceleration columns found for jerk calculation.")
        return dataframe

    df_accel = dataframe[inst_accel_cols]

    # --- Calculate Instantaneous Jerk ---
    inst_jerk_df = (df_accel.shift(-1) - df_accel.shift(1)) / 2.0
    inst_jerk_df.columns = [f"partjerk_{part}_inst" for part in base_parts]

    # --- Calculate Rolling Averages ---
    rolling_dfs = []
    for interval in intervals:
        rolling_df = inst_jerk_df.rolling(window=interval, min_periods=1, center=True).mean()
        rolling_df.columns = [f"partjerk_{part}_{interval}" for part in base_parts]
        rolling_dfs.append(rolling_df)

    new_data = pd.concat([inst_jerk_df] + rolling_dfs, axis=1)
    dataframe = pd.concat([dataframe, new_data], axis=1)

    return dataframe


def add_keypoint_kinematics_features(input_df, poscorr_dict):
    """Per-keypoint velocity/accel/jerk from rotated coords."""
    input_df = calculate_keypoints_velocity_series(input_df, keypoints_velocity_parts, rolling_windows)
    cols = [
        c
        for c in input_df.columns
        if c.startswith("partvelocity_") or c.startswith("partvelocitystd_")
    ]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_keypoints_acceleration_series(input_df, keypoints_velocity_parts, rolling_windows)
    cols = [c for c in input_df.columns if c.startswith("partaccel_")]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_keypoints_jerk_series(input_df, keypoints_velocity_parts, rolling_windows)
    cols = [c for c in input_df.columns if c.startswith("partjerk_")]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    return input_df
