"""Pairwise angular kinematics between selected keypoint pairs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import settings
from .spec import angular_velocity_pairs

logger = logging.getLogger(__name__)

rolling_windows = settings.rolling_windows


def calculate_pairwise_angular_velocities(dataframe, intervals, pairs):
    """
    Calculates angular velocity features for multiple pairs of keypoints.

    For each specified pair (a base keypoint and a moving keypoint), this computes the
    orientation (in degrees) of the vector from base -> moving for each frame using
    the rotated keypoint coordinates.

    Angular velocity is computed using a central-difference style estimate based on
    backward and forward angular differences, with wrap-around normalized into [-180, 180].

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing rotated coordinate columns.
        intervals (list): Rolling window sizes for smoothing.
        pairs (list or dict): Keypoint pairs to compute angles for.

    Returns:
        pd.DataFrame: Input dataframe with added pairwise angular velocity feature columns.
    """

    if intervals is None:
        intervals = []

    # Support both dictionary and list inputs for pairs.
    if isinstance(pairs, dict):
        pair_items = pairs.items()
    elif isinstance(pairs, list):
        pair_items = [(f"{base}_{moving}", (base, moving)) for base, moving in pairs]
    else:
        raise ValueError("pairs must be a list of tuples or a dict of (base, moving) tuples.")

    new_columns = {}

    for feature_suffix, (base_key, moving_key) in pair_items:
        new_column_name = f"partangular_velocity_{feature_suffix}"
        
        # Use rotated coordinates for all keypoints.
        # IMPORTANT: rotated coordinates are already centered on the fly centroid,
        # so the centroid is (0, 0) in this frame.
        if base_key.lower() == 'centroid':
            base_x = base_y = None  # origin
        else:
            base_x = f"{base_key}_rotated_x"
            base_y = f"{base_key}_rotated_y"

        if moving_key.lower() == 'centroid':
            moving_x = moving_y = None  # origin
        else:
            moving_x = f"{moving_key}_rotated_x"
            moving_y = f"{moving_key}_rotated_y"

        # If both base and moving are centroid, the vector is undefined; skip.
        required_cols = []
        if base_x is not None:
            required_cols += [base_x, base_y]
        if moving_x is not None:
            required_cols += [moving_x, moving_y]

        if not required_cols:
            logger.warning(f"Warning: Pair '{feature_suffix}' is centroid->centroid; skipping.")
            continue

        if not all(col in dataframe.columns for col in required_cols):
            logger.warning(f"Warning: Missing required columns for '{base_key}' and '{moving_key}'. Skipping pair.")
            continue

        # Compute dx, dy and valid frames consistently
        if base_x is None:
            # Vector from centroid(origin) -> moving keypoint
            valid_mask = dataframe[moving_x].notna() & dataframe[moving_y].notna()
            dx = dataframe[moving_x]
            dy = dataframe[moving_y]
        elif moving_x is None:
            # Vector from base keypoint -> centroid(origin)
            valid_mask = dataframe[base_x].notna() & dataframe[base_y].notna()
            dx = -dataframe[base_x]
            dy = -dataframe[base_y]
        else:
            valid_mask = (dataframe[base_x].notna() & dataframe[base_y].notna() &
                          dataframe[moving_x].notna() & dataframe[moving_y].notna())
            dx = dataframe[moving_x] - dataframe[base_x]
            dy = dataframe[moving_y] - dataframe[base_y]

        pair_angle = np.degrees(np.arctan2(dy, dx))
        pair_angle_series = pd.Series(pair_angle, index=dataframe.index).where(valid_mask, np.nan)

        # Compute backward and forward differences.
        angle_diff_prev = pair_angle_series.diff()
        angle_diff_next = (pair_angle_series.shift(-1) - pair_angle_series).fillna(0)

        # Normalize differences to the range [-180, 180].
        angle_diff_prev = (angle_diff_prev + 180) % 360 - 180
        angle_diff_next = (angle_diff_next + 180) % 360 - 180

        # For a valid angular change, ensure current, previous, and next frames are valid.
        valid_for_change = valid_mask & valid_mask.shift(1, fill_value=False) & valid_mask.shift(-1, fill_value=False)
        avg_angular_velocity = (angle_diff_prev + angle_diff_next) / 2.0
        avg_angular_velocity = avg_angular_velocity.where(valid_for_change, np.nan)

        # Store computed angular velocity with _inst suffix for consistency
        inst_column_name = f"{new_column_name}_inst"
        new_columns[inst_column_name] = avg_angular_velocity

        # Compute rolling mean angular velocity over each specified interval.
        for interval in intervals:
            interval_col = f"{new_column_name}_{interval}"
            new_columns[interval_col] = avg_angular_velocity.rolling(window=interval, min_periods=1, center=True).mean()
        # Compute rolling standard deviation over each specified interval
        for interval in intervals:
            interval_std_col = f"partangular_velocitystd_{feature_suffix}_{interval}"
            new_columns[interval_std_col] = avg_angular_velocity.rolling(window=interval, min_periods=1, center=True).std()

    # Concatenate the new columns with the source DataFrame.
    dataframe = pd.concat([dataframe, pd.DataFrame(new_columns, index=dataframe.index)], axis=1)
    
    return dataframe


def calculate_pairwise_angular_acceleration(dataframe, intervals, pairs):
    """
    Calculates angular acceleration for multiple pairs of keypoints.
    """
    if isinstance(pairs, dict):
        pair_items = pairs.items()
    elif isinstance(pairs, list):
        pair_items = [(f"{base}_{moving}", (base, moving)) for base, moving in pairs]
    else:
        raise ValueError("pairs must be a list of tuples or a dict of (base, moving) tuples.")

    new_columns = {}

    for feature_suffix, (base_key, moving_key) in pair_items:
        velocity_col = f'partangular_velocity_{feature_suffix}_inst'
        
        if velocity_col in dataframe.columns:
            # Compute instantaneous acceleration
            basic_accel = ((dataframe[velocity_col].shift(-1) - dataframe[velocity_col].shift(1)) / 2.0)
            valid_rows = dataframe[velocity_col].notna() & dataframe[velocity_col].shift(1).notna() & dataframe[velocity_col].shift(-1).notna()
            basic_accel = basic_accel.where(valid_rows, np.nan)
            
            accel_col_name = f'partangular_accel_{feature_suffix}_inst'
            new_columns[accel_col_name] = basic_accel

            for interval in intervals:
                interval_col = f'partangular_accel_{feature_suffix}_{interval}'
                new_columns[interval_col] = basic_accel.rolling(window=interval, min_periods=1, center=True).mean()
                            
    dataframe = pd.concat([dataframe, pd.DataFrame(new_columns, index=dataframe.index)], axis=1)
        
    return dataframe


def calculate_pairwise_angular_jerk(dataframe, intervals, pairs):
    """
    Calculates angular jerk for multiple pairs of keypoints.
    """
    if isinstance(pairs, dict):
        pair_items = pairs.items()
    elif isinstance(pairs, list):
        pair_items = [(f"{base}_{moving}", (base, moving)) for base, moving in pairs]
    else:
        raise ValueError("pairs must be a list of tuples or a dict of (base, moving) tuples.")

    new_columns = {}

    for feature_suffix, (base_key, moving_key) in pair_items:
        accel_col = f'partangular_accel_{feature_suffix}_inst'

        if accel_col in dataframe.columns:
            basic_jerk = ((dataframe[accel_col].shift(-1) - dataframe[accel_col].shift(1)) / 2.0)
            valid_rows = dataframe[accel_col].notna() & dataframe[accel_col].shift(1).notna() & dataframe[accel_col].shift(-1).notna()
            basic_jerk = basic_jerk.where(valid_rows, np.nan)

            jerk_col_name = f'partangular_jerk_{feature_suffix}_inst'
            new_columns[jerk_col_name] = basic_jerk

            for interval in intervals:
                interval_col = f'partangular_jerk_{feature_suffix}_{interval}'
                new_columns[interval_col] = basic_jerk.rolling(window=interval, min_periods=1, center=True).mean()
                
    dataframe = pd.concat([dataframe, pd.DataFrame(new_columns, index=dataframe.index)], axis=1)

    return dataframe


def add_pairwise_angular_kinematics_features(input_df, poscorr_dict):
    """Pairwise angular velocity/accel/jerk between selected keypoint pairs."""
    input_df = calculate_pairwise_angular_velocities(input_df, rolling_windows, angular_velocity_pairs)
    cols = [c for c in input_df.columns if c.startswith('partangular_velocity_') or c.startswith('partangular_velocitystd_')]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_pairwise_angular_acceleration(input_df, rolling_windows, angular_velocity_pairs)
    cols = [c for c in input_df.columns if c.startswith('partangular_accel_')]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_pairwise_angular_jerk(input_df, rolling_windows, angular_velocity_pairs)
    cols = [c for c in input_df.columns if c.startswith('partangular_jerk_')]
    for c in cols:
        poscorr_dict[c] = input_df[c]

    return input_df
