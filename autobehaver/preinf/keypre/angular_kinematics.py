"""Angular kinematics + curvature features for keypre."""

from __future__ import annotations

import numpy as np

from . import settings

rolling_windows = settings.rolling_windows


def calculate_angular_velocity_features(dataframe, intervals):
    # Calculate the difference in angle from the previous and next frames
    angle_diff_prev = dataframe["angle"].diff().fillna(0)
    angle_diff_next = (dataframe["angle"].shift(-1) - dataframe["angle"]).fillna(0)

    # Normalize the angle changes to be within the range [-180, 180]
    angle_diff_prev = (angle_diff_prev + 180) % 360 - 180
    angle_diff_next = (angle_diff_next + 180) % 360 - 180

    # Compute the instantaneous angular velocity by averaging the neighboring differences
    angular_velocity_inst = (angle_diff_prev + angle_diff_next) / 2

    # Store the instantaneous angular velocity in the DataFrame with _inst tag
    dataframe["angular_velocity_inst"] = angular_velocity_inst

    # If intervals are provided, compute the rolling average for each interval
    for interval in intervals:
        angular_velocity_col = f"angular_velocity_{interval}"
        dataframe[angular_velocity_col] = dataframe["angular_velocity_inst"].rolling(
            window=interval, min_periods=1, center=True
        ).mean()

    # If intervals are provided, compute the rolling standard deviation for each interval
    for interval in intervals:
        angular_velocity_std_col = f"angular_velocitystd_{interval}"
        dataframe[angular_velocity_std_col] = dataframe["angular_velocity_inst"].rolling(
            window=interval, min_periods=1, center=True
        ).std()

    return dataframe


def calculate_angular_acceleration_features(dataframe, intervals):
    """Calculates the angular acceleration as the rate of change of angular velocity."""
    # Calculate as the central difference of the instantaneous angular velocity
    dataframe["angular_acceleration_inst"] = (
        (dataframe["angular_velocity_inst"].shift(-1) - dataframe["angular_velocity_inst"].shift(1))
        / 2.0
    ).fillna(0)

    # Rolling averages for angular acceleration
    for interval in intervals:
        accel_col = f"angular_acceleration_{interval}"
        dataframe[accel_col] = dataframe["angular_acceleration_inst"].rolling(
            window=interval, min_periods=1, center=True
        ).mean()

    return dataframe


def calculate_angular_jerk_features(dataframe, intervals):
    """Calculates the angular jerk as the rate of change of angular acceleration."""
    # Calculate as the central difference of the instantaneous angular acceleration
    dataframe["angular_jerk_inst"] = (
        (dataframe["angular_acceleration_inst"].shift(-1) - dataframe["angular_acceleration_inst"].shift(1))
        / 2.0
    ).fillna(0)

    # Rolling averages for angular jerk
    for interval in intervals:
        jerk_col = f"angular_jerk_{interval}"
        dataframe[jerk_col] = dataframe["angular_jerk_inst"].rolling(
            window=interval, min_periods=1, center=True
        ).mean()

    return dataframe


def calculate_curvature_features(dataframe):
    """Calculates path curvature (rad/pixel) as (angular velocity) divided by speed."""

    epsilon = 1e-6  # Small constant to prevent division by zero

    # --- 1. Instantaneous Curvature ---
    if "angular_velocity_inst" in dataframe.columns and "speed_inst" in dataframe.columns:
        omega_rad = np.deg2rad(dataframe["angular_velocity_inst"])
        dataframe["curvature_inst"] = omega_rad / (dataframe["speed_inst"] + epsilon)

    # --- 2. Rolling Interval Curvature ---
    # Use the velocity intervals as the reference for which smoothed features to calculate
    for interval in rolling_windows:
        ang_vel_col = f"angular_velocity_{interval}"
        speed_col = f"speed_{interval}"
        if ang_vel_col in dataframe.columns and speed_col in dataframe.columns:
            omega_rad = np.deg2rad(dataframe[ang_vel_col])
            dataframe[f"curvature_{interval}"] = omega_rad / (dataframe[speed_col] + epsilon)

    return dataframe


def add_angular_kinematics_and_curvature(input_df, poscorr_dict):
    """Body orientation angular velocity/accel/jerk + curvature features."""
    input_df = calculate_angular_velocity_features(input_df, rolling_windows)
    if "angular_velocity_inst" in input_df.columns:
        poscorr_dict["angular_velocity_inst"] = input_df["angular_velocity_inst"]

    for interval in rolling_windows:
        mean_col = f"angular_velocity_{interval}"
        if mean_col in input_df.columns:
            poscorr_dict[mean_col] = input_df[mean_col]

    for interval in rolling_windows:
        std_col = f"angular_velocitystd_{interval}"
        if std_col in input_df.columns:
            poscorr_dict[std_col] = input_df[std_col]

    input_df = calculate_angular_acceleration_features(input_df, rolling_windows)
    if "angular_acceleration_inst" in input_df.columns:
        poscorr_dict["angular_acceleration_inst"] = input_df["angular_acceleration_inst"]
    for interval in rolling_windows:
        col = f"angular_acceleration_{interval}"
        if col in input_df.columns:
            poscorr_dict[col] = input_df[col]

    input_df = calculate_angular_jerk_features(input_df, rolling_windows)
    if "angular_jerk_inst" in input_df.columns:
        poscorr_dict["angular_jerk_inst"] = input_df["angular_jerk_inst"]
    for interval in rolling_windows:
        col = f"angular_jerk_{interval}"
        if col in input_df.columns:
            poscorr_dict[col] = input_df[col]

    input_df = calculate_curvature_features(input_df)
    curvature_cols = [c for c in input_df.columns if c.startswith("curvature_")]
    for c in curvature_cols:
        poscorr_dict[c] = input_df[c]

    return input_df
