"""Derive centroid speed, body-axis velocity, and their derivatives for KeyPre."""

from __future__ import annotations

import logging

import numpy as np

from . import settings

logger = logging.getLogger(__name__)

# Frozen settings globals
rolling_windows = settings.rolling_windows
warn_on_internal_nans = settings.warn_on_internal_nans
edge_skip = settings.edge_skip


def calculate_vectorized_kinematics(dataframe, rolling_windows):
    """Derive centroid speed, forward/strafe velocity, acceleration, and jerk.

    Use centered displacement (t+1 minus t-1)/2 in pixels per frame; derivatives
    use pixels per frame squared/cubed. acceleration_speed_inst is the signed
    speed derivative, not Euclidean acceleration magnitude. Fill missing
    displacements and derivatives with zero and add centered rolling summaries.
    """

    # Velocity Calculation

    # Centroid displacement over 2 frames (from t-1 to t+1)
    dx_raw = dataframe["centroid_x"].shift(-1) - dataframe["centroid_x"].shift(1)
    dy_raw = dataframe["centroid_y"].shift(-1) - dataframe["centroid_y"].shift(1)

    # Optional diagnostic: warn if NaNs occur away from edges (these will become 0 due to fillna below)
    if globals().get("warn_on_internal_nans", False):
        es = globals().get("edge_skip", 0)
        min_edge = max(es, 1)  # always ignore at least the 1-frame shift edges

        interior = np.ones(len(dataframe), dtype=bool)
        if len(dataframe) > 2 * min_edge:
            interior[:min_edge] = False
            interior[-min_edge:] = False
        else:
            interior[:] = False

        bad = (dx_raw.isna() | dy_raw.isna()) & interior
        if bad.any():
            logger.info(
                f"Warning: {int(bad.sum())} interior frames have NaNs in centroid displacement; "
                "fillna(0) will create 0 speed/vel/accel/jerk at those frames."
            )

    # Fill edge/gap NaNs with 0 before calculating movement vectors
    dx = dx_raw.fillna(0)
    dy = dy_raw.fillna(0)

    # Movement vector per frame (divide by 2 for single frame interval)
    movement_vector_x = dx / 2.0
    movement_vector_y = dy / 2.0

    # Speed is the magnitude of the centered centroid velocity.
    dataframe["speed_inst"] = np.sqrt(movement_vector_x**2 + movement_vector_y**2)

    # Forward velocity is positive toward Front.
    # Body axis vector (Back -> Front)
    body_axis_x = dataframe["Front_x"] - dataframe["Back_x"]
    body_axis_y = dataframe["Front_y"] - dataframe["Back_y"]

    body_axis_mag = np.sqrt(body_axis_x**2 + body_axis_y**2).replace(
        0, 1e-6
    )  # Avoid division by zero
    norm_body_axis_x = body_axis_x / body_axis_mag
    norm_body_axis_y = body_axis_y / body_axis_mag

    dataframe["velocity_forward_inst"] = (movement_vector_x * norm_body_axis_x) + (
        movement_vector_y * norm_body_axis_y
    )

    # Project onto the perpendicular body axis for strafe velocity.
    # If body axis is the unit vector (ux, uy), a perpendicular unit vector is (-uy, ux).
    perp_body_axis_x = -norm_body_axis_y
    perp_body_axis_y = norm_body_axis_x

    dataframe["velocity_strafe_inst"] = (movement_vector_x * perp_body_axis_x) + (
        movement_vector_y * perp_body_axis_y
    )

    # Rolling velocity features (batched across columns for speed)
    vel_inst_df = dataframe[["speed_inst", "velocity_forward_inst", "velocity_strafe_inst"]]

    for interval in rolling_windows:
        roll = vel_inst_df.rolling(window=interval, min_periods=1, center=True)

        mean_df = roll.mean()
        mean_df.columns = [
            f"speed_{interval}",
            f"velocity_forward_{interval}",
            f"velocity_strafe_{interval}",
        ]
        dataframe[mean_df.columns] = mean_df

        std_df = roll.std()
        std_df.columns = [
            f"speedstd_{interval}",
            f"velocity_forwardstd_{interval}",
            f"velocity_strafestd_{interval}",
        ]
        dataframe[std_df.columns] = std_df

    # Acceleration Calculation
    # Calculate as the central difference of the instantaneous velocity components
    dataframe["acceleration_forward_inst"] = (
        (dataframe["velocity_forward_inst"].shift(-1) - dataframe["velocity_forward_inst"].shift(1))
        / 2.0
    ).fillna(0)
    dataframe["acceleration_strafe_inst"] = (
        (dataframe["velocity_strafe_inst"].shift(-1) - dataframe["velocity_strafe_inst"].shift(1))
        / 2.0
    ).fillna(0)
    dataframe["acceleration_speed_inst"] = (
        (dataframe["speed_inst"].shift(-1) - dataframe["speed_inst"].shift(1)) / 2.0
    ).fillna(0)

    # Rolling averages for acceleration features (batched)
    acc_inst_df = dataframe[
        ["acceleration_forward_inst", "acceleration_strafe_inst", "acceleration_speed_inst"]
    ]

    for interval in rolling_windows:
        mean_df = acc_inst_df.rolling(window=interval, min_periods=1, center=True).mean()
        mean_df.columns = [
            f"acceleration_forward_{interval}",
            f"acceleration_strafe_{interval}",
            f"acceleration_speed_{interval}",
        ]
        dataframe[mean_df.columns] = mean_df

    # Jerk Calculation
    # Calculate as the central difference of the instantaneous acceleration components
    dataframe["jerk_forward_inst"] = (
        (dataframe["acceleration_forward_inst"].shift(-1) - dataframe["acceleration_forward_inst"].shift(1))
        / 2.0
    ).fillna(0)
    dataframe["jerk_strafe_inst"] = (
        (dataframe["acceleration_strafe_inst"].shift(-1) - dataframe["acceleration_strafe_inst"].shift(1))
        / 2.0
    ).fillna(0)
    dataframe["jerk_speed_inst"] = (
        (dataframe["acceleration_speed_inst"].shift(-1) - dataframe["acceleration_speed_inst"].shift(1))
        / 2.0
    ).fillna(0)

    # Rolling averages for jerk features (batched)
    jerk_inst_df = dataframe[["jerk_forward_inst", "jerk_strafe_inst", "jerk_speed_inst"]]

    for interval in rolling_windows:
        mean_df = jerk_inst_df.rolling(window=interval, min_periods=1, center=True).mean()
        mean_df.columns = [
            f"jerk_forward_{interval}",
            f"jerk_strafe_{interval}",
            f"jerk_speed_{interval}",
        ]
        dataframe[mean_df.columns] = mean_df

    return dataframe


def add_centroid_kinematics_features(input_df, poscorr_dict):
    """Add centroid speed, forward/strafe components, and derivatives to the feature mapping."""

    input_df = calculate_vectorized_kinematics(input_df, rolling_windows=rolling_windows)

    kinematic_prefixes = [
        "speed",
        "velocity_forward",
        "velocity_strafe",
        "acceleration_forward",
        "acceleration_strafe",
        "acceleration_speed",
        "jerk_forward",
        "jerk_strafe",
        "jerk_speed",
    ]

    for prefix in kinematic_prefixes:
        inst_col = f"{prefix}_inst"
        if inst_col in input_df.columns:
            poscorr_dict[inst_col] = input_df[inst_col]

        intervals = rolling_windows
        if "acceleration" in prefix:
            intervals = rolling_windows
        elif "jerk" in prefix:
            intervals = rolling_windows

        # mean features
        for interval in intervals:
            interval_col = f"{prefix}_{interval}"
            if interval_col in input_df.columns:
                poscorr_dict[interval_col] = input_df[interval_col]

        # Standard deviations are defined only for speed and velocity components.
        for interval in intervals:
            if prefix == "speed":
                std_col = f"speedstd_{interval}"
            elif prefix == "velocity_forward":
                std_col = f"velocity_forwardstd_{interval}"
            elif prefix == "velocity_strafe":
                std_col = f"velocity_strafestd_{interval}"
            else:
                std_col = f"{prefix}_std_{interval}"
            if std_col in input_df.columns:
                poscorr_dict[std_col] = input_df[std_col]

    return input_df
