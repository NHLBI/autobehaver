"""Derive centroid position, body bearing, and frame-centered polar features."""

from __future__ import annotations

import logging

import numpy as np

from . import settings

logger = logging.getLogger(__name__)


# Runtime config (set by worker.set_run_config / init_worker)
movie_dim: int = 600

# Feature switches (frozen settings)
include_position = settings.include_position


def calculate_centroid(dataframe):
    """Calculate the centroid from the derived Front and Back points."""
    front_x = dataframe["Front_x"]
    front_y = dataframe["Front_y"]
    back_x = dataframe["Back_x"]
    back_y = dataframe["Back_y"]
    centroid_x = (front_x + back_x) / 2
    centroid_y = (front_y + back_y) / 2
    return centroid_x, centroid_y


def calculate_angle(dataframe):
    # Back -> Front vector in image coordinates (+y is down)
    """Calculate the Back-to-Front bearing: 0 north, 90 east, 180 south, 270 west.

    Use image coordinates with +x right and +y down.
    """
    dx = dataframe["Front_x"] - dataframe["Back_x"]
    dy = dataframe["Front_y"] - dataframe["Back_y"]

    # θ_img in degrees: 0=East, +90=South, 180=West, -90=North
    theta_img_deg = np.degrees(np.arctan2(dy, dx))

    # Navigation bearing b: 0=North, 90=East, 180=South, 270=West
    angle_nav = (theta_img_deg + 90) % 360
    return angle_nav


def calculate_polar_coordinates(dataframe, movie_dim):
    """Normalize centroid radius by the square frame half-width (movie_dim - 1)/2.

    Place the frame center at that coordinate in both axes; +x is right and +y
    is down. Use bearings 0 north, 90 east, 180 south, and 270 west. Return radius
    and sine/cosine columns for radial bearing and body bearing minus radial bearing.
    """

    # DLC coordinates are in pixel units with indices ~[0 .. movie_dim-1],
    # so the geometric center is at (movie_dim - 1) / 2.
    arena_center_x = (movie_dim - 1) / 2.0
    arena_center_y = (movie_dim - 1) / 2.0
    arena_radius = (movie_dim - 1) / 2.0

    fly_x = dataframe["centroid_x"]
    fly_y = dataframe["centroid_y"]
    fly_angle_nav = dataframe["angle"]  # 0 = North

    # Vector from arena center to fly (image coords, +y is down)
    dx = fly_x - arena_center_x
    dy = fly_y - arena_center_y

    dist = np.sqrt(dx**2 + dy**2)
    radial_position = dist / arena_radius

    # Bearing from center to fly in NAV coordinates (0=N, 90=E)
    # arctan2(dy, dx) in image coords gives: 0=E, 90=S, -90=N
    theta_img_deg = np.degrees(np.arctan2(dy, dx))
    radial_bearing_nav = (theta_img_deg + 90) % 360

    # Wrap body bearing relative to radial bearing to [-180, 180).
    relative_angle = (fly_angle_nav - radial_bearing_nav + 180) % 360 - 180

    radial_angle_rad = np.radians(radial_bearing_nav)
    relative_angle_rad = np.radians(relative_angle)

    return {
        "radial_position": radial_position,
        "radial_angle_sin": np.sin(radial_angle_rad),
        "radial_angle_cos": np.cos(radial_angle_rad),
        "relative_angle_sin": np.sin(relative_angle_rad),
        "relative_angle_cos": np.cos(relative_angle_rad),
    }


def add_pose_features(input_df, poscorr_dict):
    """Calculate centroid and body bearing, then add position and polar features to the mapping."""

    input_df["centroid_x"], input_df["centroid_y"] = calculate_centroid(input_df)
    input_df["angle"] = calculate_angle(input_df)

    if include_position:
        poscorr_dict["centroid_x"] = input_df["centroid_x"]
        poscorr_dict["centroid_y"] = input_df["centroid_y"]

        angle_rad = np.radians(input_df["angle"])
        poscorr_dict["angle_sin"] = np.sin(angle_rad)
        poscorr_dict["angle_cos"] = np.cos(angle_rad)

        polar_features = calculate_polar_coordinates(input_df, movie_dim)
        poscorr_dict["radial_position"] = polar_features["radial_position"]
        poscorr_dict["radial_angle_sin"] = polar_features["radial_angle_sin"]
        poscorr_dict["radial_angle_cos"] = polar_features["radial_angle_cos"]
        poscorr_dict["relative_angle_sin"] = polar_features["relative_angle_sin"]
        poscorr_dict["relative_angle_cos"] = polar_features["relative_angle_cos"]

    return input_df
