"""Core pose + coordinate features for keypre."""

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
    front_x = dataframe["Front_x"]
    front_y = dataframe["Front_y"]
    back_x = dataframe["Back_x"]
    back_y = dataframe["Back_y"]
    centroid_x = (front_x + back_x) / 2
    centroid_y = (front_y + back_y) / 2
    return centroid_x, centroid_y


def calculate_angle(dataframe):
    # Back -> Front vector in image coordinates (+y is down)
    dx = dataframe["Front_x"] - dataframe["Back_x"]
    dy = dataframe["Front_y"] - dataframe["Back_y"]

    # θ_img in degrees: 0=East, +90=South, 180=West, -90=North
    theta_img_deg = np.degrees(np.arctan2(dy, dx))

    # Navigation bearing b: 0=North, 90=East, 180=South, 270=West
    angle_nav = (theta_img_deg + 90) % 360
    return angle_nav


def calculate_polar_coordinates(dataframe, movie_dim):
    """Calculates polar coordinate features relative to the arena center.

    Assumptions:
      - The arena is centered in a square frame of width/height = movie_dim pixels.
      - DLC pixel coordinates behave like indices ~[0 .. movie_dim-1], so the center is
        at (movie_dim - 1) / 2 in both x and y.

    Coordinate conventions used here:
      - Image coordinates: +x right, +y down.
      - Navigational/bearing angles: 0 = North (up), 90 = East (right),
        180 = South (down), 270 = West (left).

    Returns:
      - radial_position: normalized distance from arena center (0 at center, ~1 at arena edge)
      - radial_angle_*: bearing from arena center -> fly centroid (sin/cos)
      - relative_angle_*: (fly_angle_nav - radial_bearing_nav) (sin/cos)
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

    # Normalized radial distance
    dist = np.sqrt(dx**2 + dy**2)
    radial_position = dist / arena_radius

    # Bearing from center to fly in NAV coordinates (0=N, 90=E)
    # arctan2(dy, dx) in image coords gives: 0=E, 90=S, -90=N
    theta_img_deg = np.degrees(np.arctan2(dy, dx))
    radial_bearing_nav = (theta_img_deg + 90) % 360

    # Relative angle (signed, [-180, 180])
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
    """Compute centroid + angle (always, because downstream depends on them).

    Store position/polar features to poscorr_dict.
    """

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
