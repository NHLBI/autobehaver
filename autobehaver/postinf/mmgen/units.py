"""Unit conversion helpers for mmgen.

This module contains small conversion helpers that transform positional_calcs
columns from pixel-per-frame units into real units (mm/s, mm/s^2, etc.).

This module infers units and emits unit manifests for the master matrix outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _convert_kinematic_columns_to_real_units(
    df: pd.DataFrame, *, mm_per_px: float, fps: float
) -> pd.DataFrame:
    """Convert kinematic columns from per-frame pixel units into real units.

    Expected upstream units (from keypre positional_calcs):
      - speed / velocity_* / partvelocity_*      : px/frame
      - acceleration_* / partaccel_*             : px/frame^2
      - jerk_* / partjerk_*                      : px/frame^3
      - angular_velocity_* / partangular_velocity_* : deg/frame
      - angular_acceleration_* / partangular_accel_* : deg/frame^2
      - angular_jerk_* / partangular_jerk_*         : deg/frame^3
      - curvature_*                              : rad/px

    Converted units:
      - linear velocity     : mm/s
      - linear acceleration : mm/s^2
      - linear jerk         : mm/s^3
      - angular velocity    : deg/s
      - angular acceleration: deg/s^2
      - angular jerk        : deg/s^3
      - curvature           : rad/mm
    """

    out = df.copy()

    scale_vel = mm_per_px * fps
    scale_acc = mm_per_px * (fps**2)
    scale_jerk = mm_per_px * (fps**3)

    scale_ang_vel = fps
    scale_ang_acc = fps**2
    scale_ang_jerk = fps**3

    scale_curv = 1.0 / mm_per_px  # rad/px -> rad/mm

    for col in out.columns:
        # Curvature first (rad/px)
        if col.startswith("curvature_"):
            out[col] = out[col] * scale_curv
            continue

        # Angular kinematics (deg/frame^n)
        if col.startswith("angular_jerk_") or col.startswith("partangular_jerk_"):
            out[col] = out[col] * scale_ang_jerk
            continue
        if col.startswith("angular_acceleration_") or col.startswith(
            "partangular_accel_"
        ):
            out[col] = out[col] * scale_ang_acc
            continue
        if col.startswith("angular_velocity_") or col.startswith(
            "partangular_velocity_"
        ):
            out[col] = out[col] * scale_ang_vel
            continue

        # Linear kinematics (px/frame^n)
        if col.startswith("jerk_") or col.startswith("partjerk_"):
            out[col] = out[col] * scale_jerk
            continue
        if col.startswith("acceleration_") or col.startswith("partaccel_"):
            out[col] = out[col] * scale_acc
            continue
        if col.startswith("speed") or col.startswith("velocity_") or col.startswith(
            "partvelocity_"
        ):
            out[col] = out[col] * scale_vel
            continue

    return out
