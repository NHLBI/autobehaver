"""Worker helpers for building the per-frame positional substrate."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def build_positional_df(
    *,
    config: dict,
    masked_poses: np.ndarray,
    masked_orients: Optional[np.ndarray],
    has_orient: bool,
    masked_pose_valid: np.ndarray,
    masked_orient_valid: Optional[np.ndarray],
    masked_pose_orient_valid: Optional[np.ndarray],
    aligned_poscalc_df: pd.DataFrame,
    aligned_poscalc_df_kin: pd.DataFrame,
    group: str,
    individual_id: str,
    source_file: str,
) -> Optional[pd.DataFrame]:
    """Build the positional_df used by occupancy and other spatial analyses.

    The helper assembles the per-frame positional substrate used by occupancy and downstream spatial analyses.
    """

    if not config.get("calculate_occupancy"):
        return None

    num_frames = len(masked_poses)

    data_dict = {
        "group": group,
        "individual": individual_id,
        "source_file": source_file,
        "frame_idx": np.arange(num_frames, dtype=int),
        # Raw (smoothed) labels
        "masked_pose": masked_poses,
    }

    if has_orient and masked_orients is not None:
        data_dict["masked_orient"] = masked_orients
        data_dict["masked_pose_orient"] = [
            f"{p} + {o}" if pd.notna(p) and pd.notna(o) else np.nan
            for p, o in zip(masked_poses, masked_orients)
        ]
    else:
        data_dict["masked_orient"] = np.nan
        data_dict["masked_pose_orient"] = np.nan

    # Valid (bout-filtered) labels
    data_dict["masked_pose_valid"] = masked_pose_valid
    data_dict["masked_orient_valid"] = masked_orient_valid if masked_orient_valid is not None else np.nan
    data_dict["masked_pose_orient_valid"] = (
        masked_pose_orient_valid if masked_pose_orient_valid is not None else np.nan
    )

    # Position columns
    for col in ["centroid_x", "centroid_y"]:
        data_dict[col] = aligned_poscalc_df[col] if col in aligned_poscalc_df.columns else np.nan

    # Compute absolute fly angle from sin/cos if present
    if "angle_sin" in aligned_poscalc_df.columns and "angle_cos" in aligned_poscalc_df.columns:
        angle_rad = np.arctan2(aligned_poscalc_df["angle_sin"], aligned_poscalc_df["angle_cos"])
        angle_deg = np.degrees(angle_rad)
        data_dict["angle"] = (angle_deg + 360) % 360
    else:
        data_dict["angle"] = np.nan

    # Per-frame centroid speed in mm/s (aligned_poscalc_df_kin already converted)
    if "speed_inst" in aligned_poscalc_df_kin.columns:
        data_dict["speed_mm_s"] = aligned_poscalc_df_kin["speed_inst"]
    else:
        data_dict["speed_mm_s"] = np.nan

    # Angle features for wing/leg spread (present in your sample_positional_calcs)
    angle_cols = [
        "anglefeature_Wings_L_R",
        "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR",
        "anglefeature_Legs_BL_BR",
    ]
    for col in angle_cols:
        data_dict[col] = aligned_poscalc_df[col] if col in aligned_poscalc_df.columns else np.nan

    return pd.DataFrame(data_dict)
