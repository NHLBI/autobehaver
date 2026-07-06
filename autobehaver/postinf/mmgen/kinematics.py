"""Kinematics domain logic for mmgen.

This module contains the kinematic feature extraction and summary writers
used by mmgen finalization.

This module groups together:

* per-frame head movement signals (for TW kinematics)
* per-bout kinematic summary feature extraction
* reconstruction of aggregated per-bout kinematics tables
* writing the BB (bout-based) and TW (time-weighted) kinematic summary outputs

The functions are structured so they can be called independently from the mmgen pipeline because they
encode the current output contract and unit conventions.

Runtime settings
----------------
Run-specific values such as FPS and bout-count thresholds are passed in
explicitly from the orchestration layer.
"""

from __future__ import annotations

import os
import warnings
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .feature_matrix import format_summary_df, save_to_excel_sheets
from .naming import collapse_individual_id


DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES: int = 6
DEFAULT_MIN_BOUT_COUNT_FOR_TRANSITIONS: int = 6

# --- Kinematic Feature Extraction Settings ---
BOUT_TRIM_FRAMES_CORE: int = 5
BOUT_FRAMES_START_END: int = BOUT_TRIM_FRAMES_CORE

# --- Context guardrails (used by TW kinematics) ---
DEFAULT_VIDEO_FPS: float = 40.0
MIN_CONTEXT_DURATION_S: float = 3.0


# --- Time-Weighted (TW) kinematics: per-frame storage schema ---

HEAD_KEYPOINTS = ["Ocelli", "Antennae", "Mouthparts", "EyeLeft", "EyeRight"]

# Core per-frame signals we want available for TW kinematics
KIN_FRAMES_SIGNAL_COLS = [
    # Centroid / body translation (mm/s, mm/s^2, mm/s^3)
    "speed_inst",
    "velocity_forward_inst",
    "velocity_strafe_inst",
    "acceleration_magnitude_inst",
    "acceleration_forward_inst",
    "acceleration_strafe_inst",
    "jerk_magnitude_inst",
    "jerk_forward_inst",
    "jerk_strafe_inst",

    # Body angle (deg/s, deg/s^2, deg/s^3)
    "angular_velocity_inst",
    "angular_acceleration_inst",
    "angular_jerk_inst",

    # Walking curvature (rad/mm)
    "curvature_inst",

    # Head (derived from multiple keypoints; mm/s, mm/s^2, mm/s^3)
    "head_velocity_inst",
    "head_acceleration_inst",
    "head_jerk_inst",
]

# Add leg translational + angular columns programmatically (Front/Mid/Back; Left/Right)
_LEG_SPECS = {
    "Front": ("FrontTarsus", "FrontJoint"),
    "Mid": ("MidTarsus", "MidJoint"),
    "Back": ("BackTarsus", "BackJoint"),
}
for _leg, (_tarsus, _joint) in _LEG_SPECS.items():
    for _base in ("partvelocity", "partaccel", "partjerk"):
        KIN_FRAMES_SIGNAL_COLS.extend(
            [
                f"{_base}_{_tarsus}Left_inst",
                f"{_base}_{_tarsus}Right_inst",
            ]
        )
    for _base in ("partangular_velocity", "partangular_accel", "partangular_jerk"):
        KIN_FRAMES_SIGNAL_COLS.extend(
            [
                f"{_base}_{_joint}Left_{_tarsus}Left_inst",
                f"{_base}_{_joint}Right_{_tarsus}Right_inst",
            ]
        )

# Full per-frame schema for the HDF5 kin_frames table
PREDEFINED_KIN_FRAMES_COLS = [
    "group",
    "individual",
    "sex",
    "source_file",
    "frame_idx",
    "kin_context_valid",  # Pose+Orient valid label + Jump/Scramble
] + KIN_FRAMES_SIGNAL_COLS


def _finalize_mean_std(sum_, sum_sq, count_):
    """Finalize mean/std from aggregated sum, sum_sq, count.

    """

    sum_ = np.asarray(sum_, dtype="float64")
    sum_sq = np.asarray(sum_sq, dtype="float64")
    count_ = np.asarray(count_, dtype="int64")
    safe_n = np.maximum(count_, 1)
    mean = sum_ / safe_n
    # sample variance with ddof=1 when n>1; guard for n<=1
    var_num = np.maximum(sum_sq - (np.square(sum_) / safe_n), 0.0)
    denom = np.where(count_ > 1, count_ - 1, 1)
    std = np.sqrt(var_num / denom)
    return mean, std


# ---------------------------------------------------------------------------
# Core implementation helpers
# ---------------------------------------------------------------------------
def _safe_skewness(mean_val, median_val, std_val):
    """
    Calculates Pearson's non-parametric skewness: (mean - median) / std
    Returns np.nan if std is 0 or invalid.
    """
    if np.isnan(mean_val) or np.isnan(median_val) or np.isnan(std_val):
        return np.nan
    if std_val == 0 or std_val <= 1e-10:  # Avoid division by zero
        return np.nan
    return (mean_val - median_val) / std_val


def _safe_tail_means_from_array(arr, pmin=5, pmax=95, min_tail_frames=3):
    """Compute robust tail means from an array.

    This is used for the TW "min" and "max" kinematic statistics.

    Definition (per context / per signal):
      - "min" = mean of values in the bottom pmin% of the distribution
      - "max" = mean of values in the top  (100 - pmax)% of the distribution

    Unlike percentile *thresholds*, tail means reflect the average magnitude of
    the extreme values themselves.

    Guardrail:
      - We require at least `min_tail_frames` values to contribute to each tail.
        For short contexts this prevents tail means from collapsing to a single
        frame (which is overly sensitive to noise).

    Implementation:
      - Remove NaNs
      - Sort values
      - Take the bottom/top K values, where:
          K = max(min_tail_frames, ceil(tail_fraction * n))
        (tail_fraction derived from pmin/pmax)
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < int(min_tail_frames):
        return np.nan, np.nan

    # Derive tail fractions from pmin/pmax (clamped to [0,1])
    pmin_f = float(pmin) / 100.0
    pmax_f = float(pmax) / 100.0
    pmin_f = max(0.0, min(1.0, pmin_f))
    pmax_f = max(0.0, min(1.0, pmax_f))

    low_frac = pmin_f
    high_frac = max(0.0, 1.0 - pmax_f)

    # Use a tiny epsilon to avoid float round-up (e.g., 5.00000000004 -> 6).
    eps = 1e-12
    low_k = int(math.ceil(low_frac * n - eps))
    high_k = int(math.ceil(high_frac * n - eps))
    low_k = max(low_k, int(min_tail_frames))
    high_k = max(high_k, int(min_tail_frames))
    low_k = min(low_k, n)
    high_k = min(high_k, n)

    sorted_vals = np.sort(arr)
    low_vals = sorted_vals[:low_k]
    high_vals = sorted_vals[-high_k:]

    low_mean = float(np.nanmean(low_vals)) if low_vals.size else np.nan
    high_mean = float(np.nanmean(high_vals)) if high_vals.size else np.nan
    return low_mean, high_mean


def _safe_percentile_thresholds_from_array(arr, pmin=5, pmax=95):
    """Return percentile-threshold min/max values from an array.

    This is used for BB min/max kinematic statistics.

    Definition (per bout / per signal):
      - "min" = pmin percentile threshold value
      - "max" = pmax percentile threshold value
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan

    low = float(np.nanpercentile(arr, pmin))
    high = float(np.nanpercentile(arr, pmax))
    return low, high

def _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, base_feature, leg_part):
    """
    A helper to calculate mean, std, and asymmetry for a single kinematic feature on a single leg pair.
    """
    mean_val, std_val, asym_val = np.nan, np.nan, np.nan
    
    left_col = f'{base_feature}_{leg_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{leg_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        # Calculate overall mean and std
        left_mean = bout_features[left_col].mean()
        right_mean = bout_features[right_col].mean()
        left_std = bout_features[left_col].std()
        right_std = bout_features[right_col].std()
        mean_val = np.nanmean([left_mean, right_mean])
        std_val = np.nanmean([left_std, right_std])
        
        # Only calculate asymmetry for flat surfaces
        if "Floor" in bout_type or "Ceiling" in bout_type:
            denominator = bout_features[right_col] + bout_features[left_col]
            denominator.replace(0, np.nan, inplace=True)
            instant_asymmetry = (bout_features[right_col] - bout_features[left_col]) / denominator
            asym_val = instant_asymmetry.abs().mean()
            
    return mean_val, std_val, asym_val

def _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part):
    """
    A helper for angular kinematics with a pivot_part and moving_part naming convention.
    """
    mean_val, std_val, asym_val = np.nan, np.nan, np.nan
    
    left_col = f'{base_feature}_{pivot_part}Left_{moving_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{pivot_part}Right_{moving_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        # Calculate overall mean and std
        left_mean = bout_features[left_col].mean()
        right_mean = bout_features[right_col].mean()
        left_std = bout_features[left_col].std()
        right_std = bout_features[right_col].std()
        mean_val = np.nanmean([left_mean, right_mean])
        std_val = np.nanmean([left_std, right_std])
        
        # Only calculate asymmetry for flat surfaces
        if "Floor" in bout_type or "Ceiling" in bout_type:
            denominator = bout_features[right_col] + bout_features[left_col]
            denominator.replace(0, np.nan, inplace=True)
            instant_asymmetry = (bout_features[right_col] - bout_features[left_col]) / denominator
            asym_val = instant_asymmetry.abs().mean()
            
    return mean_val, std_val, asym_val

def compute_head_movement_frame_signals(df: pd.DataFrame, *, kinematic_suffix: str = '_inst'):
    """
    Framewise head movement signals:
      - head_velocity_* : mm/s
      - head_acceleration_* : mm/s^2
      - head_jerk_* : mm/s^3

    Logic matches compute_head_movement_features:
      per frame, drop the single worst-likelihood head keypoint, then nanmean the remaining.
    """
    keypoints = HEAD_KEYPOINTS

    lk_cols   = [f"{kp}_likelihood" for kp in keypoints]
    vel_cols  = [f"partvelocity_{kp}{kinematic_suffix}" for kp in keypoints]
    acc_cols  = [f"partaccel_{kp}{kinematic_suffix}" for kp in keypoints]
    jerk_cols = [f"partjerk_{kp}{kinematic_suffix}" for kp in keypoints]

    n = len(df)
    nan_arr = np.full(n, np.nan, dtype=float)

    # Need likelihood + vel + accel to compute vel/acc.
    if not all(c in df.columns for c in (lk_cols + vel_cols + acc_cols)):
        return nan_arr, nan_arr, nan_arr

    likelihoods   = df[lk_cols].to_numpy(dtype=float, copy=True)
    velocities    = df[vel_cols].to_numpy(dtype=float, copy=True)
    accelerations = df[acc_cols].to_numpy(dtype=float, copy=True)

    jerks = None
    if all(c in df.columns for c in jerk_cols):
        jerks = df[jerk_cols].to_numpy(dtype=float, copy=True)

    # Find worst-likelihood keypoint per frame
    all_nan_rows = np.all(np.isnan(likelihoods), axis=1)
    min_indices = np.full(n, -1, dtype=int)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        min_indices[valid_rows] = np.nanargmin(likelihoods[valid_rows], axis=1)

    rows = np.arange(n)
    if np.any(valid_rows):
        velocities[rows[valid_rows], min_indices[valid_rows]] = np.nan
        accelerations[rows[valid_rows], min_indices[valid_rows]] = np.nan
        if jerks is not None:
            jerks[rows[valid_rows], min_indices[valid_rows]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_vel = np.nanmean(velocities, axis=1)
        frame_acc = np.nanmean(accelerations, axis=1)
        frame_jerk = np.nanmean(jerks, axis=1) if jerks is not None else nan_arr

    return frame_vel, frame_acc, frame_jerk

def compute_head_movement_features(bout_features, kinematic_suffix="_inst"):
    """
    Computes head movement features for a given bout using vectorized operations.

    Returns:
        head_velocity_mean, head_velocity_std,
        head_acceleration_mean, head_acceleration_std,
        head_jerk_mean, head_jerk_std

    Notes:
        - Uses HEAD_KEYPOINTS.
        - Drops the worst-likelihood keypoint each frame and averages remaining.
        - std uses ddof=1 to match pandas' Series.std() default used elsewhere.
        - Jerk is "optional-safe": if jerk columns are missing, jerk outputs are NaN
          but velocity/acceleration still compute.
    """
    keypoints = HEAD_KEYPOINTS

    if bout_features is None or bout_features.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    likelihood_cols = [f"{kp}_likelihood" for kp in keypoints]
    velocity_cols = [f"partvelocity_{kp}{kinematic_suffix}" for kp in keypoints]
    accel_cols = [f"partaccel_{kp}{kinematic_suffix}" for kp in keypoints]
    jerk_cols = [f"partjerk_{kp}{kinematic_suffix}" for kp in keypoints]

    required_cols = likelihood_cols + velocity_cols + accel_cols
    if not all(col in bout_features.columns for col in required_cols):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    likelihoods = bout_features[likelihood_cols].to_numpy()
    velocities = bout_features[velocity_cols].to_numpy()
    accels = bout_features[accel_cols].to_numpy()

    # Jerk optional-safe
    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy()
    else:
        jerks = np.full_like(velocities, np.nan)

    # For each frame, find the index of the worst likelihood keypoint.
    # If an entire row is NaN likelihoods, skip dropping for that row.
    all_nan_rows = np.all(np.isnan(likelihoods), axis=1)
    min_indices = np.full(likelihoods.shape[0], -1, dtype=int)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        min_indices[valid_rows] = np.nanargmin(likelihoods[valid_rows], axis=1)

    rows = np.arange(likelihoods.shape[0])
    if np.any(valid_rows):
        velocities[rows[valid_rows], min_indices[valid_rows]] = np.nan
        accels[rows[valid_rows], min_indices[valid_rows]] = np.nan
        jerks[rows[valid_rows], min_indices[valid_rows]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_velocities = np.nanmean(velocities, axis=1)
        frame_accels = np.nanmean(accels, axis=1)
        frame_jerks = np.nanmean(jerks, axis=1)

    head_velocity_mean = np.nanmean(frame_velocities)
    n_vel = int(np.sum(~np.isnan(frame_velocities)))
    head_velocity_std = np.nanstd(frame_velocities, ddof=1) if n_vel > 1 else np.nan

    head_acceleration_mean = np.nanmean(frame_accels)
    n_acc = int(np.sum(~np.isnan(frame_accels)))
    head_acceleration_std = np.nanstd(frame_accels, ddof=1) if n_acc > 1 else np.nan

    head_jerk_mean = np.nanmean(frame_jerks)
    n_jerk = int(np.sum(~np.isnan(frame_jerks)))
    head_jerk_std = np.nanstd(frame_jerks, ddof=1) if n_jerk > 1 else np.nan

    return (
        head_velocity_mean, head_velocity_std,
        head_acceleration_mean, head_acceleration_std,
        head_jerk_mean, head_jerk_std
    )

def _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, base_feature, leg_part, mean_val, std_val):
    """
    Returns median and skewness for leg kinematics.
    Combines left and right legs, computes median, then skewness using provided mean/std.
    """
    median_val, skewness_val = np.nan, np.nan
    
    left_col = f'{base_feature}_{leg_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{leg_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        # Combine left and right
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if len(combined) > 0:
            median_val = combined.median()
            # Use provided mean and std for skewness calculation
            skewness_val = _safe_skewness(mean_val, median_val, std_val)
    
    return median_val, skewness_val

def _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part, mean_val, std_val):
    """
    Returns median and skewness for angular leg kinematics.
    Combines left and right legs, computes median, then skewness using provided mean/std.
    """
    median_val, skewness_val = np.nan, np.nan
    
    left_col = f'{base_feature}_{pivot_part}Left_{moving_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{pivot_part}Right_{moving_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        # Combine left and right
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if len(combined) > 0:
            median_val = combined.median()
            # Use provided mean and std for skewness calculation
            skewness_val = _safe_skewness(mean_val, median_val, std_val)
    
    return median_val, skewness_val

def compute_head_movement_features_median_skewness(
    bout_features,
    kinematic_suffix,
    velocity_mean, velocity_std,
    acceleration_mean, acceleration_std,
    jerk_mean, jerk_std
):
    """
    Computes median and skewness of head velocity, acceleration, and jerk.
    Skewness is computed as Pearson's second coefficient: (mean - median) / std.

    Jerk is optional-safe: if jerk columns are missing, jerk median/skewness return NaN.
    """
    head_keypoints = HEAD_KEYPOINTS
    likelihood_cols = [f'{kp}_likelihood' for kp in head_keypoints]
    velocity_cols = [f'partvelocity_{kp}{kinematic_suffix}' for kp in head_keypoints]
    accel_cols = [f'partaccel_{kp}{kinematic_suffix}' for kp in head_keypoints]
    jerk_cols = [f'partjerk_{kp}{kinematic_suffix}' for kp in head_keypoints]

    if not all(col in bout_features.columns for col in likelihood_cols + velocity_cols + accel_cols):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    likelihoods = bout_features[likelihood_cols].to_numpy()
    velocities = bout_features[velocity_cols].to_numpy()
    accels = bout_features[accel_cols].to_numpy()

    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy()
    else:
        jerks = np.full_like(velocities, np.nan)

    all_nan_rows = np.all(np.isnan(likelihoods), axis=1)
    min_indices = np.full(likelihoods.shape[0], -1, dtype=int)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        min_indices[valid_rows] = np.nanargmin(likelihoods[valid_rows], axis=1)

    rows = np.arange(len(min_indices))
    if np.any(valid_rows):
        velocities[rows[valid_rows], min_indices[valid_rows]] = np.nan
        accels[rows[valid_rows], min_indices[valid_rows]] = np.nan
        jerks[rows[valid_rows], min_indices[valid_rows]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_velocities = np.nanmean(velocities, axis=1)
        frame_accels = np.nanmean(accels, axis=1)
        frame_jerks = np.nanmean(jerks, axis=1)

    if np.any(~np.isnan(frame_velocities)):
        head_velocity_median = np.nanmedian(frame_velocities)
        head_velocity_skewness = _safe_skewness(velocity_mean, head_velocity_median, velocity_std)
    else:
        head_velocity_median, head_velocity_skewness = np.nan, np.nan

    if np.any(~np.isnan(frame_accels)):
        head_acceleration_median = np.nanmedian(frame_accels)
        head_acceleration_skewness = _safe_skewness(acceleration_mean, head_acceleration_median, acceleration_std)
    else:
        head_acceleration_median, head_acceleration_skewness = np.nan, np.nan

    if np.any(~np.isnan(frame_jerks)):
        head_jerk_median = np.nanmedian(frame_jerks)
        head_jerk_skewness = _safe_skewness(jerk_mean, head_jerk_median, jerk_std)
    else:
        head_jerk_median, head_jerk_skewness = np.nan, np.nan

    return (
        head_velocity_median, head_velocity_skewness,
        head_acceleration_median, head_acceleration_skewness,
        head_jerk_median, head_jerk_skewness
    )

def _calculate_leg_kinematics_minmax(
    bout_features,
    bout_type,
    kinematic_suffix,
    base_feature,
    leg_part,
    *,
    use_tail_means: bool = False,
    min_tail_frames: int = 3,
):
    """Return min/max for paired-leg kinematics.

    By default this returns BB percentile thresholds (p05/p95). When
    ``use_tail_means=True`` it returns TW-style tail means.
    """
    min_val, max_val = np.nan, np.nan

    left_col = f'{base_feature}_{leg_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{leg_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if not combined.empty:
            arr = combined.to_numpy()
            if use_tail_means:
                min_val, max_val = _safe_tail_means_from_array(
                    arr, pmin=5, pmax=95, min_tail_frames=min_tail_frames
                )
            else:
                min_val, max_val = _safe_percentile_thresholds_from_array(
                    arr, pmin=5, pmax=95
                )

    return min_val, max_val

def _calculate_angular_leg_kinematics_minmax(
    bout_features,
    bout_type,
    kinematic_suffix,
    base_feature,
    pivot_part,
    moving_part,
    *,
    use_tail_means: bool = False,
    min_tail_frames: int = 3,
):
    """Return min/max for paired angular-leg kinematics.

    By default this returns BB percentile thresholds (p05/p95). When
    ``use_tail_means=True`` it returns TW-style tail means.
    """
    min_val, max_val = np.nan, np.nan

    left_col = f'{base_feature}_{pivot_part}Left_{moving_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{pivot_part}Right_{moving_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if not combined.empty:
            arr = combined.to_numpy()
            if use_tail_means:
                min_val, max_val = _safe_tail_means_from_array(
                    arr, pmin=5, pmax=95, min_tail_frames=min_tail_frames
                )
            else:
                min_val, max_val = _safe_percentile_thresholds_from_array(
                    arr, pmin=5, pmax=95
                )

    return min_val, max_val

def compute_head_movement_features_minmax(
    bout_features,
    kinematic_suffix,
    percentile_min=5,
    percentile_max=95,
    *,
    use_tail_means: bool = False,
    min_tail_frames: int = 3,
):
    """Compute head min/max features.

    By default this returns BB percentile thresholds (p05/p95). When
    ``use_tail_means=True`` it returns TW-style tail means.

    Jerk is optional-safe: if jerk columns are missing, jerk min/max return NaN.
    """
    head_keypoints = HEAD_KEYPOINTS
    likelihood_cols = [f'{kp}_likelihood' for kp in head_keypoints]
    velocity_cols = [f'partvelocity_{kp}{kinematic_suffix}' for kp in head_keypoints]
    accel_cols = [f'partaccel_{kp}{kinematic_suffix}' for kp in head_keypoints]
    jerk_cols = [f'partjerk_{kp}{kinematic_suffix}' for kp in head_keypoints]

    if not all(col in bout_features.columns for col in likelihood_cols + velocity_cols + accel_cols):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    likelihoods = bout_features[likelihood_cols].to_numpy()
    velocities = bout_features[velocity_cols].to_numpy()
    accels = bout_features[accel_cols].to_numpy()

    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy()
    else:
        jerks = np.full_like(velocities, np.nan)

    all_nan_rows = np.all(np.isnan(likelihoods), axis=1)
    min_indices = np.full(likelihoods.shape[0], -1, dtype=int)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        min_indices[valid_rows] = np.nanargmin(likelihoods[valid_rows], axis=1)

    rows = np.arange(len(min_indices))
    if np.any(valid_rows):
        velocities[rows[valid_rows], min_indices[valid_rows]] = np.nan
        accels[rows[valid_rows], min_indices[valid_rows]] = np.nan
        jerks[rows[valid_rows], min_indices[valid_rows]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frame_velocities = np.nanmean(velocities, axis=1)
        frame_accels = np.nanmean(accels, axis=1)
        frame_jerks = np.nanmean(jerks, axis=1)

    vel_clean = frame_velocities[~np.isnan(frame_velocities)]
    if len(vel_clean) >= 1:
        if use_tail_means:
            head_velocity_min, head_velocity_max = _safe_tail_means_from_array(
                frame_velocities,
                pmin=percentile_min,
                pmax=percentile_max,
                min_tail_frames=min_tail_frames,
            )
        else:
            head_velocity_min, head_velocity_max = _safe_percentile_thresholds_from_array(
                frame_velocities, pmin=percentile_min, pmax=percentile_max
            )
    else:
        head_velocity_min, head_velocity_max = np.nan, np.nan

    acc_clean = frame_accels[~np.isnan(frame_accels)]
    if len(acc_clean) >= 1:
        if use_tail_means:
            head_acceleration_min, head_acceleration_max = _safe_tail_means_from_array(
                frame_accels,
                pmin=percentile_min,
                pmax=percentile_max,
                min_tail_frames=min_tail_frames,
            )
        else:
            head_acceleration_min, head_acceleration_max = _safe_percentile_thresholds_from_array(
                frame_accels, pmin=percentile_min, pmax=percentile_max
            )
    else:
        head_acceleration_min, head_acceleration_max = np.nan, np.nan

    jerk_clean = frame_jerks[~np.isnan(frame_jerks)]
    if len(jerk_clean) >= 1:
        if use_tail_means:
            head_jerk_min, head_jerk_max = _safe_tail_means_from_array(
                frame_jerks,
                pmin=percentile_min,
                pmax=percentile_max,
                min_tail_frames=min_tail_frames,
            )
        else:
            head_jerk_min, head_jerk_max = _safe_percentile_thresholds_from_array(
                frame_jerks, pmin=percentile_min, pmax=percentile_max
            )
    else:
        head_jerk_min, head_jerk_max = np.nan, np.nan

    return (
        head_velocity_min, head_velocity_max,
        head_acceleration_min, head_acceleration_max,
        head_jerk_min, head_jerk_max
    )

def extract_bout_features_simple(aligned_poscalc_df, bouts, individual_id, group_label, sex_label, source_file, category, kinematic_suffix='_inst', trim_frames_core=BOUT_TRIM_FRAMES_CORE, trim_frames_start_end=None):
    """
    Extracts mean and std for kinematic features, but only if they are physically
    valid for the given orientation of the bout.
    
    Walking bouts: Compute core (trimmed), start, and end features.
    - Core features are computed on a trimmed window; bouts shorter than
      3 * trim_frames_core are skipped when trimming is enabled.
    - Start/End features use the first/last N frames; bouts shorter than
      2 * trim_frames_start_end will have start/end features set to NaN.
    
    Non-walking bouts: compute features on the full bout without trimming.
    """
    def _safe_mean(df, col): return df[col].mean() if col in df.columns else np.nan
    def _safe_std(df, col):  return df[col].std()  if col in df.columns else np.nan
    def _safe_percentile(df, col, percentiles=[5, 95]):
        """Return percentile-threshold min/max values for a bout signal."""
        if col not in df.columns:
            return np.nan, np.nan
        values = df[col].dropna()
        if len(values) == 0:
            return np.nan, np.nan
        # NOTE: min/max are only computed when the caller has already enforced
        # the minimum qualifying bout length (bout_length >= 20).
        return _safe_percentile_thresholds_from_array(
            values.to_numpy(), pmin=percentiles[0], pmax=percentiles[1]
        )
    def _safe_median(df, col):
        """Returns median if column exists, else np.nan"""
        if col not in df.columns:
            return np.nan
        return df[col].median()

    # Default start/end window to core trim if not specified
    if trim_frames_start_end is None:
        trim_frames_start_end = trim_frames_core

    bout_feature_rows = []
    skipped_core_bouts = 0
    skipped_start_end_bouts = 0
    total_bouts = len(bouts)

    walking_bouts_count = 0
    non_walking_bouts_count = 0

    for bout in bouts:
        start, end, label = bout['start'], bout['end'], bout['label']
        bout_type = bout.get('bout_type', label)
        bout_length = end - start + 1

        # Detect if this is a walking bout
        is_walking = "Walking" in bout_type

        # Determine window based on walking status
        if is_walking:
            # Walking: use trimmed core window + start/end slices
            if trim_frames_core > 0:
                if bout_length < 3 * trim_frames_core:
                    skipped_core_bouts += 1
                    continue  # Skip bouts that are too short to retain a core segment
                start_i = start + trim_frames_core
                end_i = end - trim_frames_core
            else:
                start_i = start
                end_i = end

            bout_features = aligned_poscalc_df.iloc[start_i:end_i+1]

            # Start/End windows for walking
            has_start_end = bout_length >= 2 * trim_frames_start_end
            if has_start_end and trim_frames_start_end > 0:
                start_slice = aligned_poscalc_df.iloc[start : start + trim_frames_start_end]
                end_slice   = aligned_poscalc_df.iloc[end - trim_frames_start_end + 1 : end + 1]
            else:
                start_slice = None
                end_slice = None
                if trim_frames_start_end > 0:
                    skipped_start_end_bouts += 1
            walking_bouts_count += 1
        else:
            # Non-walking: use full bout, no trimming
            start_i = start
            end_i = end
            bout_features = aligned_poscalc_df.iloc[start_i:end_i+1]
            start_slice = None
            end_slice = None
            non_walking_bouts_count += 1

        # Full bout window for min/max (no trimming, used for all bout types)
        bout_features_full = aligned_poscalc_df.iloc[start:end+1]
        has_min_max = bout_length >= 20  # Guardrail: require at least 20 frames for BB min/max

        # 1. Initialize all kinematic variables to NaN
        # Original variables (for non-walking bouts)
        speed_mean, speed_std = np.nan, np.nan
        accel_mag_mean, accel_mag_std = np.nan, np.nan
        velocity_fwd_mean, velocity_fwd_std = np.nan, np.nan
        velocity_str_mean, velocity_str_std = np.nan, np.nan
        accel_fwd_mean, accel_fwd_std = np.nan, np.nan
        accel_str_mean, accel_str_std = np.nan, np.nan
        jerk_mag_mean, jerk_mag_std = np.nan, np.nan
        jerk_fwd_mean, jerk_fwd_std = np.nan, np.nan
        jerk_str_mean, jerk_str_std = np.nan, np.nan
        angular_vel_mean, angular_vel_std = np.nan, np.nan
        angular_accel_mean, angular_accel_std = np.nan, np.nan
        angular_jerk_mean, angular_jerk_std = np.nan, np.nan
        frontleg_velocity_mean, frontleg_velocity_std, frontleg_velocity_asym = np.nan, np.nan, np.nan
        frontleg_acceleration_mean, frontleg_acceleration_std, frontleg_acceleration_asym = np.nan, np.nan, np.nan
        frontleg_jerk_mean, frontleg_jerk_std, frontleg_jerk_asym = np.nan, np.nan, np.nan
        frontleg_angvel_mean, frontleg_angvel_std, frontleg_angvel_asym = np.nan, np.nan, np.nan
        frontleg_angaccel_mean, frontleg_angaccel_std, frontleg_angaccel_asym = np.nan, np.nan, np.nan
        frontleg_angjerk_mean, frontleg_angjerk_std, frontleg_angjerk_asym = np.nan, np.nan, np.nan
        midleg_velocity_mean, midleg_velocity_std, midleg_velocity_asym = np.nan, np.nan, np.nan
        midleg_acceleration_mean, midleg_acceleration_std, midleg_acceleration_asym = np.nan, np.nan, np.nan
        midleg_jerk_mean, midleg_jerk_std, midleg_jerk_asym = np.nan, np.nan, np.nan
        midleg_angvel_mean, midleg_angvel_std, midleg_angvel_asym = np.nan, np.nan, np.nan
        midleg_angaccel_mean, midleg_angaccel_std, midleg_angaccel_asym = np.nan, np.nan, np.nan
        midleg_angjerk_mean, midleg_angjerk_std, midleg_angjerk_asym = np.nan, np.nan, np.nan
        backleg_velocity_mean, backleg_velocity_std, backleg_velocity_asym = np.nan, np.nan, np.nan
        backleg_acceleration_mean, backleg_acceleration_std, backleg_acceleration_asym = np.nan, np.nan, np.nan
        backleg_jerk_mean, backleg_jerk_std, backleg_jerk_asym = np.nan, np.nan, np.nan
        backleg_angvel_mean, backleg_angvel_std, backleg_angvel_asym = np.nan, np.nan, np.nan
        backleg_angaccel_mean, backleg_angaccel_std, backleg_angaccel_asym = np.nan, np.nan, np.nan
        backleg_angjerk_mean, backleg_angjerk_std, backleg_angjerk_asym = np.nan, np.nan, np.nan
        head_velocity_mean, head_velocity_std = np.nan, np.nan
        head_acceleration_mean, head_acceleration_std = np.nan, np.nan
        head_jerk_mean, head_jerk_std = np.nan, np.nan


        # Core/start/end variables (for walking bouts)
        speed_core_mean, speed_core_std = np.nan, np.nan
        accel_mag_core_mean, accel_mag_core_std = np.nan, np.nan
        velocity_fwd_core_mean, velocity_fwd_core_std = np.nan, np.nan
        velocity_str_core_mean, velocity_str_core_std = np.nan, np.nan
        accel_fwd_core_mean, accel_fwd_core_std = np.nan, np.nan
        accel_str_core_mean, accel_str_core_std = np.nan, np.nan
        jerk_mag_core_mean, jerk_mag_core_std = np.nan, np.nan
        jerk_fwd_core_mean, jerk_fwd_core_std = np.nan, np.nan
        jerk_str_core_mean, jerk_str_core_std = np.nan, np.nan
        angular_vel_core_mean, angular_vel_core_std = np.nan, np.nan
        angular_accel_core_mean, angular_accel_core_std = np.nan, np.nan
        angular_jerk_core_mean, angular_jerk_core_std = np.nan, np.nan
        frontleg_velocity_core_mean, frontleg_velocity_core_std, frontleg_velocity_core_asym = np.nan, np.nan, np.nan
        frontleg_acceleration_core_mean, frontleg_acceleration_core_std, frontleg_acceleration_core_asym = np.nan, np.nan, np.nan
        frontleg_jerk_core_mean, frontleg_jerk_core_std, frontleg_jerk_core_asym = np.nan, np.nan, np.nan
        frontleg_angvel_core_mean, frontleg_angvel_core_std, frontleg_angvel_core_asym = np.nan, np.nan, np.nan
        frontleg_angaccel_core_mean, frontleg_angaccel_core_std, frontleg_angaccel_core_asym = np.nan, np.nan, np.nan
        frontleg_angjerk_core_mean, frontleg_angjerk_core_std, frontleg_angjerk_core_asym = np.nan, np.nan, np.nan
        midleg_velocity_core_mean, midleg_velocity_core_std, midleg_velocity_core_asym = np.nan, np.nan, np.nan
        midleg_acceleration_core_mean, midleg_acceleration_core_std, midleg_acceleration_core_asym = np.nan, np.nan, np.nan
        midleg_jerk_core_mean, midleg_jerk_core_std, midleg_jerk_core_asym = np.nan, np.nan, np.nan
        midleg_angvel_core_mean, midleg_angvel_core_std, midleg_angvel_core_asym = np.nan, np.nan, np.nan
        midleg_angaccel_core_mean, midleg_angaccel_core_std, midleg_angaccel_core_asym = np.nan, np.nan, np.nan
        midleg_angjerk_core_mean, midleg_angjerk_core_std, midleg_angjerk_core_asym = np.nan, np.nan, np.nan
        backleg_velocity_core_mean, backleg_velocity_core_std, backleg_velocity_core_asym = np.nan, np.nan, np.nan
        backleg_acceleration_core_mean, backleg_acceleration_core_std, backleg_acceleration_core_asym = np.nan, np.nan, np.nan
        backleg_jerk_core_mean, backleg_jerk_core_std, backleg_jerk_core_asym = np.nan, np.nan, np.nan
        backleg_angvel_core_mean, backleg_angvel_core_std, backleg_angvel_core_asym = np.nan, np.nan, np.nan
        backleg_angaccel_core_mean, backleg_angaccel_core_std, backleg_angaccel_core_asym = np.nan, np.nan, np.nan
        backleg_angjerk_core_mean, backleg_angjerk_core_std, backleg_angjerk_core_asym = np.nan, np.nan, np.nan
        head_velocity_core_mean, head_velocity_core_std = np.nan, np.nan
        head_acceleration_core_mean, head_acceleration_core_std = np.nan, np.nan
        head_jerk_core_mean, head_jerk_core_std = np.nan, np.nan


        # Initialize start/end kinematic variables
        speed_start_mean, speed_start_std = np.nan, np.nan
        speed_end_mean, speed_end_std = np.nan, np.nan
        accel_mag_start_mean, accel_mag_start_std = np.nan, np.nan
        accel_mag_end_mean, accel_mag_end_std = np.nan, np.nan
        velocity_fwd_start_mean, velocity_fwd_start_std = np.nan, np.nan
        velocity_fwd_end_mean, velocity_fwd_end_std = np.nan, np.nan
        velocity_str_start_mean, velocity_str_start_std = np.nan, np.nan
        velocity_str_end_mean, velocity_str_end_std = np.nan, np.nan
        accel_fwd_start_mean, accel_fwd_start_std = np.nan, np.nan
        accel_fwd_end_mean, accel_fwd_end_std = np.nan, np.nan
        accel_str_start_mean, accel_str_start_std = np.nan, np.nan
        accel_str_end_mean, accel_str_end_std = np.nan, np.nan
        jerk_mag_start_mean, jerk_mag_start_std = np.nan, np.nan
        jerk_mag_end_mean, jerk_mag_end_std = np.nan, np.nan
        jerk_fwd_start_mean, jerk_fwd_start_std = np.nan, np.nan
        jerk_fwd_end_mean, jerk_fwd_end_std = np.nan, np.nan
        jerk_str_start_mean, jerk_str_start_std = np.nan, np.nan
        jerk_str_end_mean, jerk_str_end_std = np.nan, np.nan
        angular_vel_start_mean, angular_vel_start_std = np.nan, np.nan
        angular_vel_end_mean, angular_vel_end_std = np.nan, np.nan
        angular_accel_start_mean, angular_accel_start_std = np.nan, np.nan
        angular_accel_end_mean, angular_accel_end_std = np.nan, np.nan
        angular_jerk_start_mean, angular_jerk_start_std = np.nan, np.nan
        angular_jerk_end_mean, angular_jerk_end_std = np.nan, np.nan

        # Legs start/end
        frontleg_velocity_start_mean, frontleg_velocity_start_std, frontleg_velocity_start_asym = np.nan, np.nan, np.nan
        frontleg_velocity_end_mean, frontleg_velocity_end_std, frontleg_velocity_end_asym = np.nan, np.nan, np.nan
        frontleg_acceleration_start_mean, frontleg_acceleration_start_std, frontleg_acceleration_start_asym = np.nan, np.nan, np.nan
        frontleg_acceleration_end_mean, frontleg_acceleration_end_std, frontleg_acceleration_end_asym = np.nan, np.nan, np.nan
        frontleg_jerk_start_mean, frontleg_jerk_start_std, frontleg_jerk_start_asym = np.nan, np.nan, np.nan
        frontleg_jerk_end_mean, frontleg_jerk_end_std, frontleg_jerk_end_asym = np.nan, np.nan, np.nan
        frontleg_angvel_start_mean, frontleg_angvel_start_std, frontleg_angvel_start_asym = np.nan, np.nan, np.nan
        frontleg_angvel_end_mean, frontleg_angvel_end_std, frontleg_angvel_end_asym = np.nan, np.nan, np.nan
        frontleg_angaccel_start_mean, frontleg_angaccel_start_std, frontleg_angaccel_start_asym = np.nan, np.nan, np.nan
        frontleg_angaccel_end_mean, frontleg_angaccel_end_std, frontleg_angaccel_end_asym = np.nan, np.nan, np.nan
        frontleg_angjerk_start_mean, frontleg_angjerk_start_std, frontleg_angjerk_start_asym = np.nan, np.nan, np.nan
        frontleg_angjerk_end_mean, frontleg_angjerk_end_std, frontleg_angjerk_end_asym = np.nan, np.nan, np.nan

        midleg_velocity_start_mean, midleg_velocity_start_std, midleg_velocity_start_asym = np.nan, np.nan, np.nan
        midleg_velocity_end_mean, midleg_velocity_end_std, midleg_velocity_end_asym = np.nan, np.nan, np.nan
        midleg_acceleration_start_mean, midleg_acceleration_start_std, midleg_acceleration_start_asym = np.nan, np.nan, np.nan
        midleg_acceleration_end_mean, midleg_acceleration_end_std, midleg_acceleration_end_asym = np.nan, np.nan, np.nan
        midleg_jerk_start_mean, midleg_jerk_start_std, midleg_jerk_start_asym = np.nan, np.nan, np.nan
        midleg_jerk_end_mean, midleg_jerk_end_std, midleg_jerk_end_asym = np.nan, np.nan, np.nan
        midleg_angvel_start_mean, midleg_angvel_start_std, midleg_angvel_start_asym = np.nan, np.nan, np.nan
        midleg_angvel_end_mean, midleg_angvel_end_std, midleg_angvel_end_asym = np.nan, np.nan, np.nan
        midleg_angaccel_start_mean, midleg_angaccel_start_std, midleg_angaccel_start_asym = np.nan, np.nan, np.nan
        midleg_angaccel_end_mean, midleg_angaccel_end_std, midleg_angaccel_end_asym = np.nan, np.nan, np.nan
        midleg_angjerk_start_mean, midleg_angjerk_start_std, midleg_angjerk_start_asym = np.nan, np.nan, np.nan
        midleg_angjerk_end_mean, midleg_angjerk_end_std, midleg_angjerk_end_asym = np.nan, np.nan, np.nan

        backleg_velocity_start_mean, backleg_velocity_start_std, backleg_velocity_start_asym = np.nan, np.nan, np.nan
        backleg_velocity_end_mean, backleg_velocity_end_std, backleg_velocity_end_asym = np.nan, np.nan, np.nan
        backleg_acceleration_start_mean, backleg_acceleration_start_std, backleg_acceleration_start_asym = np.nan, np.nan, np.nan
        backleg_acceleration_end_mean, backleg_acceleration_end_std, backleg_acceleration_end_asym = np.nan, np.nan, np.nan
        backleg_jerk_start_mean, backleg_jerk_start_std, backleg_jerk_start_asym = np.nan, np.nan, np.nan
        backleg_jerk_end_mean, backleg_jerk_end_std, backleg_jerk_end_asym = np.nan, np.nan, np.nan
        backleg_angvel_start_mean, backleg_angvel_start_std, backleg_angvel_start_asym = np.nan, np.nan, np.nan
        backleg_angvel_end_mean, backleg_angvel_end_std, backleg_angvel_end_asym = np.nan, np.nan, np.nan
        backleg_angaccel_start_mean, backleg_angaccel_start_std, backleg_angaccel_start_asym = np.nan, np.nan, np.nan
        backleg_angaccel_end_mean, backleg_angaccel_end_std, backleg_angaccel_end_asym = np.nan, np.nan, np.nan
        backleg_angjerk_start_mean, backleg_angjerk_start_std, backleg_angjerk_start_asym = np.nan, np.nan, np.nan
        backleg_angjerk_end_mean, backleg_angjerk_end_std, backleg_angjerk_end_asym = np.nan, np.nan, np.nan

        # Head start/end
        head_velocity_start_mean, head_velocity_start_std = np.nan, np.nan
        head_velocity_end_mean, head_velocity_end_std = np.nan, np.nan
        head_acceleration_start_mean, head_acceleration_start_std = np.nan, np.nan
        head_acceleration_end_mean, head_acceleration_end_std = np.nan, np.nan
        head_jerk_start_mean, head_jerk_start_std = np.nan, np.nan
        head_jerk_end_mean, head_jerk_end_std = np.nan, np.nan


        # Min/Max variables (tail means of bottom/top 5%, full bout, all bout types)
        speed_min, speed_max = np.nan, np.nan
        accel_mag_min, accel_mag_max = np.nan, np.nan
        velocity_fwd_min, velocity_fwd_max = np.nan, np.nan
        velocity_str_min, velocity_str_max = np.nan, np.nan
        accel_fwd_min, accel_fwd_max = np.nan, np.nan
        accel_str_min, accel_str_max = np.nan, np.nan
        jerk_mag_min, jerk_mag_max = np.nan, np.nan
        jerk_fwd_min, jerk_fwd_max = np.nan, np.nan
        jerk_str_min, jerk_str_max = np.nan, np.nan
        angular_vel_min, angular_vel_max = np.nan, np.nan
        angular_accel_min, angular_accel_max = np.nan, np.nan
        angular_jerk_min, angular_jerk_max = np.nan, np.nan
        frontleg_velocity_min, frontleg_velocity_max = np.nan, np.nan
        frontleg_acceleration_min, frontleg_acceleration_max = np.nan, np.nan
        frontleg_jerk_min, frontleg_jerk_max = np.nan, np.nan
        frontleg_angvel_min, frontleg_angvel_max = np.nan, np.nan
        frontleg_angaccel_min, frontleg_angaccel_max = np.nan, np.nan
        frontleg_angjerk_min, frontleg_angjerk_max = np.nan, np.nan
        midleg_velocity_min, midleg_velocity_max = np.nan, np.nan
        midleg_acceleration_min, midleg_acceleration_max = np.nan, np.nan
        midleg_jerk_min, midleg_jerk_max = np.nan, np.nan
        midleg_angvel_min, midleg_angvel_max = np.nan, np.nan
        midleg_angaccel_min, midleg_angaccel_max = np.nan, np.nan
        midleg_angjerk_min, midleg_angjerk_max = np.nan, np.nan
        backleg_velocity_min, backleg_velocity_max = np.nan, np.nan
        backleg_acceleration_min, backleg_acceleration_max = np.nan, np.nan
        backleg_jerk_min, backleg_jerk_max = np.nan, np.nan
        backleg_angvel_min, backleg_angvel_max = np.nan, np.nan
        backleg_angaccel_min, backleg_angaccel_max = np.nan, np.nan
        backleg_angjerk_min, backleg_angjerk_max = np.nan, np.nan
        head_velocity_min, head_velocity_max = np.nan, np.nan
        head_acceleration_min, head_acceleration_max = np.nan, np.nan
        head_jerk_min, head_jerk_max = np.nan, np.nan


        # Median and Skewness variables (all bout types)
        speed_median, speed_skewness = np.nan, np.nan
        accel_mag_median, accel_mag_skewness = np.nan, np.nan
        velocity_fwd_median, velocity_fwd_skewness = np.nan, np.nan
        velocity_str_median, velocity_str_skewness = np.nan, np.nan
        accel_fwd_median, accel_fwd_skewness = np.nan, np.nan
        accel_str_median, accel_str_skewness = np.nan, np.nan
        jerk_mag_median, jerk_mag_skewness = np.nan, np.nan
        jerk_fwd_median, jerk_fwd_skewness = np.nan, np.nan
        jerk_str_median, jerk_str_skewness = np.nan, np.nan
        angular_vel_median, angular_vel_skewness = np.nan, np.nan
        angular_accel_median, angular_accel_skewness = np.nan, np.nan
        angular_jerk_median, angular_jerk_skewness = np.nan, np.nan
        frontleg_velocity_median, frontleg_velocity_skewness = np.nan, np.nan
        frontleg_acceleration_median, frontleg_acceleration_skewness = np.nan, np.nan
        frontleg_jerk_median, frontleg_jerk_skewness = np.nan, np.nan
        frontleg_angvel_median, frontleg_angvel_skewness = np.nan, np.nan
        frontleg_angaccel_median, frontleg_angaccel_skewness = np.nan, np.nan
        frontleg_angjerk_median, frontleg_angjerk_skewness = np.nan, np.nan
        midleg_velocity_median, midleg_velocity_skewness = np.nan, np.nan
        midleg_acceleration_median, midleg_acceleration_skewness = np.nan, np.nan
        midleg_jerk_median, midleg_jerk_skewness = np.nan, np.nan
        midleg_angvel_median, midleg_angvel_skewness = np.nan, np.nan
        midleg_angaccel_median, midleg_angaccel_skewness = np.nan, np.nan
        midleg_angjerk_median, midleg_angjerk_skewness = np.nan, np.nan
        backleg_velocity_median, backleg_velocity_skewness = np.nan, np.nan
        backleg_acceleration_median, backleg_acceleration_skewness = np.nan, np.nan
        backleg_jerk_median, backleg_jerk_skewness = np.nan, np.nan
        backleg_angvel_median, backleg_angvel_skewness = np.nan, np.nan
        backleg_angaccel_median, backleg_angaccel_skewness = np.nan, np.nan
        backleg_angjerk_median, backleg_angjerk_skewness = np.nan, np.nan
        head_velocity_median, head_velocity_skewness = np.nan, np.nan
        head_acceleration_median, head_acceleration_skewness = np.nan, np.nan
        head_jerk_median, head_jerk_skewness = np.nan, np.nan


        # Curvature variables (walking bouts only)
        curvature_mean, curvature_median, curvature_std, curvature_skewness = np.nan, np.nan, np.nan, np.nan
        curvature_min, curvature_max = np.nan, np.nan

        # 2. Check orientation to determine which features are valid
        is_wall_up = "Wall-Up" in bout_type
        is_wall_down = "Wall-Down" in bout_type
        is_special_case = bout_type.startswith('Scramble') or bout_type.startswith('Jump')

        # 3. Conditionally calculate valid features based on walking status
        if is_walking:
            # Walking bouts: compute core + start + end features
            # Body movement is invalid for Wall-Up/Down due to Z-axis motion
            if not is_wall_up and not is_wall_down:
                speed_core_mean = _safe_mean(bout_features, f'speed{kinematic_suffix}')
                speed_core_std = _safe_std(bout_features, f'speed{kinematic_suffix}')
            velocity_fwd_core_mean = _safe_mean(bout_features, f'velocity_forward{kinematic_suffix}')
            velocity_fwd_core_std = _safe_std(bout_features, f'velocity_forward{kinematic_suffix}')
            velocity_str_core_mean = _safe_mean(bout_features, f'velocity_strafe{kinematic_suffix}')
            velocity_str_core_std = _safe_std(bout_features, f'velocity_strafe{kinematic_suffix}')
            accel_mag_core_mean = _safe_mean(bout_features, f'acceleration_magnitude{kinematic_suffix}')
            accel_mag_core_std = _safe_std(bout_features, f'acceleration_magnitude{kinematic_suffix}')
            accel_fwd_core_mean = _safe_mean(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_fwd_core_std = _safe_std(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_str_core_mean = _safe_mean(bout_features, f'acceleration_strafe{kinematic_suffix}')
            accel_str_core_std = _safe_std(bout_features, f'acceleration_strafe{kinematic_suffix}')
            jerk_mag_core_mean = _safe_mean(bout_features, f'jerk_magnitude{kinematic_suffix}')
            jerk_mag_core_std = _safe_std(bout_features, f'jerk_magnitude{kinematic_suffix}')
            jerk_fwd_core_mean = _safe_mean(bout_features, f'jerk_forward{kinematic_suffix}')
            jerk_fwd_core_std = _safe_std(bout_features, f'jerk_forward{kinematic_suffix}')
            jerk_str_core_mean = _safe_mean(bout_features, f'jerk_strafe{kinematic_suffix}')
            jerk_str_core_std = _safe_std(bout_features, f'jerk_strafe{kinematic_suffix}')

            if "Floor" in bout_type or "Ceiling" in bout_type:
                angular_vel_core_mean = _safe_mean(bout_features, f'angular_velocity{kinematic_suffix}')
                angular_vel_core_std = _safe_std(bout_features, f'angular_velocity{kinematic_suffix}')
                angular_accel_core_mean = _safe_mean(bout_features, f'angular_acceleration{kinematic_suffix}')
                angular_accel_core_std = _safe_std(bout_features, f'angular_acceleration{kinematic_suffix}')
                angular_jerk_core_mean = _safe_mean(bout_features, f'angular_jerk{kinematic_suffix}')
                angular_jerk_core_std = _safe_std(bout_features, f'angular_jerk{kinematic_suffix}')

            # Front Legs & Head
            if not is_wall_down and not is_special_case:
                (frontleg_velocity_core_mean, frontleg_velocity_core_std, frontleg_velocity_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                (frontleg_acceleration_core_mean, frontleg_acceleration_core_std, frontleg_acceleration_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus')
                (frontleg_jerk_core_mean, frontleg_jerk_core_std, frontleg_jerk_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus')
                (frontleg_angvel_core_mean, frontleg_angvel_core_std, frontleg_angvel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                (frontleg_angaccel_core_mean, frontleg_angaccel_core_std, frontleg_angaccel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                (frontleg_angjerk_core_mean, frontleg_angjerk_core_std, frontleg_angjerk_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                
                head_velocity_core_mean, head_velocity_core_std, head_acceleration_core_mean, head_acceleration_core_std, head_jerk_core_mean, head_jerk_core_std = compute_head_movement_features(bout_features, kinematic_suffix)

            # Middle Legs
            if not is_wall_up and not is_wall_down and not is_special_case:
                (midleg_velocity_core_mean, midleg_velocity_core_std, midleg_velocity_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus')
                (midleg_acceleration_core_mean, midleg_acceleration_core_std, midleg_acceleration_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'MidTarsus')
                (midleg_jerk_core_mean, midleg_jerk_core_std, midleg_jerk_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'MidTarsus')
                (midleg_angvel_core_mean, midleg_angvel_core_std, midleg_angvel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                (midleg_angaccel_core_mean, midleg_angaccel_core_std, midleg_angaccel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                (midleg_angjerk_core_mean, midleg_angjerk_core_std, midleg_angjerk_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

            # Back Legs
            if not is_wall_up and not is_special_case:
                (backleg_velocity_core_mean, backleg_velocity_core_std, backleg_velocity_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus')
                (backleg_acceleration_core_mean, backleg_acceleration_core_std, backleg_acceleration_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'BackTarsus')
                (backleg_jerk_core_mean, backleg_jerk_core_std, backleg_jerk_core_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'BackTarsus')
                (backleg_angvel_core_mean, backleg_angvel_core_std, backleg_angvel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                (backleg_angaccel_core_mean, backleg_angaccel_core_std, backleg_angaccel_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                (backleg_angjerk_core_mean, backleg_angjerk_core_std, backleg_angjerk_core_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')           

                # Compute curvature features (walking only, using core window)
                curvature_mean = _safe_mean(bout_features, 'curvature_inst')
                curvature_median = _safe_median(bout_features, 'curvature_inst')
                curvature_std = _safe_std(bout_features, 'curvature_inst')
                curvature_skewness = _safe_skewness(curvature_mean, curvature_median, curvature_std)
                
                # Min/Max (tail means of bottom/top 5%) - same threshold as other min/max features
                if bout_length >= 20:
                    curvature_min, curvature_max = _safe_percentile(bout_features, 'curvature_inst')

                # Compute start/end features if slices are available
                if has_start_end and trim_frames_start_end > 0:
                    # Body (start/end)
                    if not is_wall_up and not is_wall_down:
                        speed_start_mean = _safe_mean(start_slice, f'speed{kinematic_suffix}')
                        speed_start_std  = _safe_std(start_slice, f'speed{kinematic_suffix}')
                        speed_end_mean   = _safe_mean(end_slice, f'speed{kinematic_suffix}')
                        speed_end_std    = _safe_std(end_slice, f'speed{kinematic_suffix}')

                        velocity_fwd_start_mean = _safe_mean(start_slice, f'velocity_forward{kinematic_suffix}')
                        velocity_fwd_start_std  = _safe_std(start_slice, f'velocity_forward{kinematic_suffix}')
                        velocity_fwd_end_mean   = _safe_mean(end_slice, f'velocity_forward{kinematic_suffix}')
                        velocity_fwd_end_std    = _safe_std(end_slice, f'velocity_forward{kinematic_suffix}')

                        velocity_str_start_mean = _safe_mean(start_slice, f'velocity_strafe{kinematic_suffix}')
                        velocity_str_start_std  = _safe_std(start_slice, f'velocity_strafe{kinematic_suffix}')
                        velocity_str_end_mean   = _safe_mean(end_slice, f'velocity_strafe{kinematic_suffix}')
                        velocity_str_end_std    = _safe_std(end_slice, f'velocity_strafe{kinematic_suffix}')

                        accel_mag_start_mean = _safe_mean(start_slice, f'acceleration_magnitude{kinematic_suffix}')
                        accel_mag_start_std  = _safe_std(start_slice, f'acceleration_magnitude{kinematic_suffix}')
                        accel_mag_end_mean   = _safe_mean(end_slice, f'acceleration_magnitude{kinematic_suffix}')
                        accel_mag_end_std    = _safe_std(end_slice, f'acceleration_magnitude{kinematic_suffix}')

                        accel_fwd_start_mean = _safe_mean(start_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_start_std  = _safe_std(start_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_end_mean   = _safe_mean(end_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_end_std    = _safe_std(end_slice, f'acceleration_forward{kinematic_suffix}')

                        accel_str_start_mean = _safe_mean(start_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_start_std  = _safe_std(start_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_end_mean   = _safe_mean(end_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_end_std    = _safe_std(end_slice, f'acceleration_strafe{kinematic_suffix}')

                        jerk_mag_start_mean = _safe_mean(start_slice, f'jerk_magnitude{kinematic_suffix}')
                        jerk_mag_start_std  = _safe_std(start_slice, f'jerk_magnitude{kinematic_suffix}')
                        jerk_mag_end_mean   = _safe_mean(end_slice, f'jerk_magnitude{kinematic_suffix}')
                        jerk_mag_end_std    = _safe_std(end_slice, f'jerk_magnitude{kinematic_suffix}')

                        jerk_fwd_start_mean = _safe_mean(start_slice, f'jerk_forward{kinematic_suffix}')
                        jerk_fwd_start_std  = _safe_std(start_slice, f'jerk_forward{kinematic_suffix}')
                        jerk_fwd_end_mean   = _safe_mean(end_slice, f'jerk_forward{kinematic_suffix}')
                        jerk_fwd_end_std    = _safe_std(end_slice, f'jerk_forward{kinematic_suffix}')

                        jerk_str_start_mean = _safe_mean(start_slice, f'jerk_strafe{kinematic_suffix}')
                        jerk_str_start_std  = _safe_std(start_slice, f'jerk_strafe{kinematic_suffix}')
                        jerk_str_end_mean   = _safe_mean(end_slice, f'jerk_strafe{kinematic_suffix}')
                        jerk_str_end_std    = _safe_std(end_slice, f'jerk_strafe{kinematic_suffix}')

                    # Angular (start/end)
                    if "Floor" in bout_type or "Ceiling" in bout_type:
                        angular_vel_start_mean = _safe_mean(start_slice, f'angular_velocity{kinematic_suffix}')
                        angular_vel_start_std  = _safe_std(start_slice, f'angular_velocity{kinematic_suffix}')
                        angular_vel_end_mean   = _safe_mean(end_slice, f'angular_velocity{kinematic_suffix}')
                        angular_vel_end_std    = _safe_std(end_slice, f'angular_velocity{kinematic_suffix}')

                        angular_accel_start_mean = _safe_mean(start_slice, f'angular_acceleration{kinematic_suffix}')
                        angular_accel_start_std  = _safe_std(start_slice, f'angular_acceleration{kinematic_suffix}')
                        angular_accel_end_mean   = _safe_mean(end_slice, f'angular_acceleration{kinematic_suffix}')
                        angular_accel_end_std    = _safe_std(end_slice, f'angular_acceleration{kinematic_suffix}')

                        angular_jerk_start_mean = _safe_mean(start_slice, f'angular_jerk{kinematic_suffix}')
                        angular_jerk_start_std  = _safe_std(start_slice, f'angular_jerk{kinematic_suffix}')
                        angular_jerk_end_mean   = _safe_mean(end_slice, f'angular_jerk{kinematic_suffix}')
                        angular_jerk_end_std    = _safe_std(end_slice, f'angular_jerk{kinematic_suffix}')

                    # Front legs & head
                    if not is_wall_down and not is_special_case:
                        (frontleg_velocity_start_mean, frontleg_velocity_start_std, frontleg_velocity_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                        (frontleg_velocity_end_mean,   frontleg_velocity_end_std,   frontleg_velocity_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus')

                        (frontleg_acceleration_start_mean, frontleg_acceleration_start_std, frontleg_acceleration_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus')
                        (frontleg_acceleration_end_mean,   frontleg_acceleration_end_std,   frontleg_acceleration_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus')

                        (frontleg_jerk_start_mean, frontleg_jerk_start_std, frontleg_jerk_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus')
                        (frontleg_jerk_end_mean,   frontleg_jerk_end_std,   frontleg_jerk_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus')

                        (frontleg_angvel_start_mean, frontleg_angvel_start_std, frontleg_angvel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angvel_end_mean,   frontleg_angvel_end_std,   frontleg_angvel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')

                        (frontleg_angaccel_start_mean, frontleg_angaccel_start_std, frontleg_angaccel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angaccel_end_mean,   frontleg_angaccel_end_std,   frontleg_angaccel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')

                        (frontleg_angjerk_start_mean, frontleg_angjerk_start_std, frontleg_angjerk_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angjerk_end_mean,   frontleg_angjerk_end_std,   frontleg_angjerk_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')

                        head_velocity_start_mean, head_velocity_start_std, head_acceleration_start_mean, head_acceleration_start_std, head_jerk_start_mean, head_jerk_start_std = compute_head_movement_features(start_slice, kinematic_suffix)
                        head_velocity_end_mean, head_velocity_end_std, head_acceleration_end_mean, head_acceleration_end_std, head_jerk_end_mean, head_jerk_end_std = compute_head_movement_features(end_slice, kinematic_suffix)

                    # Middle legs
                    if not is_wall_up and not is_wall_down and not is_special_case:
                        (midleg_velocity_start_mean, midleg_velocity_start_std, midleg_velocity_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus')
                        (midleg_velocity_end_mean,   midleg_velocity_end_std,   midleg_velocity_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus')

                        (midleg_acceleration_start_mean, midleg_acceleration_start_std, midleg_acceleration_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partaccel', 'MidTarsus')
                        (midleg_acceleration_end_mean,   midleg_acceleration_end_std,   midleg_acceleration_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partaccel', 'MidTarsus')

                        (midleg_jerk_start_mean, midleg_jerk_start_std, midleg_jerk_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partjerk', 'MidTarsus')
                        (midleg_jerk_end_mean,   midleg_jerk_end_std,   midleg_jerk_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partjerk', 'MidTarsus')

                        (midleg_angvel_start_mean, midleg_angvel_start_std, midleg_angvel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                        (midleg_angvel_end_mean,   midleg_angvel_end_std,   midleg_angvel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')

                        (midleg_angaccel_start_mean, midleg_angaccel_start_std, midleg_angaccel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                        (midleg_angaccel_end_mean,   midleg_angaccel_end_std,   midleg_angaccel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')

                        (midleg_angjerk_start_mean, midleg_angjerk_start_std, midleg_angjerk_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')
                        (midleg_angjerk_end_mean,   midleg_angjerk_end_std,   midleg_angjerk_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

                    # Back legs
                    if not is_wall_up and not is_special_case:
                        (backleg_velocity_start_mean, backleg_velocity_start_std, backleg_velocity_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus')
                        (backleg_velocity_end_mean,   backleg_velocity_end_std,   backleg_velocity_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus')

                        (backleg_acceleration_start_mean, backleg_acceleration_start_std, backleg_acceleration_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partaccel', 'BackTarsus')
                        (backleg_acceleration_end_mean,   backleg_acceleration_end_std,   backleg_acceleration_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partaccel', 'BackTarsus')

                        (backleg_jerk_start_mean, backleg_jerk_start_std, backleg_jerk_start_asym) = _calculate_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partjerk', 'BackTarsus')
                        (backleg_jerk_end_mean,   backleg_jerk_end_std,   backleg_jerk_end_asym)   = _calculate_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partjerk', 'BackTarsus')

                        (backleg_angvel_start_mean, backleg_angvel_start_std, backleg_angvel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                        (backleg_angvel_end_mean,   backleg_angvel_end_std,   backleg_angvel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')

                        (backleg_angaccel_start_mean, backleg_angaccel_start_std, backleg_angaccel_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                        (backleg_angaccel_end_mean,   backleg_angaccel_end_std,   backleg_angaccel_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')

                        (backleg_angjerk_start_mean, backleg_angjerk_start_std, backleg_angjerk_start_asym) = _calculate_angular_leg_kinematics(start_slice, bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')
                        (backleg_angjerk_end_mean,   backleg_angjerk_end_std,   backleg_angjerk_end_asym)   = _calculate_angular_leg_kinematics(end_slice,   bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')
        else:
            # Non-walking bouts: compute features on the full bout
            # Body movement is invalid for Wall-Up/Down due to Z-axis motion
            if not is_wall_up and not is_wall_down:
                speed_mean = _safe_mean(bout_features, f'speed{kinematic_suffix}')
                speed_std = _safe_std(bout_features, f'speed{kinematic_suffix}')
                velocity_fwd_mean = _safe_mean(bout_features, f'velocity_forward{kinematic_suffix}')
                velocity_fwd_std = _safe_std(bout_features, f'velocity_forward{kinematic_suffix}')
                velocity_str_mean = _safe_mean(bout_features, f'velocity_strafe{kinematic_suffix}')
                velocity_str_std = _safe_std(bout_features, f'velocity_strafe{kinematic_suffix}')
                accel_mag_mean = _safe_mean(bout_features, f'acceleration_magnitude{kinematic_suffix}')
                accel_mag_std = _safe_std(bout_features, f'acceleration_magnitude{kinematic_suffix}')
                accel_fwd_mean = _safe_mean(bout_features, f'acceleration_forward{kinematic_suffix}')
                accel_fwd_std = _safe_std(bout_features, f'acceleration_forward{kinematic_suffix}')
                accel_str_mean = _safe_mean(bout_features, f'acceleration_strafe{kinematic_suffix}')
                accel_str_std = _safe_std(bout_features, f'acceleration_strafe{kinematic_suffix}')
                jerk_mag_mean = _safe_mean(bout_features, f'jerk_magnitude{kinematic_suffix}')
                jerk_mag_std = _safe_std(bout_features, f'jerk_magnitude{kinematic_suffix}')
                jerk_fwd_mean = _safe_mean(bout_features, f'jerk_forward{kinematic_suffix}')
                jerk_fwd_std = _safe_std(bout_features, f'jerk_forward{kinematic_suffix}')
                jerk_str_mean = _safe_mean(bout_features, f'jerk_strafe{kinematic_suffix}')
                jerk_str_std = _safe_std(bout_features, f'jerk_strafe{kinematic_suffix}')

            if "Floor" in bout_type or "Ceiling" in bout_type:
                angular_vel_mean = _safe_mean(bout_features, f'angular_velocity{kinematic_suffix}')
                angular_vel_std = _safe_std(bout_features, f'angular_velocity{kinematic_suffix}')
                angular_accel_mean = _safe_mean(bout_features, f'angular_acceleration{kinematic_suffix}')
                angular_accel_std = _safe_std(bout_features, f'angular_acceleration{kinematic_suffix}')
                angular_jerk_mean = _safe_mean(bout_features, f'angular_jerk{kinematic_suffix}')
                angular_jerk_std = _safe_std(bout_features, f'angular_jerk{kinematic_suffix}')

            # Front Legs & Head
            if not is_wall_down and not is_special_case:
                (frontleg_velocity_mean, frontleg_velocity_std, frontleg_velocity_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                (frontleg_acceleration_mean, frontleg_acceleration_std, frontleg_acceleration_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus')
                (frontleg_jerk_mean, frontleg_jerk_std, frontleg_jerk_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus')
                (frontleg_angvel_mean, frontleg_angvel_std, frontleg_angvel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                (frontleg_angaccel_mean, frontleg_angaccel_std, frontleg_angaccel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                (frontleg_angjerk_mean, frontleg_angjerk_std, frontleg_angjerk_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                
                head_velocity_mean, head_velocity_std, head_acceleration_mean, head_acceleration_std, head_jerk_mean, head_jerk_std = compute_head_movement_features(bout_features, kinematic_suffix)

            # Middle Legs
            if not is_wall_up and not is_wall_down and not is_special_case:
                (midleg_velocity_mean, midleg_velocity_std, midleg_velocity_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus')
                (midleg_acceleration_mean, midleg_acceleration_std, midleg_acceleration_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'MidTarsus')
                (midleg_jerk_mean, midleg_jerk_std, midleg_jerk_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'MidTarsus')
                (midleg_angvel_mean, midleg_angvel_std, midleg_angvel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                (midleg_angaccel_mean, midleg_angaccel_std, midleg_angaccel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                (midleg_angjerk_mean, midleg_angjerk_std, midleg_angjerk_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

            # Back Legs
            if not is_wall_up and not is_special_case:
                (backleg_velocity_mean, backleg_velocity_std, backleg_velocity_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus')
                (backleg_acceleration_mean, backleg_acceleration_std, backleg_acceleration_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partaccel', 'BackTarsus')
                (backleg_jerk_mean, backleg_jerk_std, backleg_jerk_asym) = _calculate_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partjerk', 'BackTarsus')
                (backleg_angvel_mean, backleg_angvel_std, backleg_angvel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                (backleg_angaccel_mean, backleg_angaccel_std, backleg_angaccel_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                (backleg_angjerk_mean, backleg_angjerk_std, backleg_angjerk_asym) = _calculate_angular_leg_kinematics(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')

        # Compute min/max features (tail means of bottom/top 5%) on full bout for all bout types
        if has_min_max:
            # Body movement (same conditions as mean/std)
            if not is_wall_up and not is_wall_down:
                speed_min, speed_max = _safe_percentile(bout_features_full, f'speed{kinematic_suffix}')
                velocity_fwd_min, velocity_fwd_max = _safe_percentile(bout_features_full, f'velocity_forward{kinematic_suffix}')
                velocity_str_min, velocity_str_max = _safe_percentile(bout_features_full, f'velocity_strafe{kinematic_suffix}')
                accel_mag_min, accel_mag_max = _safe_percentile(bout_features_full, f'acceleration_magnitude{kinematic_suffix}')
                accel_fwd_min, accel_fwd_max = _safe_percentile(bout_features_full, f'acceleration_forward{kinematic_suffix}')
                accel_str_min, accel_str_max = _safe_percentile(bout_features_full, f'acceleration_strafe{kinematic_suffix}')
                jerk_mag_min, jerk_mag_max = _safe_percentile(bout_features_full, f'jerk_magnitude{kinematic_suffix}')
                jerk_fwd_min, jerk_fwd_max = _safe_percentile(bout_features_full, f'jerk_forward{kinematic_suffix}')
                jerk_str_min, jerk_str_max = _safe_percentile(bout_features_full, f'jerk_strafe{kinematic_suffix}')

            # Angular (same conditions as mean/std)
            if "Floor" in bout_type or "Ceiling" in bout_type:
                angular_vel_min, angular_vel_max = _safe_percentile(bout_features_full, f'angular_velocity{kinematic_suffix}')
                angular_accel_min, angular_accel_max = _safe_percentile(bout_features_full, f'angular_acceleration{kinematic_suffix}')
                angular_jerk_min, angular_jerk_max = _safe_percentile(bout_features_full, f'angular_jerk{kinematic_suffix}')

            # Front Legs & Head
            if not is_wall_down and not is_special_case:
                frontleg_velocity_min, frontleg_velocity_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                frontleg_acceleration_min, frontleg_acceleration_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus')
                frontleg_jerk_min, frontleg_jerk_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus')
                frontleg_angvel_min, frontleg_angvel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                frontleg_angaccel_min, frontleg_angaccel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                frontleg_angjerk_min, frontleg_angjerk_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                
                head_velocity_min, head_velocity_max, head_acceleration_min, head_acceleration_max, head_jerk_min, head_jerk_max = compute_head_movement_features_minmax(bout_features_full, kinematic_suffix)

            # Middle Legs
            if not is_wall_up and not is_wall_down and not is_special_case:
                midleg_velocity_min, midleg_velocity_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus')
                midleg_acceleration_min, midleg_acceleration_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partaccel', 'MidTarsus')
                midleg_jerk_min, midleg_jerk_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partjerk', 'MidTarsus')
                midleg_angvel_min, midleg_angvel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                midleg_angaccel_min, midleg_angaccel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                midleg_angjerk_min, midleg_angjerk_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

            # Back Legs
            if not is_wall_up and not is_special_case:
                backleg_velocity_min, backleg_velocity_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus')
                backleg_acceleration_min, backleg_acceleration_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partaccel', 'BackTarsus')
                backleg_jerk_min, backleg_jerk_max = _calculate_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partjerk', 'BackTarsus')
                backleg_angvel_min, backleg_angvel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                backleg_angaccel_min, backleg_angaccel_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                backleg_angjerk_min, backleg_angjerk_max = _calculate_angular_leg_kinematics_minmax(bout_features_full, bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')

        # Compute median and skewness features (all bout types, using appropriate window)
        # For walking: use bout_features (core window), for non-walking: use bout_features (full bout)
        # Body movement (same conditions as mean/std)
        if not is_wall_up and not is_wall_down:
            speed_median = _safe_median(bout_features, f'speed{kinematic_suffix}')
            speed_skewness = _safe_skewness(speed_mean if not is_walking else speed_core_mean, speed_median, speed_std if not is_walking else speed_core_std)
            velocity_fwd_median = _safe_median(bout_features, f'velocity_forward{kinematic_suffix}')
            velocity_fwd_skewness = _safe_skewness(velocity_fwd_mean if not is_walking else velocity_fwd_core_mean, velocity_fwd_median, velocity_fwd_std if not is_walking else velocity_fwd_core_std)
            velocity_str_median = _safe_median(bout_features, f'velocity_strafe{kinematic_suffix}')
            velocity_str_skewness = _safe_skewness(velocity_str_mean if not is_walking else velocity_str_core_mean, velocity_str_median, velocity_str_std if not is_walking else velocity_str_core_std)
            accel_mag_median = _safe_median(bout_features, f'acceleration_magnitude{kinematic_suffix}')
            accel_mag_skewness = _safe_skewness(accel_mag_mean if not is_walking else accel_mag_core_mean, accel_mag_median, accel_mag_std if not is_walking else accel_mag_core_std)
            accel_fwd_median = _safe_median(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_fwd_skewness = _safe_skewness(accel_fwd_mean if not is_walking else accel_fwd_core_mean, accel_fwd_median, accel_fwd_std if not is_walking else accel_fwd_core_std)
            accel_str_median = _safe_median(bout_features, f'acceleration_strafe{kinematic_suffix}')
            accel_str_skewness = _safe_skewness(accel_str_mean if not is_walking else accel_str_core_mean, accel_str_median, accel_str_std if not is_walking else accel_str_core_std)
            jerk_mag_median = _safe_median(bout_features, f'jerk_magnitude{kinematic_suffix}')
            jerk_mag_skewness = _safe_skewness(jerk_mag_mean if not is_walking else jerk_mag_core_mean, jerk_mag_median, jerk_mag_std if not is_walking else jerk_mag_core_std)
            jerk_fwd_median = _safe_median(bout_features, f'jerk_forward{kinematic_suffix}')
            jerk_fwd_skewness = _safe_skewness(jerk_fwd_mean if not is_walking else jerk_fwd_core_mean, jerk_fwd_median, jerk_fwd_std if not is_walking else jerk_fwd_core_std)
            jerk_str_median = _safe_median(bout_features, f'jerk_strafe{kinematic_suffix}')
            jerk_str_skewness = _safe_skewness(jerk_str_mean if not is_walking else jerk_str_core_mean, jerk_str_median, jerk_str_std if not is_walking else jerk_str_core_std)

        # Angular (same conditions as mean/std)
        if "Floor" in bout_type or "Ceiling" in bout_type:
            angular_vel_median = _safe_median(bout_features, f'angular_velocity{kinematic_suffix}')
            angular_vel_skewness = _safe_skewness(angular_vel_mean if not is_walking else angular_vel_core_mean, angular_vel_median, angular_vel_std if not is_walking else angular_vel_core_std)
            angular_accel_median = _safe_median(bout_features, f'angular_acceleration{kinematic_suffix}')
            angular_accel_skewness = _safe_skewness(angular_accel_mean if not is_walking else angular_accel_core_mean, angular_accel_median, angular_accel_std if not is_walking else angular_accel_core_std)
            angular_jerk_median = _safe_median(bout_features, f'angular_jerk{kinematic_suffix}')
            angular_jerk_skewness = _safe_skewness(angular_jerk_mean if not is_walking else angular_jerk_core_mean, angular_jerk_median, angular_jerk_std if not is_walking else angular_jerk_core_std)

        # Front Legs & Head
        if not is_wall_down and not is_special_case:
            frontleg_velocity_median, frontleg_velocity_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'FrontTarsus', 
                                                                                                            frontleg_velocity_mean if not is_walking else frontleg_velocity_core_mean, 
                                                                                                            frontleg_velocity_std if not is_walking else frontleg_velocity_core_std)
            frontleg_acceleration_median, frontleg_acceleration_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partaccel', 'FrontTarsus',
                                                                                                                     frontleg_acceleration_mean if not is_walking else frontleg_acceleration_core_mean,
                                                                                                                     frontleg_acceleration_std if not is_walking else frontleg_acceleration_core_std)
            frontleg_jerk_median, frontleg_jerk_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partjerk', 'FrontTarsus',
                                                                                                    frontleg_jerk_mean if not is_walking else frontleg_jerk_core_mean,
                                                                                                    frontleg_jerk_std if not is_walking else frontleg_jerk_core_std)
            frontleg_angvel_median, frontleg_angvel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus',
                                                                                                                  frontleg_angvel_mean if not is_walking else frontleg_angvel_core_mean,
                                                                                                                  frontleg_angvel_std if not is_walking else frontleg_angvel_core_std)
            frontleg_angaccel_median, frontleg_angaccel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus',
                                                                                                                     frontleg_angaccel_mean if not is_walking else frontleg_angaccel_core_mean,
                                                                                                                     frontleg_angaccel_std if not is_walking else frontleg_angaccel_core_std)
            frontleg_angjerk_median, frontleg_angjerk_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus',
                                                                                                                    frontleg_angjerk_mean if not is_walking else frontleg_angjerk_core_mean,
                                                                                                                    frontleg_angjerk_std if not is_walking else frontleg_angjerk_core_std)
            
            head_velocity_median, head_velocity_skewness, head_acceleration_median, head_acceleration_skewness, head_jerk_median, head_jerk_skewness = compute_head_movement_features_median_skewness(
                bout_features, kinematic_suffix,
                head_velocity_mean if not is_walking else head_velocity_core_mean,
                head_velocity_std if not is_walking else head_velocity_core_std,
                head_acceleration_mean if not is_walking else head_acceleration_core_mean,
                head_acceleration_std if not is_walking else head_acceleration_core_std,
                head_jerk_mean if not is_walking else head_jerk_core_mean,
                head_jerk_std if not is_walking else head_jerk_core_std
            )

        # Middle Legs
        if not is_wall_up and not is_wall_down and not is_special_case:
            midleg_velocity_median, midleg_velocity_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'MidTarsus',
                                                                                                          midleg_velocity_mean if not is_walking else midleg_velocity_core_mean,
                                                                                                          midleg_velocity_std if not is_walking else midleg_velocity_core_std)
            midleg_acceleration_median, midleg_acceleration_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partaccel', 'MidTarsus',
                                                                                                                  midleg_acceleration_mean if not is_walking else midleg_acceleration_core_mean,
                                                                                                                  midleg_acceleration_std if not is_walking else midleg_acceleration_core_std)
            midleg_jerk_median, midleg_jerk_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partjerk', 'MidTarsus',
                                                                                                 midleg_jerk_mean if not is_walking else midleg_jerk_core_mean,
                                                                                                 midleg_jerk_std if not is_walking else midleg_jerk_core_std)
            midleg_angvel_median, midleg_angvel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus',
                                                                                                            midleg_angvel_mean if not is_walking else midleg_angvel_core_mean,
                                                                                                            midleg_angvel_std if not is_walking else midleg_angvel_core_std)
            midleg_angaccel_median, midleg_angaccel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus',
                                                                                                                 midleg_angaccel_mean if not is_walking else midleg_angaccel_core_mean,
                                                                                                                 midleg_angaccel_std if not is_walking else midleg_angaccel_core_std)
            midleg_angjerk_median, midleg_angjerk_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus',
                                                                                                                midleg_angjerk_mean if not is_walking else midleg_angjerk_core_mean,
                                                                                                                midleg_angjerk_std if not is_walking else midleg_angjerk_core_std)

        # Back Legs
        if not is_wall_up and not is_special_case:
            backleg_velocity_median, backleg_velocity_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partvelocity', 'BackTarsus',
                                                                                                            backleg_velocity_mean if not is_walking else backleg_velocity_core_mean,
                                                                                                            backleg_velocity_std if not is_walking else backleg_velocity_core_std)
            backleg_acceleration_median, backleg_acceleration_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partaccel', 'BackTarsus',
                                                                                                                    backleg_acceleration_mean if not is_walking else backleg_acceleration_core_mean,
                                                                                                                    backleg_acceleration_std if not is_walking else backleg_acceleration_core_std)
            backleg_jerk_median, backleg_jerk_skewness = _calculate_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partjerk', 'BackTarsus',
                                                                                                   backleg_jerk_mean if not is_walking else backleg_jerk_core_mean,
                                                                                                   backleg_jerk_std if not is_walking else backleg_jerk_core_std)
            backleg_angvel_median, backleg_angvel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus',
                                                                                                                backleg_angvel_mean if not is_walking else backleg_angvel_core_mean,
                                                                                                                backleg_angvel_std if not is_walking else backleg_angvel_core_std)
            backleg_angaccel_median, backleg_angaccel_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus',
                                                                                                                   backleg_angaccel_mean if not is_walking else backleg_angaccel_core_mean,
                                                                                                                   backleg_angaccel_std if not is_walking else backleg_angaccel_core_std)
            backleg_angjerk_median, backleg_angjerk_skewness = _calculate_angular_leg_kinematics_median_skewness(bout_features, bout_type, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus',
                                                                                                                   backleg_angjerk_mean if not is_walking else backleg_angjerk_core_mean,
                                                                                                                   backleg_angjerk_std if not is_walking else backleg_angjerk_core_std)
    
        # Build output dictionary conditionally based on walking status
        base_dict = {
            'individual': individual_id, 'group': group_label, 'sex': sex_label, 'source_file': source_file,
            'category': category, 'bout_type': bout_type,
            'start': start, 'end': end, 'length': bout_length,
        }
        
        if is_walking:
            # Walking bouts: include core + start + end features
            base_dict.update({
            'speed_core_mean': speed_core_mean, 'speed_core_std': speed_core_std,
            'velocity_fwd_core_mean': velocity_fwd_core_mean, 'velocity_fwd_core_std': velocity_fwd_core_std,
            'velocity_str_core_mean': velocity_str_core_mean, 'velocity_str_core_std': velocity_str_core_std,
            'accel_mag_core_mean': accel_mag_core_mean, 'accel_mag_core_std': accel_mag_core_std,
            'accel_fwd_core_mean': accel_fwd_core_mean, 'accel_fwd_core_std': accel_fwd_core_std,
            'accel_str_core_mean': accel_str_core_mean, 'accel_str_core_std': accel_str_core_std,
            'jerk_mag_core_mean': jerk_mag_core_mean, 'jerk_mag_core_std': jerk_mag_core_std,
            'jerk_fwd_core_mean': jerk_fwd_core_mean, 'jerk_fwd_core_std': jerk_fwd_core_std,
            'jerk_str_core_mean': jerk_str_core_mean, 'jerk_str_core_std': jerk_str_core_std,
            'angular_vel_core_mean': angular_vel_core_mean, 'angular_vel_core_std': angular_vel_core_std,
            'angular_accel_core_mean': angular_accel_core_mean, 'angular_accel_core_std': angular_accel_core_std,
            'angular_jerk_core_mean': angular_jerk_core_mean, 'angular_jerk_core_std': angular_jerk_core_std,

            'frontleg_velocity_core_mean': frontleg_velocity_core_mean, 'frontleg_velocity_core_std': frontleg_velocity_core_std, 'frontleg_velocity_core_asym': frontleg_velocity_core_asym,
            'frontleg_acceleration_core_mean': frontleg_acceleration_core_mean, 'frontleg_acceleration_core_std': frontleg_acceleration_core_std, 'frontleg_acceleration_core_asym': frontleg_acceleration_core_asym,
            'frontleg_jerk_core_mean': frontleg_jerk_core_mean, 'frontleg_jerk_core_std': frontleg_jerk_core_std, 'frontleg_jerk_core_asym': frontleg_jerk_core_asym,
            'frontleg_angvel_core_mean': frontleg_angvel_core_mean, 'frontleg_angvel_core_std': frontleg_angvel_core_std, 'frontleg_angvel_core_asym': frontleg_angvel_core_asym,
            'frontleg_angaccel_core_mean': frontleg_angaccel_core_mean, 'frontleg_angaccel_core_std': frontleg_angaccel_core_std, 'frontleg_angaccel_core_asym': frontleg_angaccel_core_asym,
            'frontleg_angjerk_core_mean': frontleg_angjerk_core_mean, 'frontleg_angjerk_core_std': frontleg_angjerk_core_std, 'frontleg_angjerk_core_asym': frontleg_angjerk_core_asym,

            'midleg_velocity_core_mean': midleg_velocity_core_mean, 'midleg_velocity_core_std': midleg_velocity_core_std, 'midleg_velocity_core_asym': midleg_velocity_core_asym,
            'midleg_acceleration_core_mean': midleg_acceleration_core_mean, 'midleg_acceleration_core_std': midleg_acceleration_core_std, 'midleg_acceleration_core_asym': midleg_acceleration_core_asym,
            'midleg_jerk_core_mean': midleg_jerk_core_mean, 'midleg_jerk_core_std': midleg_jerk_core_std, 'midleg_jerk_core_asym': midleg_jerk_core_asym,
            'midleg_angvel_core_mean': midleg_angvel_core_mean, 'midleg_angvel_core_std': midleg_angvel_core_std, 'midleg_angvel_core_asym': midleg_angvel_core_asym,
            'midleg_angaccel_core_mean': midleg_angaccel_core_mean, 'midleg_angaccel_core_std': midleg_angaccel_core_std, 'midleg_angaccel_core_asym': midleg_angaccel_core_asym,
            'midleg_angjerk_core_mean': midleg_angjerk_core_mean, 'midleg_angjerk_core_std': midleg_angjerk_core_std, 'midleg_angjerk_core_asym': midleg_angjerk_core_asym,
            
            'backleg_velocity_core_mean': backleg_velocity_core_mean, 'backleg_velocity_core_std': backleg_velocity_core_std, 'backleg_velocity_core_asym': backleg_velocity_core_asym,
            'backleg_acceleration_core_mean': backleg_acceleration_core_mean, 'backleg_acceleration_core_std': backleg_acceleration_core_std, 'backleg_acceleration_core_asym': backleg_acceleration_core_asym,
            'backleg_jerk_core_mean': backleg_jerk_core_mean, 'backleg_jerk_core_std': backleg_jerk_core_std, 'backleg_jerk_core_asym': backleg_jerk_core_asym,
            'backleg_angvel_core_mean': backleg_angvel_core_mean, 'backleg_angvel_core_std': backleg_angvel_core_std, 'backleg_angvel_core_asym': backleg_angvel_core_asym,
            'backleg_angaccel_core_mean': backleg_angaccel_core_mean, 'backleg_angaccel_core_std': backleg_angaccel_core_std, 'backleg_angaccel_core_asym': backleg_angaccel_core_asym,
            'backleg_angjerk_core_mean': backleg_angjerk_core_mean, 'backleg_angjerk_core_std': backleg_angjerk_core_std, 'backleg_angjerk_core_asym': backleg_angjerk_core_asym,

            'head_velocity_core_mean': head_velocity_core_mean, 'head_velocity_core_std': head_velocity_core_std,
            'head_acceleration_core_mean': head_acceleration_core_mean, 'head_acceleration_core_std': head_acceleration_core_std,
            'head_jerk_core_mean': head_jerk_core_mean, 'head_jerk_core_std': head_jerk_core_std,

            # Start features
            'speed_start_mean': speed_start_mean, 'speed_start_std': speed_start_std,
            'velocity_fwd_start_mean': velocity_fwd_start_mean, 'velocity_fwd_start_std': velocity_fwd_start_std,
            'velocity_str_start_mean': velocity_str_start_mean, 'velocity_str_start_std': velocity_str_start_std,
            'accel_mag_start_mean': accel_mag_start_mean, 'accel_mag_start_std': accel_mag_start_std,
            'accel_fwd_start_mean': accel_fwd_start_mean, 'accel_fwd_start_std': accel_fwd_start_std,
            'accel_str_start_mean': accel_str_start_mean, 'accel_str_start_std': accel_str_start_std,
            'jerk_mag_start_mean': jerk_mag_start_mean, 'jerk_mag_start_std': jerk_mag_start_std,
            'jerk_fwd_start_mean': jerk_fwd_start_mean, 'jerk_fwd_start_std': jerk_fwd_start_std,
            'jerk_str_start_mean': jerk_str_start_mean, 'jerk_str_start_std': jerk_str_start_std,
            'angular_vel_start_mean': angular_vel_start_mean, 'angular_vel_start_std': angular_vel_start_std,
            'angular_accel_start_mean': angular_accel_start_mean, 'angular_accel_start_std': angular_accel_start_std,
            'angular_jerk_start_mean': angular_jerk_start_mean, 'angular_jerk_start_std': angular_jerk_start_std,

            'frontleg_velocity_start_mean': frontleg_velocity_start_mean, 'frontleg_velocity_start_std': frontleg_velocity_start_std, 'frontleg_velocity_start_asym': frontleg_velocity_start_asym,
            'frontleg_acceleration_start_mean': frontleg_acceleration_start_mean, 'frontleg_acceleration_start_std': frontleg_acceleration_start_std, 'frontleg_acceleration_start_asym': frontleg_acceleration_start_asym,
            'frontleg_jerk_start_mean': frontleg_jerk_start_mean, 'frontleg_jerk_start_std': frontleg_jerk_start_std, 'frontleg_jerk_start_asym': frontleg_jerk_start_asym,
            'frontleg_angvel_start_mean': frontleg_angvel_start_mean, 'frontleg_angvel_start_std': frontleg_angvel_start_std, 'frontleg_angvel_start_asym': frontleg_angvel_start_asym,
            'frontleg_angaccel_start_mean': frontleg_angaccel_start_mean, 'frontleg_angaccel_start_std': frontleg_angaccel_start_std, 'frontleg_angaccel_start_asym': frontleg_angaccel_start_asym,
            'frontleg_angjerk_start_mean': frontleg_angjerk_start_mean, 'frontleg_angjerk_start_std': frontleg_angjerk_start_std, 'frontleg_angjerk_start_asym': frontleg_angjerk_start_asym,

            'midleg_velocity_start_mean': midleg_velocity_start_mean, 'midleg_velocity_start_std': midleg_velocity_start_std, 'midleg_velocity_start_asym': midleg_velocity_start_asym,
            'midleg_acceleration_start_mean': midleg_acceleration_start_mean, 'midleg_acceleration_start_std': midleg_acceleration_start_std, 'midleg_acceleration_start_asym': midleg_acceleration_start_asym,
            'midleg_jerk_start_mean': midleg_jerk_start_mean, 'midleg_jerk_start_std': midleg_jerk_start_std, 'midleg_jerk_start_asym': midleg_jerk_start_asym,
            'midleg_angvel_start_mean': midleg_angvel_start_mean, 'midleg_angvel_start_std': midleg_angvel_start_std, 'midleg_angvel_start_asym': midleg_angvel_start_asym,
            'midleg_angaccel_start_mean': midleg_angaccel_start_mean, 'midleg_angaccel_start_std': midleg_angaccel_start_std, 'midleg_angaccel_start_asym': midleg_angaccel_start_asym,
            'midleg_angjerk_start_mean': midleg_angjerk_start_mean, 'midleg_angjerk_start_std': midleg_angjerk_start_std, 'midleg_angjerk_start_asym': midleg_angjerk_start_asym,

            'backleg_velocity_start_mean': backleg_velocity_start_mean, 'backleg_velocity_start_std': backleg_velocity_start_std, 'backleg_velocity_start_asym': backleg_velocity_start_asym,
            'backleg_acceleration_start_mean': backleg_acceleration_start_mean, 'backleg_acceleration_start_std': backleg_acceleration_start_std, 'backleg_acceleration_start_asym': backleg_acceleration_start_asym,
            'backleg_jerk_start_mean': backleg_jerk_start_mean, 'backleg_jerk_start_std': backleg_jerk_start_std, 'backleg_jerk_start_asym': backleg_jerk_start_asym,
            'backleg_angvel_start_mean': backleg_angvel_start_mean, 'backleg_angvel_start_std': backleg_angvel_start_std, 'backleg_angvel_start_asym': backleg_angvel_start_asym,
            'backleg_angaccel_start_mean': backleg_angaccel_start_mean, 'backleg_angaccel_start_std': backleg_angaccel_start_std, 'backleg_angaccel_start_asym': backleg_angaccel_start_asym,
            'backleg_angjerk_start_mean': backleg_angjerk_start_mean, 'backleg_angjerk_start_std': backleg_angjerk_start_std, 'backleg_angjerk_start_asym': backleg_angjerk_start_asym,

            'head_velocity_start_mean': head_velocity_start_mean, 'head_velocity_start_std': head_velocity_start_std,
            'head_acceleration_start_mean': head_acceleration_start_mean, 'head_acceleration_start_std': head_acceleration_start_std,
            'head_jerk_start_mean': head_jerk_start_mean, 'head_jerk_start_std': head_jerk_start_std,

            # End features
            'speed_end_mean': speed_end_mean, 'speed_end_std': speed_end_std,
            'velocity_fwd_end_mean': velocity_fwd_end_mean, 'velocity_fwd_end_std': velocity_fwd_end_std,
            'velocity_str_end_mean': velocity_str_end_mean, 'velocity_str_end_std': velocity_str_end_std,
            'accel_mag_end_mean': accel_mag_end_mean, 'accel_mag_end_std': accel_mag_end_std,
            'accel_fwd_end_mean': accel_fwd_end_mean, 'accel_fwd_end_std': accel_fwd_end_std,
            'accel_str_end_mean': accel_str_end_mean, 'accel_str_end_std': accel_str_end_std,
            'jerk_mag_end_mean': jerk_mag_end_mean, 'jerk_mag_end_std': jerk_mag_end_std,
            'jerk_fwd_end_mean': jerk_fwd_end_mean, 'jerk_fwd_end_std': jerk_fwd_end_std,
            'jerk_str_end_mean': jerk_str_end_mean, 'jerk_str_end_std': jerk_str_end_std,
            'angular_vel_end_mean': angular_vel_end_mean, 'angular_vel_end_std': angular_vel_end_std,
            'angular_accel_end_mean': angular_accel_end_mean, 'angular_accel_end_std': angular_accel_end_std,
            'angular_jerk_end_mean': angular_jerk_end_mean, 'angular_jerk_end_std': angular_jerk_end_std,

            'frontleg_velocity_end_mean': frontleg_velocity_end_mean, 'frontleg_velocity_end_std': frontleg_velocity_end_std, 'frontleg_velocity_end_asym': frontleg_velocity_end_asym,
            'frontleg_acceleration_end_mean': frontleg_acceleration_end_mean, 'frontleg_acceleration_end_std': frontleg_acceleration_end_std, 'frontleg_acceleration_end_asym': frontleg_acceleration_end_asym,
            'frontleg_jerk_end_mean': frontleg_jerk_end_mean, 'frontleg_jerk_end_std': frontleg_jerk_end_std, 'frontleg_jerk_end_asym': frontleg_jerk_end_asym,
            'frontleg_angvel_end_mean': frontleg_angvel_end_mean, 'frontleg_angvel_end_std': frontleg_angvel_end_std, 'frontleg_angvel_end_asym': frontleg_angvel_end_asym,
            'frontleg_angaccel_end_mean': frontleg_angaccel_end_mean, 'frontleg_angaccel_end_std': frontleg_angaccel_end_std, 'frontleg_angaccel_end_asym': frontleg_angaccel_end_asym,
            'frontleg_angjerk_end_mean': frontleg_angjerk_end_mean, 'frontleg_angjerk_end_std': frontleg_angjerk_end_std, 'frontleg_angjerk_end_asym': frontleg_angjerk_end_asym,

            'midleg_velocity_end_mean': midleg_velocity_end_mean, 'midleg_velocity_end_std': midleg_velocity_end_std, 'midleg_velocity_end_asym': midleg_velocity_end_asym,
            'midleg_acceleration_end_mean': midleg_acceleration_end_mean, 'midleg_acceleration_end_std': midleg_acceleration_end_std, 'midleg_acceleration_end_asym': midleg_acceleration_end_asym,
            'midleg_jerk_end_mean': midleg_jerk_end_mean, 'midleg_jerk_end_std': midleg_jerk_end_std, 'midleg_jerk_end_asym': midleg_jerk_end_asym,
            'midleg_angvel_end_mean': midleg_angvel_end_mean, 'midleg_angvel_end_std': midleg_angvel_end_std, 'midleg_angvel_end_asym': midleg_angvel_end_asym,
            'midleg_angaccel_end_mean': midleg_angaccel_end_mean, 'midleg_angaccel_end_std': midleg_angaccel_end_std, 'midleg_angaccel_end_asym': midleg_angaccel_end_asym,
            'midleg_angjerk_end_mean': midleg_angjerk_end_mean, 'midleg_angjerk_end_std': midleg_angjerk_end_std, 'midleg_angjerk_end_asym': midleg_angjerk_end_asym,

            'backleg_velocity_end_mean': backleg_velocity_end_mean, 'backleg_velocity_end_std': backleg_velocity_end_std, 'backleg_velocity_end_asym': backleg_velocity_end_asym,
            'backleg_acceleration_end_mean': backleg_acceleration_end_mean, 'backleg_acceleration_end_std': backleg_acceleration_end_std, 'backleg_acceleration_end_asym': backleg_acceleration_end_asym,
            'backleg_jerk_end_mean': backleg_jerk_end_mean, 'backleg_jerk_end_std': backleg_jerk_end_std, 'backleg_jerk_end_asym': backleg_jerk_end_asym,
            'backleg_angvel_end_mean': backleg_angvel_end_mean, 'backleg_angvel_end_std': backleg_angvel_end_std, 'backleg_angvel_end_asym': backleg_angvel_end_asym,
            'backleg_angaccel_end_mean': backleg_angaccel_end_mean, 'backleg_angaccel_end_std': backleg_angaccel_end_std, 'backleg_angaccel_end_asym': backleg_angaccel_end_asym,
            'backleg_angjerk_end_mean': backleg_angjerk_end_mean, 'backleg_angjerk_end_std': backleg_angjerk_end_std, 'backleg_angjerk_end_asym': backleg_angjerk_end_asym,

            'head_velocity_end_mean': head_velocity_end_mean, 'head_velocity_end_std': head_velocity_end_std,
            'head_acceleration_end_mean': head_acceleration_end_mean, 'head_acceleration_end_std': head_acceleration_end_std,
            'head_jerk_end_mean': head_jerk_end_mean, 'head_jerk_end_std': head_jerk_end_std,

            # Min/Max features (full bout, tail means of bottom/top 5%)
            'speed_min': speed_min, 'speed_max': speed_max,
            'velocity_fwd_min': velocity_fwd_min, 'velocity_fwd_max': velocity_fwd_max,
            'velocity_str_min': velocity_str_min, 'velocity_str_max': velocity_str_max,
            'accel_mag_min': accel_mag_min, 'accel_mag_max': accel_mag_max,
            'accel_fwd_min': accel_fwd_min, 'accel_fwd_max': accel_fwd_max,
            'accel_str_min': accel_str_min, 'accel_str_max': accel_str_max,
            'jerk_mag_min': jerk_mag_min, 'jerk_mag_max': jerk_mag_max,
            'jerk_fwd_min': jerk_fwd_min, 'jerk_fwd_max': jerk_fwd_max,
            'jerk_str_min': jerk_str_min, 'jerk_str_max': jerk_str_max,
            'angular_vel_min': angular_vel_min, 'angular_vel_max': angular_vel_max,
            'angular_accel_min': angular_accel_min, 'angular_accel_max': angular_accel_max,
            'angular_jerk_min': angular_jerk_min, 'angular_jerk_max': angular_jerk_max,

            'frontleg_velocity_min': frontleg_velocity_min, 'frontleg_velocity_max': frontleg_velocity_max,
            'frontleg_acceleration_min': frontleg_acceleration_min, 'frontleg_acceleration_max': frontleg_acceleration_max,
            'frontleg_jerk_min': frontleg_jerk_min, 'frontleg_jerk_max': frontleg_jerk_max,
            'frontleg_angvel_min': frontleg_angvel_min, 'frontleg_angvel_max': frontleg_angvel_max,
            'frontleg_angaccel_min': frontleg_angaccel_min, 'frontleg_angaccel_max': frontleg_angaccel_max,
            'frontleg_angjerk_min': frontleg_angjerk_min, 'frontleg_angjerk_max': frontleg_angjerk_max,

            'midleg_velocity_min': midleg_velocity_min, 'midleg_velocity_max': midleg_velocity_max,
            'midleg_acceleration_min': midleg_acceleration_min, 'midleg_acceleration_max': midleg_acceleration_max,
            'midleg_jerk_min': midleg_jerk_min, 'midleg_jerk_max': midleg_jerk_max,
            'midleg_angvel_min': midleg_angvel_min, 'midleg_angvel_max': midleg_angvel_max,
            'midleg_angaccel_min': midleg_angaccel_min, 'midleg_angaccel_max': midleg_angaccel_max,
            'midleg_angjerk_min': midleg_angjerk_min, 'midleg_angjerk_max': midleg_angjerk_max,

            'backleg_velocity_min': backleg_velocity_min, 'backleg_velocity_max': backleg_velocity_max,
            'backleg_acceleration_min': backleg_acceleration_min, 'backleg_acceleration_max': backleg_acceleration_max,
            'backleg_jerk_min': backleg_jerk_min, 'backleg_jerk_max': backleg_jerk_max,
            'backleg_angvel_min': backleg_angvel_min, 'backleg_angvel_max': backleg_angvel_max,
            'backleg_angaccel_min': backleg_angaccel_min, 'backleg_angaccel_max': backleg_angaccel_max,
            'backleg_angjerk_min': backleg_angjerk_min, 'backleg_angjerk_max': backleg_angjerk_max,

            'head_velocity_min': head_velocity_min, 'head_velocity_max': head_velocity_max,
            'head_acceleration_min': head_acceleration_min, 'head_acceleration_max': head_acceleration_max,

            # Median and Skewness features (core window for walking)
            'speed_median': speed_median, 'speed_skewness': speed_skewness,
            'velocity_fwd_median': velocity_fwd_median, 'velocity_fwd_skewness': velocity_fwd_skewness,
            'velocity_str_median': velocity_str_median, 'velocity_str_skewness': velocity_str_skewness,
            'accel_mag_median': accel_mag_median, 'accel_mag_skewness': accel_mag_skewness,
            'accel_fwd_median': accel_fwd_median, 'accel_fwd_skewness': accel_fwd_skewness,
            'accel_str_median': accel_str_median, 'accel_str_skewness': accel_str_skewness,
            'jerk_mag_median': jerk_mag_median, 'jerk_mag_skewness': jerk_mag_skewness,
            'jerk_fwd_median': jerk_fwd_median, 'jerk_fwd_skewness': jerk_fwd_skewness,
            'jerk_str_median': jerk_str_median, 'jerk_str_skewness': jerk_str_skewness,
            'angular_vel_median': angular_vel_median, 'angular_vel_skewness': angular_vel_skewness,
            'angular_accel_median': angular_accel_median, 'angular_accel_skewness': angular_accel_skewness,
            'angular_jerk_median': angular_jerk_median, 'angular_jerk_skewness': angular_jerk_skewness,

            'frontleg_velocity_median': frontleg_velocity_median, 'frontleg_velocity_skewness': frontleg_velocity_skewness,
            'frontleg_acceleration_median': frontleg_acceleration_median, 'frontleg_acceleration_skewness': frontleg_acceleration_skewness,
            'frontleg_jerk_median': frontleg_jerk_median, 'frontleg_jerk_skewness': frontleg_jerk_skewness,
            'frontleg_angvel_median': frontleg_angvel_median, 'frontleg_angvel_skewness': frontleg_angvel_skewness,
            'frontleg_angaccel_median': frontleg_angaccel_median, 'frontleg_angaccel_skewness': frontleg_angaccel_skewness,
            'frontleg_angjerk_median': frontleg_angjerk_median, 'frontleg_angjerk_skewness': frontleg_angjerk_skewness,

            'midleg_velocity_median': midleg_velocity_median, 'midleg_velocity_skewness': midleg_velocity_skewness,
            'midleg_acceleration_median': midleg_acceleration_median, 'midleg_acceleration_skewness': midleg_acceleration_skewness,
            'midleg_jerk_median': midleg_jerk_median, 'midleg_jerk_skewness': midleg_jerk_skewness,
            'midleg_angvel_median': midleg_angvel_median, 'midleg_angvel_skewness': midleg_angvel_skewness,
            'midleg_angaccel_median': midleg_angaccel_median, 'midleg_angaccel_skewness': midleg_angaccel_skewness,
            'midleg_angjerk_median': midleg_angjerk_median, 'midleg_angjerk_skewness': midleg_angjerk_skewness,

            'backleg_velocity_median': backleg_velocity_median, 'backleg_velocity_skewness': backleg_velocity_skewness,
            'backleg_acceleration_median': backleg_acceleration_median, 'backleg_acceleration_skewness': backleg_acceleration_skewness,
            'backleg_jerk_median': backleg_jerk_median, 'backleg_jerk_skewness': backleg_jerk_skewness,
            'backleg_angvel_median': backleg_angvel_median, 'backleg_angvel_skewness': backleg_angvel_skewness,
            'backleg_angaccel_median': backleg_angaccel_median, 'backleg_angaccel_skewness': backleg_angaccel_skewness,
            'backleg_angjerk_median': backleg_angjerk_median, 'backleg_angjerk_skewness': backleg_angjerk_skewness,

            'head_velocity_median': head_velocity_median, 'head_velocity_skewness': head_velocity_skewness,
            'head_acceleration_median': head_acceleration_median, 'head_acceleration_skewness': head_acceleration_skewness,
            'head_jerk_median': head_jerk_median, 'head_jerk_skewness': head_jerk_skewness,

            # Curvature features (walking only, core window)
            'curvature_mean': curvature_mean,
            'curvature_median': curvature_median,
            'curvature_std': curvature_std,
            'curvature_skewness': curvature_skewness,
            'curvature_min': curvature_min,
            'curvature_max': curvature_max
            })
        else:
            # Non-walking bouts: include full-bout features
            base_dict.update({
            'speed_mean': speed_mean, 'speed_std': speed_std,
            'velocity_fwd_mean': velocity_fwd_mean, 'velocity_fwd_std': velocity_fwd_std,
            'velocity_str_mean': velocity_str_mean, 'velocity_str_std': velocity_str_std,
            'accel_mag_mean': accel_mag_mean, 'accel_mag_std': accel_mag_std,
            'accel_fwd_mean': accel_fwd_mean, 'accel_fwd_std': accel_fwd_std,
            'accel_str_mean': accel_str_mean, 'accel_str_std': accel_str_std,
            'jerk_mag_mean': jerk_mag_mean, 'jerk_mag_std': jerk_mag_std,
            'jerk_fwd_mean': jerk_fwd_mean, 'jerk_fwd_std': jerk_fwd_std,
            'jerk_str_mean': jerk_str_mean, 'jerk_str_std': jerk_str_std,
            'angular_vel_mean': angular_vel_mean, 'angular_vel_std': angular_vel_std,
            'angular_accel_mean': angular_accel_mean, 'angular_accel_std': angular_accel_std,
            'angular_jerk_mean': angular_jerk_mean, 'angular_jerk_std': angular_jerk_std,

            'frontleg_velocity_mean': frontleg_velocity_mean, 'frontleg_velocity_std': frontleg_velocity_std, 'frontleg_velocity_asym': frontleg_velocity_asym,
            'frontleg_acceleration_mean': frontleg_acceleration_mean, 'frontleg_acceleration_std': frontleg_acceleration_std, 'frontleg_acceleration_asym': frontleg_acceleration_asym,
            'frontleg_jerk_mean': frontleg_jerk_mean, 'frontleg_jerk_std': frontleg_jerk_std, 'frontleg_jerk_asym': frontleg_jerk_asym,
            'frontleg_angvel_mean': frontleg_angvel_mean, 'frontleg_angvel_std': frontleg_angvel_std, 'frontleg_angvel_asym': frontleg_angvel_asym,
            'frontleg_angaccel_mean': frontleg_angaccel_mean, 'frontleg_angaccel_std': frontleg_angaccel_std, 'frontleg_angaccel_asym': frontleg_angaccel_asym,
            'frontleg_angjerk_mean': frontleg_angjerk_mean, 'frontleg_angjerk_std': frontleg_angjerk_std, 'frontleg_angjerk_asym': frontleg_angjerk_asym,

            'midleg_velocity_mean': midleg_velocity_mean, 'midleg_velocity_std': midleg_velocity_std, 'midleg_velocity_asym': midleg_velocity_asym,
            'midleg_acceleration_mean': midleg_acceleration_mean, 'midleg_acceleration_std': midleg_acceleration_std, 'midleg_acceleration_asym': midleg_acceleration_asym,
            'midleg_jerk_mean': midleg_jerk_mean, 'midleg_jerk_std': midleg_jerk_std, 'midleg_jerk_asym': midleg_jerk_asym,
            'midleg_angvel_mean': midleg_angvel_mean, 'midleg_angvel_std': midleg_angvel_std, 'midleg_angvel_asym': midleg_angvel_asym,
            'midleg_angaccel_mean': midleg_angaccel_mean, 'midleg_angaccel_std': midleg_angaccel_std, 'midleg_angaccel_asym': midleg_angaccel_asym,
            'midleg_angjerk_mean': midleg_angjerk_mean, 'midleg_angjerk_std': midleg_angjerk_std, 'midleg_angjerk_asym': midleg_angjerk_asym,
            
            'backleg_velocity_mean': backleg_velocity_mean, 'backleg_velocity_std': backleg_velocity_std, 'backleg_velocity_asym': backleg_velocity_asym,
            'backleg_acceleration_mean': backleg_acceleration_mean, 'backleg_acceleration_std': backleg_acceleration_std, 'backleg_acceleration_asym': backleg_acceleration_asym,
            'backleg_jerk_mean': backleg_jerk_mean, 'backleg_jerk_std': backleg_jerk_std, 'backleg_jerk_asym': backleg_jerk_asym,
            'backleg_angvel_mean': backleg_angvel_mean, 'backleg_angvel_std': backleg_angvel_std, 'backleg_angvel_asym': backleg_angvel_asym,
            'backleg_angaccel_mean': backleg_angaccel_mean, 'backleg_angaccel_std': backleg_angaccel_std, 'backleg_angaccel_asym': backleg_angaccel_asym,
            'backleg_angjerk_mean': backleg_angjerk_mean, 'backleg_angjerk_std': backleg_angjerk_std, 'backleg_angjerk_asym': backleg_angjerk_asym,

            'head_velocity_mean': head_velocity_mean, 'head_velocity_std': head_velocity_std,
            'head_acceleration_mean': head_acceleration_mean, 'head_acceleration_std': head_acceleration_std,
            'head_jerk_mean': head_jerk_mean, 'head_jerk_std': head_jerk_std,

            # Min/Max features (full bout, tail means of bottom/top 5%)
            'speed_min': speed_min, 'speed_max': speed_max,
            'velocity_fwd_min': velocity_fwd_min, 'velocity_fwd_max': velocity_fwd_max,
            'velocity_str_min': velocity_str_min, 'velocity_str_max': velocity_str_max,
            'accel_mag_min': accel_mag_min, 'accel_mag_max': accel_mag_max,
            'accel_fwd_min': accel_fwd_min, 'accel_fwd_max': accel_fwd_max,
            'accel_str_min': accel_str_min, 'accel_str_max': accel_str_max,
            'jerk_mag_min': jerk_mag_min, 'jerk_mag_max': jerk_mag_max,
            'jerk_fwd_min': jerk_fwd_min, 'jerk_fwd_max': jerk_fwd_max,
            'jerk_str_min': jerk_str_min, 'jerk_str_max': jerk_str_max,
            'angular_vel_min': angular_vel_min, 'angular_vel_max': angular_vel_max,
            'angular_accel_min': angular_accel_min, 'angular_accel_max': angular_accel_max,
            'angular_jerk_min': angular_jerk_min, 'angular_jerk_max': angular_jerk_max,

            'frontleg_velocity_min': frontleg_velocity_min, 'frontleg_velocity_max': frontleg_velocity_max,
            'frontleg_acceleration_min': frontleg_acceleration_min, 'frontleg_acceleration_max': frontleg_acceleration_max,
            'frontleg_jerk_min': frontleg_jerk_min, 'frontleg_jerk_max': frontleg_jerk_max,
            'frontleg_angvel_min': frontleg_angvel_min, 'frontleg_angvel_max': frontleg_angvel_max,
            'frontleg_angaccel_min': frontleg_angaccel_min, 'frontleg_angaccel_max': frontleg_angaccel_max,
            'frontleg_angjerk_min': frontleg_angjerk_min, 'frontleg_angjerk_max': frontleg_angjerk_max,

            'midleg_velocity_min': midleg_velocity_min, 'midleg_velocity_max': midleg_velocity_max,
            'midleg_acceleration_min': midleg_acceleration_min, 'midleg_acceleration_max': midleg_acceleration_max,
            'midleg_jerk_min': midleg_jerk_min, 'midleg_jerk_max': midleg_jerk_max,
            'midleg_angvel_min': midleg_angvel_min, 'midleg_angvel_max': midleg_angvel_max,
            'midleg_angaccel_min': midleg_angaccel_min, 'midleg_angaccel_max': midleg_angaccel_max,
            'midleg_angjerk_min': midleg_angjerk_min, 'midleg_angjerk_max': midleg_angjerk_max,

            'backleg_velocity_min': backleg_velocity_min, 'backleg_velocity_max': backleg_velocity_max,
            'backleg_acceleration_min': backleg_acceleration_min, 'backleg_acceleration_max': backleg_acceleration_max,
            'backleg_jerk_min': backleg_jerk_min, 'backleg_jerk_max': backleg_jerk_max,
            'backleg_angvel_min': backleg_angvel_min, 'backleg_angvel_max': backleg_angvel_max,
            'backleg_angaccel_min': backleg_angaccel_min, 'backleg_angaccel_max': backleg_angaccel_max,
            'backleg_angjerk_min': backleg_angjerk_min, 'backleg_angjerk_max': backleg_angjerk_max,

            'head_velocity_min': head_velocity_min, 'head_velocity_max': head_velocity_max,
            'head_acceleration_min': head_acceleration_min, 'head_acceleration_max': head_acceleration_max,
            'head_jerk_min': head_jerk_min, 'head_jerk_max': head_jerk_max,

            # Median and Skewness features (full bout for non-walking)
            'speed_median': speed_median, 'speed_skewness': speed_skewness,
            'velocity_fwd_median': velocity_fwd_median, 'velocity_fwd_skewness': velocity_fwd_skewness,
            'velocity_str_median': velocity_str_median, 'velocity_str_skewness': velocity_str_skewness,
            'accel_mag_median': accel_mag_median, 'accel_mag_skewness': accel_mag_skewness,
            'accel_fwd_median': accel_fwd_median, 'accel_fwd_skewness': accel_fwd_skewness,
            'accel_str_median': accel_str_median, 'accel_str_skewness': accel_str_skewness,
            'jerk_mag_median': jerk_mag_median, 'jerk_mag_skewness': jerk_mag_skewness,
            'jerk_fwd_median': jerk_fwd_median, 'jerk_fwd_skewness': jerk_fwd_skewness,
            'jerk_str_median': jerk_str_median, 'jerk_str_skewness': jerk_str_skewness,
            'angular_vel_median': angular_vel_median, 'angular_vel_skewness': angular_vel_skewness,
            'angular_accel_median': angular_accel_median, 'angular_accel_skewness': angular_accel_skewness,
            'angular_jerk_median': angular_jerk_median, 'angular_jerk_skewness': angular_jerk_skewness,

            'frontleg_velocity_median': frontleg_velocity_median, 'frontleg_velocity_skewness': frontleg_velocity_skewness,
            'frontleg_acceleration_median': frontleg_acceleration_median, 'frontleg_acceleration_skewness': frontleg_acceleration_skewness,
            'frontleg_jerk_median': frontleg_jerk_median, 'frontleg_jerk_skewness': frontleg_jerk_skewness,
            'frontleg_angvel_median': frontleg_angvel_median, 'frontleg_angvel_skewness': frontleg_angvel_skewness,
            'frontleg_angaccel_median': frontleg_angaccel_median, 'frontleg_angaccel_skewness': frontleg_angaccel_skewness,
            'frontleg_angjerk_median': frontleg_angjerk_median, 'frontleg_angjerk_skewness': frontleg_angjerk_skewness,

            'midleg_velocity_median': midleg_velocity_median, 'midleg_velocity_skewness': midleg_velocity_skewness,
            'midleg_acceleration_median': midleg_acceleration_median, 'midleg_acceleration_skewness': midleg_acceleration_skewness,
            'midleg_jerk_median': midleg_jerk_median, 'midleg_jerk_skewness': midleg_jerk_skewness,
            'midleg_angvel_median': midleg_angvel_median, 'midleg_angvel_skewness': midleg_angvel_skewness,
            'midleg_angaccel_median': midleg_angaccel_median, 'midleg_angaccel_skewness': midleg_angaccel_skewness,
            'midleg_angjerk_median': midleg_angjerk_median, 'midleg_angjerk_skewness': midleg_angjerk_skewness,

            'backleg_velocity_median': backleg_velocity_median, 'backleg_velocity_skewness': backleg_velocity_skewness,
            'backleg_acceleration_median': backleg_acceleration_median, 'backleg_acceleration_skewness': backleg_acceleration_skewness,
            'backleg_jerk_median': backleg_jerk_median, 'backleg_jerk_skewness': backleg_jerk_skewness,
            'backleg_angvel_median': backleg_angvel_median, 'backleg_angvel_skewness': backleg_angvel_skewness,
            'backleg_angaccel_median': backleg_angaccel_median, 'backleg_angaccel_skewness': backleg_angaccel_skewness,
            'backleg_angjerk_median': backleg_angjerk_median, 'backleg_angjerk_skewness': backleg_angjerk_skewness,

            'head_velocity_median': head_velocity_median, 'head_velocity_skewness': head_velocity_skewness,
            'head_acceleration_median': head_acceleration_median, 'head_acceleration_skewness': head_acceleration_skewness,
            'head_jerk_median': head_jerk_median, 'head_jerk_skewness': head_jerk_skewness,
            })

        # ------------------------------------------------------------------
        # Feature pruning (forward-only): remove categories that are repeatedly
        # observed to be highly redundant across datasets.
        #
        # Rationale:
        #   - Angular acceleration/jerk std features are often nearly identical
        #     to angular velocity std (and can dominate redundancy reports).
        #   - Walking start/end windows are short; accel/jerk std in these
        #     segments is unstable and frequently redundant.
        #
        # IMPORTANT: We only remove the exported per-bout columns; internal
        # std values may still be computed to support skewness calculations.
        # ------------------------------------------------------------------
        for k in list(base_dict.keys()):
            # 1) Drop ALL angular-accel / angular-jerk std features (body + legs)
            if k.endswith('_std') and (
                ('angaccel' in k) or ('angjerk' in k) or ('angular_accel' in k) or ('angular_jerk' in k)
            ):
                base_dict.pop(k, None)
                continue
            # 2) Drop walking start/end accel + jerk std (body + legs + head)
            if k.endswith(('_start_std', '_end_std')) and (
                ('accel' in k) or ('acceleration' in k) or ('jerk' in k)
            ):
                base_dict.pop(k, None)
                continue
        
        bout_feature_rows.append(base_dict)

    return pd.DataFrame(bout_feature_rows)

def _reconstruct_kinematic_table_from_aggregates(agg_tables, *, min_bout_count_for_features: int = DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES):
    # Build a compact per-(individual,group,bout_type) table with the four stats:
    results = {}
    for suffix, df in agg_tables.items():
        if df is None or df.empty:
            results[suffix] = pd.DataFrame()
            continue
        base_keys = ['individual','group','bout_type']
        kin_sum_cols   = [c for c in df.columns if c.endswith('__sum')]
        kin_sumsq_cols = [c for c in df.columns if c.endswith('__sum_sq')]
        kin_cnt_cols   = [c for c in df.columns if c.endswith('__count')]

        recs = []
        for key, block in df.groupby(base_keys, observed=True):
            bt = key[2]
            row = {
                'individual': key[0],
                'group': key[1],
                # Jump/Scramble are collapsed contexts but treated as Pose+Orient kinematics
                'category': ('Pose+Orient' if ('+' in str(bt) or str(bt) in ['Jump', 'Scramble']) else 'Pose'),
                'bout_type': bt
            }
            # aggregate across files
            blk_sum   = block[kin_sum_cols].sum(numeric_only=True)
            blk_sumsq = block[kin_sumsq_cols].sum(numeric_only=True)
            blk_cnt   = block[kin_cnt_cols].sum(numeric_only=True)
            for base in [c[:-5] for c in kin_sum_cols]:  # strip '__sum'
                s   = blk_sum.get(f'{base}__sum',   0.0)
                ss  = blk_sumsq.get(f'{base}__sum_sq', 0.0)
                cnt = blk_cnt.get(f'{base}__count', 0)
                # Enforce minimum of 3 bouts for statistical validity (especially important for std_std)
                if cnt < min_bout_count_for_features:
                    mean, std = np.nan, np.nan
                else:
                    mean, std = _finalize_mean_std(s, ss, cnt)
                if base.endswith('_mean'):
                    head = base[:-5]  # drop '_mean'
                    row[f'{head}_mean_mean'] = mean
                    row[f'{head}_mean_std']  = std
                elif base.endswith('_std'):
                    head = base[:-4]  # drop '_std'
                    row[f'{head}_std_mean'] = mean
                    row[f'{head}_std_std']  = std
                elif base.endswith('_asym'):
                    head = base[:-5]  # drop '_asym'
                    row[f'{head}_asym_mean'] = mean
                    row[f'{head}_asym_std']  = std
                elif base.endswith('_min'):
                    head = base[:-4]  # drop '_min'
                    row[f'{head}_min_mean'] = mean
                elif base.endswith('_max'):
                    head = base[:-4]  # drop '_max'
                    row[f'{head}_max_mean'] = mean
                elif base.endswith('_median'):
                    head = base[:-7]  # drop '_median'
                    row[f'{head}_median_mean'] = mean
                    row[f'{head}_median_std']  = std
                elif base.endswith('_skewness'):
                    head = base[:-9]  # drop '_skewness'
                    row[f'{head}_skewness_mean'] = mean
                    row[f'{head}_skewness_std']  = std
            recs.append(row)
        results[suffix] = pd.DataFrame.from_records(recs)
    return results

def calculate_and_save_kinematic_summary(master_dfs_by_suffix, output_dir, *, min_bout_count_for_features: int = DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES, save_outputs: bool = True):
    """
    Aggregates kinematic features and saves them to a single consolidated Excel file,
    with sheets for each smoothing level and behavior category (pose/pose+orient).
    """
    print("\nCalculating and saving consolidated kinematic feature summary to Excel...")
    excel_path = os.path.join(output_dir, '5_Kinematic_Features.xlsx')

    sheets_to_save = {}

    for suffix, master_df in master_dfs_by_suffix.items():
        # Create a clean label for the smoothing level, e.g., "(inst)", "(5)"
        smoothing_label = f"({suffix.lstrip('_')})"
        
        if master_df.empty:
            print(f"   - Skipping sheets for smoothing level {smoothing_label} as no data is available.")
            continue
        
        print(f"  Processing data for: {smoothing_label}")
        # master_df already has per-individual rows with columns
        # like *_mean_mean, *_mean_std, *_std_mean, *_std_std.
        # Pivot them directly:
        value_cols = [c for c in master_df.columns 
                      if any(c.endswith(suf) for suf in 
                             ('_mean_mean','_mean_std','_std_mean','_std_std',
                              '_asym_mean','_asym_std','_min_mean','_max_mean',
                              '_median_mean','_median_std','_skewness_mean','_skewness_std'))]
        tidy = master_df[['individual','group','bout_type'] + value_cols].copy()
        pivot_df = tidy.pivot_table(index=['individual','group'],
                                    columns='bout_type',
                                    values=value_cols,
                                    observed=True)
        # Flatten MultiIndex columns "values x bout_type" → "bout_type_value"
        # Standardize feature names: remove spaces from bout_type (e.g., "Walking + Floor" -> "Walking+Floor")
        pivot_df.columns = [f'{bt.replace(" ", "")}_{val}' for val, bt in pivot_df.columns.to_flat_index()]
        pivot_df.reset_index(inplace=True)

        # Drop any column that is entirely empty (all NaN)
        pivot_df.dropna(axis=1, how='all', inplace=True)
        
        pivot_df = pivot_df.copy() # De-fragment the DataFrame to improve performance

        # Define sorting logic for columns
        pose_order = ['Walking', 'Standing', 'Scramble', 'Groom-FrontLegs', 'Groom-Head', 'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs', 'Jump']
        orient_order = ['Floor', 'Ceiling', 'Wall-Side', 'Wall-Up', 'Wall-Down']
        behavior_map = {b: i for i, b in enumerate(pose_order + [f"{p} + {o}" for p in pose_order if p != 'Jump' for o in orient_order])}
        feature_prefix_map = {'speed': 0, 'velocity_fwd': 1, 'velocity_str': 2, 'accel_mag': 3, 'accel_fwd': 4, 'accel_str': 5, 'frontleg': 6, 'backleg': 7, 'head': 8}

        # Normalize bout_types to remove spaces (matching the column name format from line 3707)
        all_bout_types = sorted([b.replace(" ", "") for b in master_df['bout_type'].unique() if pd.notna(b)], key=len, reverse=True)
        def get_parts(col):
            for bt in all_bout_types:
                if col.startswith(bt): return bt, col[len(bt):].lstrip('_')
            return None, col
        def custom_sort_key(col):
            bt, feat = get_parts(col)
            f_prefix = next((p for p in feature_prefix_map if feat.startswith(p)), 'z_other')
            return (feature_prefix_map.get(f_prefix, 999), behavior_map.get(bt, 999), feat)

        # Define multiple unique separators for the statistical blocks
        separators = ['---_1', '---_2', '---_3', '---_4', '---_5', '---_6', '---_7', '---_8', '---_9', '---_10', '---_11'] 
        for sep in separators:
            pivot_df[sep] = '' # Add the blank columns to the DataFrame

        all_feature_cols = [col for col in pivot_df.columns if col not in ['individual', 'group'] + separators]

        def _is_pose_only_bt(bt):
            bt = str(bt)
            # Jump/Scramble are collapsed contexts but belong in Pose+Orient kinematics outputs.
            if bt in ['Jump', 'Scramble']:
                return False
            return '+' not in bt

        pose_cols = [c for c in all_feature_cols if _is_pose_only_bt(get_parts(c)[0])]
        orient_cols = [c for c in all_feature_cols if not _is_pose_only_bt(get_parts(c)[0])]

        # Process and save the Pose-only sheet with the new column order
        if pose_cols:
            # 1. Create lists for each statistical measure
            pose_mean_mean_cols = sorted([c for c in pose_cols if c.endswith('_mean_mean')], key=custom_sort_key)
            pose_mean_std_cols  = sorted([c for c in pose_cols if c.endswith('_mean_std')], key=custom_sort_key)
            pose_std_mean_cols  = sorted([c for c in pose_cols if c.endswith('_std_mean')], key=custom_sort_key)
            pose_std_std_cols   = sorted([c for c in pose_cols if c.endswith('_std_std')], key=custom_sort_key)
            pose_asym_mean_cols = sorted([c for c in pose_cols if c.endswith('_asym_mean')], key=custom_sort_key)
            pose_asym_std_cols  = sorted([c for c in pose_cols if c.endswith('_asym_std')], key=custom_sort_key)
            pose_min_mean_cols  = sorted([c for c in pose_cols if c.endswith('_min_mean')], key=custom_sort_key)
            pose_max_mean_cols  = sorted([c for c in pose_cols if c.endswith('_max_mean')], key=custom_sort_key)
            pose_median_mean_cols = sorted([c for c in pose_cols if c.endswith('_median_mean')], key=custom_sort_key)
            pose_median_std_cols  = sorted([c for c in pose_cols if c.endswith('_median_std')], key=custom_sort_key)
            pose_skewness_mean_cols = sorted([c for c in pose_cols if c.endswith('_skewness_mean')], key=custom_sort_key)
            pose_skewness_std_cols  = sorted([c for c in pose_cols if c.endswith('_skewness_std')], key=custom_sort_key)

            # 2. Define the final column order with separators
            pose_desired_order = (pose_mean_mean_cols + [separators[0]] + 
                                  pose_mean_std_cols  + [separators[1]] +
                                  pose_std_mean_cols  + [separators[2]] +
                                  pose_std_std_cols   + [separators[3]] +
                                  pose_asym_mean_cols + [separators[4]] +
                                  pose_asym_std_cols  + [separators[5]] +
                                  pose_min_mean_cols  + [separators[6]] +
                                  pose_max_mean_cols  + [separators[7]] +
                                  pose_median_mean_cols + [separators[8]] +
                                  pose_median_std_cols  + [separators[9]] +
                                  pose_skewness_mean_cols + [separators[10]] +
                                  pose_skewness_std_cols)
                                  
            df_to_format = pivot_df[['individual', 'group'] + pose_desired_order].copy()
            
            # Filter out columns that are entirely zero or NaN
            cols_to_drop = []
            for col in df_to_format.columns:
                # Skip identifier and separator columns
                if col in ['individual', 'group'] or str(col).startswith('---_'):
                    continue
                # If a column, after filling NaNs with 0, contains only 0s, mark for dropping
                if (df_to_format[col].fillna(0) == 0).all():
                    cols_to_drop.append(col)
            
            df_to_format.drop(columns=cols_to_drop, inplace=True)
            pose_desired_order_filtered = [c for c in pose_desired_order if c not in cols_to_drop]

            sheets_to_save[f'Pose {smoothing_label}'] = format_summary_df(df_to_format, pose_desired_order_filtered)

        # Process and save the Pose+Orient sheet with the new column order
        if orient_cols:
            # 1. Create lists for each statistical measure
            orient_mean_mean_cols = sorted([c for c in orient_cols if c.endswith('_mean_mean')], key=custom_sort_key)
            orient_mean_std_cols  = sorted([c for c in orient_cols if c.endswith('_mean_std')], key=custom_sort_key)
            orient_std_mean_cols  = sorted([c for c in orient_cols if c.endswith('_std_mean')], key=custom_sort_key)
            orient_std_std_cols   = sorted([c for c in orient_cols if c.endswith('_std_std')], key=custom_sort_key)
            orient_asym_mean_cols = sorted([c for c in orient_cols if c.endswith('_asym_mean')], key=custom_sort_key)
            orient_asym_std_cols  = sorted([c for c in orient_cols if c.endswith('_asym_std')], key=custom_sort_key)
            orient_min_mean_cols  = sorted([c for c in orient_cols if c.endswith('_min_mean')], key=custom_sort_key)
            orient_max_mean_cols  = sorted([c for c in orient_cols if c.endswith('_max_mean')], key=custom_sort_key)
            orient_median_mean_cols = sorted([c for c in orient_cols if c.endswith('_median_mean')], key=custom_sort_key)
            orient_median_std_cols  = sorted([c for c in orient_cols if c.endswith('_median_std')], key=custom_sort_key)
            orient_skewness_mean_cols = sorted([c for c in orient_cols if c.endswith('_skewness_mean')], key=custom_sort_key)
            orient_skewness_std_cols  = sorted([c for c in orient_cols if c.endswith('_skewness_std')], key=custom_sort_key)

            # 2. Define the final column order with separators
            orient_desired_order = (orient_mean_mean_cols + [separators[0]] + 
                                    orient_mean_std_cols  + [separators[1]] +
                                    orient_std_mean_cols  + [separators[2]] +
                                    orient_std_std_cols   + [separators[3]] +
                                    orient_asym_mean_cols + [separators[4]] +
                                    orient_asym_std_cols  + [separators[5]] +
                                    orient_min_mean_cols  + [separators[6]] +
                                    orient_max_mean_cols  + [separators[7]] +
                                    orient_median_mean_cols + [separators[8]] +
                                    orient_median_std_cols  + [separators[9]] +
                                    orient_skewness_mean_cols + [separators[10]] +
                                    orient_skewness_std_cols)

            df_to_format = pivot_df[['individual', 'group'] + orient_desired_order].copy()

            # Filter out columns that are entirely zero or NaN
            cols_to_drop = []
            for col in df_to_format.columns:
                if col in ['individual', 'group'] or str(col).startswith('---_'):
                    continue
                if (df_to_format[col].fillna(0) == 0).all():
                    cols_to_drop.append(col)

            df_to_format.drop(columns=cols_to_drop, inplace=True)
            orient_desired_order_filtered = [c for c in orient_desired_order if c not in cols_to_drop]
            
            sheets_to_save[f'Pose + Orient {smoothing_label}'] = format_summary_df(df_to_format, orient_desired_order_filtered)

    if save_outputs:
        save_to_excel_sheets(sheets_to_save, excel_path)
    return sheets_to_save

def _ordered_chunks_for_workers(items, n_chunks: int):
    items = list(items)
    if not items:
        return []
    # Use more tasks than workers so Pool.map(chunksize=1) can balance uneven
    # animals better, while still avoiding one tiny process per animal.
    n_chunks = max(1, min(int(n_chunks) * 4, len(items)))
    base, rem = divmod(len(items), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        stop = start + size
        if stop > start:
            chunks.append(items[start:stop])
        start = stop
    return chunks


def _tw_finalize_sheets_from_long(tw_long: pd.DataFrame, *, out_path: str, save_outputs: bool):
    sheets = {}

    pose_df = tw_long[tw_long["category"] == "Pose"].copy()
    if not pose_df.empty:
        feature_cols = [c for c in pose_df.columns if c not in ["category", "bout_type", "sex", "individual", "group"]]
        wide = pose_df.pivot_table(index=["individual", "group"], columns="bout_type", values=feature_cols, aggfunc="first")
        if isinstance(wide.columns, pd.MultiIndex) and wide.columns.nlevels >= 2:
            wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
            wide.columns = [f"{bout}_{feat}" for (bout, feat) in wide.columns]
        else:
            wide.columns = [str(c) for c in wide.columns]
        wide = wide.reset_index()
        sheets["Pose (inst)"] = wide

    po_df = tw_long[tw_long["category"] == "Pose+Orient"].copy()
    if not po_df.empty:
        feature_cols = [c for c in po_df.columns if c not in ["category", "bout_type", "sex", "individual", "group"]]
        wide = po_df.pivot_table(index=["individual", "group"], columns="bout_type", values=feature_cols, aggfunc="first")
        if isinstance(wide.columns, pd.MultiIndex) and wide.columns.nlevels >= 2:
            wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
            wide.columns = [f"{bout}_{feat}" for (bout, feat) in wide.columns]
        else:
            wide.columns = [str(c) for c in wide.columns]
        wide = wide.reset_index()
        sheets["Pose + Orient (inst)"] = wide

    if not sheets:
        print("TW kinematic output has no sheets to write; skipping.")
        return

    if save_outputs:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet_name, df_sheet in sheets.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved TW kinematic features: {out_path}")
    return sheets


def _tw_chunk_worker(args):
    temp_store_path, output_dir, video_fps, min_context_frames, selected_individuals, individual_to_animal = args
    return calculate_and_save_timeweighted_kinematic_summary(
        temp_store_path,
        output_dir,
        video_fps=float(video_fps),
        min_context_frames=(None if min_context_frames is None else int(min_context_frames)),
        save_outputs=False,
        num_workers=1,
        selected_individuals=list(selected_individuals),
        individual_to_animal=dict(individual_to_animal or {}),
        _return_long_rows=True,
    )


def _collapse_tw_individual_id(individual_id: str, individual_to_animal: dict[str, str] | None = None) -> str:
    """Collapse a per-recording individual ID to the final animal-level row ID."""

    individual_id = str(individual_id)
    if individual_to_animal and individual_id in individual_to_animal:
        return str(individual_to_animal[individual_id])
    return collapse_individual_id(individual_id)


def _make_individual_where_clause(individual_ids: list[str]) -> str:
    """Return a PyTables where-clause matching one or more individual IDs."""

    safe_ids = [str(ind).replace("\\", "\\\\").replace("'", "\\'") for ind in individual_ids if ind is not None]
    if not safe_ids:
        return ""
    if len(safe_ids) == 1:
        return f"individual=='{safe_ids[0]}'"
    return " | ".join([f"individual=='{safe_id}'" for safe_id in safe_ids])


def calculate_and_save_timeweighted_kinematic_summary(temp_store_path, output_dir, *, video_fps: float = DEFAULT_VIDEO_FPS, min_context_frames: int | None = None, save_outputs: bool = True, num_workers: int = 1, selected_individuals: list[str] | None = None, individual_to_animal: dict[str, str] | None = None, _return_long_rows: bool = False):
    """
    Time-weighted (TW) kinematic features computed directly from per-frame signals in the HDF5 /kin_frames table.

    This function aggregates per-frame kinematic signals over *all frames* in a context, rather than
    aggregating per-bout summaries (BB). It is intended to produce a mirrored set of kinematic features
    that are directly comparable to the existing BB kinematic feature families.

    Importantly, TW aggregation is performed at the final animal-level row identity, pooling frames from
    all recordings (movie0/movie1/...) that belong to the same animal before any summary statistics are
    finalized. This keeps the TW outputs aligned with the collapsed animal-level master matrix.

    Rules:
      - Non-walking contexts: compute one set of stats over ALL frames in that context.
      - Walking contexts:
          * "core" mean/std and median/skew computed on CORE frames only (trimmed by BOUT_TRIM_FRAMES_CORE)
          * "start/end" mean/std computed on the first/last BOUT_FRAMES_START_END frames of each bout
          * min/max (tail means of bottom/top 5%) computed on FULL walking frames
          * curvature stats computed on CORE frames (mirrors BB where curvature is computed on trimmed walking)
      - Guardrail:
          * For full/core context-level stats, require at least min_context_frames frames.
          * For start/end segments, we compute if any frames exist (reliability is handled downstream by bout-count filters).

    Output:
      - Writes: 5b_KinematicTW_Features.xlsx
      - Sheets: "Pose (inst)" and "Pose + Orient (inst)"
    """
    # NOTE: HDF_KEYS is defined inside run_post_inference_analysis (local scope), so we hardcode here.
    kin_key = "/kin_frames"
    bouts_key = "/bouts_inst"
    meta_key = "/bout_summary"  # optional; may not exist

    out_path = os.path.join(output_dir, "5b_KinematicTW_Features.xlsx")
    kinematic_suffix = "_inst"
    video_fps = float(video_fps)
    if min_context_frames is None:
        min_context_frames = int(round(MIN_CONTEXT_DURATION_S * video_fps))
    else:
        min_context_frames = int(min_context_frames)
    individual_to_animal = {
        str(k): str(v)
        for k, v in (individual_to_animal or {}).items()
        if k is not None and v is not None
    }

    # --- Helpers ---
    def _n_non_nan(arr):
        return int(np.sum(~np.isnan(arr)))

    def _safe_mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        n = _n_non_nan(arr)
        if n == 0:
            return np.nan, np.nan
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr, ddof=1)) if n > 1 else np.nan
        return mean_val, std_val

    def _safe_median_skew(arr, mean_val, std_val):
        arr = np.asarray(arr, dtype=float)
        n = _n_non_nan(arr)
        if n == 0:
            return np.nan, np.nan
        median_val = float(np.nanmedian(arr))
        skew_val = _safe_skewness(mean_val, median_val, std_val)
        return median_val, skew_val

    def _safe_p5_p95(arr, pmin=5, pmax=95, min_tail_frames=3):
        """Return tail means (bottom/top 5%) for consistency with bout features."""
        return _safe_tail_means_from_array(arr, pmin=pmin, pmax=pmax, min_tail_frames=min_tail_frames)

    def _is_special_case(bout_type):
        # Mirrors BB: Scramble/Jump skip head + leg kinematics
        s = str(bout_type)
        return s.startswith("Scramble") or s.startswith("Jump")

    def _is_floor_or_ceiling(bout_type):
        s = str(bout_type)
        return ("Floor" in s) or ("Ceiling" in s)

    def _is_pose_orient_label(bout_type):
        # robust against either "Walking + Floor" or "Walking+Floor" naming
        return "+" in str(bout_type)

    # Body translational metrics (column mapping)
    body_cols = {
        "speed": f"speed{kinematic_suffix}",
        "velocity_fwd": f"velocity_forward{kinematic_suffix}",
        "velocity_str": f"velocity_strafe{kinematic_suffix}",
        "accel_mag": f"acceleration_magnitude{kinematic_suffix}",
        "accel_fwd": f"acceleration_forward{kinematic_suffix}",
        "accel_str": f"acceleration_strafe{kinematic_suffix}",
        "jerk_mag": f"jerk_magnitude{kinematic_suffix}",
        "jerk_fwd": f"jerk_forward{kinematic_suffix}",
        "jerk_str": f"jerk_strafe{kinematic_suffix}",
    }

    angular_cols = {
        "angular_vel": f"angular_velocity{kinematic_suffix}",
        "angular_accel": f"angular_acceleration{kinematic_suffix}",
        "angular_jerk": f"angular_jerk{kinematic_suffix}",
    }

    head_cols = {
        "head_velocity": f"head_velocity{kinematic_suffix}",
        "head_acceleration": f"head_acceleration{kinematic_suffix}",
        "head_jerk": f"head_jerk{kinematic_suffix}",
    }

    curvature_col = f"curvature{kinematic_suffix}"

    # Translational legs use leg_part naming
    leg_trans_parts = [
        ("frontleg", "FrontTarsus"),
        ("midleg", "MidTarsus"),
        ("backleg", "BackTarsus"),
    ]

    # Angular legs use pivot + moving naming (mirrors BB helper signature)
    leg_ang_parts = [
        ("frontleg", "FrontJoint", "FrontTarsus"),
        ("midleg", "MidJoint", "MidTarsus"),
        ("backleg", "BackJoint", "BackTarsus"),
    ]

    trans_features = [
        ("velocity", "partvelocity"),
        ("acceleration", "partaccel"),
        ("jerk", "partjerk"),
    ]
    ang_features = [
        ("angvel", "partangular_velocity"),
        ("angaccel", "partangular_accel"),
        ("angjerk", "partangular_jerk"),
    ]

    tw_rows = []

    # --- Load per-individual data from HDF ---
    with pd.HDFStore(temp_store_path, mode="r") as store:
        if kin_key not in store:
            print("No /kin_frames table found; skipping TW kinematic features.")
            return

        # Build list of individuals
        if meta_key in store:
            meta_df = store.select(meta_key, columns=["group", "individual", "sex"]).drop_duplicates()
        elif bouts_key in store:
            meta_df = store.select(bouts_key, columns=["group", "individual", "sex"]).drop_duplicates()
        else:
            meta_df = store.select(kin_key, columns=["group", "individual", "sex"]).drop_duplicates()

        meta_df = meta_df.dropna(subset=["individual"])
        if meta_df.empty:
            print("No individuals found; skipping TW kinematic features.")
            return

        meta_df = meta_df.copy()
        meta_df["individual_raw"] = meta_df["individual"].astype(str)
        meta_df["individual"] = meta_df["individual_raw"].map(
            lambda ind: _collapse_tw_individual_id(ind, individual_to_animal)
        )
        meta_df = meta_df.dropna(subset=["individual"])
        if meta_df.empty:
            print("No individuals found after animal-level collapse; skipping TW kinematic features.")
            return

        member_ids_by_individual = (
            meta_df.groupby("individual", sort=False)["individual_raw"]
            .agg(lambda vals: list(dict.fromkeys([str(v) for v in vals if pd.notna(v)])))
            .to_dict()
        )
        meta_df = meta_df.drop_duplicates(subset=["individual"], keep="first")
        meta_df = meta_df.sort_values(["individual"], kind="stable").reset_index(drop=True)

        if selected_individuals is not None:
            selected_order = [str(x) for x in selected_individuals]
            order_map = {ind: idx for idx, ind in enumerate(selected_order)}
            meta_df = meta_df[meta_df["individual"].isin(order_map)].copy()
            if meta_df.empty:
                if _return_long_rows:
                    return pd.DataFrame()
                print("No selected individuals found; skipping TW kinematic features.")
                return
            meta_df["__ord"] = meta_df["individual"].map(order_map)
            meta_df = meta_df.sort_values(["__ord"], kind="stable").drop(columns=["__ord"]).reset_index(drop=True)

        if selected_individuals is None and int(num_workers) > 1 and len(meta_df) > 1:
            worker_count = max(1, min(int(num_workers), len(meta_df)))
            chunks = _ordered_chunks_for_workers(meta_df["individual"].tolist(), worker_count)
            with Pool(processes=worker_count) as pool:
                chunk_results = pool.map(
                    _tw_chunk_worker,
                    [
                        (temp_store_path, output_dir, float(video_fps), int(min_context_frames), chunk, individual_to_animal)
                        for chunk in chunks
                    ],
                    chunksize=1,
                )
            chunk_frames = [df for df in chunk_results if df is not None and not df.empty]
            if not chunk_frames:
                print("No TW kinematic rows computed; skipping TW kinematic output.")
                return
            tw_long = pd.concat(chunk_frames, ignore_index=True, sort=False)
            if _return_long_rows:
                return tw_long
            return _tw_finalize_sheets_from_long(tw_long, out_path=out_path, save_outputs=save_outputs)

        for _, meta_row in meta_df.iterrows():
            group = meta_row.get("group", np.nan)
            individual = meta_row.get("individual", np.nan)
            sex = meta_row.get("sex", np.nan)
            member_ids = member_ids_by_individual.get(individual, [individual])

            where_str = _make_individual_where_clause(member_ids)
            if not where_str:
                continue

            try:
                kin_ind = store.select(kin_key, where=where_str)
            except Exception:
                continue

            if kin_ind is None or kin_ind.empty:
                continue

            # Required columns for segmentation and context filtering
            required = {"kin_context_valid", "frame_idx", "source_file"}
            if not required.issubset(set(kin_ind.columns)):
                continue

            # Pull bouts for this individual (used for walking segmentation)
            bouts_ind = pd.DataFrame()
            if bouts_key in store:
                try:
                    bouts_ind = store.select(
                        bouts_key,
                        where=where_str,
                        columns=["source_file", "category", "bout_type", "start", "end"],
                    )
                except Exception:
                    bouts_ind = pd.DataFrame()

            # --- Build walking segment frame sets (full/core/start/end) ---
            walk_full_list, walk_core_list, walk_start_list, walk_end_list = [], [], [], []

            if bouts_ind is not None and (not bouts_ind.empty):
                bouts_walk = bouts_ind[
                    (bouts_ind["category"] == "Pose+Orient") &
                    (bouts_ind["bout_type"].astype(str).str.startswith("Walking"))
                ].copy()

                if not bouts_walk.empty:
                    for source_file, kf_file in kin_ind.groupby("source_file", sort=False):
                        # Only walking contexts from this file
                        kf_walk = kf_file[kf_file["kin_context_valid"].astype(str).str.startswith("Walking")]
                        if kf_walk.empty:
                            continue

                        bouts_file = bouts_walk[bouts_walk["source_file"] == source_file].copy()
                        if bouts_file.empty:
                            continue

                        bouts_file = bouts_file.sort_values("start")
                        starts = bouts_file["start"].to_numpy(dtype=int)
                        ends = bouts_file["end"].to_numpy(dtype=int)

                        frames = kf_walk["frame_idx"].to_numpy(dtype=int)
                        idx = np.searchsorted(starts, frames, side="right") - 1
                        valid = (idx >= 0) & (frames <= ends[idx])
                        if not np.any(valid):
                            continue

                        kf_walk_valid = kf_walk.loc[valid].copy()
                        idx_valid = idx[valid]
                        starts_v = starts[idx_valid]
                        ends_v = ends[idx_valid]
                        bout_len = (ends_v - starts_v + 1)
                        offset = frames[valid] - starts_v

                        core_mask = (frames[valid] >= (starts_v + BOUT_TRIM_FRAMES_CORE)) & (frames[valid] <= (ends_v - BOUT_TRIM_FRAMES_CORE))
                        has_start_end = bout_len >= (2 * BOUT_FRAMES_START_END)
                        start_mask = has_start_end & (offset < BOUT_FRAMES_START_END)
                        end_mask = has_start_end & (frames[valid] >= (ends_v - BOUT_FRAMES_START_END + 1))

                        walk_full_list.append(kf_walk_valid)
                        walk_core_list.append(kf_walk_valid.loc[core_mask])
                        walk_start_list.append(kf_walk_valid.loc[start_mask])
                        walk_end_list.append(kf_walk_valid.loc[end_mask])

            walk_full = pd.concat(walk_full_list, ignore_index=True) if walk_full_list else pd.DataFrame()
            walk_core = pd.concat(walk_core_list, ignore_index=True) if walk_core_list else pd.DataFrame()
            walk_start = pd.concat(walk_start_list, ignore_index=True) if walk_start_list else pd.DataFrame()
            walk_end = pd.concat(walk_end_list, ignore_index=True) if walk_end_list else pd.DataFrame()

            # --- Non-walking contexts: compute on full context frames ---
            all_contexts = sorted(kin_ind["kin_context_valid"].astype(str).unique())
            non_walking_contexts = [c for c in all_contexts if not c.startswith("Walking")]

            for bout_type in non_walking_contexts:
                df_ctx = kin_ind[kin_ind["kin_context_valid"].astype(str) == bout_type]
                if df_ctx.empty:
                    continue

                row = {
                    "individual": individual,
                    "group": group,
                    "sex": sex,
                    "bout_type": bout_type,
                    # Jump/Scramble are collapsed contexts but belong in Pose+Orient kinematics.
                    "category": ("Pose+Orient" if (_is_pose_orient_label(bout_type) or _is_special_case(bout_type)) else "Pose"),
                }

                # Guardrail for context-level (non-walking)
                if len(df_ctx) < min_context_frames:
                    tw_rows.append(row)
                    continue

                # Body metrics
                for feat_name, col in body_cols.items():
                    if col not in df_ctx.columns:
                        continue
                    arr = df_ctx[col].to_numpy()
                    mean_val, std_val = _safe_mean_std(arr)
                    med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                    p5, p95 = _safe_p5_p95(arr)
                    row[f"{feat_name}_mean"] = mean_val
                    row[f"{feat_name}_std"] = std_val
                    row[f"{feat_name}_median"] = med_val
                    row[f"{feat_name}_skewness"] = skew_val
                    row[f"{feat_name}_min"] = p5
                    row[f"{feat_name}_max"] = p95

                # Angular metrics (only Floor/Ceiling)
                if _is_floor_or_ceiling(bout_type):
                    for feat_name, col in angular_cols.items():
                        if col not in df_ctx.columns:
                            continue
                        arr = df_ctx[col].to_numpy()
                        mean_val, std_val = _safe_mean_std(arr)
                        med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                        p5, p95 = _safe_p5_p95(arr)
                        row[f"{feat_name}_mean"] = mean_val
                        row[f"{feat_name}_std"] = std_val
                        row[f"{feat_name}_median"] = med_val
                        row[f"{feat_name}_skewness"] = skew_val
                        row[f"{feat_name}_min"] = p5
                        row[f"{feat_name}_max"] = p95

                # Legs + Head: skip for special-case contexts (mirrors BB)
                if not _is_special_case(bout_type):
                    # Translational legs
                    for leg_prefix, leg_part in leg_trans_parts:
                        for out_name, base_feature in trans_features:
                            mean_val, std_val, asym_val = _calculate_leg_kinematics(df_ctx, bout_type, kinematic_suffix, base_feature, leg_part)
                            row[f"{leg_prefix}_{out_name}_mean"] = mean_val
                            row[f"{leg_prefix}_{out_name}_std"] = std_val
                            row[f"{leg_prefix}_{out_name}_asym"] = asym_val

                            p5, p95 = _calculate_leg_kinematics_minmax(
                                df_ctx,
                                bout_type,
                                kinematic_suffix,
                                base_feature,
                                leg_part,
                                use_tail_means=True,
                            )
                            row[f"{leg_prefix}_{out_name}_min"] = p5
                            row[f"{leg_prefix}_{out_name}_max"] = p95

                            med_val, skew_val = _calculate_leg_kinematics_median_skewness(df_ctx, bout_type, kinematic_suffix, base_feature, leg_part, mean_val, std_val)
                            row[f"{leg_prefix}_{out_name}_median"] = med_val
                            row[f"{leg_prefix}_{out_name}_skewness"] = skew_val

                    # Angular legs
                    for leg_prefix, pivot_part, moving_part in leg_ang_parts:
                        for out_name, base_feature in ang_features:
                            mean_val, std_val, asym_val = _calculate_angular_leg_kinematics(df_ctx, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part)
                            row[f"{leg_prefix}_{out_name}_mean"] = mean_val
                            row[f"{leg_prefix}_{out_name}_std"] = std_val
                            row[f"{leg_prefix}_{out_name}_asym"] = asym_val

                            p5, p95 = _calculate_angular_leg_kinematics_minmax(
                                df_ctx,
                                bout_type,
                                kinematic_suffix,
                                base_feature,
                                pivot_part,
                                moving_part,
                                use_tail_means=True,
                            )
                            row[f"{leg_prefix}_{out_name}_min"] = p5
                            row[f"{leg_prefix}_{out_name}_max"] = p95

                            med_val, skew_val = _calculate_angular_leg_kinematics_median_skewness(df_ctx, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part, mean_val, std_val)
                            row[f"{leg_prefix}_{out_name}_median"] = med_val
                            row[f"{leg_prefix}_{out_name}_skewness"] = skew_val

                    # Head metrics (use per-frame derived columns)
                    for feat_name, col in head_cols.items():
                        if col not in df_ctx.columns:
                            continue
                        arr = df_ctx[col].to_numpy()
                        mean_val, std_val = _safe_mean_std(arr)
                        med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                        p5, p95 = _safe_p5_p95(arr)
                        row[f"{feat_name}_mean"] = mean_val
                        row[f"{feat_name}_std"] = std_val
                        row[f"{feat_name}_median"] = med_val
                        row[f"{feat_name}_skewness"] = skew_val
                        row[f"{feat_name}_min"] = p5
                        row[f"{feat_name}_max"] = p95

                tw_rows.append(row)

            # --- Walking contexts: core/start/end/full rules ---
            if not walk_full.empty:
                walking_contexts = sorted(walk_full["kin_context_valid"].astype(str).unique())
                for bout_type in walking_contexts:
                    df_full = walk_full[walk_full["kin_context_valid"].astype(str) == bout_type]
                    df_core = walk_core[walk_core["kin_context_valid"].astype(str) == bout_type] if not walk_core.empty else pd.DataFrame()
                    df_start = walk_start[walk_start["kin_context_valid"].astype(str) == bout_type] if not walk_start.empty else pd.DataFrame()
                    df_end = walk_end[walk_end["kin_context_valid"].astype(str) == bout_type] if not walk_end.empty else pd.DataFrame()

                    row = {
                        "individual": individual,
                        "group": group,
                        "sex": sex,
                        "bout_type": bout_type,
                        "category": "Pose+Orient",
                    }

                    # Guardrails per segment
                    core_ok = (df_core is not None) and (not df_core.empty) and (len(df_core) >= min_context_frames)
                    full_ok = (df_full is not None) and (not df_full.empty) and (len(df_full) >= min_context_frames)

                    # Start/end: compute if any data exists (downstream bout-count filters handle stability)
                    start_ok = (df_start is not None) and (not df_start.empty)
                    end_ok = (df_end is not None) and (not df_end.empty)

                    # Body metrics: core mean/std + core median/skew; full min/max; start/end mean/std
                    for feat_name, col in body_cols.items():
                        if col not in kin_ind.columns:
                            continue

                        if core_ok and col in df_core.columns:
                            arr = df_core[col].to_numpy()
                            mean_val, std_val = _safe_mean_std(arr)
                            med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                            row[f"{feat_name}_core_mean"] = mean_val
                            row[f"{feat_name}_core_std"] = std_val
                            row[f"{feat_name}_median"] = med_val
                            row[f"{feat_name}_skewness"] = skew_val

                        if full_ok and col in df_full.columns:
                            p5, p95 = _safe_p5_p95(df_full[col].to_numpy())
                            row[f"{feat_name}_min"] = p5
                            row[f"{feat_name}_max"] = p95

                        if start_ok and col in df_start.columns:
                            m, s = _safe_mean_std(df_start[col].to_numpy())
                            row[f"{feat_name}_start_mean"] = m
                            row[f"{feat_name}_start_std"] = s

                        if end_ok and col in df_end.columns:
                            m, s = _safe_mean_std(df_end[col].to_numpy())
                            row[f"{feat_name}_end_mean"] = m
                            row[f"{feat_name}_end_std"] = s

                    # Angular metrics: only Floor/Ceiling contexts
                    if _is_floor_or_ceiling(bout_type):
                        for feat_name, col in angular_cols.items():
                            if col not in kin_ind.columns:
                                continue

                            if core_ok and col in df_core.columns:
                                arr = df_core[col].to_numpy()
                                mean_val, std_val = _safe_mean_std(arr)
                                med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                                row[f"{feat_name}_core_mean"] = mean_val
                                row[f"{feat_name}_core_std"] = std_val
                                row[f"{feat_name}_median"] = med_val
                                row[f"{feat_name}_skewness"] = skew_val

                            if full_ok and col in df_full.columns:
                                p5, p95 = _safe_p5_p95(df_full[col].to_numpy())
                                row[f"{feat_name}_min"] = p5
                                row[f"{feat_name}_max"] = p95

                            if start_ok and col in df_start.columns:
                                m, s = _safe_mean_std(df_start[col].to_numpy())
                                row[f"{feat_name}_start_mean"] = m
                                row[f"{feat_name}_start_std"] = s

                            if end_ok and col in df_end.columns:
                                m, s = _safe_mean_std(df_end[col].to_numpy())
                                row[f"{feat_name}_end_mean"] = m
                                row[f"{feat_name}_end_std"] = s

                    # Curvature (BB-style: computed on core)
                    if core_ok and (curvature_col in df_core.columns):
                        arr = df_core[curvature_col].to_numpy()
                        mean_val, std_val = _safe_mean_std(arr)
                        med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                        p5, p95 = _safe_p5_p95(arr)
                        row["curvature_mean"] = mean_val
                        row["curvature_std"] = std_val
                        row["curvature_median"] = med_val
                        row["curvature_skewness"] = skew_val
                        row["curvature_min"] = p5
                        row["curvature_max"] = p95

                    # Legs (core mean/std/asym, start/end mean/std/asym, full min/max, core median/skew)
                    if core_ok:
                        for leg_prefix, leg_part in leg_trans_parts:
                            for out_name, base_feature in trans_features:
                                mean_val, std_val, asym_val = _calculate_leg_kinematics(df_core, bout_type, kinematic_suffix, base_feature, leg_part)
                                row[f"{leg_prefix}_{out_name}_core_mean"] = mean_val
                                row[f"{leg_prefix}_{out_name}_core_std"] = std_val
                                row[f"{leg_prefix}_{out_name}_core_asym"] = asym_val

                                med_val, skew_val = _calculate_leg_kinematics_median_skewness(df_core, bout_type, kinematic_suffix, base_feature, leg_part, mean_val, std_val)
                                row[f"{leg_prefix}_{out_name}_median"] = med_val
                                row[f"{leg_prefix}_{out_name}_skewness"] = skew_val

                                if full_ok:
                                    p5, p95 = _calculate_leg_kinematics_minmax(
                                        df_full,
                                        bout_type,
                                        kinematic_suffix,
                                        base_feature,
                                        leg_part,
                                        use_tail_means=True,
                                    )
                                    row[f"{leg_prefix}_{out_name}_min"] = p5
                                    row[f"{leg_prefix}_{out_name}_max"] = p95

                                if start_ok:
                                    m, s, a = _calculate_leg_kinematics(df_start, bout_type, kinematic_suffix, base_feature, leg_part)
                                    row[f"{leg_prefix}_{out_name}_start_mean"] = m
                                    row[f"{leg_prefix}_{out_name}_start_std"] = s
                                    row[f"{leg_prefix}_{out_name}_start_asym"] = a

                                if end_ok:
                                    m, s, a = _calculate_leg_kinematics(df_end, bout_type, kinematic_suffix, base_feature, leg_part)
                                    row[f"{leg_prefix}_{out_name}_end_mean"] = m
                                    row[f"{leg_prefix}_{out_name}_end_std"] = s
                                    row[f"{leg_prefix}_{out_name}_end_asym"] = a

                        for leg_prefix, pivot_part, moving_part in leg_ang_parts:
                            for out_name, base_feature in ang_features:
                                mean_val, std_val, asym_val = _calculate_angular_leg_kinematics(df_core, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part)
                                row[f"{leg_prefix}_{out_name}_core_mean"] = mean_val
                                row[f"{leg_prefix}_{out_name}_core_std"] = std_val
                                row[f"{leg_prefix}_{out_name}_core_asym"] = asym_val

                                med_val, skew_val = _calculate_angular_leg_kinematics_median_skewness(df_core, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part, mean_val, std_val)
                                row[f"{leg_prefix}_{out_name}_median"] = med_val
                                row[f"{leg_prefix}_{out_name}_skewness"] = skew_val

                                if full_ok:
                                    p5, p95 = _calculate_angular_leg_kinematics_minmax(
                                        df_full,
                                        bout_type,
                                        kinematic_suffix,
                                        base_feature,
                                        pivot_part,
                                        moving_part,
                                        use_tail_means=True,
                                    )
                                    row[f"{leg_prefix}_{out_name}_min"] = p5
                                    row[f"{leg_prefix}_{out_name}_max"] = p95

                                if start_ok:
                                    m, s, a = _calculate_angular_leg_kinematics(df_start, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part)
                                    row[f"{leg_prefix}_{out_name}_start_mean"] = m
                                    row[f"{leg_prefix}_{out_name}_start_std"] = s
                                    row[f"{leg_prefix}_{out_name}_start_asym"] = a

                                if end_ok:
                                    m, s, a = _calculate_angular_leg_kinematics(df_end, bout_type, kinematic_suffix, base_feature, pivot_part, moving_part)
                                    row[f"{leg_prefix}_{out_name}_end_mean"] = m
                                    row[f"{leg_prefix}_{out_name}_end_std"] = s
                                    row[f"{leg_prefix}_{out_name}_end_asym"] = a

                    # Head (core mean/std + median/skew; full min/max; start/end mean/std)
                    for feat_name, col in head_cols.items():
                        if col not in kin_ind.columns:
                            continue

                        if core_ok and col in df_core.columns:
                            arr = df_core[col].to_numpy()
                            mean_val, std_val = _safe_mean_std(arr)
                            med_val, skew_val = _safe_median_skew(arr, mean_val, std_val)
                            row[f"{feat_name}_core_mean"] = mean_val
                            row[f"{feat_name}_core_std"] = std_val
                            row[f"{feat_name}_median"] = med_val
                            row[f"{feat_name}_skewness"] = skew_val

                        if full_ok and col in df_full.columns:
                            p5, p95 = _safe_p5_p95(df_full[col].to_numpy())
                            row[f"{feat_name}_min"] = p5
                            row[f"{feat_name}_max"] = p95

                        if start_ok and col in df_start.columns:
                            m, s = _safe_mean_std(df_start[col].to_numpy())
                            row[f"{feat_name}_start_mean"] = m
                            row[f"{feat_name}_start_std"] = s

                        if end_ok and col in df_end.columns:
                            m, s = _safe_mean_std(df_end[col].to_numpy())
                            row[f"{feat_name}_end_mean"] = m
                            row[f"{feat_name}_end_std"] = s

                    tw_rows.append(row)

    if not tw_rows:
        print("No TW kinematic rows computed; skipping TW kinematic output.")
        return pd.DataFrame() if _return_long_rows else None

    tw_long = pd.DataFrame(tw_rows)
    if _return_long_rows:
        return tw_long
    return _tw_finalize_sheets_from_long(tw_long, out_path=out_path, save_outputs=save_outputs)
