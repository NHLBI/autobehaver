"""Extract and summarize bout-based kinematics in real units.

Minima/maxima use per-bout percentile thresholds. Pass eligibility thresholds explicitly.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .feature_matrix import format_summary_df, save_to_excel_sheets
from .naming import collapse_individual_id


DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES: int = 6
DEFAULT_MIN_BOUT_COUNT_FOR_TRANSITIONS: int = 6

# Kinematic Feature Extraction Settings
BOUT_TRIM_FRAMES_CORE: int = 5
BOUT_FRAMES_START_END: int = BOUT_TRIM_FRAMES_CORE

HEAD_KEYPOINTS = ["Ocelli", "Antennae", "Mouthparts", "EyeLeft", "EyeRight"]

# Input signals shared by the bout-based kinematic calculations
KINEMATIC_SIGNAL_COLS = [
    # Centroid / body translation (mm/s, mm/s^2, mm/s^3)
    "speed_inst",
    "velocity_forward_inst",
    "velocity_strafe_inst",
    "acceleration_speed_inst",
    "acceleration_forward_inst",
    "acceleration_strafe_inst",
    "jerk_speed_inst",
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
        KINEMATIC_SIGNAL_COLS.extend(
            [
                f"{_base}_{_tarsus}Left_inst",
                f"{_base}_{_tarsus}Right_inst",
            ]
        )
    for _base in ("partangular_velocity", "partangular_accel", "partangular_jerk"):
        KINEMATIC_SIGNAL_COLS.extend(
            [
                f"{_base}_{_joint}Left_{_tarsus}Left_inst",
                f"{_base}_{_joint}Right_{_tarsus}Right_inst",
            ]
        )

def _finalize_mean_std(sum_, sum_sq, count_):
    """Calculate means and guarded sample deviations from sum, squared sum, and count.

    Keep an unobserved mean and a sample deviation with fewer than two
    observations missing. Clamp negative variance roundoff to zero.
    """

    sum_ = np.asarray(sum_, dtype="float64")
    sum_sq = np.asarray(sum_sq, dtype="float64")
    count_ = np.asarray(count_, dtype="int64")
    safe_n = np.maximum(count_, 1)
    mean = np.where(count_ > 0, sum_ / safe_n, np.nan)
    # sample variance with ddof=1 when n>1; guard for n<=1
    var_num = np.maximum(sum_sq - (np.square(sum_) / safe_n), 0.0)
    denom = np.where(count_ > 1, count_ - 1, 1)
    std = np.where(count_ > 1, np.sqrt(var_num / denom), np.nan)
    if mean.ndim == 0:
        return mean.item(), std.item()
    return mean, std


# Core implementation helpers
def _safe_skewness(mean_val, median_val, std_val):
    """Return (mean - median) / std, or NaN for NaN inputs or std <= 1e-10."""
    if np.isnan(mean_val) or np.isnan(median_val) or np.isnan(std_val):
        return np.nan
    if std_val == 0 or std_val <= 1e-10:  # Avoid division by zero
        return np.nan
    return (mean_val - median_val) / std_val




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

def _calculate_leg_kinematics(bout_features, kinematic_suffix, base_feature, leg_part):
    """Calculate paired-leg mean and deviation."""
    mean_val, std_val = np.nan, np.nan
    
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
        
            
    return mean_val, std_val

def _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, base_feature, pivot_part, moving_part):
    """Calculate paired angular-leg statistics using the pivot/moving-part column
    convention.
    """
    mean_val, std_val = np.nan, np.nan
    
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
        
            
    return mean_val, std_val

def compute_head_movement_frame_signals(df: pd.DataFrame, *, kinematic_suffix: str = '_inst'):
    """Compute framewise head velocity, acceleration, and jerk in mm/s, mm/s^2, and
    mm/s^3.

    Drop the lowest-likelihood head keypoint each frame and average the remaining
    signals.
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

def _checked_head_frame_signals(frame_signals, length):
    values = tuple(np.asarray(x) for x in frame_signals)
    if len(values) != 3 or any(x.ndim != 1 or len(x) != length for x in values):
        raise ValueError("Head frame signals must match the selected window.")
    return values


def compute_head_movement_features(bout_features, kinematic_suffix="_inst", *, frame_signals=None):
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
    if frame_signals is not None:
        values = _checked_head_frame_signals(frame_signals, len(bout_features))
        result = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for arr in values:
                result.extend((np.nanmean(arr), np.nanstd(arr, ddof=1)
                               if np.sum(~np.isnan(arr)) > 1 else np.nan))
        return tuple(result)
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

    likelihoods = bout_features[likelihood_cols].to_numpy(dtype=float, copy=True)
    velocities = bout_features[velocity_cols].to_numpy(dtype=float, copy=True)
    accels = bout_features[accel_cols].to_numpy(dtype=float, copy=True)

    # Jerk optional-safe
    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy(dtype=float, copy=True)
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
    """Return median and skewness for leg kinematics.
    Combines left and right legs, computes median, then skewness using provided
    mean/std.
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
    """Return median and skewness for angular leg kinematics.
    Combines left and right legs, computes median, then skewness using provided
    mean/std.
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
    jerk_mean, jerk_std,
    *, frame_signals=None
):
    """Calculate head-signal medians and (mean - median) / std statistics.

    Missing jerk inputs leave jerk outputs unavailable.
    """
    if frame_signals is not None:
        values = _checked_head_frame_signals(frame_signals, len(bout_features))
        result = []
        for arr, mean, std in zip(values, (velocity_mean, acceleration_mean, jerk_mean),
                                  (velocity_std, acceleration_std, jerk_std)):
            median = np.nanmedian(arr) if np.any(~np.isnan(arr)) else np.nan
            result.extend((median, _safe_skewness(mean, median, std)))
        return tuple(result)
    head_keypoints = HEAD_KEYPOINTS
    likelihood_cols = [f'{kp}_likelihood' for kp in head_keypoints]
    velocity_cols = [f'partvelocity_{kp}{kinematic_suffix}' for kp in head_keypoints]
    accel_cols = [f'partaccel_{kp}{kinematic_suffix}' for kp in head_keypoints]
    jerk_cols = [f'partjerk_{kp}{kinematic_suffix}' for kp in head_keypoints]

    if not all(col in bout_features.columns for col in likelihood_cols + velocity_cols + accel_cols):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    likelihoods = bout_features[likelihood_cols].to_numpy(dtype=float, copy=True)
    velocities = bout_features[velocity_cols].to_numpy(dtype=float, copy=True)
    accels = bout_features[accel_cols].to_numpy(dtype=float, copy=True)

    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy(dtype=float, copy=True)
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
):
    """Return min/max for paired-leg kinematics.

    Return per-bout percentile thresholds (p05/p95).
    """
    min_val, max_val = np.nan, np.nan

    left_col = f'{base_feature}_{leg_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{leg_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if not combined.empty:
            arr = combined.to_numpy()
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
):
    """Return min/max for paired angular-leg kinematics.

    Return per-bout percentile thresholds (p05/p95).
    """
    min_val, max_val = np.nan, np.nan

    left_col = f'{base_feature}_{pivot_part}Left_{moving_part}Left{kinematic_suffix}'
    right_col = f'{base_feature}_{pivot_part}Right_{moving_part}Right{kinematic_suffix}'

    if all(c in bout_features.columns for c in [left_col, right_col]):
        combined = pd.concat([bout_features[left_col], bout_features[right_col]]).dropna()
        if not combined.empty:
            arr = combined.to_numpy()
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
    frame_signals=None,
):
    """Compute head min/max features.

    Return per-bout percentile thresholds (p05/p95).

    Jerk is optional-safe: if jerk columns are missing, jerk min/max return NaN.
    """
    if frame_signals is not None:
        values = _checked_head_frame_signals(frame_signals, len(bout_features))
        result = []
        for arr in values:
            stats = _safe_percentile_thresholds_from_array(arr,
                pmin=percentile_min, pmax=percentile_max)
            result.extend(stats)
        return tuple(result)
    head_keypoints = HEAD_KEYPOINTS
    likelihood_cols = [f'{kp}_likelihood' for kp in head_keypoints]
    velocity_cols = [f'partvelocity_{kp}{kinematic_suffix}' for kp in head_keypoints]
    accel_cols = [f'partaccel_{kp}{kinematic_suffix}' for kp in head_keypoints]
    jerk_cols = [f'partjerk_{kp}{kinematic_suffix}' for kp in head_keypoints]

    if not all(col in bout_features.columns for col in likelihood_cols + velocity_cols + accel_cols):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    likelihoods = bout_features[likelihood_cols].to_numpy(dtype=float, copy=True)
    velocities = bout_features[velocity_cols].to_numpy(dtype=float, copy=True)
    accels = bout_features[accel_cols].to_numpy(dtype=float, copy=True)

    if all(col in bout_features.columns for col in jerk_cols):
        jerks = bout_features[jerk_cols].to_numpy(dtype=float, copy=True)
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
        head_velocity_min, head_velocity_max = _safe_percentile_thresholds_from_array(
            frame_velocities, pmin=percentile_min, pmax=percentile_max
        )
    else:
        head_velocity_min, head_velocity_max = np.nan, np.nan

    acc_clean = frame_accels[~np.isnan(frame_accels)]
    if len(acc_clean) >= 1:
        head_acceleration_min, head_acceleration_max = _safe_percentile_thresholds_from_array(
            frame_accels, pmin=percentile_min, pmax=percentile_max
        )
    else:
        head_acceleration_min, head_acceleration_max = np.nan, np.nan

    jerk_clean = frame_jerks[~np.isnan(frame_jerks)]
    if len(jerk_clean) >= 1:
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

def extract_bout_features_simple(aligned_poscalc_df, bouts, individual_id, group_label, sex_label, source_file, category, kinematic_suffix='_inst', trim_frames_core=BOUT_TRIM_FRAMES_CORE, trim_frames_start_end=None, *, head_frame_signals=None):
    """Extract orientation-eligible bout kinematics.

    Walking means, deviations, medians, and skewness use a trimmed core; start/end
    statistics require sufficiently long bouts. Non-walking summaries and bout-based
    percentile minima/maxima use full qualifying bouts.
    """
    def _safe_mean(df, col): return df[col].mean() if col in df.columns else np.nan
    def _safe_std(df, col):  return df[col].std()  if col in df.columns else np.nan
    def _safe_percentile(df, col, percentiles=[5, 95]):
        """Return the requested percentile threshold for an available qualifying bout
        signal.
        """
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
        """Return the column median or NaN when the column is absent."""
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
    if head_frame_signals is None:
        head_frame_signals = compute_head_movement_frame_signals(
            aligned_poscalc_df, kinematic_suffix=kinematic_suffix)
    head_frame_signals = _checked_head_frame_signals(head_frame_signals, len(aligned_poscalc_df))

    for bout in bouts:
        if not bout.get('complete', True):
            continue
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

            # Walking start/end windows require the configured bout length.
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
        head_signals = tuple(x[start_i:end_i+1] for x in head_frame_signals)
        head_full_signals = tuple(x[start:end+1] for x in head_frame_signals)
        head_start_signals = tuple(x[start:start+trim_frames_start_end] for x in head_frame_signals)
        head_end_signals = tuple(x[end-trim_frames_start_end+1:end+1] for x in head_frame_signals)
        has_min_max = bout_length >= 20  # Guardrail: require at least 20 frames for BB min/max

        # 1. Initialize all kinematic variables to NaN
        speed_mean, speed_std = np.nan, np.nan
        accel_speed_mean, accel_speed_std = np.nan, np.nan
        velocity_fwd_mean, velocity_fwd_std = np.nan, np.nan
        velocity_str_mean, velocity_str_std = np.nan, np.nan
        accel_fwd_mean, accel_fwd_std = np.nan, np.nan
        accel_str_mean, accel_str_std = np.nan, np.nan
        jerk_speed_mean, jerk_speed_std = np.nan, np.nan
        jerk_fwd_mean, jerk_fwd_std = np.nan, np.nan
        jerk_str_mean, jerk_str_std = np.nan, np.nan
        angular_vel_mean, angular_vel_std = np.nan, np.nan
        angular_accel_mean, angular_accel_std = np.nan, np.nan
        angular_jerk_mean, angular_jerk_std = np.nan, np.nan
        frontleg_velocity_mean, frontleg_velocity_std = np.nan, np.nan
        frontleg_acceleration_mean, frontleg_acceleration_std = np.nan, np.nan
        frontleg_jerk_mean, frontleg_jerk_std = np.nan, np.nan
        frontleg_angvel_mean, frontleg_angvel_std = np.nan, np.nan
        frontleg_angaccel_mean, frontleg_angaccel_std = np.nan, np.nan
        frontleg_angjerk_mean, frontleg_angjerk_std = np.nan, np.nan
        midleg_velocity_mean, midleg_velocity_std = np.nan, np.nan
        midleg_acceleration_mean, midleg_acceleration_std = np.nan, np.nan
        midleg_jerk_mean, midleg_jerk_std = np.nan, np.nan
        midleg_angvel_mean, midleg_angvel_std = np.nan, np.nan
        midleg_angaccel_mean, midleg_angaccel_std = np.nan, np.nan
        midleg_angjerk_mean, midleg_angjerk_std = np.nan, np.nan
        backleg_velocity_mean, backleg_velocity_std = np.nan, np.nan
        backleg_acceleration_mean, backleg_acceleration_std = np.nan, np.nan
        backleg_jerk_mean, backleg_jerk_std = np.nan, np.nan
        backleg_angvel_mean, backleg_angvel_std = np.nan, np.nan
        backleg_angaccel_mean, backleg_angaccel_std = np.nan, np.nan
        backleg_angjerk_mean, backleg_angjerk_std = np.nan, np.nan
        head_velocity_mean, head_velocity_std = np.nan, np.nan
        head_acceleration_mean, head_acceleration_std = np.nan, np.nan
        head_jerk_mean, head_jerk_std = np.nan, np.nan


        # Core/start/end variables (for walking bouts)
        speed_core_mean, speed_core_std = np.nan, np.nan
        accel_speed_core_mean, accel_speed_core_std = np.nan, np.nan
        velocity_fwd_core_mean, velocity_fwd_core_std = np.nan, np.nan
        velocity_str_core_mean, velocity_str_core_std = np.nan, np.nan
        accel_fwd_core_mean, accel_fwd_core_std = np.nan, np.nan
        accel_str_core_mean, accel_str_core_std = np.nan, np.nan
        jerk_speed_core_mean, jerk_speed_core_std = np.nan, np.nan
        jerk_fwd_core_mean, jerk_fwd_core_std = np.nan, np.nan
        jerk_str_core_mean, jerk_str_core_std = np.nan, np.nan
        angular_vel_core_mean, angular_vel_core_std = np.nan, np.nan
        angular_accel_core_mean, angular_accel_core_std = np.nan, np.nan
        angular_jerk_core_mean, angular_jerk_core_std = np.nan, np.nan
        frontleg_velocity_core_mean, frontleg_velocity_core_std = np.nan, np.nan
        frontleg_acceleration_core_mean, frontleg_acceleration_core_std = np.nan, np.nan
        frontleg_jerk_core_mean, frontleg_jerk_core_std = np.nan, np.nan
        frontleg_angvel_core_mean, frontleg_angvel_core_std = np.nan, np.nan
        frontleg_angaccel_core_mean, frontleg_angaccel_core_std = np.nan, np.nan
        frontleg_angjerk_core_mean, frontleg_angjerk_core_std = np.nan, np.nan
        midleg_velocity_core_mean, midleg_velocity_core_std = np.nan, np.nan
        midleg_acceleration_core_mean, midleg_acceleration_core_std = np.nan, np.nan
        midleg_jerk_core_mean, midleg_jerk_core_std = np.nan, np.nan
        midleg_angvel_core_mean, midleg_angvel_core_std = np.nan, np.nan
        midleg_angaccel_core_mean, midleg_angaccel_core_std = np.nan, np.nan
        midleg_angjerk_core_mean, midleg_angjerk_core_std = np.nan, np.nan
        backleg_velocity_core_mean, backleg_velocity_core_std = np.nan, np.nan
        backleg_acceleration_core_mean, backleg_acceleration_core_std = np.nan, np.nan
        backleg_jerk_core_mean, backleg_jerk_core_std = np.nan, np.nan
        backleg_angvel_core_mean, backleg_angvel_core_std = np.nan, np.nan
        backleg_angaccel_core_mean, backleg_angaccel_core_std = np.nan, np.nan
        backleg_angjerk_core_mean, backleg_angjerk_core_std = np.nan, np.nan
        head_velocity_core_mean, head_velocity_core_std = np.nan, np.nan
        head_acceleration_core_mean, head_acceleration_core_std = np.nan, np.nan
        head_jerk_core_mean, head_jerk_core_std = np.nan, np.nan


        # Initialize start/end kinematic variables
        speed_start_mean, speed_start_std = np.nan, np.nan
        speed_end_mean, speed_end_std = np.nan, np.nan
        accel_speed_start_mean, accel_speed_start_std = np.nan, np.nan
        accel_speed_end_mean, accel_speed_end_std = np.nan, np.nan
        velocity_fwd_start_mean, velocity_fwd_start_std = np.nan, np.nan
        velocity_fwd_end_mean, velocity_fwd_end_std = np.nan, np.nan
        velocity_str_start_mean, velocity_str_start_std = np.nan, np.nan
        velocity_str_end_mean, velocity_str_end_std = np.nan, np.nan
        accel_fwd_start_mean, accel_fwd_start_std = np.nan, np.nan
        accel_fwd_end_mean, accel_fwd_end_std = np.nan, np.nan
        accel_str_start_mean, accel_str_start_std = np.nan, np.nan
        accel_str_end_mean, accel_str_end_std = np.nan, np.nan
        jerk_speed_start_mean, jerk_speed_start_std = np.nan, np.nan
        jerk_speed_end_mean, jerk_speed_end_std = np.nan, np.nan
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
        frontleg_velocity_start_mean, frontleg_velocity_start_std = np.nan, np.nan
        frontleg_velocity_end_mean, frontleg_velocity_end_std = np.nan, np.nan
        frontleg_acceleration_start_mean, frontleg_acceleration_start_std = np.nan, np.nan
        frontleg_acceleration_end_mean, frontleg_acceleration_end_std = np.nan, np.nan
        frontleg_jerk_start_mean, frontleg_jerk_start_std = np.nan, np.nan
        frontleg_jerk_end_mean, frontleg_jerk_end_std = np.nan, np.nan
        frontleg_angvel_start_mean, frontleg_angvel_start_std = np.nan, np.nan
        frontleg_angvel_end_mean, frontleg_angvel_end_std = np.nan, np.nan
        frontleg_angaccel_start_mean, frontleg_angaccel_start_std = np.nan, np.nan
        frontleg_angaccel_end_mean, frontleg_angaccel_end_std = np.nan, np.nan
        frontleg_angjerk_start_mean, frontleg_angjerk_start_std = np.nan, np.nan
        frontleg_angjerk_end_mean, frontleg_angjerk_end_std = np.nan, np.nan

        midleg_velocity_start_mean, midleg_velocity_start_std = np.nan, np.nan
        midleg_velocity_end_mean, midleg_velocity_end_std = np.nan, np.nan
        midleg_acceleration_start_mean, midleg_acceleration_start_std = np.nan, np.nan
        midleg_acceleration_end_mean, midleg_acceleration_end_std = np.nan, np.nan
        midleg_jerk_start_mean, midleg_jerk_start_std = np.nan, np.nan
        midleg_jerk_end_mean, midleg_jerk_end_std = np.nan, np.nan
        midleg_angvel_start_mean, midleg_angvel_start_std = np.nan, np.nan
        midleg_angvel_end_mean, midleg_angvel_end_std = np.nan, np.nan
        midleg_angaccel_start_mean, midleg_angaccel_start_std = np.nan, np.nan
        midleg_angaccel_end_mean, midleg_angaccel_end_std = np.nan, np.nan
        midleg_angjerk_start_mean, midleg_angjerk_start_std = np.nan, np.nan
        midleg_angjerk_end_mean, midleg_angjerk_end_std = np.nan, np.nan

        backleg_velocity_start_mean, backleg_velocity_start_std = np.nan, np.nan
        backleg_velocity_end_mean, backleg_velocity_end_std = np.nan, np.nan
        backleg_acceleration_start_mean, backleg_acceleration_start_std = np.nan, np.nan
        backleg_acceleration_end_mean, backleg_acceleration_end_std = np.nan, np.nan
        backleg_jerk_start_mean, backleg_jerk_start_std = np.nan, np.nan
        backleg_jerk_end_mean, backleg_jerk_end_std = np.nan, np.nan
        backleg_angvel_start_mean, backleg_angvel_start_std = np.nan, np.nan
        backleg_angvel_end_mean, backleg_angvel_end_std = np.nan, np.nan
        backleg_angaccel_start_mean, backleg_angaccel_start_std = np.nan, np.nan
        backleg_angaccel_end_mean, backleg_angaccel_end_std = np.nan, np.nan
        backleg_angjerk_start_mean, backleg_angjerk_start_std = np.nan, np.nan
        backleg_angjerk_end_mean, backleg_angjerk_end_std = np.nan, np.nan

        # Head start/end
        head_velocity_start_mean, head_velocity_start_std = np.nan, np.nan
        head_velocity_end_mean, head_velocity_end_std = np.nan, np.nan
        head_acceleration_start_mean, head_acceleration_start_std = np.nan, np.nan
        head_acceleration_end_mean, head_acceleration_end_std = np.nan, np.nan
        head_jerk_start_mean, head_jerk_start_std = np.nan, np.nan
        head_jerk_end_mean, head_jerk_end_std = np.nan, np.nan


        # Bout-based min/max use percentile thresholds from full qualifying bouts.
        speed_min, speed_max = np.nan, np.nan
        accel_speed_min, accel_speed_max = np.nan, np.nan
        velocity_fwd_min, velocity_fwd_max = np.nan, np.nan
        velocity_str_min, velocity_str_max = np.nan, np.nan
        accel_fwd_min, accel_fwd_max = np.nan, np.nan
        accel_str_min, accel_str_max = np.nan, np.nan
        jerk_speed_min, jerk_speed_max = np.nan, np.nan
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
        accel_speed_median, accel_speed_skewness = np.nan, np.nan
        velocity_fwd_median, velocity_fwd_skewness = np.nan, np.nan
        velocity_str_median, velocity_str_skewness = np.nan, np.nan
        accel_fwd_median, accel_fwd_skewness = np.nan, np.nan
        accel_str_median, accel_str_skewness = np.nan, np.nan
        jerk_speed_median, jerk_speed_skewness = np.nan, np.nan
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
            accel_speed_core_mean = _safe_mean(bout_features, f'acceleration_speed{kinematic_suffix}')
            accel_speed_core_std = _safe_std(bout_features, f'acceleration_speed{kinematic_suffix}')
            accel_fwd_core_mean = _safe_mean(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_fwd_core_std = _safe_std(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_str_core_mean = _safe_mean(bout_features, f'acceleration_strafe{kinematic_suffix}')
            accel_str_core_std = _safe_std(bout_features, f'acceleration_strafe{kinematic_suffix}')
            jerk_speed_core_mean = _safe_mean(bout_features, f'jerk_speed{kinematic_suffix}')
            jerk_speed_core_std = _safe_std(bout_features, f'jerk_speed{kinematic_suffix}')
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
                (frontleg_velocity_core_mean, frontleg_velocity_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                (frontleg_acceleration_core_mean, frontleg_acceleration_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'FrontTarsus')
                (frontleg_jerk_core_mean, frontleg_jerk_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'FrontTarsus')
                (frontleg_angvel_core_mean, frontleg_angvel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                (frontleg_angaccel_core_mean, frontleg_angaccel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                (frontleg_angjerk_core_mean, frontleg_angjerk_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                
                head_velocity_core_mean, head_velocity_core_std, head_acceleration_core_mean, head_acceleration_core_std, head_jerk_core_mean, head_jerk_core_std = compute_head_movement_features(bout_features, kinematic_suffix, frame_signals=head_signals)

            # Middle Legs
            if not is_wall_up and not is_wall_down and not is_special_case:
                (midleg_velocity_core_mean, midleg_velocity_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'MidTarsus')
                (midleg_acceleration_core_mean, midleg_acceleration_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'MidTarsus')
                (midleg_jerk_core_mean, midleg_jerk_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'MidTarsus')
                (midleg_angvel_core_mean, midleg_angvel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                (midleg_angaccel_core_mean, midleg_angaccel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                (midleg_angjerk_core_mean, midleg_angjerk_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

            # Back Legs
            if not is_wall_up and not is_special_case:
                (backleg_velocity_core_mean, backleg_velocity_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'BackTarsus')
                (backleg_acceleration_core_mean, backleg_acceleration_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'BackTarsus')
                (backleg_jerk_core_mean, backleg_jerk_core_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'BackTarsus')
                (backleg_angvel_core_mean, backleg_angvel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                (backleg_angaccel_core_mean, backleg_angaccel_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                (backleg_angjerk_core_mean, backleg_angjerk_core_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')           

                # Compute curvature features (walking only, using core window)
                curvature_mean = _safe_mean(bout_features, 'curvature_inst')
                curvature_median = _safe_median(bout_features, 'curvature_inst')
                curvature_std = _safe_std(bout_features, 'curvature_inst')
                curvature_skewness = _safe_skewness(curvature_mean, curvature_median, curvature_std)
                
                # Curvature min/max use full-bout percentile thresholds with the same length guard.
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

                        accel_speed_start_mean = _safe_mean(start_slice, f'acceleration_speed{kinematic_suffix}')
                        accel_speed_start_std  = _safe_std(start_slice, f'acceleration_speed{kinematic_suffix}')
                        accel_speed_end_mean   = _safe_mean(end_slice, f'acceleration_speed{kinematic_suffix}')
                        accel_speed_end_std    = _safe_std(end_slice, f'acceleration_speed{kinematic_suffix}')

                        accel_fwd_start_mean = _safe_mean(start_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_start_std  = _safe_std(start_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_end_mean   = _safe_mean(end_slice, f'acceleration_forward{kinematic_suffix}')
                        accel_fwd_end_std    = _safe_std(end_slice, f'acceleration_forward{kinematic_suffix}')

                        accel_str_start_mean = _safe_mean(start_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_start_std  = _safe_std(start_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_end_mean   = _safe_mean(end_slice, f'acceleration_strafe{kinematic_suffix}')
                        accel_str_end_std    = _safe_std(end_slice, f'acceleration_strafe{kinematic_suffix}')

                        jerk_speed_start_mean = _safe_mean(start_slice, f'jerk_speed{kinematic_suffix}')
                        jerk_speed_start_std  = _safe_std(start_slice, f'jerk_speed{kinematic_suffix}')
                        jerk_speed_end_mean   = _safe_mean(end_slice, f'jerk_speed{kinematic_suffix}')
                        jerk_speed_end_std    = _safe_std(end_slice, f'jerk_speed{kinematic_suffix}')

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
                        (frontleg_velocity_start_mean, frontleg_velocity_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                        (frontleg_velocity_end_mean,   frontleg_velocity_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partvelocity', 'FrontTarsus')

                        (frontleg_acceleration_start_mean, frontleg_acceleration_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partaccel', 'FrontTarsus')
                        (frontleg_acceleration_end_mean,   frontleg_acceleration_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partaccel', 'FrontTarsus')

                        (frontleg_jerk_start_mean, frontleg_jerk_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partjerk', 'FrontTarsus')
                        (frontleg_jerk_end_mean,   frontleg_jerk_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partjerk', 'FrontTarsus')

                        (frontleg_angvel_start_mean, frontleg_angvel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angvel_end_mean,   frontleg_angvel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')

                        (frontleg_angaccel_start_mean, frontleg_angaccel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angaccel_end_mean,   frontleg_angaccel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')

                        (frontleg_angjerk_start_mean, frontleg_angjerk_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                        (frontleg_angjerk_end_mean,   frontleg_angjerk_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')

                        head_velocity_start_mean, head_velocity_start_std, head_acceleration_start_mean, head_acceleration_start_std, head_jerk_start_mean, head_jerk_start_std = compute_head_movement_features(start_slice, kinematic_suffix, frame_signals=head_start_signals)
                        head_velocity_end_mean, head_velocity_end_std, head_acceleration_end_mean, head_acceleration_end_std, head_jerk_end_mean, head_jerk_end_std = compute_head_movement_features(end_slice, kinematic_suffix, frame_signals=head_end_signals)

                    # Middle legs
                    if not is_wall_up and not is_wall_down and not is_special_case:
                        (midleg_velocity_start_mean, midleg_velocity_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partvelocity', 'MidTarsus')
                        (midleg_velocity_end_mean,   midleg_velocity_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partvelocity', 'MidTarsus')

                        (midleg_acceleration_start_mean, midleg_acceleration_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partaccel', 'MidTarsus')
                        (midleg_acceleration_end_mean,   midleg_acceleration_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partaccel', 'MidTarsus')

                        (midleg_jerk_start_mean, midleg_jerk_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partjerk', 'MidTarsus')
                        (midleg_jerk_end_mean,   midleg_jerk_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partjerk', 'MidTarsus')

                        (midleg_angvel_start_mean, midleg_angvel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                        (midleg_angvel_end_mean,   midleg_angvel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')

                        (midleg_angaccel_start_mean, midleg_angaccel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                        (midleg_angaccel_end_mean,   midleg_angaccel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')

                        (midleg_angjerk_start_mean, midleg_angjerk_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')
                        (midleg_angjerk_end_mean,   midleg_angjerk_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

                    # Back legs
                    if not is_wall_up and not is_special_case:
                        (backleg_velocity_start_mean, backleg_velocity_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partvelocity', 'BackTarsus')
                        (backleg_velocity_end_mean,   backleg_velocity_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partvelocity', 'BackTarsus')

                        (backleg_acceleration_start_mean, backleg_acceleration_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partaccel', 'BackTarsus')
                        (backleg_acceleration_end_mean,   backleg_acceleration_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partaccel', 'BackTarsus')

                        (backleg_jerk_start_mean, backleg_jerk_start_std) = _calculate_leg_kinematics(start_slice, kinematic_suffix, 'partjerk', 'BackTarsus')
                        (backleg_jerk_end_mean,   backleg_jerk_end_std)   = _calculate_leg_kinematics(end_slice,   kinematic_suffix, 'partjerk', 'BackTarsus')

                        (backleg_angvel_start_mean, backleg_angvel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                        (backleg_angvel_end_mean,   backleg_angvel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')

                        (backleg_angaccel_start_mean, backleg_angaccel_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                        (backleg_angaccel_end_mean,   backleg_angaccel_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')

                        (backleg_angjerk_start_mean, backleg_angjerk_start_std) = _calculate_angular_leg_kinematics(start_slice, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')
                        (backleg_angjerk_end_mean,   backleg_angjerk_end_std)   = _calculate_angular_leg_kinematics(end_slice,   kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')
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
                accel_speed_mean = _safe_mean(bout_features, f'acceleration_speed{kinematic_suffix}')
                accel_speed_std = _safe_std(bout_features, f'acceleration_speed{kinematic_suffix}')
                accel_fwd_mean = _safe_mean(bout_features, f'acceleration_forward{kinematic_suffix}')
                accel_fwd_std = _safe_std(bout_features, f'acceleration_forward{kinematic_suffix}')
                accel_str_mean = _safe_mean(bout_features, f'acceleration_strafe{kinematic_suffix}')
                accel_str_std = _safe_std(bout_features, f'acceleration_strafe{kinematic_suffix}')
                jerk_speed_mean = _safe_mean(bout_features, f'jerk_speed{kinematic_suffix}')
                jerk_speed_std = _safe_std(bout_features, f'jerk_speed{kinematic_suffix}')
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
                (frontleg_velocity_mean, frontleg_velocity_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'FrontTarsus')
                (frontleg_acceleration_mean, frontleg_acceleration_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'FrontTarsus')
                (frontleg_jerk_mean, frontleg_jerk_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'FrontTarsus')
                (frontleg_angvel_mean, frontleg_angvel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'FrontJoint', 'FrontTarsus')
                (frontleg_angaccel_mean, frontleg_angaccel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'FrontJoint', 'FrontTarsus')
                (frontleg_angjerk_mean, frontleg_angjerk_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'FrontJoint', 'FrontTarsus')
                
                head_velocity_mean, head_velocity_std, head_acceleration_mean, head_acceleration_std, head_jerk_mean, head_jerk_std = compute_head_movement_features(bout_features, kinematic_suffix, frame_signals=head_signals)

            # Middle Legs
            if not is_wall_up and not is_wall_down and not is_special_case:
                (midleg_velocity_mean, midleg_velocity_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'MidTarsus')
                (midleg_acceleration_mean, midleg_acceleration_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'MidTarsus')
                (midleg_jerk_mean, midleg_jerk_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'MidTarsus')
                (midleg_angvel_mean, midleg_angvel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'MidJoint', 'MidTarsus')
                (midleg_angaccel_mean, midleg_angaccel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'MidJoint', 'MidTarsus')
                (midleg_angjerk_mean, midleg_angjerk_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'MidJoint', 'MidTarsus')

            # Back Legs
            if not is_wall_up and not is_special_case:
                (backleg_velocity_mean, backleg_velocity_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partvelocity', 'BackTarsus')
                (backleg_acceleration_mean, backleg_acceleration_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partaccel', 'BackTarsus')
                (backleg_jerk_mean, backleg_jerk_std) = _calculate_leg_kinematics(bout_features, kinematic_suffix, 'partjerk', 'BackTarsus')
                (backleg_angvel_mean, backleg_angvel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_velocity', 'BackJoint', 'BackTarsus')
                (backleg_angaccel_mean, backleg_angaccel_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_accel', 'BackJoint', 'BackTarsus')
                (backleg_angjerk_mean, backleg_angjerk_std) = _calculate_angular_leg_kinematics(bout_features, kinematic_suffix, 'partangular_jerk', 'BackJoint', 'BackTarsus')

        # Calculate bout-based percentile min/max on full qualifying bouts.
        if has_min_max:
            # Body movement (same conditions as mean/std)
            if not is_wall_up and not is_wall_down:
                speed_min, speed_max = _safe_percentile(bout_features_full, f'speed{kinematic_suffix}')
                velocity_fwd_min, velocity_fwd_max = _safe_percentile(bout_features_full, f'velocity_forward{kinematic_suffix}')
                velocity_str_min, velocity_str_max = _safe_percentile(bout_features_full, f'velocity_strafe{kinematic_suffix}')
                accel_speed_min, accel_speed_max = _safe_percentile(bout_features_full, f'acceleration_speed{kinematic_suffix}')
                accel_fwd_min, accel_fwd_max = _safe_percentile(bout_features_full, f'acceleration_forward{kinematic_suffix}')
                accel_str_min, accel_str_max = _safe_percentile(bout_features_full, f'acceleration_strafe{kinematic_suffix}')
                jerk_speed_min, jerk_speed_max = _safe_percentile(bout_features_full, f'jerk_speed{kinematic_suffix}')
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
                
                head_velocity_min, head_velocity_max, head_acceleration_min, head_acceleration_max, head_jerk_min, head_jerk_max = compute_head_movement_features_minmax(bout_features_full, kinematic_suffix, frame_signals=head_full_signals)

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
            accel_speed_median = _safe_median(bout_features, f'acceleration_speed{kinematic_suffix}')
            accel_speed_skewness = _safe_skewness(accel_speed_mean if not is_walking else accel_speed_core_mean, accel_speed_median, accel_speed_std if not is_walking else accel_speed_core_std)
            accel_fwd_median = _safe_median(bout_features, f'acceleration_forward{kinematic_suffix}')
            accel_fwd_skewness = _safe_skewness(accel_fwd_mean if not is_walking else accel_fwd_core_mean, accel_fwd_median, accel_fwd_std if not is_walking else accel_fwd_core_std)
            accel_str_median = _safe_median(bout_features, f'acceleration_strafe{kinematic_suffix}')
            accel_str_skewness = _safe_skewness(accel_str_mean if not is_walking else accel_str_core_mean, accel_str_median, accel_str_std if not is_walking else accel_str_core_std)
            jerk_speed_median = _safe_median(bout_features, f'jerk_speed{kinematic_suffix}')
            jerk_speed_skewness = _safe_skewness(jerk_speed_mean if not is_walking else jerk_speed_core_mean, jerk_speed_median, jerk_speed_std if not is_walking else jerk_speed_core_std)
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
                head_jerk_std if not is_walking else head_jerk_core_std,
                frame_signals=head_signals
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
            'accel_speed_core_mean': accel_speed_core_mean, 'accel_speed_core_std': accel_speed_core_std,
            'accel_fwd_core_mean': accel_fwd_core_mean, 'accel_fwd_core_std': accel_fwd_core_std,
            'accel_str_core_mean': accel_str_core_mean, 'accel_str_core_std': accel_str_core_std,
            'jerk_speed_core_mean': jerk_speed_core_mean, 'jerk_speed_core_std': jerk_speed_core_std,
            'jerk_fwd_core_mean': jerk_fwd_core_mean, 'jerk_fwd_core_std': jerk_fwd_core_std,
            'jerk_str_core_mean': jerk_str_core_mean, 'jerk_str_core_std': jerk_str_core_std,
            'angular_vel_core_mean': angular_vel_core_mean, 'angular_vel_core_std': angular_vel_core_std,
            'angular_accel_core_mean': angular_accel_core_mean, 'angular_accel_core_std': angular_accel_core_std,
            'angular_jerk_core_mean': angular_jerk_core_mean, 'angular_jerk_core_std': angular_jerk_core_std,

            'frontleg_velocity_core_mean': frontleg_velocity_core_mean, 'frontleg_velocity_core_std': frontleg_velocity_core_std,
            'frontleg_acceleration_core_mean': frontleg_acceleration_core_mean, 'frontleg_acceleration_core_std': frontleg_acceleration_core_std,
            'frontleg_jerk_core_mean': frontleg_jerk_core_mean, 'frontleg_jerk_core_std': frontleg_jerk_core_std,
            'frontleg_angvel_core_mean': frontleg_angvel_core_mean, 'frontleg_angvel_core_std': frontleg_angvel_core_std,
            'frontleg_angaccel_core_mean': frontleg_angaccel_core_mean, 'frontleg_angaccel_core_std': frontleg_angaccel_core_std,
            'frontleg_angjerk_core_mean': frontleg_angjerk_core_mean, 'frontleg_angjerk_core_std': frontleg_angjerk_core_std,

            'midleg_velocity_core_mean': midleg_velocity_core_mean, 'midleg_velocity_core_std': midleg_velocity_core_std,
            'midleg_acceleration_core_mean': midleg_acceleration_core_mean, 'midleg_acceleration_core_std': midleg_acceleration_core_std,
            'midleg_jerk_core_mean': midleg_jerk_core_mean, 'midleg_jerk_core_std': midleg_jerk_core_std,
            'midleg_angvel_core_mean': midleg_angvel_core_mean, 'midleg_angvel_core_std': midleg_angvel_core_std,
            'midleg_angaccel_core_mean': midleg_angaccel_core_mean, 'midleg_angaccel_core_std': midleg_angaccel_core_std,
            'midleg_angjerk_core_mean': midleg_angjerk_core_mean, 'midleg_angjerk_core_std': midleg_angjerk_core_std,
            
            'backleg_velocity_core_mean': backleg_velocity_core_mean, 'backleg_velocity_core_std': backleg_velocity_core_std,
            'backleg_acceleration_core_mean': backleg_acceleration_core_mean, 'backleg_acceleration_core_std': backleg_acceleration_core_std,
            'backleg_jerk_core_mean': backleg_jerk_core_mean, 'backleg_jerk_core_std': backleg_jerk_core_std,
            'backleg_angvel_core_mean': backleg_angvel_core_mean, 'backleg_angvel_core_std': backleg_angvel_core_std,
            'backleg_angaccel_core_mean': backleg_angaccel_core_mean, 'backleg_angaccel_core_std': backleg_angaccel_core_std,
            'backleg_angjerk_core_mean': backleg_angjerk_core_mean, 'backleg_angjerk_core_std': backleg_angjerk_core_std,

            'head_velocity_core_mean': head_velocity_core_mean, 'head_velocity_core_std': head_velocity_core_std,
            'head_acceleration_core_mean': head_acceleration_core_mean, 'head_acceleration_core_std': head_acceleration_core_std,
            'head_jerk_core_mean': head_jerk_core_mean, 'head_jerk_core_std': head_jerk_core_std,

            # Start features
            'speed_start_mean': speed_start_mean, 'speed_start_std': speed_start_std,
            'velocity_fwd_start_mean': velocity_fwd_start_mean, 'velocity_fwd_start_std': velocity_fwd_start_std,
            'velocity_str_start_mean': velocity_str_start_mean, 'velocity_str_start_std': velocity_str_start_std,
            'accel_speed_start_mean': accel_speed_start_mean, 'accel_speed_start_std': accel_speed_start_std,
            'accel_fwd_start_mean': accel_fwd_start_mean, 'accel_fwd_start_std': accel_fwd_start_std,
            'accel_str_start_mean': accel_str_start_mean, 'accel_str_start_std': accel_str_start_std,
            'jerk_speed_start_mean': jerk_speed_start_mean, 'jerk_speed_start_std': jerk_speed_start_std,
            'jerk_fwd_start_mean': jerk_fwd_start_mean, 'jerk_fwd_start_std': jerk_fwd_start_std,
            'jerk_str_start_mean': jerk_str_start_mean, 'jerk_str_start_std': jerk_str_start_std,
            'angular_vel_start_mean': angular_vel_start_mean, 'angular_vel_start_std': angular_vel_start_std,
            'angular_accel_start_mean': angular_accel_start_mean, 'angular_accel_start_std': angular_accel_start_std,
            'angular_jerk_start_mean': angular_jerk_start_mean, 'angular_jerk_start_std': angular_jerk_start_std,

            'frontleg_velocity_start_mean': frontleg_velocity_start_mean, 'frontleg_velocity_start_std': frontleg_velocity_start_std,
            'frontleg_acceleration_start_mean': frontleg_acceleration_start_mean, 'frontleg_acceleration_start_std': frontleg_acceleration_start_std,
            'frontleg_jerk_start_mean': frontleg_jerk_start_mean, 'frontleg_jerk_start_std': frontleg_jerk_start_std,
            'frontleg_angvel_start_mean': frontleg_angvel_start_mean, 'frontleg_angvel_start_std': frontleg_angvel_start_std,
            'frontleg_angaccel_start_mean': frontleg_angaccel_start_mean, 'frontleg_angaccel_start_std': frontleg_angaccel_start_std,
            'frontleg_angjerk_start_mean': frontleg_angjerk_start_mean, 'frontleg_angjerk_start_std': frontleg_angjerk_start_std,

            'midleg_velocity_start_mean': midleg_velocity_start_mean, 'midleg_velocity_start_std': midleg_velocity_start_std,
            'midleg_acceleration_start_mean': midleg_acceleration_start_mean, 'midleg_acceleration_start_std': midleg_acceleration_start_std,
            'midleg_jerk_start_mean': midleg_jerk_start_mean, 'midleg_jerk_start_std': midleg_jerk_start_std,
            'midleg_angvel_start_mean': midleg_angvel_start_mean, 'midleg_angvel_start_std': midleg_angvel_start_std,
            'midleg_angaccel_start_mean': midleg_angaccel_start_mean, 'midleg_angaccel_start_std': midleg_angaccel_start_std,
            'midleg_angjerk_start_mean': midleg_angjerk_start_mean, 'midleg_angjerk_start_std': midleg_angjerk_start_std,

            'backleg_velocity_start_mean': backleg_velocity_start_mean, 'backleg_velocity_start_std': backleg_velocity_start_std,
            'backleg_acceleration_start_mean': backleg_acceleration_start_mean, 'backleg_acceleration_start_std': backleg_acceleration_start_std,
            'backleg_jerk_start_mean': backleg_jerk_start_mean, 'backleg_jerk_start_std': backleg_jerk_start_std,
            'backleg_angvel_start_mean': backleg_angvel_start_mean, 'backleg_angvel_start_std': backleg_angvel_start_std,
            'backleg_angaccel_start_mean': backleg_angaccel_start_mean, 'backleg_angaccel_start_std': backleg_angaccel_start_std,
            'backleg_angjerk_start_mean': backleg_angjerk_start_mean, 'backleg_angjerk_start_std': backleg_angjerk_start_std,

            'head_velocity_start_mean': head_velocity_start_mean, 'head_velocity_start_std': head_velocity_start_std,
            'head_acceleration_start_mean': head_acceleration_start_mean, 'head_acceleration_start_std': head_acceleration_start_std,
            'head_jerk_start_mean': head_jerk_start_mean, 'head_jerk_start_std': head_jerk_start_std,

            # End features
            'speed_end_mean': speed_end_mean, 'speed_end_std': speed_end_std,
            'velocity_fwd_end_mean': velocity_fwd_end_mean, 'velocity_fwd_end_std': velocity_fwd_end_std,
            'velocity_str_end_mean': velocity_str_end_mean, 'velocity_str_end_std': velocity_str_end_std,
            'accel_speed_end_mean': accel_speed_end_mean, 'accel_speed_end_std': accel_speed_end_std,
            'accel_fwd_end_mean': accel_fwd_end_mean, 'accel_fwd_end_std': accel_fwd_end_std,
            'accel_str_end_mean': accel_str_end_mean, 'accel_str_end_std': accel_str_end_std,
            'jerk_speed_end_mean': jerk_speed_end_mean, 'jerk_speed_end_std': jerk_speed_end_std,
            'jerk_fwd_end_mean': jerk_fwd_end_mean, 'jerk_fwd_end_std': jerk_fwd_end_std,
            'jerk_str_end_mean': jerk_str_end_mean, 'jerk_str_end_std': jerk_str_end_std,
            'angular_vel_end_mean': angular_vel_end_mean, 'angular_vel_end_std': angular_vel_end_std,
            'angular_accel_end_mean': angular_accel_end_mean, 'angular_accel_end_std': angular_accel_end_std,
            'angular_jerk_end_mean': angular_jerk_end_mean, 'angular_jerk_end_std': angular_jerk_end_std,

            'frontleg_velocity_end_mean': frontleg_velocity_end_mean, 'frontleg_velocity_end_std': frontleg_velocity_end_std,
            'frontleg_acceleration_end_mean': frontleg_acceleration_end_mean, 'frontleg_acceleration_end_std': frontleg_acceleration_end_std,
            'frontleg_jerk_end_mean': frontleg_jerk_end_mean, 'frontleg_jerk_end_std': frontleg_jerk_end_std,
            'frontleg_angvel_end_mean': frontleg_angvel_end_mean, 'frontleg_angvel_end_std': frontleg_angvel_end_std,
            'frontleg_angaccel_end_mean': frontleg_angaccel_end_mean, 'frontleg_angaccel_end_std': frontleg_angaccel_end_std,
            'frontleg_angjerk_end_mean': frontleg_angjerk_end_mean, 'frontleg_angjerk_end_std': frontleg_angjerk_end_std,

            'midleg_velocity_end_mean': midleg_velocity_end_mean, 'midleg_velocity_end_std': midleg_velocity_end_std,
            'midleg_acceleration_end_mean': midleg_acceleration_end_mean, 'midleg_acceleration_end_std': midleg_acceleration_end_std,
            'midleg_jerk_end_mean': midleg_jerk_end_mean, 'midleg_jerk_end_std': midleg_jerk_end_std,
            'midleg_angvel_end_mean': midleg_angvel_end_mean, 'midleg_angvel_end_std': midleg_angvel_end_std,
            'midleg_angaccel_end_mean': midleg_angaccel_end_mean, 'midleg_angaccel_end_std': midleg_angaccel_end_std,
            'midleg_angjerk_end_mean': midleg_angjerk_end_mean, 'midleg_angjerk_end_std': midleg_angjerk_end_std,

            'backleg_velocity_end_mean': backleg_velocity_end_mean, 'backleg_velocity_end_std': backleg_velocity_end_std,
            'backleg_acceleration_end_mean': backleg_acceleration_end_mean, 'backleg_acceleration_end_std': backleg_acceleration_end_std,
            'backleg_jerk_end_mean': backleg_jerk_end_mean, 'backleg_jerk_end_std': backleg_jerk_end_std,
            'backleg_angvel_end_mean': backleg_angvel_end_mean, 'backleg_angvel_end_std': backleg_angvel_end_std,
            'backleg_angaccel_end_mean': backleg_angaccel_end_mean, 'backleg_angaccel_end_std': backleg_angaccel_end_std,
            'backleg_angjerk_end_mean': backleg_angjerk_end_mean, 'backleg_angjerk_end_std': backleg_angjerk_end_std,

            'head_velocity_end_mean': head_velocity_end_mean, 'head_velocity_end_std': head_velocity_end_std,
            'head_acceleration_end_mean': head_acceleration_end_mean, 'head_acceleration_end_std': head_acceleration_end_std,
            'head_jerk_end_mean': head_jerk_end_mean, 'head_jerk_end_std': head_jerk_end_std,

            # Min/max percentile thresholds from the full qualifying bout.
            'speed_min': speed_min, 'speed_max': speed_max,
            'velocity_fwd_min': velocity_fwd_min, 'velocity_fwd_max': velocity_fwd_max,
            'velocity_str_min': velocity_str_min, 'velocity_str_max': velocity_str_max,
            'accel_speed_min': accel_speed_min, 'accel_speed_max': accel_speed_max,
            'accel_fwd_min': accel_fwd_min, 'accel_fwd_max': accel_fwd_max,
            'accel_str_min': accel_str_min, 'accel_str_max': accel_str_max,
            'jerk_speed_min': jerk_speed_min, 'jerk_speed_max': jerk_speed_max,
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
            'accel_speed_median': accel_speed_median, 'accel_speed_skewness': accel_speed_skewness,
            'accel_fwd_median': accel_fwd_median, 'accel_fwd_skewness': accel_fwd_skewness,
            'accel_str_median': accel_str_median, 'accel_str_skewness': accel_str_skewness,
            'jerk_speed_median': jerk_speed_median, 'jerk_speed_skewness': jerk_speed_skewness,
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
            'accel_speed_mean': accel_speed_mean, 'accel_speed_std': accel_speed_std,
            'accel_fwd_mean': accel_fwd_mean, 'accel_fwd_std': accel_fwd_std,
            'accel_str_mean': accel_str_mean, 'accel_str_std': accel_str_std,
            'jerk_speed_mean': jerk_speed_mean, 'jerk_speed_std': jerk_speed_std,
            'jerk_fwd_mean': jerk_fwd_mean, 'jerk_fwd_std': jerk_fwd_std,
            'jerk_str_mean': jerk_str_mean, 'jerk_str_std': jerk_str_std,
            'angular_vel_mean': angular_vel_mean, 'angular_vel_std': angular_vel_std,
            'angular_accel_mean': angular_accel_mean, 'angular_accel_std': angular_accel_std,
            'angular_jerk_mean': angular_jerk_mean, 'angular_jerk_std': angular_jerk_std,

            'frontleg_velocity_mean': frontleg_velocity_mean, 'frontleg_velocity_std': frontleg_velocity_std,
            'frontleg_acceleration_mean': frontleg_acceleration_mean, 'frontleg_acceleration_std': frontleg_acceleration_std,
            'frontleg_jerk_mean': frontleg_jerk_mean, 'frontleg_jerk_std': frontleg_jerk_std,
            'frontleg_angvel_mean': frontleg_angvel_mean, 'frontleg_angvel_std': frontleg_angvel_std,
            'frontleg_angaccel_mean': frontleg_angaccel_mean, 'frontleg_angaccel_std': frontleg_angaccel_std,
            'frontleg_angjerk_mean': frontleg_angjerk_mean, 'frontleg_angjerk_std': frontleg_angjerk_std,

            'midleg_velocity_mean': midleg_velocity_mean, 'midleg_velocity_std': midleg_velocity_std,
            'midleg_acceleration_mean': midleg_acceleration_mean, 'midleg_acceleration_std': midleg_acceleration_std,
            'midleg_jerk_mean': midleg_jerk_mean, 'midleg_jerk_std': midleg_jerk_std,
            'midleg_angvel_mean': midleg_angvel_mean, 'midleg_angvel_std': midleg_angvel_std,
            'midleg_angaccel_mean': midleg_angaccel_mean, 'midleg_angaccel_std': midleg_angaccel_std,
            'midleg_angjerk_mean': midleg_angjerk_mean, 'midleg_angjerk_std': midleg_angjerk_std,
            
            'backleg_velocity_mean': backleg_velocity_mean, 'backleg_velocity_std': backleg_velocity_std,
            'backleg_acceleration_mean': backleg_acceleration_mean, 'backleg_acceleration_std': backleg_acceleration_std,
            'backleg_jerk_mean': backleg_jerk_mean, 'backleg_jerk_std': backleg_jerk_std,
            'backleg_angvel_mean': backleg_angvel_mean, 'backleg_angvel_std': backleg_angvel_std,
            'backleg_angaccel_mean': backleg_angaccel_mean, 'backleg_angaccel_std': backleg_angaccel_std,
            'backleg_angjerk_mean': backleg_angjerk_mean, 'backleg_angjerk_std': backleg_angjerk_std,

            'head_velocity_mean': head_velocity_mean, 'head_velocity_std': head_velocity_std,
            'head_acceleration_mean': head_acceleration_mean, 'head_acceleration_std': head_acceleration_std,
            'head_jerk_mean': head_jerk_mean, 'head_jerk_std': head_jerk_std,

            # Min/max percentile thresholds from the full qualifying bout.
            'speed_min': speed_min, 'speed_max': speed_max,
            'velocity_fwd_min': velocity_fwd_min, 'velocity_fwd_max': velocity_fwd_max,
            'velocity_str_min': velocity_str_min, 'velocity_str_max': velocity_str_max,
            'accel_speed_min': accel_speed_min, 'accel_speed_max': accel_speed_max,
            'accel_fwd_min': accel_fwd_min, 'accel_fwd_max': accel_fwd_max,
            'accel_str_min': accel_str_min, 'accel_str_max': accel_str_max,
            'jerk_speed_min': jerk_speed_min, 'jerk_speed_max': jerk_speed_max,
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
            'accel_speed_median': accel_speed_median, 'accel_speed_skewness': accel_speed_skewness,
            'accel_fwd_median': accel_fwd_median, 'accel_fwd_skewness': accel_fwd_skewness,
            'accel_str_median': accel_str_median, 'accel_str_skewness': accel_str_skewness,
            'jerk_speed_median': jerk_speed_median, 'jerk_speed_skewness': jerk_speed_skewness,
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

        # Exclude the declared redundant export families; internal deviations still support skewness.
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
    """Pool recording sums and counts into animal-level bout-based statistics and apply
    the configured bout threshold.
    """
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
                # Censor statistics below the configured qualifying-bout count.
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
    """Write bout-based animal kinematics by smoothing level and label category."""
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
                              '_min_mean','_max_mean',
                              '_median_mean','_median_std','_skewness_mean','_skewness_std'))]
        tidy = master_df[['individual','group','bout_type'] + value_cols].copy()
        pivot_df = tidy.pivot_table(index=['individual','group'],
                                    columns='bout_type',
                                    values=value_cols,
                                    observed=True)
        # Flatten MultiIndex columns "values x bout_type" â†’ "bout_type_value"
        # Standardize feature names: remove spaces from bout_type (e.g., "Walking + Floor" -> "Walking+Floor")
        pivot_df.columns = [f'{bt.replace(" ", "")}_{val}' for val, bt in pivot_df.columns.to_flat_index()]
        pivot_df.reset_index(inplace=True)

        # Drop any column that is entirely empty (all NaN)
        pivot_df.dropna(axis=1, how='all', inplace=True)
        
        pivot_df = pivot_df.copy() # De-fragment the DataFrame to improve performance

        # Define sorting logic for columns
        pose_order = ['Walking', 'Standing', 'Scramble', 'Groom-FrontLegs', 'Groom-Head', 'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs', 'Jump']
        orient_order = ['Floor', 'Ceiling', 'Wall-Side', 'Wall-Up', 'Wall-Down']
        behavior_map = {b.replace(" ", ""): i for i, b in enumerate(pose_order + [f"{p} + {o}" for p in pose_order if p != 'Jump' for o in orient_order])}
        feature_prefix_map = {'speed': 0, 'velocity_fwd': 1, 'velocity_str': 2, 'accel_speed': 3, 'accel_fwd': 4, 'accel_str': 5, 'frontleg': 6, 'backleg': 7, 'head': 8}

        # Remove label spaces to match the generated feature columns.
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
        separators = [f'---_{i}' for i in range(1, 10)] 
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

        # Export the Pose-only sheet in canonical statistical order.
        if pose_cols:
            # 1. Create lists for each statistical measure
            pose_mean_mean_cols = sorted([c for c in pose_cols if c.endswith('_mean_mean')], key=custom_sort_key)
            pose_mean_std_cols  = sorted([c for c in pose_cols if c.endswith('_mean_std')], key=custom_sort_key)
            pose_std_mean_cols  = sorted([c for c in pose_cols if c.endswith('_std_mean')], key=custom_sort_key)
            pose_std_std_cols   = sorted([c for c in pose_cols if c.endswith('_std_std')], key=custom_sort_key)
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
                                  pose_min_mean_cols  + [separators[4]] +
                                  pose_max_mean_cols  + [separators[5]] +
                                  pose_median_mean_cols + [separators[6]] +
                                  pose_median_std_cols  + [separators[7]] +
                                  pose_skewness_mean_cols + [separators[8]] +
                                  pose_skewness_std_cols)
                                  
            df_to_format = pivot_df[['individual', 'group'] + pose_desired_order].copy()
            
            # Omit unavailable report columns while retaining measured zeros.
            cols_to_drop = []
            for col in df_to_format.columns:
                # Skip identifier and separator columns
                if col in ['individual', 'group'] or str(col).startswith('---_'):
                    continue
                # Retain zero-valued observations when deciding which columns to export.
                # Measured zero is data, not an absent feature. Dropping a
                # shard's zero-only column loses those observations at merge.
                if df_to_format[col].isna().all():
                    cols_to_drop.append(col)
            
            df_to_format.drop(columns=cols_to_drop, inplace=True)
            pose_desired_order_filtered = [c for c in pose_desired_order if c not in cols_to_drop]

            sheets_to_save[f'Pose {smoothing_label}'] = format_summary_df(df_to_format, pose_desired_order_filtered)

        # Export the combined-context sheet in canonical statistical order.
        if orient_cols:
            # 1. Create lists for each statistical measure
            orient_mean_mean_cols = sorted([c for c in orient_cols if c.endswith('_mean_mean')], key=custom_sort_key)
            orient_mean_std_cols  = sorted([c for c in orient_cols if c.endswith('_mean_std')], key=custom_sort_key)
            orient_std_mean_cols  = sorted([c for c in orient_cols if c.endswith('_std_mean')], key=custom_sort_key)
            orient_std_std_cols   = sorted([c for c in orient_cols if c.endswith('_std_std')], key=custom_sort_key)
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
                                    orient_min_mean_cols  + [separators[4]] +
                                    orient_max_mean_cols  + [separators[5]] +
                                    orient_median_mean_cols + [separators[6]] +
                                    orient_median_std_cols  + [separators[7]] +
                                    orient_skewness_mean_cols + [separators[8]] +
                                    orient_skewness_std_cols)

            df_to_format = pivot_df[['individual', 'group'] + orient_desired_order].copy()

            # Omit unavailable report columns while retaining measured zeros.
            cols_to_drop = []
            for col in df_to_format.columns:
                if col in ['individual', 'group'] or str(col).startswith('---_'):
                    continue
                if df_to_format[col].isna().all():
                    cols_to_drop.append(col)

            df_to_format.drop(columns=cols_to_drop, inplace=True)
            orient_desired_order_filtered = [c for c in orient_desired_order if c not in cols_to_drop]
            
            sheets_to_save[f'Pose + Orient {smoothing_label}'] = format_summary_df(df_to_format, orient_desired_order_filtered)

    if save_outputs:
        save_to_excel_sheets(sheets_to_save, excel_path)
    return sheets_to_save
