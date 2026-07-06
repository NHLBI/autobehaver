"""Master feature matrix assembly for mmgen.

This module contains the final feature-matrix construction and related
helpers (bout-count filtering, units manifest, and Excel formatting utilities).
It is intentionally focused on assembling the canonical master matrix from the
post-aggregation source tables emitted by mmgen.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import os
import gc
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.styles.borders import Border, Side

from .naming import collapse_individual_id

# ---------------------------------------------------------------------------
# Default runtime values
# ---------------------------------------------------------------------------
DEFAULT_FILTER_FEATURES_BY_BOUT_COUNT: bool = True
DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES: int = 6
DEFAULT_MIN_BOUT_COUNT_FOR_TRANSITIONS: int = 6
DEFAULT_MM_PER_PX: float = 0.0169896744624207
DEFAULT_VIDEO_FPS: float = 40.0
# Spatial occupancy summary thresholds.
# NOTE: These constants are used when generating the spatial r_norm-derived
# summary features during master-matrix assembly.
SPATIAL_R_NORM_CENTER_THRESHOLD: float = 0.40
SPATIAL_R_NORM_PERIPHERY_THRESHOLD: float = 0.70

# Context-level guardrails.
# These are used to avoid emitting noisy/unstable per-context summary features
# when there are too few frames in a given context.
MIN_CONTEXT_DURATION_S: float = 3.0
DEFAULT_MIN_CONTEXT_FRAMES: int = int(round(MIN_CONTEXT_DURATION_S * DEFAULT_VIDEO_FPS))
MIN_CONTEXT_FRAMES: int = DEFAULT_MIN_CONTEXT_FRAMES

# Grooming episode stitching threshold.
# A grooming episode can contain short non-grooming gaps; if the gap grows
# beyond this threshold the episode is closed.
GROOM_EPISODE_MAX_NON_GROOM_GAP_S: float = 1.0
DEFAULT_GROOM_EPISODE_MAX_GAP_FRAMES: int = int(round(GROOM_EPISODE_MAX_NON_GROOM_GAP_S * DEFAULT_VIDEO_FPS))
GROOM_EPISODE_MAX_GAP_FRAMES: int = DEFAULT_GROOM_EPISODE_MAX_GAP_FRAMES

def _filter_features_by_bout_count(master_matrix, min_bout_count, min_transition_count, bout_count_df=None):
    """
    Filter BB-style bout-derived features (BoutSummary, Kinematic, and Transition).

    ``KinematicTW_*`` features are intentionally excluded because their stability
    is governed by pooled context-frame sufficiency, not by the number of bouts.

    For each eligible feature, find its corresponding bout_count column and set
    the feature value to NaN for any animal where the bout count is below the
    threshold.

    Args:
        master_matrix: DataFrame with all features (one row per animal)
        min_bout_count: Minimum number of bouts required to keep BoutSummary/Kinematic features
        min_transition_count: Minimum number of "from" bouts required to keep Transition features
        bout_count_df: Optional separate DataFrame containing bout_count columns (if None, searches in master_matrix)

    Returns:
        DataFrame: The filtered master_matrix with NaN values for low-count features
    """
    print(f"  Filtering features with bout count < {min_bout_count} (or transition count < {min_transition_count})...")

    # We map BoutSummary_* and BB KinematicBB_* features to their corresponding
    # bout_count column purely by tokens: {Prefix}_{Category}_{Behavior}_...
    # Count col: BoutSummary_{Category}_{Behavior}_bout_count
    # Transition_* uses the "from_behavior" state's bout count.

    # Determine where to look for bout_count columns
    if bout_count_df is not None:
        count_source_columns = bout_count_df.columns.tolist()
    else:
        count_source_columns = master_matrix.columns.tolist()

    feature_to_count_map = {}
    feature_threshold_map = {}
    all_columns = master_matrix.columns.tolist()

    for col in all_columns:
        if not (col.startswith('BoutSummary_') or col.startswith('KinematicBB_') or col.startswith('Kinematic_') or col.startswith('Transition_')):
            continue

        # --- Transition features (keep existing parsing pattern; it matches your naming) ---
        if col.startswith('Transition_'):
            remaining = col.replace('Transition_', '')
            if '_to_' not in remaining:
                continue

            from_part = remaining.split('_to_', 1)[0]  # e.g., "Pose_Walking" or "PoseOrient_Walking+Floor"
            from_parts = from_part.split('_', 1)
            if len(from_parts) < 2:
                continue

            subcategory = from_parts[0]       # Pose or PoseOrient
            from_behavior = from_parts[1]     # Walking or Walking+Floor

            if subcategory == 'PoseOrient':
                subcategory = 'Pose+Orient'

            count_col_name = f"BoutSummary_{subcategory}_{from_behavior}_bout_count"
            if count_col_name in count_source_columns:
                feature_to_count_map[col] = count_col_name
                feature_threshold_map[col] = min_transition_count
            continue

        # --- BoutSummary / Kinematic: token-based mapping ---
        parts = col.split('_')
        # Expect at least: Prefix_Category_Behavior_...
        if len(parts) < 3:
            continue

        category = parts[1]
        behavior = parts[2]

        count_col_name = f"BoutSummary_{category}_{behavior}_bout_count"
        if count_col_name in count_source_columns:
            feature_to_count_map[col] = count_col_name
            feature_threshold_map[col] = min_bout_count

    
    print(f"  Found {len(feature_to_count_map)} features that can be filtered by bout count.")
    
    # Apply the filter
    filtered_count = 0
    filtered_examples = []  # Store examples for debugging
    max_examples = 5  # Show up to 5 examples
    
    # If using separate bout_count_df, create a merged lookup dataframe
    if bout_count_df is not None:
        # Merge bout_count_df with master_matrix on 'individual' for lookups
        # Only merge the count columns we need
        count_cols_to_merge = ['individual'] + list(set(feature_to_count_map.values()))
        # Create a lookup dataframe with individual and all needed count columns
        count_lookup = pd.merge(
            master_matrix[['individual']], 
            bout_count_df[count_cols_to_merge], 
            on='individual', 
            how='left'
        )
    else:
        # Use master_matrix directly for count lookups
        count_lookup = master_matrix
    
    # Build indexed lookup ONCE (faster and avoids repeated index creation)
    count_lookup_indexed = count_lookup.set_index('individual')

    for feature_col, count_col in feature_to_count_map.items():
        threshold = feature_threshold_map.get(feature_col, min_bout_count)

        if count_col not in count_lookup_indexed.columns:
            continue

        aligned_counts = master_matrix['individual'].map(count_lookup_indexed[count_col])
        low_count_mask = (aligned_counts < threshold) | aligned_counts.isna()

        n_filtered = int(low_count_mask.sum())
        if n_filtered > 0:
            filtered_animals = master_matrix.loc[low_count_mask, 'individual'].tolist()[:3]
            bout_counts = aligned_counts.loc[low_count_mask].tolist()[:3]

            if len(filtered_examples) < max_examples:
                filtered_examples.append({
                    'feature': feature_col,
                    'count_col': count_col,
                    'n_filtered': n_filtered,
                    'animals': filtered_animals,
                    'counts': bout_counts
                })

            master_matrix.loc[low_count_mask, feature_col] = np.nan
            filtered_count += n_filtered

    
    # Print debug information
    print(f"  Set {filtered_count} feature values to NaN due to insufficient bout counts.")
    
    if filtered_examples:
        print(f"\n  Debug: Examples of filtered features (showing up to {len(filtered_examples)}):")
        for i, example in enumerate(filtered_examples, 1):
            animals_str = ', '.join([
                f"{a} (count={0 if pd.isna(c) else int(c)})" 
                for a, c in zip(example['animals'], example['counts'])
            ])
            print(f"    {i}. {example['feature']}")
            print(f"       -> {example['n_filtered']} animals filtered (e.g., {animals_str})")
            print(f"       -> Count column: {example['count_col']}")
    else:
        print(f"  Debug: No features were filtered (all bout counts >= {min_bout_count}).")
    
    return master_matrix



def _master_matrix_family_sort_key(column_name: str) -> tuple[int, str]:
    """Return a canonical family-grouped sort key for master-matrix columns."""
    family_prefixes = [
        "BoutSummary_",
        "TimeBudget_",
        "StdMeasure_",
        "Postural_",
        "KinematicBB_",
        "Kinematic_",
        "KinematicTW_",
        "Transition_",
        "SleepFeat_",
        "Spatial_",
        "Global_",
    ]
    for rank, prefix in enumerate(family_prefixes):
        if str(column_name).startswith(prefix):
            return (rank, str(column_name))
    return (len(family_prefixes), str(column_name))


def reorder_master_matrix_columns(master_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return *master_matrix* with stable family-grouped column ordering."""
    id_cols = [c for c in ["individual", "group", "sex"] if c in master_matrix.columns]
    remaining = [c for c in master_matrix.columns if c not in id_cols]
    ordered = id_cols + sorted(remaining, key=_master_matrix_family_sort_key)
    return master_matrix.loc[:, ordered].copy()



# ============================================================================
# FEATURE UNITS MANIFEST
# ============================================================================

def _infer_unit_for_feature_name(feature_name: str) -> tuple[str, str]:
    """
    Infer the physical unit for a MASTER_FEATURE_MATRIX column based on naming conventions.
    Returns: (unit, notes)
    Unit strings are short: 'mm', 'mm^2', 'mm/s', 'mm/s^2', 'mm/s^3', 'deg', 'deg/s', 'deg/s^2', 'deg/s^3', 'rad/mm', 'unitless', etc.
    """
    # Identifier columns
    if feature_name in {"individual", "group", "sex"}:
        return ("ID", "identifier column")

    # ------------------------------------------------------------------------
    # Bout summary
    # BoutSummary_{Category}_{Behavior}_{metric}
    # ------------------------------------------------------------------------
    if feature_name.startswith("BoutSummary_"):
        # Grooming episode metrics (treated as BoutSummary family for downstream grouping)
        if feature_name.startswith("BoutSummary_GroomEpisode_"):
            if feature_name.endswith("_EpisodesPerMin"):
                return ("episodes/min", "episodes per minute")
            if feature_name.endswith("_MeanDuration_s"):
                return ("s", "seconds")
            if feature_name.endswith("_SubtypeEntropyNorm"):
                return ("unitless", "normalized entropy in [0,1]")
            return ("unitless", "groom episode metric")

        if feature_name.endswith("_bout_frequency"):
            return ("1/s", "bouts per second")
        if feature_name.endswith("_bout_duration_mean_s"):
            return ("s", "seconds")
        if feature_name.endswith("_inter_bout_interval_mean_s"):
            return ("s", "seconds")
        if feature_name.endswith("_inter_bout_interval_std_s"):
            return ("s", "seconds")
        if feature_name.endswith("_bout_duration_coeff_of_var"):
            return ("unitless", "coefficient of variation")
        if feature_name.endswith("_initiation_prob"):
            return ("unitless", "probability in [0,1]")
        return ("unknown", "unrecognized BoutSummary metric suffix")

    # ------------------------------------------------------------------------
    # Time budgets
    # ------------------------------------------------------------------------
    if feature_name.startswith("TimeBudget_"):
        return ("unitless", "fraction/time budget in [0,1]")

    # ------------------------------------------------------------------------
    # Transition probabilities
    # ------------------------------------------------------------------------
    if feature_name.startswith("Transition_"):
        # Repertoire structure metrics are grouped under Transition_ for downstream convenience
        if feature_name.startswith("Transition_Repertoire_"):
            if "SwitchRate_per_min" in feature_name:
                return ("1/min", "switches per minute")
            if "EntropyNorm" in feature_name:
                return ("unitless", "normalized entropy in [0,1]")
            return ("unitless", "repertoire metric")

        return ("unitless", "transition probability in [0,1]")

    # ------------------------------------------------------------------------
    # Spatial center/periphery fractions
    # ------------------------------------------------------------------------
    if feature_name.startswith("SpatialFrac_"):
        return ("unitless", "fraction of frames in zone (center/periphery)")

    # ------------------------------------------------------------------------
    # Global movement summary
    # ------------------------------------------------------------------------
    if feature_name == "Global_MeanSpeed_mm_s":
        return ("mm/s", "mean centroid speed across all frames")

    # ------------------------------------------------------------------------
    # Repertoire structure
    # ------------------------------------------------------------------------
    if feature_name.startswith("Repertoire_"):
        if "SwitchRate_per_min" in feature_name:
            return ("1/min", "switches per minute")
        if "EntropyNorm" in feature_name:
            return ("unitless", "normalized entropy in [0,1]")
        return ("unitless", "repertoire metric")

    # ------------------------------------------------------------------------
    # Grooming episode metrics
    # ------------------------------------------------------------------------
    if feature_name.startswith("GroomEpisode_"):
        if feature_name.endswith("_EpisodesPerMin"):
            return ("episodes/min", "episodes per minute")
        if feature_name.endswith("_MeanDuration_s"):
            return ("s", "seconds")
        if feature_name.endswith("_SubtypeEntropyNorm"):
            return ("unitless", "normalized entropy in [0,1]")
        return ("unitless", "groom episode metric")

    # ------------------------------------------------------------------------
    # Wing/leg spread angles
    # ------------------------------------------------------------------------
    if feature_name.startswith("WingSpread_") or feature_name.startswith("LegSpread_"):
        return ("deg", "degrees")

    # ------------------------------------------------------------------------
    # Sleep features
    # SleepFeat_{bout_type}_{metric}
    # ------------------------------------------------------------------------
    if feature_name.startswith("SleepFeat_"):
        if feature_name.endswith("_n_sleep_bouts"):
            return ("count", "number of sleep bouts")
        if feature_name.endswith("_sleep_bout_duration_mean_s"):
            return ("s", "seconds")
        if feature_name.endswith("_sleep_bout_duration_std_s"):
            return ("s", "seconds")
        if feature_name.endswith("_sleep_time_budget"):
            return ("unitless", "fraction/time budget in [0,1]")
        return ("unknown", "unrecognized SleepFeat metric suffix")

    # ------------------------------------------------------------------------
    # Spatial features (currently r_norm + relative angle summaries)
    # ------------------------------------------------------------------------
    if feature_name.startswith("Spatial_"):
        # Center/periphery fractions (merged into Spatial_ family)
        if feature_name.startswith("Spatial_Frac_"):
            return ("unitless", "fraction of frames in zone (center/periphery)")

        if feature_name.endswith("_r_norm"):
            return ("unitless", "normalized radius (0=center, 1=wall)")
        if feature_name.endswith("MeanAbsRelAngle") or feature_name.endswith("StdAbsRelAngle"):
            return ("deg", "absolute relative angle in degrees")
        return ("unknown", "unrecognized Spatial feature")

    # ------------------------------------------------------------------------
    # Standardized measurements
    # StdMeasure_{measurement}_{stat}...
    # ------------------------------------------------------------------------
    if feature_name.startswith("StdMeasure_"):
        # Normalized/ratio versions are dimensionless
        if "_norm" in feature_name or "_ratio" in feature_name:
            return ("unitless", "normalized/ratio measurement")

        # Areas
        if "body_area" in feature_name:
            return ("mm^2", "area")

        # Everything else in StdMeasure is a length scale
        return ("mm", "linear measurement")

    # ------------------------------------------------------------------------
    # Postural
    # Postural_{Orient}_{feature}_{stat}
    # ------------------------------------------------------------------------
    if feature_name.startswith("Postural_"):
        # Wing/leg spread angles (added as Postural_* family features)
        if "WingSpread" in feature_name or "LegSpread" in feature_name:
            return ("deg", "degrees")

        if "stance_area" in feature_name:
            return ("mm^2", "area")
        if "extension_norm" in feature_name:
            return ("unitless", "normalized by body length normalizer")
        if "extension_raw" in feature_name:
            return ("mm", "linear distance")
        if "asymmetry_magnitude" in feature_name or "overall_asymmetry" in feature_name:
            return ("mm", "linear distance")
        return ("unknown", "unrecognized Postural feature")

    # ------------------------------------------------------------------------
    # Kinematics
    # KinematicBB_{Category}_{Behavior}_{metric} / KinematicTW_{Category}_{Behavior}_{metric}
    # ------------------------------------------------------------------------
    if feature_name.startswith(("KinematicBB_", "Kinematic_", "KinematicTW_")):
        # Distribution shape measures
        if "_skewness_" in feature_name or feature_name.endswith("_skewness_mean") or feature_name.endswith("_skewness_std"):
            return ("unitless", "skewness is dimensionless")

        # Asymmetry indices are dimensionless ratios
        if "_asym" in feature_name:
            return ("unitless", "left-right asymmetry index")

        # Curvature is angle / distance
        if "curvature" in feature_name:
            return ("rad/mm", "curvature")

        # Angular motion
        if "angular_jerk" in feature_name or "angjerk" in feature_name:
            return ("deg/s^3", "angular jerk")
        if "angular_acceleration" in feature_name or "angular_accel" in feature_name or "angaccel" in feature_name:
            return ("deg/s^2", "angular acceleration")
        if "angular_velocity" in feature_name or "angular_vel" in feature_name or "angvel" in feature_name:
            return ("deg/s", "angular velocity")

        # Linear motion (order matters: jerk before accel before vel)
        if "jerk" in feature_name:
            return ("mm/s^3", "linear jerk")
        if "acceleration" in feature_name or "accel" in feature_name:
            return ("mm/s^2", "linear acceleration")
        if "velocity" in feature_name or "speed" in feature_name:
            return ("mm/s", "linear velocity/speed")

        return ("unknown", "unrecognized kinematic metric")

    return ("unknown", "no rule matched")

def write_feature_units_manifest(master_matrix: pd.DataFrame,
                                output_dir: str,
                                units_csv_name: str = "MASTER_FEATURE_MATRIX_UNITS.csv",
                                readme_name: str = "MASTER_FEATURE_MATRIX_UNITS_README.txt",
                                mm_per_px: float = DEFAULT_MM_PER_PX,
                                fps: float = 40.0) -> None:
    """
    Writes a manifest mapping every column in MASTER_FEATURE_MATRIX to its unit.
    Designed to be called right before/after saving MASTER_FEATURE_MATRIX.csv.
    """
    records = []
    for col in master_matrix.columns:
        unit, notes = _infer_unit_for_feature_name(col)
        family = col.split("_", 1)[0] if "_" in col else "UNKNOWN"
        records.append({
            "feature": col,
            "family": family,
            "unit": unit,
            "notes": notes
        })

    units_df = pd.DataFrame(records)

    units_path = os.path.join(output_dir, units_csv_name)
    units_df.to_csv(units_path, index=False)

    # README with conventions / calibration
    readme_path = os.path.join(output_dir, readme_name)
    with open(readme_path, "w") as f:
        f.write("MASTER FEATURE MATRIX UNIT MANIFEST\n")
        f.write("=================================\n\n")
        f.write(f"MM_PER_PX = {mm_per_px}\n")
        f.write(f"FPS (used for time conversions) = {fps}\n\n")
        f.write("Unit conventions:\n")
        f.write("- Time budgets / probabilities / normalized metrics: unitless\n")
        f.write("- StdMeasure lengths: mm; StdMeasure areas: mm^2\n")
        f.write("- Postural extensions/asymmetries: mm; stance_area: mm^2\n")
        f.write("- Kinematic speed/velocity: mm/s; accel: mm/s^2; jerk: mm/s^3\n")
        f.write("- Kinematic angular velocity/accel/jerk: deg/s, deg/s^2, deg/s^3\n")
        f.write("- Kinematic curvature: rad/mm\n")
        f.write("- Spatial relative angles: deg; r_norm: unitless\n\n")
        f.write("Any 'unknown' units should be reviewed and added to _infer_unit_for_feature_name().\n")

    # Console summary / sanity warning
    unit_counts = units_df["unit"].value_counts(dropna=False)
    print(f"\nSaved feature unit manifest to: {units_path}")
    print("Unit breakdown:")
    print(unit_counts.to_string())

    unknown_df = units_df[units_df["unit"] == "unknown"]
    if not unknown_df.empty:
        sample = unknown_df["feature"].head(25).tolist()
        print(f"\nWARNING: {len(unknown_df)} features have unit='unknown'. Sample (first {len(sample)}):")
        for s in sample:
            print(f"  - {s}")
        print("Add rules to _infer_unit_for_feature_name() to eliminate these.")

@dataclass
class MMGenFinalizationSources:
    """Canonical source tables used to assemble the mmgen master feature matrix.

    These sources may come directly from temp-store-derived in-memory tables during
    pipeline finalization, or from previously emitted mmgen output files
    when using the wrapper ``create_feature_matrix``.
    """

    bout_summary_df: pd.DataFrame | None = None
    time_budget_sheets: dict[str, pd.DataFrame] | None = None
    measurements_df: pd.DataFrame | None = None
    postural_summary_df: pd.DataFrame | None = None
    kinematic_sheets: dict[str, pd.DataFrame] | None = None
    tw_kinematic_sheets: dict[str, pd.DataFrame] | None = None
    transition_pose_tidy_df: pd.DataFrame | None = None
    transition_pose_orient_tidy_df: pd.DataFrame | None = None
    sleep_summary_df: pd.DataFrame | None = None
    spatial_df: pd.DataFrame | None = None
    spatial_data_path: str | None = None



def _ordered_chunks_for_workers(items: list[str], n_chunks: int) -> list[list[str]]:
    if not items:
        return []
    n_chunks = max(1, min(int(n_chunks), len(items)))
    return [items[i::n_chunks] for i in range(n_chunks) if items[i::n_chunks]]


def _merge_feature_frame(base_df: pd.DataFrame, feat_df: pd.DataFrame | None) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return base_df
    return pd.merge(base_df, feat_df, on='individual', how='left')


def _compute_spatial_feature_summary_core(
    spatial_df: pd.DataFrame,
    bout_summary_df: pd.DataFrame | None,
    master_matrix: pd.DataFrame,
    *,
    video_fps: float,
    min_context_frames: int,
    groom_episode_max_gap_frames: int,
) -> pd.DataFrame:
    summary_df = master_matrix[['individual']].drop_duplicates().reset_index(drop=True).copy()
    if spatial_df is None or spatial_df.empty:
        return summary_df

    pose_col = 'masked_pose_valid' if 'masked_pose_valid' in spatial_df.columns else 'masked_pose'
    orient_col = 'masked_orient_valid' if 'masked_orient_valid' in spatial_df.columns else 'masked_orient'
    pose_orient_col = 'masked_pose_orient_valid' if 'masked_pose_orient_valid' in spatial_df.columns else 'masked_pose_orient'

    orient_like_cols = {'masked_orient', 'masked_pose_orient', 'masked_orient_valid', 'masked_pose_orient_valid'}

    # Spatial r_norm summaries
    for beh_col in [pose_col, orient_col, pose_orient_col]:
        if beh_col not in spatial_df.columns:
            continue
        clean_spatial = spatial_df.dropna(subset=[beh_col, 'r_norm'])
        if beh_col in orient_like_cols:
            exclude_mask = clean_spatial[beh_col].astype(str).str.contains('Wall-Up|Wall-Down|Wall-Side', case=False, na=False, regex=True)
            clean_spatial = clean_spatial.loc[~exclude_mask].copy()
        if clean_spatial.empty:
            continue
        grouped = clean_spatial.groupby(['individual', beh_col])['r_norm']
        r_norm_stats = grouped.agg(['mean', 'median', 'std']).reset_index()
        pivoted_r_norm = r_norm_stats.pivot(index='individual', columns=beh_col, values=['mean', 'median', 'std'])
        cat_prefix_map = {
            'masked_pose': 'Pose', 'masked_pose_valid': 'Pose',
            'masked_orient': 'Orient', 'masked_orient_valid': 'Orient',
            'masked_pose_orient': 'PoseOrient', 'masked_pose_orient_valid': 'PoseOrient',
        }
        prefix = f"Spatial_{cat_prefix_map.get(beh_col, 'Unknown')}"
        pivoted_r_norm.columns = [f"{prefix}_{behavior}_{stat}_r_norm".replace('+', 'plus') for stat, behavior in pivoted_r_norm.columns]
        pivoted_r_norm.reset_index(inplace=True)
        summary_df = _merge_feature_frame(summary_df, pivoted_r_norm)

    # Spatial relative angle summaries
    if 'relative_angle' in spatial_df.columns:
        filtered_spatial_df = spatial_df[spatial_df['r_norm'] >= 0.1].copy()
        if not filtered_spatial_df.empty:
            pose_col_f = 'masked_pose_valid' if 'masked_pose_valid' in filtered_spatial_df.columns else 'masked_pose'
            orient_col_f = 'masked_orient_valid' if 'masked_orient_valid' in filtered_spatial_df.columns else 'masked_orient'
            pose_orient_col_f = 'masked_pose_orient_valid' if 'masked_pose_orient_valid' in filtered_spatial_df.columns else 'masked_pose_orient'
            for beh_col in [pose_col_f, orient_col_f, pose_orient_col_f]:
                if beh_col not in filtered_spatial_df.columns:
                    continue
                clean_rel_angle_df = filtered_spatial_df.dropna(subset=[beh_col, 'relative_angle']).copy()
                if beh_col in orient_like_cols:
                    exclude_mask = clean_rel_angle_df[beh_col].astype(str).str.contains('Wall-Up|Wall-Down|Wall-Side', case=False, na=False, regex=True)
                    clean_rel_angle_df = clean_rel_angle_df.loc[~exclude_mask].copy()
                if clean_rel_angle_df.empty:
                    continue
                clean_rel_angle_df['abs_relative_angle'] = clean_rel_angle_df['relative_angle'].abs()
                grouped = clean_rel_angle_df.groupby(['individual', beh_col])
                rel_angle_stats = grouped['abs_relative_angle'].agg(['mean', 'std']).reset_index()
                rel_angle_stats.rename(columns={'mean': 'MeanAbsRelAngle', 'std': 'StdAbsRelAngle'}, inplace=True)
                pivoted_rel_angle = rel_angle_stats.pivot(index='individual', columns=beh_col, values=['MeanAbsRelAngle', 'StdAbsRelAngle'])
                cat_prefix_map = {
                    'masked_pose': 'Pose', 'masked_pose_valid': 'Pose',
                    'masked_orient': 'Orient', 'masked_orient_valid': 'Orient',
                    'masked_pose_orient': 'PoseOrient', 'masked_pose_orient_valid': 'PoseOrient',
                }
                prefix = f"Spatial_{cat_prefix_map.get(beh_col, 'Unknown')}"
                pivoted_rel_angle.columns = [f"{prefix}_{behavior}_{stat}".replace('+', 'plus') for stat, behavior in pivoted_rel_angle.columns]
                pivoted_rel_angle.reset_index(inplace=True)
                summary_df = _merge_feature_frame(summary_df, pivoted_rel_angle)

    total_frames = spatial_df.groupby('individual').size()
    total_minutes = total_frames / video_fps / 60.0

    # Global mean speed
    if 'speed_mm_s' in spatial_df.columns:
        global_speed = spatial_df.groupby('individual')['speed_mm_s'].mean().reset_index()
        global_speed.rename(columns={'speed_mm_s': 'Global_MeanSpeed_mm_s'}, inplace=True)
        summary_df = _merge_feature_frame(summary_df, global_speed)

    def _spatial_frac_df(context_name, mask):
        sub = spatial_df.loc[mask, ['individual', 'r_norm']].dropna(subset=['r_norm'])
        if sub.empty:
            return pd.DataFrame(columns=['individual', f'Spatial_Frac_{context_name}_Center', f'Spatial_Frac_{context_name}_Periphery'])
        counts = sub.groupby('individual').size()
        center_counts = sub[sub['r_norm'] <= SPATIAL_R_NORM_CENTER_THRESHOLD].groupby('individual').size()
        periph_counts = sub[sub['r_norm'] >= SPATIAL_R_NORM_PERIPHERY_THRESHOLD].groupby('individual').size()
        center_frac = center_counts.reindex(counts.index, fill_value=0) / counts
        periph_frac = periph_counts.reindex(counts.index, fill_value=0) / counts
        out = pd.DataFrame({
            'individual': counts.index,
            f'Spatial_Frac_{context_name}_Center': center_frac.values,
            f'Spatial_Frac_{context_name}_Periphery': periph_frac.values,
        }).reset_index(drop=True)
        too_small = counts < min_context_frames
        if too_small.any():
            out.loc[too_small.values, [f'Spatial_Frac_{context_name}_Center', f'Spatial_Frac_{context_name}_Periphery']] = np.nan
        return out

    masks = {'Overall': spatial_df['r_norm'].notna()}
    if pose_col in spatial_df.columns:
        pose_series = spatial_df[pose_col].astype(str)
        masks.update({
            'Walking': spatial_df[pose_col] == 'Walking',
            'Scramble': spatial_df[pose_col] == 'Scramble',
            'Grooming': pose_series.str.startswith('Groom-'),
        })
    if pose_orient_col in spatial_df.columns:
        po = spatial_df[pose_orient_col].astype(str)
        masks.update({
            'Walking_Floor': spatial_df[pose_orient_col] == 'Walking + Floor',
            'Walking_Ceiling': spatial_df[pose_orient_col] == 'Walking + Ceiling',
            'Standing_Floor': spatial_df[pose_orient_col] == 'Standing + Floor',
            'Standing_Ceiling': spatial_df[pose_orient_col] == 'Standing + Ceiling',
            'Grooming_Floor': po.str.startswith('Groom-') & po.str.endswith(' + Floor'),
            'Grooming_Ceiling': po.str.startswith('Groom-') & po.str.endswith(' + Ceiling'),
        })
    for ctx, m in masks.items():
        summary_df = _merge_feature_frame(summary_df, _spatial_frac_df(ctx, m))

    def _angle_stats(context_name, mask, col, prefix):
        sub = spatial_df.loc[mask, ['individual', col]].dropna(subset=[col])
        if sub.empty:
            return pd.DataFrame(columns=['individual', f'{prefix}_{context_name}_Mean_deg', f'{prefix}_{context_name}_P95_deg'])
        counts = sub.groupby('individual').size()
        mean_vals = sub.groupby('individual')[col].mean()
        p95_vals = sub.groupby('individual')[col].quantile(0.95)
        out = pd.DataFrame({
            'individual': counts.index,
            f'{prefix}_{context_name}_Mean_deg': mean_vals.reindex(counts.index).values,
            f'{prefix}_{context_name}_P95_deg': p95_vals.reindex(counts.index).values,
        }).reset_index(drop=True)
        too_small = counts < min_context_frames
        if too_small.any():
            out.loc[too_small.values, [f'{prefix}_{context_name}_Mean_deg', f'{prefix}_{context_name}_P95_deg']] = np.nan
        return out

    if pose_orient_col in spatial_df.columns:
        if 'anglefeature_Wings_L_R' in spatial_df.columns:
            wing_contexts = {
                'Standing_Floor': spatial_df[pose_orient_col] == 'Standing + Floor',
                'Standing_Ceiling': spatial_df[pose_orient_col] == 'Standing + Ceiling',
                'GroomWings_Floor': spatial_df[pose_orient_col] == 'Groom-Wings + Floor',
                'GroomWings_Ceiling': spatial_df[pose_orient_col] == 'Groom-Wings + Ceiling',
            }
            for cname, cmask in wing_contexts.items():
                summary_df = _merge_feature_frame(summary_df, _angle_stats(cname, cmask, 'anglefeature_Wings_L_R', prefix='Postural_WingSpread'))

        leg_contexts = {
            'Walking_Floor': spatial_df[pose_orient_col] == 'Walking + Floor',
            'Walking_Ceiling': spatial_df[pose_orient_col] == 'Walking + Ceiling',
            'Standing_Floor': spatial_df[pose_orient_col] == 'Standing + Floor',
            'Standing_Ceiling': spatial_df[pose_orient_col] == 'Standing + Ceiling',
        }
        leg_cols = [
            ('anglefeature_Legs_FL_FR', 'Postural_LegSpread_FL_FR'),
            ('anglefeature_Legs_ML_MR', 'Postural_LegSpread_ML_MR'),
            ('anglefeature_Legs_BL_BR', 'Postural_LegSpread_BL_BR'),
        ]
        for col, pref in leg_cols:
            if col not in spatial_df.columns:
                continue
            for cname, cmask in leg_contexts.items():
                summary_df = _merge_feature_frame(summary_df, _angle_stats(cname, cmask, col, prefix=pref))

    needed = {'individual', 'source_file', 'frame_idx', pose_col, orient_col}
    if needed.issubset(set(spatial_df.columns)):
        groom_pose_types = ['Groom-FrontLegs', 'Groom-Head', 'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs']
        K = len(groom_pose_types)

        def _collapse_wall(o):
            if o in ('Wall-Up', 'Wall-Down'):
                return 'Wall-Side'
            return o

        episode_rows = []
        for (ind, src), g in spatial_df.groupby(['individual', 'source_file']):
            g = g.sort_values('frame_idx')
            pose_seq = g[pose_col].astype(object).reset_index(drop=True)
            orient_seq = g[orient_col].astype(object).reset_index(drop=True)
            is_groom = pose_seq.astype(str).str.startswith('Groom-')
            in_ep = False
            ep_start = None
            last_groom = None
            gap = 0
            for i_frame, flag in enumerate(is_groom.values):
                if flag:
                    if not in_ep:
                        in_ep = True
                        ep_start = i_frame
                    last_groom = i_frame
                    gap = 0
                else:
                    if in_ep:
                        gap += 1
                        if gap > groom_episode_max_gap_frames:
                            if ep_start is not None and last_groom is not None:
                                s = ep_start
                                e = last_groom
                                dur_s = (e - s + 1) / video_fps
                                o_slice = orient_seq.iloc[s:e+1].dropna().astype(str).map(_collapse_wall)
                                dom_orient = o_slice.mode().iloc[0] if not o_slice.empty else 'Unknown'
                                p_slice = pose_seq.iloc[s:e+1].dropna().astype(str)
                                p_slice = p_slice[p_slice.isin(groom_pose_types)]
                                if p_slice.empty or K <= 1:
                                    ent_norm = np.nan
                                else:
                                    counts = p_slice.value_counts().values.astype(float)
                                    probs = counts / counts.sum()
                                    ent = -np.sum([p * np.log(p) for p in probs if p > 0])
                                    ent_norm = ent / np.log(K)
                                episode_rows.append({'individual': ind, 'dominant_orient': dom_orient, 'duration_s': dur_s, 'subtype_entropy_norm': ent_norm})
                            in_ep = False
                            ep_start = None
                            last_groom = None
                            gap = 0
            if in_ep and ep_start is not None and last_groom is not None:
                s = ep_start
                e = last_groom
                dur_s = (e - s + 1) / video_fps
                o_slice = orient_seq.iloc[s:e+1].dropna().astype(str).map(_collapse_wall)
                dom_orient = o_slice.mode().iloc[0] if not o_slice.empty else 'Unknown'
                p_slice = pose_seq.iloc[s:e+1].dropna().astype(str)
                p_slice = p_slice[p_slice.isin(groom_pose_types)]
                if p_slice.empty or K <= 1:
                    ent_norm = np.nan
                else:
                    counts = p_slice.value_counts().values.astype(float)
                    probs = counts / counts.sum()
                    ent = -np.sum([p * np.log(p) for p in probs if p > 0])
                    ent_norm = ent / np.log(K)
                episode_rows.append({'individual': ind, 'dominant_orient': dom_orient, 'duration_s': dur_s, 'subtype_entropy_norm': ent_norm})

        episodes_df = pd.DataFrame(episode_rows)

        def _agg_episode(ctx_name, orient_value=None):
            if episodes_df.empty:
                return pd.DataFrame(columns=['individual', f'BoutSummary_GroomEpisode_{ctx_name}_EpisodesPerMin', f'BoutSummary_GroomEpisode_{ctx_name}_MeanDuration_s', f'BoutSummary_GroomEpisode_{ctx_name}_SubtypeEntropyNorm'])
            dfc = episodes_df
            if orient_value is not None:
                dfc = dfc[dfc['dominant_orient'] == orient_value]
            if dfc.empty:
                return pd.DataFrame(columns=['individual', f'BoutSummary_GroomEpisode_{ctx_name}_EpisodesPerMin', f'BoutSummary_GroomEpisode_{ctx_name}_MeanDuration_s', f'BoutSummary_GroomEpisode_{ctx_name}_SubtypeEntropyNorm'])
            n_ep = dfc.groupby('individual').size()
            sum_dur = dfc.groupby('individual')['duration_s'].sum()
            mean_dur = dfc.groupby('individual')['duration_s'].mean()
            ep_per_min = n_ep / total_minutes.reindex(n_ep.index)
            w_num = (dfc['subtype_entropy_norm'] * dfc['duration_s']).groupby(dfc['individual']).sum()
            ent_w = w_num / sum_dur
            out = pd.DataFrame({
                'individual': n_ep.index,
                f'BoutSummary_GroomEpisode_{ctx_name}_EpisodesPerMin': ep_per_min.values,
                f'BoutSummary_GroomEpisode_{ctx_name}_MeanDuration_s': mean_dur.reindex(n_ep.index).values,
                f'BoutSummary_GroomEpisode_{ctx_name}_SubtypeEntropyNorm': ent_w.reindex(n_ep.index).values,
            }).reset_index(drop=True)
            too_small = sum_dur < MIN_CONTEXT_DURATION_S
            if too_small.any():
                out.loc[too_small.values, [f'BoutSummary_GroomEpisode_{ctx_name}_EpisodesPerMin', f'BoutSummary_GroomEpisode_{ctx_name}_MeanDuration_s', f'BoutSummary_GroomEpisode_{ctx_name}_SubtypeEntropyNorm']] = np.nan
            return out

        for ctx_name, orient_value in [('All', None), ('Floor', 'Floor'), ('Ceiling', 'Ceiling'), ('WallSide', 'Wall-Side')]:
            summary_df = _merge_feature_frame(summary_df, _agg_episode(ctx_name, orient_value))

    if bout_summary_df is not None and not bout_summary_df.empty:
        try:
            rep = bout_summary_df.groupby(['individual', 'category'])['n_bouts'].sum().unstack('category')
            def _switch_rate(colname, outname):
                if colname not in rep.columns:
                    return None
                transitions = (rep[colname] - 1).clip(lower=0)
                rate = transitions / total_minutes.reindex(transitions.index)
                tmp = rate.reset_index()
                tmp.columns = ['individual', outname]
                return tmp
            for colname, outname in [('Pose', 'Transition_Repertoire_Pose_SwitchRate_per_min'), ('Orient', 'Transition_Repertoire_Orient_SwitchRate_per_min'), ('Pose+Orient', 'Transition_Repertoire_PoseOrient_SwitchRate_per_min')]:
                summary_df = _merge_feature_frame(summary_df, _switch_rate(colname, outname))
        except Exception:
            pass

    def _entropy_from_timebudgets(prefix):
        cols = [c for c in master_matrix.columns if c.startswith(prefix)]
        if not cols:
            return None
        mat = master_matrix[cols].to_numpy(dtype=float)
        mat = np.where(np.isfinite(mat) & (mat > 0), mat, 0.0)
        row_sums = mat.sum(axis=1, keepdims=True)
        k = mat.shape[1]
        if k <= 1:
            return np.full((mat.shape[0],), np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.where(row_sums > 0, mat / row_sums, np.nan)
            logp = np.where(p > 0, np.log(p), 0.0)
            H = -np.nansum(np.where(p > 0, p * logp, 0.0), axis=1)
            Hn = H / np.log(k)
            Hn = np.where(row_sums.flatten() > 0, Hn, np.nan)
        return Hn

    ent_pose = _entropy_from_timebudgets('TimeBudget_Pose_')
    if ent_pose is not None:
        summary_df['Transition_Repertoire_Pose_EntropyNorm'] = ent_pose
    ent_or = _entropy_from_timebudgets('TimeBudget_Orient_')
    if ent_or is not None:
        summary_df['Transition_Repertoire_Orient_EntropyNorm'] = ent_or
    ent_po = _entropy_from_timebudgets('TimeBudget_PoseOrient_')
    if ent_po is not None:
        summary_df['Transition_Repertoire_PoseOrient_EntropyNorm'] = ent_po

    return summary_df


def _compute_spatial_feature_summary_worker(task: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, float, int, int]) -> pd.DataFrame:
    spatial_df, bout_summary_df, master_matrix, video_fps, min_context_frames, groom_episode_max_gap_frames = task
    return _compute_spatial_feature_summary_core(
        spatial_df,
        bout_summary_df,
        master_matrix,
        video_fps=float(video_fps),
        min_context_frames=int(min_context_frames),
        groom_episode_max_gap_frames=int(groom_episode_max_gap_frames),
    )


def compute_spatial_feature_summary_df(
    spatial_df: pd.DataFrame | None,
    bout_summary_df: pd.DataFrame | None,
    master_matrix: pd.DataFrame,
    *,
    video_fps: float = DEFAULT_VIDEO_FPS,
    num_workers: int = 1,
) -> pd.DataFrame | None:
    if spatial_df is None or spatial_df.empty:
        return None

    cols_to_load = [
        'individual', 'source_file', 'frame_idx', 'r_norm', 'relative_angle', 'speed_mm_s',
        'masked_pose', 'masked_orient', 'masked_pose_orient',
        'masked_pose_valid', 'masked_orient_valid', 'masked_pose_orient_valid',
        'anglefeature_Wings_L_R', 'anglefeature_Legs_FL_FR', 'anglefeature_Legs_ML_MR', 'anglefeature_Legs_BL_BR',
    ]
    spatial_df = spatial_df[[c for c in cols_to_load if c in spatial_df.columns]].copy()
    if spatial_df.empty:
        return None

    individuals = sorted(pd.unique(spatial_df['individual'].astype(str)).tolist())
    if not individuals:
        return None

    min_context_frames = int(round(MIN_CONTEXT_DURATION_S * float(video_fps)))
    groom_gap_frames = int(round(GROOM_EPISODE_MAX_NON_GROOM_GAP_S * float(video_fps)))

    worker_count = max(1, min(int(num_workers), len(individuals)))
    if worker_count <= 1 or len(individuals) <= 1:
        return _compute_spatial_feature_summary_core(
            spatial_df,
            bout_summary_df,
            master_matrix,
            video_fps=float(video_fps),
            min_context_frames=min_context_frames,
            groom_episode_max_gap_frames=groom_gap_frames,
        )

    n_chunks = max(worker_count * 2, worker_count)
    chunks = _ordered_chunks_for_workers(individuals, n_chunks)
    tasks = []
    for chunk in chunks:
        chunk_set = set(chunk)
        spatial_chunk = spatial_df[spatial_df['individual'].astype(str).isin(chunk_set)].copy()
        if spatial_chunk.empty:
            continue
        bout_chunk = None
        if bout_summary_df is not None and not bout_summary_df.empty:
            bout_chunk = bout_summary_df[bout_summary_df['individual'].astype(str).isin(chunk_set)].copy()
        master_chunk = master_matrix[master_matrix['individual'].astype(str).isin(chunk_set)].copy()
        tasks.append((spatial_chunk, bout_chunk, master_chunk, float(video_fps), int(min_context_frames), int(groom_gap_frames)))

    if not tasks:
        return None

    with Pool(processes=worker_count) as pool:
        results = pool.map(_compute_spatial_feature_summary_worker, tasks, chunksize=1)

    union_columns = ['individual']
    for df in results:
        for col in df.columns:
            if col not in union_columns:
                union_columns.append(col)

    aligned = [df.reindex(columns=union_columns) for df in results if df is not None and not df.empty]
    if not aligned:
        return None
    out = pd.concat(aligned, axis=0, ignore_index=True, sort=False)
    order = pd.Index(master_matrix['individual'].astype(str).drop_duplicates())
    out = out.set_index('individual').reindex(order).reset_index()
    return out

def _read_excel_sheets(path: str) -> dict[str, pd.DataFrame] | None:
    if not os.path.exists(path):
        return None
    xls = pd.ExcelFile(path)
    return {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}


def load_finalization_sources_from_output_folder(output_dir: str) -> MMGenFinalizationSources:
    """Load previously emitted mmgen output files into an ``MMGenFinalizationSources`` bundle."""

    def _read_csv_optional(filename: str) -> pd.DataFrame | None:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    bout_summary_df = _read_csv_optional('1_Individual_Bout_Summary.csv')
    measurements_df = None
    measurements_path = os.path.join(output_dir, '3_Standardized_Measurements.xlsx')
    if os.path.exists(measurements_path):
        try:
            measurements_df = pd.read_excel(measurements_path, sheet_name='Combined Measurements')
        except Exception:
            measurements_df = None

    spatial_data_path = os.path.join(output_dir, 'temp_transformed_positional_data.csv')
    if not os.path.exists(spatial_data_path):
        spatial_data_path = None

    return MMGenFinalizationSources(
        bout_summary_df=bout_summary_df,
        time_budget_sheets=_read_excel_sheets(os.path.join(output_dir, '2_Time_Budgets.xlsx')),
        measurements_df=measurements_df,
        postural_summary_df=_read_csv_optional('4_Postural_Features.csv'),
        kinematic_sheets=_read_excel_sheets(os.path.join(output_dir, '5_Kinematic_Features.xlsx')),
        tw_kinematic_sheets=_read_excel_sheets(os.path.join(output_dir, '5b_KinematicTW_Features.xlsx')),
        transition_pose_tidy_df=_read_csv_optional('6_Pose_All_Transitions_Tidy.csv'),
        transition_pose_orient_tidy_df=_read_csv_optional('7_Pose+Orient_All_Transitions_Tidy.csv'),
        sleep_summary_df=_read_csv_optional('12_Sleep_Features.csv'),
        spatial_data_path=spatial_data_path,
    )


def create_feature_matrix_from_sources(output_dir: str, sources: MMGenFinalizationSources, *, filter_features_by_bout_count: bool = DEFAULT_FILTER_FEATURES_BY_BOUT_COUNT, min_bout_count_for_features: int = DEFAULT_MIN_BOUT_COUNT_FOR_FEATURES, min_bout_count_for_transitions: int = DEFAULT_MIN_BOUT_COUNT_FOR_TRANSITIONS, mm_per_px: float = DEFAULT_MM_PER_PX, video_fps: float = DEFAULT_VIDEO_FPS, drop_all_na_columns: bool = True, spatial_num_workers: int = 1):
    """Assemble the master feature matrix directly from finalized mmgen tables."""
    print("Assembling master feature matrix from finalization sources...")

    mm_per_px = float(mm_per_px)
    video_fps = float(video_fps)
    min_context_frames = int(round(MIN_CONTEXT_DURATION_S * video_fps))
    groom_episode_max_gap_frames = int(round(GROOM_EPISODE_MAX_NON_GROOM_GAP_S * video_fps))

    bout_summary_df = sources.bout_summary_df.copy() if sources.bout_summary_df is not None else None
    if bout_summary_df is None or bout_summary_df.empty:
        print("Error: no bout summary source data found. Cannot create feature matrix.")
        return None

    # --- Step 1 & 2: Establish Base Table from Bout Summary ---
    base_df = bout_summary_df[['individual', 'group', 'sex']].drop_duplicates().reset_index(drop=True)
    print(f"Base table created with {len(base_df)} unique individuals.")

    # --- Step 3: Process and Pivot Bout Summary Features ---
    id_vars = ['individual', 'group', 'sex', 'category', 'bout_type']
    exclude_vars = ['n_bouts', 'total_length_frames']
    value_vars = [col for col in bout_summary_df.columns if col not in id_vars + exclude_vars]

    pivoted_bout_summary = bout_summary_df.pivot_table(
        index='individual',
        columns=['category', 'bout_type'],
        values=value_vars,
        observed=True
    )
    pivoted_bout_summary.columns = [f"BoutSummary_{cat}_{bout.replace(' ', '')}_{val}" for val, cat, bout in pivoted_bout_summary.columns]
    pivoted_bout_summary.reset_index(inplace=True)
    master_matrix = pd.merge(base_df, pivoted_bout_summary, on='individual', how='left')

    bout_count_pivot = bout_summary_df.pivot_table(
        index='individual',
        columns=['category', 'bout_type'],
        values='n_bouts',
        observed=True
    )
    bout_count_pivot.columns = [f"BoutSummary_{cat}_{bout.replace(' ', '')}_bout_count" for cat, bout in bout_count_pivot.columns]
    bout_count_pivot.reset_index(inplace=True)

    for bt in ['Jump', 'Scramble']:
        src = f"BoutSummary_Pose_{bt}_bout_count"
        dst = f"BoutSummary_Pose+Orient_{bt}_bout_count"
        if src in bout_count_pivot.columns and dst not in bout_count_pivot.columns:
            bout_count_pivot[dst] = bout_count_pivot[src]

    # --- Step 4: Process and Merge Time Budget Features ---
    time_budget_sheets = sources.time_budget_sheets or {}
    if time_budget_sheets:
        print("Processing time budget features...")

        def _process_sheet(df, sheet_name, prefix):
            df = df[~df['individual'].astype(str).str.contains('--- AVERAGE ---', na=False)]
            df = df.dropna(subset=['individual'])
            id_cols = ['individual', 'group']
            feature_cols = [col for col in df.columns if col not in id_cols and str(col).strip() != '' and '---' not in str(col)]
            rename_dict = {col: f"{prefix}_{str(col).replace(' | ', '_')}" for col in feature_cols}
            df = df.copy()
            df.rename(columns=rename_dict, inplace=True)
            cols_to_select = ['individual'] + list(rename_dict.values())
            return df[cols_to_select]

        sheets_to_process = {
            'Pose (Overall)': 'TimeBudget_Pose',
            'Orientation (Overall)': 'TimeBudget_Orient',
            'Pose-Orient (Overall)': 'TimeBudget_PoseOrient',
            'Pose given Orient (Overall)': 'TimeBudget_PoseGivenOrient',
            'Orient given Pose (Overall)': 'TimeBudget_OrientGivenPose',
        }
        for sheet_name, prefix in sheets_to_process.items():
            if sheet_name in time_budget_sheets:
                try:
                    processed_df = _process_sheet(time_budget_sheets[sheet_name].copy(), sheet_name, prefix)
                    master_matrix = pd.merge(master_matrix, processed_df, on='individual', how='left')
                    print(f"  - Merged features from '{sheet_name}' sheet.")
                except Exception as e:
                    print(f"  - Error processing sheet '{sheet_name}': {e}")
    else:
        print("Warning: time budget source sheets not found. Skipping time budget features.")

    # --- Step 5: Process and Merge Standardized Measurement Features ---
    measurements_source_df = sources.measurements_df
    if measurements_source_df is not None and not measurements_source_df.empty:
        print("Processing standardized measurement features...")
        try:
            measurements_df = measurements_source_df.copy()
            if {'individual_id', 'group_label'}.issubset(measurements_df.columns):
                measurement_order = [
                    'length_avg', 'headwidth_avg', 'thorax_width_avg', 'abdomenwidth_avg', 'winglength_avg',
                    'length_norm', 'headwidth_norm', 'abdomenwidth_norm', 'winglength_norm', 'bodyarea_top_norm', 'bodyarea_side_norm',
                    'length_side', 'headheight_side', 'thoraxheight_side', 'abdomenheight_side', 'body_area_top', 'body_area_side'
                ]
                blank_spacer_cols = [' ', '  ']
                combined_df = measurements_df.drop(columns=['sex'], errors='ignore').copy()
                means_cols = [f'{name}_mean' for name in measurement_order if f'{name}_mean' in combined_df.columns]
                std_cols = [f'{name}_std' for name in measurement_order if f'{name}_std' in combined_df.columns]
                n_samples_cols = [f'{name}_n_samples' for name in measurement_order if f'{name}_n_samples' in combined_df.columns]
                combined_df[blank_spacer_cols[0]], combined_df[blank_spacer_cols[1]] = '', ''
                desired_order = means_cols + [blank_spacer_cols[0]] + std_cols + [blank_spacer_cols[1]] + n_samples_cols
                combined_df.rename(columns={'individual_id': 'individual', 'group_label': 'group'}, inplace=True)
                measurements_df = format_summary_df(combined_df, desired_order)

            measurements_df = measurements_df[~measurements_df['individual'].astype(str).str.contains('--- AVERAGE ---', na=False)]
            measurements_df = measurements_df.dropna(subset=['individual'])
            id_cols = ['individual', 'group']
            feature_cols = [col for col in measurements_df.columns if col not in id_cols and not str(col).endswith('_n_samples') and str(col).strip() != '' and '---' not in str(col)]
            rename_dict = {col: f"StdMeasure_{col}" for col in feature_cols}
            measurements_df.rename(columns=rename_dict, inplace=True)
            cols_to_merge = ['individual'] + list(rename_dict.values())
            master_matrix = pd.merge(master_matrix, measurements_df[cols_to_merge], on='individual', how='left')
            print("  - Merged measurement features.")
        except Exception as e:
            print(f"  - Error processing standardized measurements: {e}")
    else:
        print("Warning: measurement source data not found. Skipping standardized measurement features.")

    # --- Step 6: Process and Merge Postural Features ---
    postural_df = sources.postural_summary_df
    if postural_df is not None and not postural_df.empty:
        print("Processing postural features...")
        try:
            postural_df = postural_df.copy()
            id_cols = ['individual', 'group']
            feature_cols = [col for col in postural_df.columns if col not in id_cols]
            feature_cols = [col for col in feature_cols if col != 'group']
            rename_dict = {col: f"Postural_{col}" for col in feature_cols}
            if 'group' in feature_cols or 'Postural_group' in rename_dict.values():
                raise ValueError("ERROR: 'group' column is being treated as a feature! This should not happen.")
            postural_df.rename(columns=rename_dict, inplace=True)
            cols_to_merge = ['individual'] + list(rename_dict.values())
            master_matrix = pd.merge(master_matrix, postural_df[cols_to_merge], on='individual', how='left')
            print("  - Merged postural features.")
        except Exception as e:
            print(f"  - Error processing postural features: {e}")
    else:
        print("Warning: postural source data not found. Skipping postural features.")

    # --- Step 7: Process and Merge Kinematic Features ---
    kinematic_sheets = sources.kinematic_sheets or {}
    if kinematic_sheets:
        print("Processing kinematic features...")
        for sheet_name, kin_df in kinematic_sheets.items():
            match = re.search(r'(Pose|Pose \+ Orient) \((inst|5|9)\)', sheet_name)
            if not match:
                continue
            category = match.group(1).replace(' + ', '+')
            try:
                kin_df = kin_df.copy()
                kin_df = kin_df[~kin_df['individual'].astype(str).str.contains('--- AVERAGE ---', na=False)]
                kin_df = kin_df.dropna(subset=['individual'])
                id_cols = ['individual', 'group']
                feature_cols = [col for col in kin_df.columns if col not in id_cols and '---' not in str(col) and str(col).strip() != '']
                rename_dict = {col: f"KinematicBB_{category}_{str(col).replace(' ', '')}" for col in feature_cols}
                kin_df.rename(columns=rename_dict, inplace=True)
                cols_to_merge = ['individual'] + list(rename_dict.values())
                master_matrix = pd.merge(master_matrix, kin_df[cols_to_merge], on='individual', how='left')
                print(f"  - Merged features from '{sheet_name}' sheet.")
            except Exception as e:
                print(f"  - Error processing sheet '{sheet_name}': {e}")
    else:
        print("Warning: kinematic source sheets not found. Skipping kinematic features.")

    # --- Step 7b: Process and Merge TW Kinematic Features ---
    tw_kinematic_sheets = sources.tw_kinematic_sheets or {}
    if tw_kinematic_sheets:
        print("Processing TW kinematic features...")
        for sheet_name, kin_df in tw_kinematic_sheets.items():
            match = re.search(r'(Pose|Pose \+ Orient) \((inst)\)', sheet_name)
            if not match:
                continue
            category = match.group(1).replace(' + ', '+')
            try:
                kin_df = kin_df.copy()
                if 'individual' in kin_df.columns:
                    kin_df = kin_df[~kin_df['individual'].astype(str).str.contains('--- AVERAGE ---', na=False)]
                    kin_df = kin_df.dropna(subset=['individual'])
                    raw_ids = kin_df['individual'].astype(str)
                    collapsed_ids = raw_ids.map(collapse_individual_id)
                    if (raw_ids != collapsed_ids).any():
                        raise ValueError(
                            "TW kinematic sheets must already be aggregated at the collapsed animal-level "
                            "individual ID before master-matrix assembly."
                        )
                    if collapsed_ids.duplicated().any():
                        dupes = collapsed_ids[collapsed_ids.duplicated()].unique().tolist()[:5]
                        raise ValueError(
                            "TW kinematic sheets contain duplicate collapsed individual IDs; "
                            f"example duplicates: {dupes}"
                        )
                id_cols = ['individual', 'group']
                feature_cols = [col for col in kin_df.columns if col not in id_cols and '---' not in str(col) and str(col).strip() != '']
                rename_dict = {col: f"KinematicTW_{category}_{str(col).replace(' ', '')}" for col in feature_cols}
                kin_df.rename(columns=rename_dict, inplace=True)
                cols_to_merge = ['individual'] + list(rename_dict.values())
                master_matrix = pd.merge(master_matrix, kin_df[cols_to_merge], on='individual', how='left')
                print(f"  - Merged TW features from '{sheet_name}' sheet.")
            except ValueError:
                raise
            except Exception as e:
                print(f"  - Error processing TW sheet '{sheet_name}': {e}")
    else:
        print("Warning: TW kinematic source sheets not found. Skipping TW kinematic features.")

    # --- Step 8: Process and Merge Transition Features ---
    transition_sources = {
        'Transition_Pose': sources.transition_pose_tidy_df,
        'Transition_PoseOrient': sources.transition_pose_orient_tidy_df,
    }
    for prefix, trans_df in transition_sources.items():
        if trans_df is None or trans_df.empty:
            continue
        print(f"Processing transition features from '{prefix}'...")
        try:
            trans_df = trans_df.copy()
            trans_df['transition'] = trans_df['from_behavior'] + "_to_" + trans_df['to_behavior']
            pivoted_trans = trans_df.pivot_table(
                index='individual',
                columns='transition',
                values='probability',
                observed=True
            )
            pivoted_trans.columns = [f"{prefix}_{col.replace(' ', '')}" for col in pivoted_trans.columns]
            pivoted_trans.reset_index(inplace=True)
            master_matrix = pd.merge(master_matrix, pivoted_trans, on='individual', how='left')
            print(f"  - Merged features from '{prefix}'.")
        except Exception as e:
            print(f"  - Error processing transition features from '{prefix}': {e}")

    # --- Step 8.5: Process and Merge Sleep Features ---
    sleep_df = sources.sleep_summary_df
    if sleep_df is not None and not sleep_df.empty:
        print("Processing sleep features...")
        try:
            sleep_df = sleep_df.copy()
            sleep_df = sleep_df[~sleep_df['individual'].astype(str).str.contains('--- AVERAGE ---', na=False)]
            sleep_df = sleep_df.dropna(subset=['individual'])
            id_cols = ['individual', 'group', 'sex', 'bout_type']
            feature_cols = [col for col in sleep_df.columns if col not in id_cols and '---' not in str(col) and str(col).strip() != '']
            if feature_cols:
                sleep_pivot = sleep_df.pivot_table(
                    index='individual',
                    columns='bout_type',
                    values=feature_cols,
                    observed=True
                )
                sleep_pivot.columns = [f"SleepFeat_{bout.replace(' ', '')}_{val}" for val, bout in sleep_pivot.columns]
                sleep_pivot.reset_index(inplace=True)
                master_matrix = pd.merge(master_matrix, sleep_pivot, on='individual', how='left')
                print("  - Merged sleep features.")
            else:
                print("  - Warning: No sleep feature columns found in source data.")
        except Exception as e:
            print(f"  - Error processing sleep features: {e}")
    else:
        print("Warning: sleep feature source data not found. Skipping sleep features.")

    # --- Step 9: Process and Merge Spatial / Occupancy-Derived Features ---
    spatial_data_path = sources.spatial_data_path
    spatial_df = sources.spatial_df.copy() if sources.spatial_df is not None else None
    if spatial_df is not None and spatial_df.empty:
        spatial_df = None
    if spatial_df is not None or (spatial_data_path and os.path.exists(spatial_data_path)):
        spatial_source_label = 'in-memory transformed spatial data' if spatial_df is not None else spatial_data_path
        print("Processing spatial-derived features...")
        try:
            if spatial_df is None:
                cols_to_load = [
                    'individual', 'source_file', 'frame_idx', 'r_norm', 'relative_angle', 'speed_mm_s',
                    'masked_pose', 'masked_orient', 'masked_pose_orient',
                    'masked_pose_valid', 'masked_orient_valid', 'masked_pose_orient_valid',
                    'anglefeature_Wings_L_R', 'anglefeature_Legs_FL_FR', 'anglefeature_Legs_ML_MR', 'anglefeature_Legs_BL_BR',
                ]
                spatial_df = pd.read_csv(spatial_data_path, usecols=lambda c: c in cols_to_load)

            spatial_summary_df = compute_spatial_feature_summary_df(
                spatial_df,
                bout_summary_df=bout_summary_df,
                master_matrix=master_matrix,
                video_fps=video_fps,
                num_workers=int(spatial_num_workers),
            )
            if spatial_summary_df is not None and not spatial_summary_df.empty:
                spatial_feature_cols = [c for c in spatial_summary_df.columns if c != 'individual']
                master_matrix = pd.merge(master_matrix, spatial_summary_df, on='individual', how='left')
                print(f"  - Merged {len(spatial_feature_cols)} spatial-derived features.")
            else:
                print("  - No spatial-derived features were computed.")
        except Exception as e:
            print(f"Error processing spatial-derived features from {spatial_source_label}: {e}")
            print("Skipping spatial-derived features.")
    else:
        print("Warning: spatial source data not found. Skipping spatial-derived features.")




    # --- Step 10: Filter Features by Bout Count (if enabled) ---
    if filter_features_by_bout_count:
        print(f"\nFiltering features with minimum bout count of {min_bout_count_for_features} (transitions: {min_bout_count_for_transitions})...")
        # Pass the separate bout_count_pivot DataFrame (created in Step 3) for filtering
        # This allows filtering without including bout_count as a feature in the master matrix
        master_matrix = _filter_features_by_bout_count(
            master_matrix, 
            min_bout_count_for_features, 
            min_bout_count_for_transitions,
            bout_count_df=bout_count_pivot
        )

    # --- Final Cleanup: Remove empty and redundant columns ---
    print("\nCleaning final master matrix before saving...")

    # 1. Optionally drop any columns that are fully empty
    if bool(drop_all_na_columns):
        cols_before_nan_drop = len(master_matrix.columns)
        master_matrix.dropna(axis=1, how='all', inplace=True)
        cols_after_nan_drop = len(master_matrix.columns)
        if cols_before_nan_drop != cols_after_nan_drop:
            print(f"  -> Dropped {cols_before_nan_drop - cols_after_nan_drop} fully empty columns.")

    # 2. Define and drop your list of redundant features (optional)
    redundant_features_to_drop = [
        # Add any feature names here if needed later
    ]
    if redundant_features_to_drop: # Only drop if list is not empty
        cols_before_manual_drop = len(master_matrix.columns)
        master_matrix.drop(columns=redundant_features_to_drop, inplace=True, errors='ignore')
        cols_after_manual_drop = len(master_matrix.columns)
        if cols_before_manual_drop != cols_after_manual_drop:
            print(f"  -> Dropped {cols_before_manual_drop - cols_after_manual_drop} predefined redundant features.")

    # Validate stable row identity before saving.
    if 'individual' not in master_matrix.columns:
        raise ValueError("MASTER_FEATURE_MATRIX is missing required row identity column 'individual'.")
    dup_mask = master_matrix['individual'].astype(str).duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(master_matrix.loc[dup_mask, 'individual'].astype(str).unique().tolist())
        raise ValueError(
            "Duplicate master-matrix row identities detected in 'individual': "
            + ", ".join(dupes[:10])
            + (" ..." if len(dupes) > 10 else "")
        )

    # Impose canonical family-grouped column ordering before saving.
    master_matrix = reorder_master_matrix_columns(master_matrix)

    # Save the final master matrix
    output_path_final = os.path.join(output_dir, 'MASTER_FEATURE_MATRIX.csv')
    master_matrix.to_csv(output_path_final, index=False)
    print(f"\nMaster feature matrix is complete and saved to {output_path_final}")

    # Save feature units manifest alongside the matrix
    try:
        write_feature_units_manifest(
            master_matrix=master_matrix,
            output_dir=output_dir,
            mm_per_px=mm_per_px,
            fps=video_fps
        )
    except Exception as e:
        print(f"WARNING: Failed to write MASTER_FEATURE_MATRIX unit manifest: {e}")

    gc.collect()

    return master_matrix

def create_feature_matrix(output_dir):
    """Wrapper that loads emitted mmgen output files and assembles the master matrix."""
    sources = load_finalization_sources_from_output_folder(output_dir)
    return create_feature_matrix_from_sources(output_dir, sources)

def validate_feature_matrix(matrix_path, required_id_cols=['individual', 'group', 'sex']):
    """
    Validates that a feature matrix has the expected structure.
    
    Args:
        matrix_path: Path to the feature matrix CSV file
        required_id_cols: List of required ID columns that must be present
    
    Returns:
        tuple: (is_valid, error_message, df)
            - is_valid: Boolean indicating if matrix is valid
            - error_message: None if valid, error string if invalid
            - df: Loaded DataFrame if valid, None if invalid
    """
    try:
        df = pd.read_csv(matrix_path)
        
        # Check required ID columns
        missing_cols = [col for col in required_id_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}", None
        
        # Check for feature columns
        feature_cols = [col for col in df.columns if col not in required_id_cols]
        if len(feature_cols) == 0:
            return False, "No feature columns found", None
        
        # Check for empty dataframe
        if len(df) == 0:
            return False, "DataFrame is empty", None
        
        return True, None, df
        
    except Exception as e:
        return False, f"Error loading matrix: {str(e)}", None




# ============================================================================
# FILE I/O AND FORMATTING
# ============================================================================

def format_summary_df(pivot_df, desired_order, id_cols=['individual', 'group']):
    """
    A general utility to format a summary DataFrame with group averages and separators.
    Handles NaNs, which become blank cells in Excel.
    """
    # Ensure all id_cols are present as columns, not index
    if any(col in pivot_df.index.names for col in id_cols):
        pivot_df = pivot_df.reset_index()

    # Include non-feature separator columns in the final set
    final_cols = id_cols + [col for col in desired_order if col in pivot_df.columns]
    # Reindex and ensure all id_cols are present
    pivot_df = pivot_df.reindex(columns=final_cols)

    # 1. Identify numeric columns and grouping columns
    numeric_cols = pivot_df.select_dtypes(include=np.number).columns.tolist()
    group_by_cols = [c for c in id_cols if c != 'individual']
    if not group_by_cols:
        group_by_cols = ['group'] if 'group' in id_cols else []
    
    group_summary = pd.DataFrame()
    
    # 2. Calculate the group average. The .mean() function correctly ignores NaNs.
    if group_by_cols:
        pivot_df_copy = pivot_df.copy() # De-fragment before groupby
        group_summary = pivot_df_copy.groupby(group_by_cols, as_index=False, observed=True)[numeric_cols].mean()
        group_summary.insert(0, 'individual', '--- AVERAGE ---')
        group_summary = group_summary.reindex(columns=final_cols, fill_value='')

    # Sort by all identifier columns for consistent ordering
    pivot_df = pivot_df.sort_values(by=id_cols)
    
    processed_dfs = []
    separator = pd.DataFrame([[''] * len(final_cols)], columns=final_cols)
    
    # Use 'group' for inserting separators if it exists, otherwise don't separate
    primary_group_col = 'group' if 'group' in group_by_cols else None
    
    if primary_group_col:
        for i, (group_name, group_df) in enumerate(pivot_df.groupby(primary_group_col, observed=True)):
            processed_dfs.append(group_df)
            if i < len(pivot_df[primary_group_col].unique()) - 1:
                processed_dfs.append(separator)
        individual_data = pd.concat(processed_dfs, ignore_index=True)
    else:
        individual_data = pivot_df # No grouping, just use the sorted df

    # The final DataFrame will have NaNs, which is what we want.
    return pd.concat([group_summary, separator, individual_data], ignore_index=True)

def save_to_excel_sheets(sheets_dict, output_path, sheet_order=None):
    """
    Saves multiple DataFrames to a single Excel file, each on its own sheet.

    Notes:
    - Excel sheet titles are limited to 31 characters. To avoid openpyxl warnings
      sheet names are sanitized, truncated, and made unique.
    - Header cells are left-aligned for readability.
    """
    sheets_to_process = sheet_order if sheet_order else list(sheets_dict.keys())

    def _sanitize_sheet_name(name: str) -> str:
        # Excel forbids: : \ / ? * [ ]
        invalid = [':', '\\', '/', '?', '*', '[', ']']
        for ch in invalid:
            name = name.replace(ch, '_')
        name = name.strip()
        return name or 'Sheet'

    def _make_unique_sheet_name(name: str, used: set[str]) -> str:
        name = _sanitize_sheet_name(str(name))
        max_len = 31
        base = name[:max_len]
        if base not in used:
            used.add(base)
            return base

        # If collision, append a numeric suffix while keeping <= 31 chars.
        for i in range(1, 1000):
            suffix = f"_{i}"
            avail = max_len - len(suffix)
            candidate = f"{name[:avail]}{suffix}"
            if candidate not in used:
                used.add(candidate)
                return candidate

        # Extremely unlikely unless thousands of collisions occur.
        raise ValueError(f"Could not generate a unique Excel sheet name for: {name!r}")

    used_names: set[str] = set()
    name_map: dict[str, str] = {}

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for orig_name in sheets_to_process:
            if orig_name not in sheets_dict:
                continue
            df = sheets_dict[orig_name]
            if df is None or df.empty:
                continue

            safe_name = _make_unique_sheet_name(orig_name, used_names)
            name_map[str(orig_name)] = safe_name
            df.to_excel(writer, sheet_name=safe_name, index=False)

        # Apply left alignment to headers in all written sheets
        workbook = writer.book
        for safe_name in name_map.values():
            if safe_name in workbook.sheetnames:
                worksheet = workbook[safe_name]
                for cell in worksheet[1]:
                    cell.alignment = Alignment(horizontal='left')

    print(f"Saved Excel file with {len(name_map)} sheets to {output_path}")

