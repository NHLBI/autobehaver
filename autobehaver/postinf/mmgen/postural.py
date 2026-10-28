"""Extract standing-frame posture and summarize it by animal and orientation.

Use centered KeyPre keypoints and explicit millimeters-per-pixel calibration.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .naming import collapse_individual_id
from .feature_schema import feature_columns

DEFAULT_MM_PER_PX: float = 0.0169896744624207

def _calculate_postural_features(poscalc_df, masked_poses, masked_orients, video_path, group, individual, *, mm_per_px: float = DEFAULT_MM_PER_PX, likelihood_threshold=0.25, verbose_debug=False):
    """Extract pre-calculated postural features and calculates summary metrics.
    - Calculates average leg extensions for front, mid, and back pairs.
    - Reads pre-calculated stance_area and detailed asymmetry indices.
    - Calculates combined asymmetry magnitudes and overall asymmetry scores.
    """
    if verbose_debug:
        print(f"\n--- Postural Analysis for: {os.path.basename(video_path)} ---")
        print(f"Total frames received: {len(poscalc_df)}")

    # 1. Define Keypoints & Feature Columns
    leg_tip_kps = ['FrontTarsusLeft', 'FrontTarsusRight', 'MidTarsusLeft', 'MidTarsusRight', 'BackTarsusLeft', 'BackTarsusRight']
    body_axis_kps = ['Antennae', 'AbdomenTip']
    
    # Define the leg pairs for averaging
    leg_pairs = {
        'FrontLegs': ('FrontTarsusLeft', 'FrontTarsusRight'),
        'MidLegs': ('MidTarsusLeft', 'MidTarsusRight'),
        'BackLegs': ('BackTarsusLeft', 'BackTarsusRight')
    }
    
    precalc_features = [
        'stance_area',
        'asym_lr_x_Eyes', 'asym_lr_y_Eyes', 'asym_lr_x_Thorax', 'asym_lr_y_Thorax',
        'asym_lr_x_Abdomen', 'asym_lr_y_Abdomen', 'asym_lr_x_Wings', 'asym_lr_y_Wings',
        'asym_lr_x_FrontLegs', 'asym_lr_y_FrontLegs', 'asym_lr_x_MidLegs', 'asym_lr_y_MidLegs',
        'asym_lr_x_BackLegs', 'asym_lr_y_BackLegs', 'asym_tb_x_Head', 'asym_tb_y_Head',
        'asym_tb_x_Thorax', 'asym_tb_y_Thorax', 'asym_tb_x_Abdomen', 'asym_tb_y_Abdomen'
    ]
    
    extension_req_cols = [f"{kp}_x" for kp in leg_tip_kps + body_axis_kps] + \
                         [f"{kp}_y" for kp in leg_tip_kps + body_axis_kps] + \
                         [f"{kp}_likelihood" for kp in leg_tip_kps + body_axis_kps] 
    
    missing_cols = [col for col in extension_req_cols + precalc_features if col not in poscalc_df.columns]
    if missing_cols:
        if verbose_debug:
            print(f"Result: Prerequisite columns are missing. Skipping postural analysis.")
            print(f"Missing columns: {missing_cols}")
        return None

    # Only these inputs can contribute to postural output. Avoid copying the
    # entire engineered-feature table for every Standing/orientation subset.
    needed = set(extension_req_cols + precalc_features)
    needed.update(c for c in poscalc_df if c.endswith('_likelihood'))
    needed.add('orient_Wall-Side')
    poscalc_df = poscalc_df[[c for c in poscalc_df if c in needed]]

    # 2. Filter for High-Quality "Standing" Frames (Floor, Ceiling, and Wall-Side)
    likelihood_cols = [f"{kp}_likelihood" for kp in leg_tip_kps + body_axis_kps]
    
    # Helper function to calculate features for Floor or Ceiling (stance and leg extensions)
    def calculate_stance_and_extensions(orient_filter):
        """Extract eligible standing-frame stance and paired-leg extension features for
        one orientation.
        """
        behavior_mask = (masked_poses == 'Standing') & (masked_orients == orient_filter)
        valid_indices = np.where(behavior_mask)[0]
        
        if verbose_debug: print(f"Frames that are 'Standing' AND '{orient_filter}': {len(valid_indices)}")
        if len(valid_indices) == 0: return None

        poscalc_df_filtered = poscalc_df.iloc[valid_indices].copy()
        likelihood_mask = (poscalc_df_filtered[likelihood_cols] > likelihood_threshold).all(axis=1)
        final_poscalc_df = poscalc_df_filtered[likelihood_mask]

        if verbose_debug: print(f"Frames passing likelihood threshold for '{orient_filter}': {len(final_poscalc_df)}")
        if final_poscalc_df.empty: return None

        # Exclude body_area features (they're handled by StdMeasure, not Postural)
        cols_to_keep = [col for col in final_poscalc_df.columns 
                        if col not in ['body_area_top', 'body_area_side']]
        results_df = final_poscalc_df[cols_to_keep].copy()

        # KeyPre exports centered/rotated keypoints: the centroid is (0, 0).
        # Absolute image centroid columns must never be subtracted here.
        # Average tarsus-to-centroid distance in this common coordinate frame.
        for pair_name, (kp_left, kp_right) in leg_pairs.items():
            dist_left = np.hypot(results_df[f'{kp_left}_x'], results_df[f'{kp_left}_y'])
            dist_right = np.hypot(results_df[f'{kp_right}_x'], results_df[f'{kp_right}_y'])
            results_df[f'extension_raw_{pair_name}_avg'] = (dist_left + dist_right) / 2

        # Add orientation column
        results_df['masked_orient'] = orient_filter
        
        return results_df
    
    # Helper function to calculate LR asymmetry for Floor (top-down view)
    def calculate_lr_asymmetry():
        """Extract standing Floor left/right asymmetries and their aggregate magnitudes."""
        behavior_mask = (masked_poses == 'Standing') & (masked_orients == 'Floor')
        valid_indices = np.where(behavior_mask)[0]
        
        if verbose_debug: print(f"Frames that are 'Standing' AND 'Floor' (for LR asymmetry): {len(valid_indices)}")
        if len(valid_indices) == 0: return None

        poscalc_df_filtered = poscalc_df.iloc[valid_indices].copy()
        likelihood_mask = (poscalc_df_filtered[likelihood_cols] > likelihood_threshold).all(axis=1)
        final_poscalc_df = poscalc_df_filtered[likelihood_mask]

        if verbose_debug: print(f"Frames passing likelihood threshold for Floor (LR asymmetry): {len(final_poscalc_df)}")
        if final_poscalc_df.empty: return None

        # Exclude body_area features
        cols_to_keep = [col for col in final_poscalc_df.columns 
                        if col not in ['body_area_top', 'body_area_side']]
        results_df = final_poscalc_df[cols_to_keep].copy()

        # Calculate LR Asymmetry Magnitudes (Left-Right, from top-down view)
        lr_asym_pairs = {
            'Eyes_LR': ('asym_lr_x_Eyes', 'asym_lr_y_Eyes'),
            'Thorax_LR': ('asym_lr_x_Thorax', 'asym_lr_y_Thorax'),
            'Abdomen_LR': ('asym_lr_x_Abdomen', 'asym_lr_y_Abdomen'),
            'Wings_LR': ('asym_lr_x_Wings', 'asym_lr_y_Wings'),
            'FrontLegs_LR': ('asym_lr_x_FrontLegs', 'asym_lr_y_FrontLegs'),
            'MidLegs_LR': ('asym_lr_x_MidLegs', 'asym_lr_y_MidLegs'),
            'BackLegs_LR': ('asym_lr_x_BackLegs', 'asym_lr_y_BackLegs')
        }
        for name, (x_col, y_col) in lr_asym_pairs.items():
            results_df[f'asymmetry_magnitude_{name}'] = np.sqrt(results_df[x_col]**2 + results_df[y_col]**2)

        # Calculate Overall LR Asymmetry Score
        lr_mag_cols = [f'asymmetry_magnitude_{name}' for name in lr_asym_pairs.keys()]
        results_df['overall_asymmetry_LR'] = results_df[lr_mag_cols].mean(axis=1)

        # Add orientation column
        results_df['masked_orient'] = 'Floor'
        
        return results_df
    
    # Helper function to calculate TB asymmetry for Wall-Side (side view)
    def calculate_tb_asymmetry():
        # First, check if orientation probability column exists
        """Extract standing Wall-Side top/bottom asymmetries and their aggregate
        magnitudes.
        """
        orient_prob_col = 'orient_Wall-Side'
        has_orient_prob = orient_prob_col in poscalc_df.columns
        
        # Create behavior mask with orientation probability filter (if available)
        if has_orient_prob:
            behavior_mask = (masked_poses == 'Standing') & \
                           (masked_orients == 'Wall-Side') & \
                           (poscalc_df[orient_prob_col] >= 0.75)
            if verbose_debug:
                print(f"Using orientation probability filter (>= 0.75) for Wall-Side")
        else:
            behavior_mask = (masked_poses == 'Standing') & (masked_orients == 'Wall-Side')
            if verbose_debug:
                print(f"Warning: '{orient_prob_col}' column not found. Using basic orientation filter only.")
        
        valid_indices = np.where(behavior_mask)[0]
        
        if verbose_debug: print(f"Frames that are 'Standing' AND 'Wall-Side' (for TB asymmetry): {len(valid_indices)}")
        if len(valid_indices) == 0: return None

        poscalc_df_filtered = poscalc_df.iloc[valid_indices].copy()
        
        # For Wall-Side, use a different set of keypoints (no leg tips needed)
        # Use keypoints relevant for side-view and TB asymmetry calculation
        side_view_kps = ['Antennae', 'AbdomenTip', 'Ocelli', 'Mouthparts', 
                         'ThoraxTop', 'ThoraxBottom', 'AbdomenTop', 'AbdomenBottom']
        side_view_likelihood_cols = [f"{kp}_likelihood" for kp in side_view_kps 
                                     if f"{kp}_likelihood" in poscalc_df_filtered.columns]
        
        # Apply likelihood filter only if we have the required columns
        if side_view_likelihood_cols:
            likelihood_mask = (poscalc_df_filtered[side_view_likelihood_cols] > likelihood_threshold).all(axis=1)
            final_poscalc_df = poscalc_df_filtered[likelihood_mask]
            if verbose_debug: 
                print(f"Frames passing side-view likelihood threshold (keypoints: {len(side_view_likelihood_cols)}) for Wall-Side (TB asymmetry): {len(final_poscalc_df)}")
        else:
            # If keypoint columns don't exist, skip likelihood filtering
            final_poscalc_df = poscalc_df_filtered
            if verbose_debug:
                print(f"Warning: Side-view keypoint likelihood columns not found. Skipping likelihood filter for Wall-Side.")

        if final_poscalc_df.empty: return None

        # Exclude body_area features
        cols_to_keep = [col for col in final_poscalc_df.columns 
                        if col not in ['body_area_top', 'body_area_side', 'stance_area']]
        results_df = final_poscalc_df[cols_to_keep].copy()

        # Calculate TB Asymmetry Magnitudes (Top-Bottom, from side view)
        tb_asym_pairs = {
            'Head_TB': ('asym_tb_x_Head', 'asym_tb_y_Head'),
            'Thorax_TB': ('asym_tb_x_Thorax', 'asym_tb_y_Thorax'),
            'Abdomen_TB': ('asym_tb_x_Abdomen', 'asym_tb_y_Abdomen')
        }
        for name, (x_col, y_col) in tb_asym_pairs.items():
            results_df[f'asymmetry_magnitude_{name}'] = np.sqrt(results_df[x_col]**2 + results_df[y_col]**2)

        # Calculate Overall TB Asymmetry Score
        tb_mag_cols = [f'asymmetry_magnitude_{name}' for name in tb_asym_pairs.keys()]
        results_df['overall_asymmetry_TB'] = results_df[tb_mag_cols].mean(axis=1)

        # Add orientation column
        results_df['masked_orient'] = 'Wall-Side'
        
        return results_df
    
    # Calculate features for Floor (stance, leg extensions, and LR asymmetry)
    floor_stance_ext_df = calculate_stance_and_extensions('Floor')
    floor_lr_asym_df = calculate_lr_asymmetry()
    
    # Calculate features for Ceiling (only stance and leg extensions, no asymmetry)
    ceiling_df = calculate_stance_and_extensions('Ceiling')
    
    # Calculate TB asymmetry for Wall-Side
    wallside_tb_asym_df = calculate_tb_asymmetry()
    
    # Combine all dataframes
    combined_results = []
    
    # For Floor: merge stance/extensions with LR asymmetry
    if floor_stance_ext_df is not None and not floor_stance_ext_df.empty:
        if floor_lr_asym_df is not None and not floor_lr_asym_df.empty:
            # Both should have the same number of rows since they're from the same filtered data
            if len(floor_stance_ext_df) == len(floor_lr_asym_df):
                # Reset index to align by position
                floor_stance_ext_reset = floor_stance_ext_df.reset_index(drop=True)
                floor_lr_asym_reset = floor_lr_asym_df.reset_index(drop=True)
                
                # Select only asymmetry columns from LR asymmetry dataframe
                lr_asym_cols = [col for col in floor_lr_asym_reset.columns 
                               if 'asymmetry' in col]
                # Merge the asymmetry columns
                floor_merged = floor_stance_ext_reset.copy()
                for col in lr_asym_cols:
                    if col in floor_lr_asym_reset.columns:
                        floor_merged[col] = floor_lr_asym_reset[col].values
                # Ensure masked_orient is set correctly
                floor_merged['masked_orient'] = 'Floor'
                combined_results.append(floor_merged)
            else:
                # Mismatch in row counts - use only stance/extensions and warn
                if verbose_debug:
                    print(f"Warning: Floor stance/extensions ({len(floor_stance_ext_df)} rows) and LR asymmetry ({len(floor_lr_asym_df)} rows) have different row counts. Using only stance/extensions.")
                combined_results.append(floor_stance_ext_df)
        else:
            # Only stance/extensions, no LR asymmetry
            combined_results.append(floor_stance_ext_df)
    
    if ceiling_df is not None and not ceiling_df.empty:
        combined_results.append(ceiling_df)
    
    if wallside_tb_asym_df is not None and not wallside_tb_asym_df.empty:
        combined_results.append(wallside_tb_asym_df)
    
    if not combined_results:
        if verbose_debug: print("Result: No valid frames found for Floor, Ceiling, or Wall-Side.")
        return None
    
    # Concatenate results
    results_df = pd.concat(combined_results, ignore_index=True)
    
    mm_per_px = float(DEFAULT_MM_PER_PX if mm_per_px is None else mm_per_px)

    # 3. Finalize and Return DataFrame
    # Define base feature columns (stance and extensions - may exist for Floor/Ceiling)
    final_feature_cols = []
    if 'stance_area' in results_df.columns:
        final_feature_cols.append('stance_area')
    final_feature_cols.extend([f'extension_raw_{pair_name}_avg' for pair_name in leg_pairs.keys() 
                              if f'extension_raw_{pair_name}_avg' in results_df.columns])
    
    # Add LR asymmetry columns (only for Floor)
    lr_asym_names = ['Eyes_LR', 'Thorax_LR', 'Abdomen_LR', 'Wings_LR', 
                     'FrontLegs_LR', 'MidLegs_LR', 'BackLegs_LR']
    for name in lr_asym_names:
        col_name = f'asymmetry_magnitude_{name}'
        if col_name in results_df.columns:
            final_feature_cols.append(col_name)
    if 'overall_asymmetry_LR' in results_df.columns:
        final_feature_cols.append('overall_asymmetry_LR')
    
    # Add TB asymmetry columns (only for Wall-Side)
    tb_asym_names = ['Head_TB', 'Thorax_TB', 'Abdomen_TB']
    for name in tb_asym_names:
        col_name = f'asymmetry_magnitude_{name}'
        if col_name in results_df.columns:
            final_feature_cols.append(col_name)
    if 'overall_asymmetry_TB' in results_df.columns:
        final_feature_cols.append('overall_asymmetry_TB')
                         
    final_results_df = results_df[final_feature_cols + ['masked_orient']].copy()
    final_results_df['group'] = group
    final_results_df['individual'] = individual

    # UNIT CONVERSION: pixels -> mm (distances) and px^2 -> mm^2 (areas)
    # stance_area: px^2 -> mm^2
    if 'stance_area' in final_results_df.columns:
        final_results_df['stance_area'] = final_results_df['stance_area'] * (mm_per_px ** 2)

    # extension_raw_* and asymmetry magnitudes: px -> mm
    for col in final_results_df.columns:
        if col.startswith('extension_raw_') or col.startswith('asymmetry_magnitude_') or col.startswith('overall_asymmetry_'):
            final_results_df[col] = final_results_df[col] * mm_per_px

    if verbose_debug: 
        floor_count = (final_results_df['masked_orient'] == 'Floor').sum()
        ceiling_count = (final_results_df['masked_orient'] == 'Ceiling').sum()
        wallside_count = (final_results_df['masked_orient'] == 'Wall-Side').sum()
        print(f"Result: Successfully extracted {len(final_results_df)} frames of postural features ({floor_count} Floor, {ceiling_count} Ceiling, {wallside_count} Wall-Side).")

    return final_results_df



def calculate_and_save_postural_summary(master_postural_df, output_dir, *, save_outputs: bool = True):
    """Summarize numeric posture features by animal and Floor, Ceiling, or Wall-Side
    orientation.
    """
    if master_postural_df is None or master_postural_df.empty:
        print("\nNo postural features were extracted. Skipping postural summary.")
        return

    print("\nCalculating and saving postural feature summary...")

    # 1. Aggregate Data by Animal and Orientation
    df = master_postural_df.copy()
    # Use collapse function for consistency (mapping not available in this function scope)
    # This function is called after mapping is created, but we need to use the collapse function
    # since we don't have access to the mapping here
    identities = {value: collapse_individual_id(value) for value in df['individual'].unique()}
    df['animal_id'] = df['individual'].map(identities)
    
    # Ensure masked_orient column exists (should be added by _calculate_postural_features)
    if 'masked_orient' not in df.columns:
        # If missing, assume all are Floor.
        df['masked_orient'] = 'Floor'
        print("Warning: 'masked_orient' column not found. Assuming all rows are 'Floor'.")

    # Per-recording identities support safe resume/purge and are metadata,
    # never measurements. Restrict summary inputs to numeric feature columns.
    excluded = {'group', 'individual', 'animal_id', 'source_file', 'sex',
                'frame_idx', 'Frame Number', 'masked_pose', 'masked_orient',
                'body_area_top', 'body_area_side', 'normalizer'}
    feature_cols = [col for col in df.select_dtypes(include=np.number).columns
                    if col not in excluded]
    
    # Group by group, animal_id, and masked_orient
    summary_df = df.groupby(['group', 'animal_id', 'masked_orient'])[feature_cols].agg(['mean', 'std']).reset_index()
    
    return _format_postural_summary(summary_df, output_dir, save_outputs=save_outputs)


def _format_postural_summary(summary_df, output_dir, *, save_outputs=True):
    # Pivot/unstack the masked_orient level to create separate columns for Floor and Ceiling
    # After groupby.agg, columns are MultiIndex: (feature_name, stat)
    # Set index to group, animal_id, masked_orient
    """Format posture statistics into orientation-prefixed feature columns and
    optionally write the workbook.
    """
    summary_df_indexed = summary_df.set_index(['group', 'animal_id', 'masked_orient'])
    
    # Unstack the masked_orient level
    # After unstack, columns become MultiIndex: (feature_name, stat, masked_orient)
    summary_df_pivoted = summary_df_indexed.unstack(level='masked_orient')
    
    # Flatten the MultiIndex columns: (feature_name, stat, orient) -> feature_name_stat_orient
    # Handle both 2-level (if only one orient exists) and 3-level MultiIndex
    flattened_cols = []
    for col in summary_df_pivoted.columns.values:
        if isinstance(col, tuple):
            if len(col) == 3:
                # (feature_name, stat, orient)
                feature_name, stat, orient = col
                flattened_cols.append(f'{feature_name}_{stat}_{orient}')
            elif len(col) == 2:
                # (feature_name, stat) - shouldn't happen after unstack, but handle it
                feature_name, stat = col
                flattened_cols.append(f'{feature_name}_{stat}')
            else:
                flattened_cols.append('_'.join(str(c) for c in col))
        else:
            flattened_cols.append(str(col))
    
    summary_df_pivoted.columns = flattened_cols
    
    # Reset index to get group and animal_id back as columns
    summary_df = summary_df_pivoted.reset_index()
    
    # Identify columns that should NOT get the Postural_ prefix (identifier columns)
    id_cols = ['group', 'animal_id']
    
    # Rename columns to include Floor/Ceiling/Wall-Side prefix (without Postural_ prefix)
    # The Postural_ prefix will be added later in create_feature_matrix, consistent with other feature categories
    # For columns that end with _Floor, _Ceiling, or _Wall-Side, add the orientation prefix before the feature name
    # IMPORTANT: Exclude identifier columns from getting the prefix
    rename_dict = {}
    for col in summary_df.columns:
        # Skip identifier columns - they should never get any prefix
        if col in id_cols:
            continue
        elif col.endswith('_Floor'):
            # e.g., 'stance_area_mean_Floor' -> 'Floor_stance_area_mean'
            feature_part = col.replace('_Floor', '')
            rename_dict[col] = f'Floor_{feature_part}'
        elif col.endswith('_Ceiling'):
            # e.g., 'stance_area_mean_Ceiling' -> 'Ceiling_stance_area_mean'
            feature_part = col.replace('_Ceiling', '')
            rename_dict[col] = f'Ceiling_{feature_part}'
        elif col.endswith('_Wall-Side'):
            # e.g., 'overall_asymmetry_TB_mean_Wall-Side' -> 'Wall-Side_overall_asymmetry_TB_mean'
            feature_part = col.replace('_Wall-Side', '')
            rename_dict[col] = f'Wall-Side_{feature_part}'
        else:
            # This shouldn't happen if pivoting worked correctly, but handle it
            # Assume it is a Floor-only feature.
            # BUT: Double-check it's not an identifier column before applying prefix
            if col not in id_cols:
                rename_dict[col] = f'Floor_{col}'
    
    summary_df.rename(columns=rename_dict, inplace=True)
    summary_df.rename(columns={'animal_id': 'individual'}, inplace=True)

    # Unstack creates empty feature/orientation combinations that do not apply,
    # such as top/bottom asymmetry on Floor. Keep declared but unavailable
    # features; reject undeclared measurements instead of discarding their data.
    declared = {name.removeprefix('Postural_') for name in feature_columns()
                if name.startswith('Postural_')}
    unexpected = sorted(set(summary_df.columns) - {'group', 'individual'} - declared)
    populated = [name for name in unexpected if summary_df[name].notna().any()]
    if populated:
        raise ValueError('Postural summary contains measurements for undeclared features: '
                         f'{["Postural_" + name for name in populated]}')
    summary_df.drop(columns=unexpected, inplace=True)
    
    # Ensure the identifier column keeps its unprefixed name.
    assert 'group' in summary_df.columns, "ERROR: 'group' column missing from postural summary"
    assert 'Postural_group' not in summary_df.columns, "ERROR: 'Postural_group' was incorrectly created! The 'group' column should never get the Postural_ prefix."

    # Ensure raw extensions are positive.
    if not summary_df.empty:
        numeric_cols = summary_df.select_dtypes(include=np.number).columns
        # Check if the entire numeric part is NaN (could happen if input was all NaN after filtering)
        assert not summary_df[numeric_cols].isnull().all().all(), "Entire postural summary DataFrame is NaN after aggregation."

        # Check a physically constrained value, e.g., raw extensions should have non-negative means
        # Column names include Floor_, Ceiling_, or Wall-Side_ prefixes before the Postural_ prefix is added.
        ext_mean_cols = [c for c in summary_df.columns if 'extension_raw_' in c and c.endswith('_mean')]
        if ext_mean_cols:
             # Check for negative values after dropping NaNs
             assert (summary_df[ext_mean_cols].dropna() >= 0).all().all(), f"Negative mean raw leg extensions found in columns: {ext_mean_cols}"
        # Check std deviations are non-negative
        std_cols = [c for c in summary_df.columns if c.endswith('_std')]
        if std_cols:
             assert (summary_df[std_cols].dropna() >= 0).all().all(), f"Negative standard deviations found in columns: {std_cols}"
        print(" - Assert Checks Passed: Basic sanity checks on final postural summary (non-negative means/stds).")

    # 2. Save Formatted Summary File
    if save_outputs:
        summary_output_path = os.path.join(output_dir, '4_Postural_Features.csv')
        summary_df.to_csv(summary_output_path, index=False)
        print(f"Saved postural feature summary to {summary_output_path}")
    return summary_df


def calculate_postural_summary_from_store(store_path, output_dir, *, normalizer_map=None,
                                         individual_to_animal=None, save_outputs=True,
                                         chunk_rows=4096):
    """Calculate animal-level postural means and sample deviations from bounded HDF
    chunks.
    """
    from .hdf_store import HDF_KEYS, open_store
    from autobehaver.preinf.keyprenorm.normparams import merge_centered_moments

    excluded = {'group', 'individual', 'animal_id', 'source_file', 'sex', 'frame_idx',
                'Frame Number', 'masked_pose', 'masked_orient', 'body_area_top',
                'body_area_side', 'normalizer'}
    moments = {}
    feature_cols = None
    identities = {}
    mapped_normalizers = 0
    with open_store(store_path, 'r') as store:
        key = HDF_KEYS['postural']
        if key not in store:
            return None
        for frame in store.select(key, iterator=True, chunksize=int(chunk_rows)):
            for value in frame['individual'].unique():
                identities.setdefault(value, collapse_individual_id(value))
            frame['animal_id'] = frame['individual'].map(identities)
            if normalizer_map is not None:
                norm_ids = frame['individual'].map({value: (individual_to_animal or {}).get(
                    str(value), identities[value]) for value in frame['individual'].unique()})
                normalizers = norm_ids.map(normalizer_map)
                valid = normalizers.dropna()
                mapped_normalizers += len(valid)
                if len(valid) and not (valid > 0).all():
                    raise ValueError('Mapped normalizer values must be positive.')
                for col in list(frame.columns):
                    if col.startswith('extension_raw_'):
                        frame[col.replace('_raw_', '_norm_')] = frame[col] / normalizers
            if feature_cols is None:
                feature_cols = [col for col in frame.select_dtypes(include=np.number)
                                if col not in excluded]
            for group, block in frame.groupby(['group', 'animal_id', 'masked_orient'],
                                              sort=False, observed=True):
                values = block[feature_cols].to_numpy(dtype=np.float64, copy=True)
                if np.isinf(values).any():
                    raise ValueError('Postural summary contains infinite feature values.')
                count = np.sum(np.isfinite(values), axis=0, dtype=np.int64)
                mean = np.divide(np.nansum(values, axis=0), count,
                                 out=np.zeros(len(feature_cols)), where=count > 0)
                np.subtract(values, mean, out=values)
                np.square(values, out=values)
                m2 = np.nansum(values, axis=0)
                if group in moments:
                    mean, m2, count = merge_centered_moments(*moments[group], mean, m2, count)
                moments[group] = (mean, m2, count)
    if not moments:
        return None
    if normalizer_map is not None and not mapped_normalizers:
        raise ValueError('Normalizer mapping failed for ALL animals!')
    groups = sorted(moments)
    rows = []
    for group in groups:
        mean, m2, count = moments[group]
        mean = np.where(count > 0, mean, np.nan)
        variance = np.divide(m2, count-1, out=np.full(len(count), np.nan), where=count > 1)
        rows.append(np.column_stack([mean, np.sqrt(variance)]).ravel())
    summary = pd.DataFrame(rows,
        index=pd.MultiIndex.from_tuples(groups, names=['group', 'animal_id', 'masked_orient']),
        columns=pd.MultiIndex.from_product([feature_cols, ['mean', 'std']])).reset_index()
    return _format_postural_summary(summary, output_dir, save_outputs=save_outputs)


