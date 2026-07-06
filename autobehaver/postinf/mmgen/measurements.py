"""Standardized measurement extraction and pooling.

This module contains the standardized measurement routines used during
mmgen finalization. It keeps the measurement extraction and pooling logic in one
place while exposing explicit calibration/runtime arguments.

Notes
-----
- Unit conversion uses an explicit ``mm_per_px`` argument.
- Reproducible pooling/sampling uses an explicit ``random_seed`` argument.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .feature_matrix import format_summary_df, save_to_excel_sheets
from .naming import collapse_individual_id

DEFAULT_MM_PER_PX: float = 0.0169896744624207
DEFAULT_RANDOM_SEED: int = 42

def extract_standardized_measurements_raw(masked_poses, masked_orients, poscalc_df, individual_id, group_label, sex):
    """
    Extracts raw data for standardized animal measurements under controlled conditions.
    Uses a pose hierarchy to find valid frames.
    """
    masked_poses = pd.Series(masked_poses).reset_index(drop=True)
    masked_orients = pd.Series(masked_orients).reset_index(drop=True)
    poscalc_df = poscalc_df.reset_index(drop=True)

    raw_data = {'individual_id': individual_id, 'group_label': group_label, 'sex': sex}

    # 1. Define the pose hierarchy as requested
    pose_priority = [
        "Standing",
        "Walking",
        "Groom-FrontLegs",
        "Groom-Head",
        "Groom-SideLeg",
        "Groom-Wings",
        "Groom-Abdomen",
        "Groom-BackLegs"
    ]

    # 2. Redefine definitions to store orientations and keypoints
    measurement_definitions = {
        # name: ([orient_list], (kps...))
        'length_top':         (["Floor"],       ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y')),
        'length_side':        (["Wall-Side"],   ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y')),
        'length_bottom':      (["Ceiling"],     ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y')),
        'headwidth_top':      (["Floor"],       ('EyeRight_x', 'EyeRight_y', 'EyeLeft_x', 'EyeLeft_y')),
        'headwidth_bottom':   (["Ceiling"],     ('EyeRight_x', 'EyeRight_y', 'EyeLeft_x', 'EyeLeft_y')),
        'headheight_side':    (["Wall-Side"],   ('Ocelli_x', 'Ocelli_y', 'Mouthparts_x', 'Mouthparts_y')),
        'thoraxwidth_top':    (["Floor"],       ('ThoraxLeft_x', 'ThoraxLeft_y', 'ThoraxRight_x', 'ThoraxRight_y')),
        'thoraxwidth_bottom': (["Ceiling"],     ('ThoraxLeft_x', 'ThoraxLeft_y', 'ThoraxRight_x', 'ThoraxRight_y')),
        'thoraxheight_side':  (["Wall-Side"],   ('ThoraxTop_x', 'ThoraxTop_y', 'ThoraxBottom_x', 'ThoraxBottom_y')),
        'abdomenwidth_top':   (["Floor"],       ('AbdomenLeft_x', 'AbdomenLeft_y', 'AbdomenRight_x', 'AbdomenRight_y')),
        'abdomenwidth_bottom':(["Ceiling"],     ('AbdomenLeft_x', 'AbdomenLeft_y', 'AbdomenRight_x', 'AbdomenRight_y')),
        'abdomenheight_side': (["Wall-Side"],   ('AbdomenTop_x', 'AbdomenTop_y', 'AbdomenBottom_x', 'AbdomenBottom_y')),
        'wingLeftlength_top': (["Floor"],       ('WingLeft_x', 'WingLeft_y', 'ThoraxLeft_x', 'ThoraxLeft_y')),
        'wingRightlength_top':(["Floor"],       ('WingRight_x', 'WingRight_y', 'ThoraxRight_x', 'ThoraxRight_y')),
        'body_area_top':      (["Floor"],       ('body_area_top', None, None, None)),
        'body_area_side':     (["Wall-Side"],   ('body_area_side', None, None, None)),
    }

    # 3. Iterate through measurements, applying the pose hierarchy
    for name, (orient_list, keypoints) in measurement_definitions.items():
        kpx1, kpy1, kpx2, kpy2 = keypoints
        
        # Combine all orientations for this measurement (e.g., ["Floor", "Ceiling"])
        orient_mask = masked_orients.isin(orient_list)
        
        # Iterate through poses in order of priority
        for pose in pose_priority:
            pose_mask = (masked_poses == pose)
            final_mask = (pose_mask & orient_mask)
            
            filtered_df = pd.DataFrame() # Reset for each pose
            
            # Check if this is a direct feature (body_area) or distance
            if kpx1 is not None and kpy1 is not None and kpx2 is not None and kpy2 is not None:
                # Traditional distance
                required_cols = [kpx1, kpy1, kpx2, kpy2]
                if all(col in poscalc_df.columns for col in required_cols):
                    filtered_df = poscalc_df.loc[final_mask, required_cols].copy().dropna()
            else:
                # Direct feature
                feature_col = kpx1
                if feature_col in poscalc_df.columns:
                    filtered_df = poscalc_df.loc[final_mask, [feature_col]].copy().dropna()

            # If we found data, store it, and stop searching for this measurement
            if not filtered_df.empty:
                raw_data[name] = filtered_df
                break  # This moves to the next measurement (e.g., 'length_side')
    
    return raw_data


def extract_standardized_measurements_summary(
    masked_poses,
    masked_orients,
    poscalc_df,
    individual_id,
    group_label,
    sex,
    *,
    max_samples: int = 100,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """Extract per-recording summary statistics for standardized measurements.

    This is a memory-safe alternative to :func:`extract_standardized_measurements_raw`.
    Instead of returning frame-level DataFrames (which can be very large and
    cause long-run memory accumulation in mmgen), this returns a single dict
    containing mean/std/n_samples for each measurement.

    The selection logic follows the current raw-extraction contract:
      - use a pose hierarchy to find valid frames
      - restrict to required orientations per measurement
      - stop at the first pose in the hierarchy that yields valid samples

    Sampling is performed *within each recording* to cap compute and keep
    deterministic behavior.
    """

    masked_poses = pd.Series(masked_poses).reset_index(drop=True)
    masked_orients = pd.Series(masked_orients).reset_index(drop=True)
    poscalc_df = poscalc_df.reset_index(drop=True)

    # Pose hierarchy used by the raw extractor
    pose_priority = [
        "Standing",
        "Walking",
        "Groom-FrontLegs",
        "Groom-Head",
        "Groom-SideLeg",
        "Groom-Wings",
        "Groom-Abdomen",
        "Groom-BackLegs",
    ]

    # Measurement definitions used by the raw extractor
    measurement_definitions = {
        # name: ([orient_list], (kps...))
        "length_top": (["Floor"], ("Antennae_x", "Antennae_y", "AbdomenTip_x", "AbdomenTip_y")),
        "length_side": (["Wall-Side"], ("Antennae_x", "Antennae_y", "AbdomenTip_x", "AbdomenTip_y")),
        "length_bottom": (["Ceiling"], ("Antennae_x", "Antennae_y", "AbdomenTip_x", "AbdomenTip_y")),
        "headwidth_top": (["Floor"], ("EyeRight_x", "EyeRight_y", "EyeLeft_x", "EyeLeft_y")),
        "headwidth_bottom": (["Ceiling"], ("EyeRight_x", "EyeRight_y", "EyeLeft_x", "EyeLeft_y")),
        "headheight_side": (["Wall-Side"], ("Ocelli_x", "Ocelli_y", "Mouthparts_x", "Mouthparts_y")),
        "thoraxwidth_top": (["Floor"], ("ThoraxLeft_x", "ThoraxLeft_y", "ThoraxRight_x", "ThoraxRight_y")),
        "thoraxwidth_bottom": (["Ceiling"], ("ThoraxLeft_x", "ThoraxLeft_y", "ThoraxRight_x", "ThoraxRight_y")),
        "thoraxheight_side": (["Wall-Side"], ("ThoraxTop_x", "ThoraxTop_y", "ThoraxBottom_x", "ThoraxBottom_y")),
        "abdomenwidth_top": (["Floor"], ("AbdomenLeft_x", "AbdomenLeft_y", "AbdomenRight_x", "AbdomenRight_y")),
        "abdomenwidth_bottom": (["Ceiling"], ("AbdomenLeft_x", "AbdomenLeft_y", "AbdomenRight_x", "AbdomenRight_y")),
        "abdomenheight_side": (["Wall-Side"], ("AbdomenTop_x", "AbdomenTop_y", "AbdomenBottom_x", "AbdomenBottom_y")),
        "wingLeftlength_top": (["Floor"], ("WingLeft_x", "WingLeft_y", "ThoraxLeft_x", "ThoraxLeft_y")),
        "wingRightlength_top": (["Floor"], ("WingRight_x", "WingRight_y", "ThoraxRight_x", "ThoraxRight_y")),
        "body_area_top": (["Floor"], ("body_area_top", None, None, None)),
        "body_area_side": (["Wall-Side"], ("body_area_side", None, None, None)),
    }

    row = {"individual_id": individual_id, "group_label": group_label, "sex": sex}

    for name, (orient_list, keypoints) in measurement_definitions.items():
        kpx1, kpy1, kpx2, kpy2 = keypoints
        orient_mask = masked_orients.isin(orient_list)
        found_vals = None

        for pose in pose_priority:
            pose_mask = masked_poses == pose
            final_mask = pose_mask & orient_mask

            if not final_mask.any():
                continue

            # Distance measurement
            if kpx1 is not None and kpy1 is not None and kpx2 is not None and kpy2 is not None:
                required_cols = [kpx1, kpy1, kpx2, kpy2]
                if not all(col in poscalc_df.columns for col in required_cols):
                    continue
                df = poscalc_df.loc[final_mask, required_cols].copy().dropna()
                if df.empty:
                    continue
                if max_samples and len(df) > max_samples:
                    df = df.sample(n=max_samples, random_state=random_seed)
                vals = np.sqrt((df[kpx1] - df[kpx2]) ** 2 + (df[kpy1] - df[kpy2]) ** 2)
                found_vals = vals
                break

            # Direct feature measurement
            feature_col = kpx1
            if feature_col is not None and feature_col in poscalc_df.columns:
                s = poscalc_df.loc[final_mask, feature_col].copy().dropna()
                if s.empty:
                    continue
                if max_samples and len(s) > max_samples:
                    s = s.sample(n=max_samples, random_state=random_seed)
                found_vals = s
                break

        if found_vals is not None and len(found_vals) > 0:
            row[f"{name}_mean"] = float(found_vals.mean())
            # Pandas std uses ddof=1; keep the established summary convention
            row[f"{name}_std"] = float(found_vals.std()) if len(found_vals) > 1 else np.nan
            row[f"{name}_n_samples"] = int(len(found_vals))
        else:
            row[f"{name}_mean"] = np.nan
            row[f"{name}_std"] = np.nan
            row[f"{name}_n_samples"] = 0

    return row



def _calculate_formatted_sex_separated_df(measurements_df, measurement_order, spacer_columns):
    """
    Creates a formatted standardized measurement summary with sex-separated columns and returns a DataFrame.
    """
    if 'sex' not in measurements_df.columns:
        print("Sex column not found in measurement data. Skipping sex-separated summary.")
        return pd.DataFrame()

    # 1. Separate data by sex
    male_df = measurements_df[measurements_df['sex'] == 'MALE'].copy()
    female_df = measurements_df[measurements_df['sex'] == 'FEMALE'].copy()

    # 2. Rename columns
    for col in male_df.columns:
        if '_mean' in col or '_std' in col or '_n_samples' in col:
            new_col = col.replace('_mean', '_MALE_mean').replace('_std', '_MALE_std').replace('_n_samples', '_MALE_n_samples')
            male_df.rename(columns={col: new_col}, inplace=True)

    for col in female_df.columns:
        if '_mean' in col or '_std' in col or '_n_samples' in col:
            new_col = col.replace('_mean', '_FEMALE_mean').replace('_std', '_FEMALE_std').replace('_n_samples', '_FEMALE_n_samples')
            female_df.rename(columns={col: new_col}, inplace=True)

    # 3. Merge into a wide-format DataFrame
    male_df.drop(columns=['sex'], inplace=True, errors='ignore')
    female_df.drop(columns=['sex'], inplace=True, errors='ignore')

    if male_df.empty:
        merged_df = female_df
    elif female_df.empty:
        merged_df = male_df
    else:
        merged_df = pd.merge(male_df, female_df, on=['individual_id', 'group_label'], how='outer')

    if merged_df.empty:
        print("No male or female data to create a sex-separated summary.")
        return pd.DataFrame()

    # 4. Define column order
    mean_cols = [item for name in measurement_order for item in (f'{name}_MALE_mean', f'{name}_FEMALE_mean') if item in merged_df.columns]
    std_cols = [item for name in measurement_order for item in (f'{name}_MALE_std', f'{name}_FEMALE_std') if item in merged_df.columns]
    n_samples_cols = [item for name in measurement_order for item in (f'{name}_MALE_n_samples', f'{name}_FEMALE_n_samples') if item in merged_df.columns]

    merged_df[spacer_columns[0]] = ''
    merged_df[spacer_columns[1]] = ''
    desired_order = mean_cols + [spacer_columns[0]] + std_cols + [spacer_columns[1]] + n_samples_cols

    # 5. Format the DataFrame and return it
    # format_summary_df returns a DataFrame for the workbook writer.
    # Standardize column names before passing to the formatter
    merged_df.rename(columns={'individual_id': 'individual', 'group_label': 'group'}, inplace=True)
    
    # Define the identifier columns for the formatter
    id_cols = ['individual', 'group'] # Add 'sex' if it exists as a column here

    return format_summary_df(merged_df, desired_order, id_cols=id_cols)



def _calculate_sex_ratio_df(measurements_df, measurement_order):
    """
    Calculates a summary of female-to-male size ratios for each group,
    including averaged and normalized features.
    """
    print("Calculating sex-based size ratios...")
    if 'sex' not in measurements_df.columns or 'group_label' not in measurements_df.columns:
        return pd.DataFrame()

    mean_cols = [col for col in measurements_df.columns if col.endswith('_mean')]
    group_sex_means = measurements_df.groupby(['group_label', 'sex'])[mean_cols].mean(numeric_only=True)

    if group_sex_means.empty:
        return pd.DataFrame()
    pivoted_means = group_sex_means.unstack(level='sex')

    ratio_df = pd.DataFrame(index=pivoted_means.index)
    for col_name in mean_cols:
        female_col = (col_name, 'FEMALE')
        male_col = (col_name, 'MALE')
        if female_col in pivoted_means.columns and male_col in pivoted_means.columns:
            ratio_df[f"{col_name.replace('_mean', '_ratio')}"] = pivoted_means[female_col] / pivoted_means[male_col]
            
    # Dynamically create the ratio names from the master measurement_order list
    final_col_order = [f"{name}_ratio" for name in measurement_order]
    
    final_cols_exist = [col for col in final_col_order if col in ratio_df.columns]
    ratio_df = ratio_df[final_cols_exist]

    return ratio_df.reset_index()



def calculate_and_save_pooled_measurements(all_measurements_raw, output_dir, max_samples=100, *, mm_per_px: float = DEFAULT_MM_PER_PX, random_seed: int = DEFAULT_RANDOM_SEED, save_outputs: bool = True):
    """
    Pools raw measurement data, calculates base statistics, computes averaged and 
    normalized features, saves a consolidated Excel summary, and returns the full feature DataFrame.
    """
    # ------------------------------------------------------------------
    # INPUT NORMALIZATION
    #
    # Earlier mmgen paths passed a list[dict] where each dict contains
    # frame-level DataFrames for each measurement (output of
    # ``extract_standardized_measurements_raw``). That approach can cause
    # long-run memory accumulation on large datasets.
    #
    # Newer mmgen runs may instead pass per-recording *summary rows* produced
    # by ``extract_standardized_measurements_summary``.
    #
    # We support both formats here.
    # ------------------------------------------------------------------

    if all_measurements_raw is None:
        return pd.DataFrame()

    # Case 1: already a table of summary rows
    if isinstance(all_measurements_raw, pd.DataFrame):
        measurements_df = all_measurements_raw.copy()

    # Case 2: list of dict summary rows
    elif isinstance(all_measurements_raw, list) and all_measurements_raw and isinstance(all_measurements_raw[0], dict) and any(
        str(k).endswith('_mean') for k in all_measurements_raw[0].keys()
    ):
        measurements_df = pd.DataFrame(all_measurements_raw)

    # Case 3: raw frame-level dict payloads
    else:
        # --- 1. Pool raw data by individual and measurement type ---
        pooled_data = {}
        for video_data in all_measurements_raw:
            individual_id = video_data['individual_id']
            group_label = video_data['group_label']
            sex = video_data.get('sex', 'Unknown')
            if individual_id not in pooled_data:
                pooled_data[individual_id] = {'group_label': group_label, 'sex': sex}
            for key, df in video_data.items():
                if key not in ['individual_id', 'group_label', 'sex']:
                    if key not in pooled_data[individual_id]:
                        pooled_data[individual_id][key] = []
                    pooled_data[individual_id][key].append(df)
        for individual_id, data in pooled_data.items():
            for key, df_list in data.items():
                if key not in ['group_label', 'sex']:
                    pooled_data[individual_id][key] = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

        # Sample, calculate base stats, and store final results
        final_results = []
        measurement_definitions = {
            'length_top': ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y'), 'length_side': ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y'),
            'length_bottom': ('Antennae_x', 'Antennae_y', 'AbdomenTip_x', 'AbdomenTip_y'), 'headwidth_top': ('EyeRight_x', 'EyeRight_y', 'EyeLeft_x', 'EyeLeft_y'),
            'headwidth_bottom': ('EyeRight_x', 'EyeRight_y', 'EyeLeft_x', 'EyeLeft_y'), 'headheight_side': ('Ocelli_x', 'Ocelli_y', 'Mouthparts_x', 'Mouthparts_y'),
            'thoraxwidth_top': ('ThoraxLeft_x', 'ThoraxLeft_y', 'ThoraxRight_x', 'ThoraxRight_y'), 'thoraxwidth_bottom': ('ThoraxLeft_x', 'ThoraxLeft_y', 'ThoraxRight_x', 'ThoraxRight_y'),
            'thoraxheight_side': ('ThoraxTop_x', 'ThoraxTop_y', 'ThoraxBottom_x', 'ThoraxBottom_y'), 'abdomenwidth_top': ('AbdomenLeft_x', 'AbdomenLeft_y', 'AbdomenRight_x', 'AbdomenRight_y'),
            'abdomenwidth_bottom': ('AbdomenLeft_x', 'AbdomenLeft_y', 'AbdomenRight_x', 'AbdomenRight_y'), 'abdomenheight_side': ('AbdomenTop_x', 'AbdomenTop_y', 'AbdomenBottom_x', 'AbdomenBottom_y'),
            'wingLeftlength_top': ('WingLeft_x', 'WingLeft_y', 'ThoraxLeft_x', 'ThoraxLeft_y'), 'wingRightlength_top': ('WingRight_x', 'WingRight_y', 'ThoraxRight_x', 'ThoraxRight_y'),
            'body_area_top': ('body_area_top', None, None, None), 'body_area_side': ('body_area_side', None, None, None),
        }
        for individual_id, data in pooled_data.items():
            result_row = {'individual_id': individual_id, 'group_label': data['group_label'], 'sex': data['sex']}
            for name, cols in measurement_definitions.items():
                if name in data and not data[name].empty:
                    df_sample = data[name].sample(n=min(max_samples, len(data[name])), random_state=random_seed)
                    if cols[1] is not None:
                        vals = np.sqrt((df_sample[cols[0]] - df_sample[cols[2]])**2 + (df_sample[cols[1]] - df_sample[cols[3]])**2)
                    else:
                        vals = df_sample[cols[0]]
                    result_row[f'{name}_mean'] = vals.mean()
                    result_row[f'{name}_std'] = vals.std()
                    result_row[f'{name}_n_samples'] = len(vals)
                else:
                    result_row[f'{name}_mean'], result_row[f'{name}_std'], result_row[f'{name}_n_samples'] = np.nan, np.nan, 0
            final_results.append(result_row)
        
        if not final_results:
            return pd.DataFrame() # Return empty DataFrame if no data
            
        measurements_df = pd.DataFrame(final_results)

    if measurements_df is None or measurements_df.empty:
        return pd.DataFrame()

    required_cols = {'individual_id', 'group_label'}
    missing_required = required_cols - set(measurements_df.columns)
    if missing_required:
        raise ValueError(f"Missing required measurement columns: {sorted(missing_required)}")
    
    # Collapse individual_id to animal_id and aggregate per animal (not per movie)
    # This ensures consistency with other outputs that use collapsed animal IDs
    measurements_df['animal_id'] = measurements_df['individual_id'].map(collapse_individual_id)
    # Aggregate by animal_id, taking mean of all measurement features
    # Keep group_label and sex (they should be consistent per animal)
    id_cols = ['animal_id', 'group_label', 'sex']
    feature_cols = [col for col in measurements_df.columns if col not in id_cols]
    measurements_df = measurements_df.groupby(id_cols)[feature_cols].mean(numeric_only=True).reset_index()
    measurements_df.rename(columns={'animal_id': 'individual_id'}, inplace=True)

    # --- Calculate Averaged and Normalized Features (with fallbacks) ---

    # 1. Calculate Averages for top/bottom views.
    #    .mean(axis=1) automatically ignores NaNs.
    #    If 'length_top_mean' is 10 and 'length_bottom_mean' is NaN, the result is 10.
    #    If both are NaN, the result is NaN.
    measurements_df['winglength_avg_mean'] = measurements_df[['wingLeftlength_top_mean', 'wingRightlength_top_mean']].mean(axis=1)
    measurements_df['length_avg_mean'] = measurements_df[['length_top_mean', 'length_bottom_mean']].mean(axis=1)
    measurements_df['headwidth_avg_mean'] = measurements_df[['headwidth_top_mean', 'headwidth_bottom_mean']].mean(axis=1)
    measurements_df['thorax_width_avg_mean'] = measurements_df[['thoraxwidth_top_mean', 'thoraxwidth_bottom_mean']].mean(axis=1)
    measurements_df['abdomenwidth_avg_mean'] = measurements_df[['abdomenwidth_top_mean', 'abdomenwidth_bottom_mean']].mean(axis=1)

    # 2. Implement the backup plan.
    #    If the average is still NaN (meaning both top/bottom failed),
    #    fill it with the corresponding side/height measurement.
    measurements_df['length_avg_mean'] = measurements_df['length_avg_mean'].fillna(measurements_df['length_side_mean'])
    measurements_df['headwidth_avg_mean'] = measurements_df['headwidth_avg_mean'].fillna(measurements_df['headheight_side_mean'])
    measurements_df['thorax_width_avg_mean'] = measurements_df['thorax_width_avg_mean'].fillna(measurements_df['thoraxheight_side_mean'])
    measurements_df['abdomenwidth_avg_mean'] = measurements_df['abdomenwidth_avg_mean'].fillna(measurements_df['abdomenheight_side_mean'])
    
    # Calculate Stdevs for Averages (by averaging variances)
    # This logic remains the same
    measurements_df['winglength_avg_std'] = np.sqrt((measurements_df['wingLeftlength_top_std']**2 + measurements_df['wingRightlength_top_std']**2) / 2)
    # (Add std calculations for other averages if needed)

    # *** ASSERT CHECK: Ensure thorax width is positive ***
    valid_thorax_widths = measurements_df['thorax_width_avg_mean'].dropna()
    if not valid_thorax_widths.empty:
        assert (valid_thorax_widths > 0).all(), "Thorax width average (normalizer) must be positive."
    print(" - Assert Check Passed: Normalizer (thorax width) is positive where calculated.")

    # 3. Calculate Normalizations (Ratios)
    #    Define the primary normalizer to be used for ratios
    normalizer = measurements_df['thorax_width_avg_mean'].replace(0, np.nan)
    
    measurements_df['length_norm_mean'] = measurements_df['length_avg_mean'] / normalizer
    measurements_df['headwidth_norm_mean'] = measurements_df['headwidth_avg_mean'] / normalizer
    measurements_df['abdomenwidth_norm_mean'] = measurements_df['abdomenwidth_avg_mean'] / normalizer
    measurements_df['winglength_norm_mean'] = measurements_df['winglength_avg_mean'] / normalizer
    measurements_df['bodyarea_top_norm_mean'] = np.sqrt(measurements_df['body_area_top_mean']) / normalizer
    measurements_df['bodyarea_side_norm_mean'] = np.sqrt(measurements_df['body_area_side_mean']) / normalizer
    
    # *** ASSERT CHECK: Ensure thorax width is positive ***
    norm_cols = [col for col in measurements_df.columns if '_norm_mean' in col]
    for col in norm_cols:
        # Check specifically for positive infinity, as negative infinity is less likely here
        assert not (measurements_df[col] == np.inf).any(), f"Infinite values found in normalized feature: {col}"
    print(" - Assert Check Passed: No infinite values in normalized measurement features.")

    # --- UNIT CONVERSION: pixels -> mm (lengths) and px^2 -> mm^2 (areas) ---
    # Convert only base mean/std measurement features. Normalized features (*_norm_mean) are unitless and remain unchanged.
    length_scale = float(mm_per_px)
    area_scale = float(mm_per_px) ** 2

    for col in measurements_df.columns:
        # Only convert numeric measurement outputs (means/stds)
        if not (col.endswith('_mean') or col.endswith('_std')):
            continue

        # Skip unitless features
        if '_norm_' in col or col.endswith('_ratio'):
            continue

        # Area features
        if col.startswith('body_area_'):
            measurements_df[col] = measurements_df[col] * area_scale
        else:
            # All other measurement means/stds in this table are lengths
            measurements_df[col] = measurements_df[col] * length_scale

    # --- 3. Generate all three DataFrames for the Excel sheets ---
    measurement_order = [
        # Averaged Features
        'length_avg', 'headwidth_avg', 'thorax_width_avg', 'abdomenwidth_avg', 'winglength_avg',
        # Normalized Features
        'length_norm', 'headwidth_norm', 'abdomenwidth_norm', 'winglength_norm', 'bodyarea_top_norm', 'bodyarea_side_norm',
        # Original component features
        'length_side', 'headheight_side',
        'thoraxheight_side', 'abdomenheight_side',
        'body_area_top', 'body_area_side'
    ]
    blank_spacer_cols = [' ', '  ']

    # --- DF for Tab 1: Sex Ratios ---
    ratio_df = _calculate_sex_ratio_df(measurements_df, measurement_order)

    # --- DF for Tab 2: Sex-Specific Measurements ---
    formatted_sex_separated_df = _calculate_formatted_sex_separated_df(measurements_df, measurement_order, blank_spacer_cols)

    # --- DF for Tab 3: Combined Measurements ---
    combined_df = measurements_df.drop(columns=['sex'], errors='ignore')
    means_cols = [f'{name}_mean' for name in measurement_order if f'{name}_mean' in combined_df.columns]
    std_cols = [f'{name}_std' for name in measurement_order if f'{name}_std' in combined_df.columns]
    n_samples_cols = [f'{name}_n_samples' for name in measurement_order if f'{name}_n_samples' in combined_df.columns]
    combined_df[blank_spacer_cols[0]], combined_df[blank_spacer_cols[1]] = '', ''
    desired_order = means_cols + [blank_spacer_cols[0]] + std_cols + [blank_spacer_cols[1]] + n_samples_cols
    combined_df.rename(columns={'individual_id': 'individual', 'group_label': 'group'}, inplace=True)
    formatted_combined_df = format_summary_df(combined_df, desired_order)

    # --- 4. Save all DataFrames to a single Excel file ---
    sheets_to_save = {
        'Combined Measurements': formatted_combined_df,
        'Sex-Specific Measurements': formatted_sex_separated_df,
        'Sex Ratios': ratio_df        
    }
    if save_outputs:
        excel_path = os.path.join(output_dir, '3_Standardized_Measurements.xlsx')
        save_to_excel_sheets(sheets_to_save, excel_path)

    # --- 5. Return the full DataFrame for downstream use ---
    return measurements_df


# 4 Postural Analysis
