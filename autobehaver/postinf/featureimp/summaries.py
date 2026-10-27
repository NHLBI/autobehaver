"""Aggregate FeatureImp SHAP values into individual and group summaries.

Support in-sample/out-of-fold and male/female comparisons with formatted exports.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from warnings import catch_warnings, simplefilter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment

from .config import DEFAULT_SETTINGS
from .runtime_settings import SummaryRuntimeSettings
from .io_utils import _sanitize_excel_sheet_name

# Module-level default used when callers do not pass explicit settings.
_DEFAULT_SUMMARY_SETTINGS = SummaryRuntimeSettings.from_mapping(DEFAULT_SETTINGS)

# Binary classification convention
# Throughout FeatureImp we treat the "positive" class as index 1 and the
# "negative" class as index 0. This matches FixedBinaryLabelEncoder, which
# encodes the training groups in a fixed order (group_a -> 0, group_b -> 1).
NEGATIVE_CLASS_INDEX: int = 0
POSITIVE_CLASS_INDEX: int = 1


def save_consolidated_summary_with_formatting(*args: Any, **kwargs: Any) -> None:
    """Save a consolidated SHAP summary Excel file with formatting.

    This wrapper provides a stable API while importing the Excel formatting
    implementation lazily to avoid circular imports.
    """

    # Local import to avoid circular imports during module initialization.
    from .reporting import save_consolidated_summary_with_formatting as _impl_save

    return _impl_save(*args, **kwargs)


def apply_settings(settings: Dict[str, Any]) -> None:
    """Update module-level summary defaults used by direct callers."""

    global _DEFAULT_SUMMARY_SETTINGS
    _DEFAULT_SUMMARY_SETTINGS = SummaryRuntimeSettings.from_mapping(settings)


def _clean_performance_dict_for_json(perf_dict):
    """Convert a performance dictionary to JSON-serializable format.
    Handles numpy types, None values, and ensures all values are basic Python types.

    Args:
        perf_dict: Dictionary with performance metrics (may contain numpy types)

    Returns:
        Dictionary with all values converted to JSON-serializable types
    """
    cleaned = {}
    for key, value in perf_dict.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            cleaned[key] = int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            cleaned[key] = float(value)
        elif isinstance(value, bool):
            cleaned[key] = bool(value)
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to lists, handling numpy arrays
            cleaned[key] = [float(v) if isinstance(v, (np.floating, np.float32, np.float64)) 
                           else int(v) if isinstance(v, (np.integer, np.int32, np.int64))
                           else str(v) for v in value]
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            cleaned[key] = value.tolist()
        else:
            # For strings and other types, convert to string if not already basic type
            cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


def _sanitize_group_name_for_filename(group_name):
    """Return a group name string that is safe for filenames."""
    if pd.isna(group_name):
        return "Unknown"
    name_str = str(group_name)
    for char in [':', '\\', '/', '?', '*', '"', '<', '>', '|', ' ', '\t', '\n']:
        name_str = name_str.replace(char, '_')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    name_str = name_str.strip('_')
    return name_str or "Unknown"


def _aggregate_shap_from_files(all_shap_file_paths, all_feature_names, df_shape, num_classes, 
                                performance_results, n_splits=5, validation_error_on_fail: Optional[bool] = None):
    """Aggregate SHAP values from multiple file paths (from specialists across all
    folds).

    Notes:
      - This is primarily used by older per-file SHAP aggregation helpers and
        used by reporting utilities.
      - Aggregation is done in the full feature space; each specialist contributes only its own feature subset.
      - Values are averaged *per feature* by the number of times that feature appears across successfully loaded files.
      - If files are missing/corrupt/misaligned, we report clearly and (optionally) raise depending on VALIDATION_ERROR_ON_FAIL.

    Args:
        all_shap_file_paths: List of .npy SHAP file paths (from all specialists, all folds)
        all_feature_names: List of all feature names (full feature space)
        df_shape: Shape tuple containing (n_samples,)
        num_classes: Number of classes (expected 2 for this pipeline)
        performance_results: List of performance dicts aligned with all_shap_file_paths (same order)
        n_splits: Unused here; kept only for signature consistency

    Returns:
        numpy array: Aggregated SHAP values, shape (n_samples, n_features)
    """
    settings_validation_error_on_fail = (
        _DEFAULT_SUMMARY_SETTINGS.validation_error_on_fail
        if validation_error_on_fail is None
        else bool(validation_error_on_fail)
    )

    n_samples = df_shape[0]
    n_all_features = len(all_feature_names)

    if num_classes != 2:
        raise ValueError(f"_aggregate_shap_from_files expects binary SHAP for this pipeline. Got num_classes={num_classes}")

    # Initialize aggregation arrays
    aggregated_shap = np.zeros((n_samples, n_all_features), dtype=np.float32)
    feature_hit_counts = np.zeros(n_all_features, dtype=np.int32)

    # Create feature index mapping
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_feature_names)}

    # Track problems explicitly (do not silently continue without reporting)
    missing_metadata = []
    missing_files = []
    load_errors = []
    shape_mismatches = []

    for result_idx, shap_file_path in enumerate(all_shap_file_paths):
        # Ensure we can map this file to its specialist_features list
        if result_idx >= len(performance_results):
            missing_metadata.append(shap_file_path)
            continue

        perf = performance_results[result_idx]
        specialist_features = perf.get('specialist_features', [])

        # Normalize specialist_features to a list of strings
        if isinstance(specialist_features, (pd.Index, np.ndarray, tuple)):
            specialist_features = list(specialist_features)
        elif not isinstance(specialist_features, list):
            specialist_features = [str(specialist_features)]

        if not shap_file_path or not isinstance(shap_file_path, str):
            load_errors.append((shap_file_path, "Invalid/empty shap_file_path"))
            continue

        if not os.path.exists(shap_file_path):
            missing_files.append(shap_file_path)
            continue

        try:
            shap_values = np.load(shap_file_path, allow_pickle=False)
        except Exception as e:
            load_errors.append((shap_file_path, f"{type(e).__name__}: {e}"))
            continue

        # Handle binary SHAP arrays saved as (n_samples, n_features, 2)
        if shap_values.ndim == 3:
            if shap_values.shape[2] <= POSITIVE_CLASS_INDEX:
                # Fallback: if only one class axis exists, use it
                shap_values = shap_values[:, :, 0]
            else:
                shap_values = shap_values[:, :, POSITIVE_CLASS_INDEX]

        # Basic shape validation
        if shap_values.ndim != 2:
            shape_mismatches.append((shap_file_path, f"Expected 2D after class-slice, got shape {shap_values.shape}"))
            continue

        if shap_values.shape[0] != n_samples:
            shape_mismatches.append((shap_file_path, f"Row mismatch: expected {n_samples}, got {shap_values.shape[0]}"))
            continue

        # Map specialist features to full feature space
        # Only map up to the number of columns present in SHAP values
        max_cols = shap_values.shape[1]
        for feat_idx, feat_name in enumerate(specialist_features[:max_cols]):
            full_idx = feature_to_idx.get(feat_name, None)
            if full_idx is None:
                continue
            aggregated_shap[:, full_idx] += shap_values[:, feat_idx].astype(np.float32, copy=False)
            feature_hit_counts[full_idx] += 1

    # Report issues clearly
    any_problems = bool(missing_metadata or missing_files or load_errors or shape_mismatches)
    if any_problems:
        msg_lines = ["[SHAP FILE AGGREGATION] Problems detected:"]
        if missing_metadata:
            msg_lines.append(f"  - {len(missing_metadata)} file(s) had no matching performance metadata (order mismatch).")
            msg_lines.append(f"    Example: {missing_metadata[:3]}")
        if missing_files:
            msg_lines.append(f"  - {len(missing_files)} file(s) missing on disk.")
            msg_lines.append(f"    Example: {missing_files[:3]}")
        if load_errors:
            msg_lines.append(f"  - {len(load_errors)} file(s) failed to load.")
            msg_lines.append(f"    Example: {load_errors[:3]}")
        if shape_mismatches:
            msg_lines.append(f"  - {len(shape_mismatches)} file(s) had unexpected shapes.")
            msg_lines.append(f"    Example: {shape_mismatches[:3]}")

        msg = "\n".join(msg_lines)

        # Honor the configured failure policy for incomplete legacy-file aggregation.
        if settings_validation_error_on_fail:
            raise RuntimeError(msg)
        else:
            print(msg)
            print("  WARNING: Continuing with partial aggregation. Results may be incomplete.")

    # Average by hit count (vectorized; avoids division by zero)
    nonzero = feature_hit_counts > 0
    if np.any(nonzero):
        aggregated_shap[:, nonzero] /= feature_hit_counts[nonzero].astype(np.float32)

    # If any feature never received any hits, it will remain zeros; warn because that usually indicates a logic error.
    zero_hit = np.where(feature_hit_counts == 0)[0]
    if len(zero_hit) > 0:
        print(f"[SHAP FILE AGGREGATION] WARNING: {len(zero_hit)} feature(s) had zero hit counts in file-based aggregation.")

    return aggregated_shap


def _generate_kfold_shap_summaries(shap_values, df, selected_features, le, paths, suffix, train_groups):
    """Generate SHAP summaries for in-sample or out-of-sample data in the same format
    as the main analysis, but with a suffix added to filenames.

    Args:
        shap_values: numpy array of SHAP values (n_samples, n_features) for binary or (n_samples, n_features, n_classes) for multi
        df: DataFrame with training data (individual, group columns)
        selected_features: list of feature names
        le: LabelEncoder
        paths: dictionary of paths
        suffix: 'insample' or 'outofsample'
        train_groups: list of training group names

    Returns:
        tuple: (individual_summary_sheets, consolidated_df)
            - individual_summary_sheets: dict of {sheet_name: DataFrame} for per-individual summary
            - consolidated_df: DataFrame with consolidated summary statistics
    """
    # Clean matplotlib state at start of function
    plt.close('all')  # Close any existing figures to ensure clean state
    
    X = df[selected_features]
    num_classes = len(le.classes_)
    
    # Per-Individual Summary
    feature_cols = list(X.columns)  # Convert to list to avoid pandas Index printing issues
    
    # Validate shap_values before creating DataFrame
    if shap_values is None or shap_values.size == 0:
        print(f"WARNING: No SHAP values for per-individual summary ({suffix}). Skipping.")
        return None, None
    
    # Ensure shap_values is a numpy array
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)
    
    # Check for inf values
    if np.any(np.isinf(shap_values)):
        print(f"WARNING: Found inf values in SHAP data ({suffix}). Replacing with NaN.")
        shap_values = np.where(np.isinf(shap_values), np.nan, shap_values)
    
    individual_summary_sheets = {}
    class_name = le.classes_[POSITIVE_CLASS_INDEX]
    class_shap_df = pd.DataFrame(shap_values, columns=feature_cols).add_prefix('SHAP_')
    base_info_df = df[['individual', 'group']].reset_index(drop=True)
    sheet_df = pd.concat([base_info_df, class_shap_df], axis=1)
    sheet_name = _sanitize_excel_sheet_name(f'SHAP_for_{class_name}')
    individual_summary_sheets[sheet_name] = sheet_df
    
    # Note: Individual summary sheets are returned but not saved here.
    # They will be combined with the other suffix's sheets into a single file.
    
    # Per-Group Summary
    group_summary_data = {}
    all_groups_in_data = df['group'].unique()
    
    for group_name in all_groups_in_data:
        if group_name not in train_groups:
            continue  # Skip non-training groups
        group_mask = (df['group'] == group_name).values
        
        group_shap_values = shap_values[group_mask, :]
        if group_shap_values.shape[0] == 0:
            continue
        
        # Binary classification - always use the positive class
        trained_class_index = POSITIVE_CLASS_INDEX
        trained_class_name = le.classes_[trained_class_index]
        
        group_summary_data[group_name] = {
            'feature_name': feature_cols,  # Use list instead of Index to avoid printing issues
            f'mean_shap_for_{trained_class_name}': np.mean(group_shap_values, axis=0),
            f'mean_abs_shap_for_{trained_class_name}': np.mean(np.abs(group_shap_values), axis=0),
            f'std_shap_for_{trained_class_name}': np.std(group_shap_values, axis=0)
        }
        
        # Generate summary plot with error handling
        plt.figure()
        try:
            # Validate data before plotting
            if group_shap_values.shape[0] == 0 or X[group_mask].shape[0] == 0:
                print(f"WARNING: Skipping SHAP plot for group '{group_name}' ({suffix}) - no data")
            elif np.any(np.isnan(group_shap_values)) or np.any(np.isinf(group_shap_values)):
                print(f"WARNING: Skipping SHAP plot for group '{group_name}' ({suffix}) - invalid SHAP values")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    plt.clf()  # Clear current figure to ensure clean state
                    shap.summary_plot(group_shap_values, X[group_mask], show=False, plot_size=(10, 12))
                plt.title(f'SHAP Summary for Group: {group_name} ({suffix})\n(Push towards {trained_class_name})')
                plt.tight_layout()
                plot_path = os.path.join(paths['analysis_folder'], f'SHAP_SUMMARY_PLOT_{group_name}_vs_{trained_class_name}_{suffix}{paths.get("file_suffix", "")}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close('all')  # Close all figures to ensure clean state
                print(f"SHAP summary plot for group '{group_name}' ({suffix}) saved to {plot_path}")
        except Exception as e:
            plt.close('all')  # Clean up on error
            print(f"WARNING: Failed to generate SHAP summary plot for group '{group_name}' ({suffix}): {e}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Continuing without this plot...")
            # Continue execution - don't crash the whole analysis
    
    # Global Importance Calculation
    global_mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    global_shap_df = pd.DataFrame({
        'feature_name': feature_cols,  # Use list instead of Index to avoid printing issues
        'Global_Importance': global_mean_abs_shap
    }).sort_values(by='Global_Importance', ascending=False)
    
    # Build Consolidated Summary
    consolidated_df = global_shap_df.copy()
    
    stats_pivoted = pd.DataFrame({'feature_name': []})

    # Merge group summaries
    for group_name in all_groups_in_data:
        if group_name not in train_groups:
            continue
        
        if group_name in group_summary_data:
            group_stats = group_summary_data[group_name]
            class_block_df = pd.DataFrame(group_stats)
            
            rename_dict_shap = {}
            for trained_class_name in le.classes_:
                mean_col = f'mean_shap_for_{trained_class_name}'
                mean_abs_col = f'mean_abs_shap_for_{trained_class_name}'
                std_col = f'std_shap_for_{trained_class_name}'
                if mean_col in class_block_df.columns:
                    rename_dict_shap[mean_col] = f'MeanSHAP_{group_name}'
                    if mean_abs_col in class_block_df.columns:
                        rename_dict_shap[mean_abs_col] = f'MeanAbsSHAP_{group_name}'
                    rename_dict_shap[std_col] = f'StdSHAP_{group_name}'
            
            class_block_df.rename(columns=rename_dict_shap, inplace=True)
            cols_to_keep = ['feature_name'] + list(rename_dict_shap.values())
            class_block_df = class_block_df[cols_to_keep]
        else:
            class_block_df = pd.DataFrame({'feature_name': feature_cols})  # Use list instead of Index to avoid printing issues
        
        # Add raw feature stats for this group
        feature_mean_col = f'FeatureMean_{group_name}'
        feature_std_col = f'FeatureStd_{group_name}'
        cols_to_get_from_stats = ['feature_name']
        if feature_mean_col in stats_pivoted.columns:
            cols_to_get_from_stats.append(feature_mean_col)
        if feature_std_col in stats_pivoted.columns:
            cols_to_get_from_stats.append(feature_std_col)
        
        if len(cols_to_get_from_stats) > 1:
            class_block_df = pd.merge(class_block_df, stats_pivoted[cols_to_get_from_stats], on='feature_name', how='left')
        
        consolidated_df = pd.merge(consolidated_df, class_block_df, on='feature_name', how='left')
    
    # Add Normalized Columns (Post-Aggregation)
    # Normalize Global_Importance
    total_global_importance = consolidated_df['Global_Importance'].sum()
    if total_global_importance > 1e-10:
        consolidated_df['Global_Importance_Normalized'] = consolidated_df['Global_Importance'] / total_global_importance
    else:
        consolidated_df['Global_Importance_Normalized'] = np.nan
    
    # Normalize MeanSHAP and MeanAbsSHAP for each group
    for group_name in all_groups_in_data:
        if group_name not in train_groups:
            continue
        
        # Normalize MeanSHAP
        mean_shap_col = f'MeanSHAP_{group_name}'
        if mean_shap_col in consolidated_df.columns:
            total_abs_mean_shap = consolidated_df[mean_shap_col].abs().sum()
            if total_abs_mean_shap > 1e-10:
                consolidated_df[f'MeanSHAP_Normalized_{group_name}'] = consolidated_df[mean_shap_col] / total_abs_mean_shap
            else:
                consolidated_df[f'MeanSHAP_Normalized_{group_name}'] = np.nan
        
        # Normalize MeanAbsSHAP
        mean_abs_shap_col = f'MeanAbsSHAP_{group_name}'
        if mean_abs_shap_col in consolidated_df.columns:
            total_mean_abs_shap = consolidated_df[mean_abs_shap_col].sum()
            if total_mean_abs_shap > 1e-10:
                consolidated_df[f'MeanAbsSHAP_Normalized_{group_name}'] = consolidated_df[mean_abs_shap_col] / total_mean_abs_shap
            else:
                consolidated_df[f'MeanAbsSHAP_Normalized_{group_name}'] = np.nan
        
    
    # Reorder columns: place normalized columns immediately after their raw counterparts
    column_order = ['feature_name', 'Global_Importance', 'Global_Importance_Normalized']
    for group_name in all_groups_in_data:
        if group_name not in train_groups:
            continue
        if f'MeanSHAP_{group_name}' in consolidated_df.columns:
            column_order.append(f'MeanSHAP_{group_name}')
            if f'MeanSHAP_Normalized_{group_name}' in consolidated_df.columns:
                column_order.append(f'MeanSHAP_Normalized_{group_name}')
        if f'MeanAbsSHAP_{group_name}' in consolidated_df.columns:
            column_order.append(f'MeanAbsSHAP_{group_name}')
            if f'MeanAbsSHAP_Normalized_{group_name}' in consolidated_df.columns:
                column_order.append(f'MeanAbsSHAP_Normalized_{group_name}')
        if f'StdSHAP_{group_name}' in consolidated_df.columns:
            column_order.append(f'StdSHAP_{group_name}')
        if f'FeatureMean_{group_name}' in consolidated_df.columns:
            column_order.append(f'FeatureMean_{group_name}')
        if f'FeatureStd_{group_name}' in consolidated_df.columns:
            column_order.append(f'FeatureStd_{group_name}')
    
    remaining_cols = [col for col in consolidated_df.columns if col not in column_order]
    column_order.extend(remaining_cols)
    consolidated_df = consolidated_df[column_order]
    
    # Sort by global importance
    consolidated_df = consolidated_df.sort_values(by='Global_Importance', ascending=False)

    # Add cumulative global importance (running cumulative sum of normalized global importance)
    if 'Global_Importance_Normalized' in consolidated_df.columns:
        consolidated_df['Global_Importance_Cumulative'] = consolidated_df['Global_Importance_Normalized'].cumsum()
        # Force last value to 1.0 when possible (floating point rounding safeguard)
        if len(consolidated_df) > 0 and not pd.isna(consolidated_df['Global_Importance_Cumulative'].iloc[-1]):
            consolidated_df.iloc[-1, consolidated_df.columns.get_loc('Global_Importance_Cumulative')] = 1.0

        # Reposition the column immediately after Global_Importance_Normalized (for readability)
        try:
            cum_col = consolidated_df.pop('Global_Importance_Cumulative')
            insert_idx = consolidated_df.columns.get_loc('Global_Importance_Normalized') + 1
            consolidated_df.insert(insert_idx, 'Global_Importance_Cumulative', cum_col)
        except Exception as e:
            print(f"WARNING: Could not reposition Global_Importance_Cumulative column ({suffix}): {e}")
    else:
        consolidated_df['Global_Importance_Cumulative'] = np.nan


    return individual_summary_sheets, consolidated_df


def _generate_male_female_comparison(df, selected_features, all_shap_file_paths, le, paths, avg_shap_values_override=None):
    """Generate a male vs female SHAP comparison for each class.
    Compares SHAP values between males and females within each class to identify
    sex-specific feature importance differences.

    Args:
        df: DataFrame with 'individual', 'group', 'sex' columns
        selected_features: List of selected feature names
        all_shap_file_paths: List of paths to SHAP value files
        le: LabelEncoder fitted on training data
        paths: Dictionary of paths
    """
    if 'sex' not in df.columns:
        print("Warning: 'sex' column not found. Cannot generate male vs female comparison.")
        return
    
    print("Loading and aggregating SHAP values for male vs female comparison...")
    
    # Load and aggregate SHAP values (similar to _aggregate_and_summarize_results)
    X = df[selected_features]
    feature_cols_list = list(X.columns)
    n_shap_samples, n_features = X.shape
    num_classes = len(le.classes_)
    
    # Validation Check: Verify data alignment
    print(f"\n[Validation Check - Male vs Female] Data alignment check:")
    print(f"  - df shape: {df.shape}, df index: {df.index[:5].tolist()}...")
    print(f"  - X shape: {X.shape}")
    print(f"  - n_shap_samples: {n_shap_samples}, n_features: {n_features}")
    
    # Load pre-aggregated file (from K-Fold committee design)
    if avg_shap_values_override is not None:
        avg_shap_values = np.asarray(avg_shap_values_override, dtype=np.float32)
        print(f"  - Using in-memory SHAP values override with shape: {avg_shap_values.shape}")
    else:
        print(f"  - Loading pre-aggregated SHAP file: {all_shap_file_paths[0]}")
        output_shape = (n_shap_samples, n_features)
        
        avg_shap_values = np.memmap(all_shap_file_paths[0], dtype=np.float32, mode='r', shape=output_shape)
        avg_shap_values = np.array(avg_shap_values)  # Convert to regular array
        print(f"  - Loaded SHAP values shape: {avg_shap_values.shape}")

    # Validation Check: Verify SHAP values shape matches df
    if avg_shap_values.shape[0] != len(df):
        print(f"\n[ERROR] Shape mismatch: avg_shap_values.shape[0]={avg_shap_values.shape[0]}, len(df)={len(df)}")
        print(f"  This will cause indexing errors. SHAP values must match df row order.")
        return
    
    # Get unique groups and sexes
    unique_groups = df['group'].unique()
    unique_sexes = df['sex'].unique()
    
    # Validation Check: Print sex and group information
    print(f"\n[Male vs Female] Data overview:")
    print(f"  - Unique groups: {unique_groups}")
    print(f"  - Unique sexes: {unique_sexes}")
    print(f"  - Total samples: {len(df)}")
    
    # Standardize sex values
    sex_mapping = {}
    for sex_val in unique_sexes:
        sex_upper = str(sex_val).upper()
        if sex_upper in ['M', 'MALE']:
            sex_mapping[sex_val] = 'MALE'
        elif sex_upper in ['F', 'FEMALE']:
            sex_mapping[sex_val] = 'FEMALE'
    
    print(f"  - Sex mapping: {sex_mapping}")
    
    df_with_sex = df.copy()
    df_with_sex['sex_standardized'] = df_with_sex['sex'].map(sex_mapping)
    
    # Check for unmapped sexes
    unmapped = df_with_sex[df_with_sex['sex_standardized'].isna()]
    if len(unmapped) > 0:
        print(f"  - WARNING: {len(unmapped)} samples with unmapped sex values")
        print(f"    Unique unmapped values: {unmapped['sex'].unique()}")

    training_groups = {str(g) for g in getattr(le, 'classes_', [])}
    train_mask = df_with_sex['group'].astype(str).isin(training_groups).values
    print(f"  - Training groups for global-importance alignment: {sorted(training_groups)}")
    print(f"  - Training rows available for global-importance alignment: {int(np.sum(train_mask))}")

    def _mean_abs_shap_for_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """Average absolute SHAP over selected rows, returning None for an empty
        selection.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != avg_shap_values.shape[0]:
            raise ValueError(
                f"Mask length {mask.shape[0]} does not match SHAP rows {avg_shap_values.shape[0]}"
            )
        if not np.any(mask):
            return None
        return np.mean(np.abs(avg_shap_values[mask, :]), axis=0)

    def _safe_std_ddof1(values: np.ndarray) -> np.ndarray:
        """Return per-feature sample std without ddof=1 warnings.

        A group/sex cell can legitimately contain a single sample. In that case
        the ddof=1 standard deviation is undefined; returning NaNs explicitly is
        preferable to emitting RuntimeWarnings during summary generation.
        """

        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D SHAP array for std computation, got shape {values.shape!r}")
        if values.shape[0] < 2:
            return np.full(values.shape[1], np.nan, dtype=np.float64)
        return np.std(values, axis=0, ddof=1)

    def _normalized_or_zero(values, denom: float) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if np.isfinite(denom) and abs(float(denom)) > 1e-10:
            return arr / float(denom)
        return np.where(np.isfinite(arr), 0.0, np.nan)
    
    # For each group, calculate mean SHAP for males and females
    male_summaries = {}
    female_summaries = {}
    
    print(f"\n[Male vs Female] Processing groups:")
    for group_name in unique_groups:
        # Get indices for this group
        group_mask = df_with_sex['group'] == group_name
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) == 0:
            print(f"  - {group_name}: No samples found")
            continue
        
        # Split by sex
        male_mask = (df_with_sex['group'] == group_name) & (df_with_sex['sex_standardized'] == 'MALE')
        female_mask = (df_with_sex['group'] == group_name) & (df_with_sex['sex_standardized'] == 'FEMALE')
        
        male_indices = np.where(male_mask)[0]
        female_indices = np.where(female_mask)[0]
        
        print(f"  - {group_name}: {len(group_indices)} total samples")
        print(f"      Male indices: {len(male_indices)}, Female indices: {len(female_indices)}")
        
        if len(male_indices) == 0 or len(female_indices) == 0:
            print(f"      WARNING: Skipping group '{group_name}' - missing one or both sexes")
            continue  # Skip groups without both sexes
        
        # Get SHAP values for males and females
        # Validation Check: Verify indices are valid
        if len(male_indices) > 0:
            max_male_idx = male_indices.max()
            if max_male_idx >= len(avg_shap_values):
                print(f"    ERROR: Male index {max_male_idx} >= SHAP array size {len(avg_shap_values)}")
                continue
        if len(female_indices) > 0:
            max_female_idx = female_indices.max()
            if max_female_idx >= len(avg_shap_values):
                print(f"    ERROR: Female index {max_female_idx} >= SHAP array size {len(avg_shap_values)}")
                continue
        
        male_shap = avg_shap_values[male_indices, :]  # (n_male_samples, n_features)
        female_shap = avg_shap_values[female_indices, :]  # (n_female_samples, n_features)
        
        # Validation Check: Verify SHAP extraction
        print(f"      Extracted SHAP shapes: male={male_shap.shape}, female={female_shap.shape}")
        print(f"      Male SHAP min/max/mean: {male_shap.min():.6f} / {male_shap.max():.6f} / {male_shap.mean():.6f}")
        print(f"      Female SHAP min/max/mean: {female_shap.min():.6f} / {female_shap.max():.6f} / {female_shap.mean():.6f}")
        
        # Calculate mean SHAP per feature for males and females
        male_mean_shap = np.mean(male_shap, axis=0)  # (n_features,)
        female_mean_shap = np.mean(female_shap, axis=0)  # (n_features,)
        male_mean_abs_shap = np.mean(np.abs(male_shap), axis=0)  # (n_features,)
        female_mean_abs_shap = np.mean(np.abs(female_shap), axis=0)  # (n_features,)
        male_std_shap = _safe_std_ddof1(male_shap)  # (n_features,)
        female_std_shap = _safe_std_ddof1(female_shap)  # (n_features,)
        
        # Validation Check: Verify summary statistics
        print(f"      Male mean SHAP: min={male_mean_shap.min():.6f}, max={male_mean_shap.max():.6f}, non-zero={np.count_nonzero(male_mean_shap)}")
        print(f"      Female mean SHAP: min={female_mean_shap.min():.6f}, max={female_mean_shap.max():.6f}, non-zero={np.count_nonzero(female_mean_shap)}")
        
        male_summaries[group_name] = {
            'mean': male_mean_shap,
            'mean_abs': male_mean_abs_shap,
            'std': male_std_shap
        }
        female_summaries[group_name] = {
            'mean': female_mean_shap,
            'mean_abs': female_mean_abs_shap,
            'std': female_std_shap
        }
    
    # Validation Check: Verify summaries were created
    print(f"\n[Male vs Female] Summary creation:")
    print(f"  - Groups in male_summaries: {list(male_summaries.keys())}")
    print(f"  - Groups in female_summaries: {list(female_summaries.keys())}")
    
    # Build consolidated DataFrames for male and female
    male_df = pd.DataFrame({'feature_name': feature_cols_list})
    female_df = pd.DataFrame({'feature_name': feature_cols_list})
    
    # Add Global Importance using the same convention as the main consolidated
    # summary: sample-level mean(abs(SHAP)) on training rows when available.
    male_train_mask = train_mask & (df_with_sex['sex_standardized'].values == 'MALE')
    female_train_mask = train_mask & (df_with_sex['sex_standardized'].values == 'FEMALE')
    male_all_mask = df_with_sex['sex_standardized'].values == 'MALE'
    female_all_mask = df_with_sex['sex_standardized'].values == 'FEMALE'

    male_global_importance = _mean_abs_shap_for_mask(male_train_mask)
    female_global_importance = _mean_abs_shap_for_mask(female_train_mask)

    if male_global_importance is None:
        male_global_importance = _mean_abs_shap_for_mask(male_all_mask)
        if male_global_importance is None:
            male_global_importance = np.full(len(feature_cols_list), np.nan)
            print("  - WARNING: No male rows available for global importance. Marked unavailable.")
        else:
            print("  - WARNING: No male training rows found; using all male rows for Global_Importance.")

    if female_global_importance is None:
        female_global_importance = _mean_abs_shap_for_mask(female_all_mask)
        if female_global_importance is None:
            female_global_importance = np.full(len(feature_cols_list), np.nan)
            print("  - WARNING: No female rows available for global importance. Marked unavailable.")
        else:
            print("  - WARNING: No female training rows found; using all female rows for Global_Importance.")
    
    male_df['Global_Importance'] = male_global_importance
    female_df['Global_Importance'] = female_global_importance

    availability = {}
    for label, reference_mask, all_mask in (
        ("male", male_train_mask, male_all_mask),
        ("female", female_train_mask, female_all_mask),
    ):
        n_reference, n_all = int(reference_mask.sum()), int(all_mask.sum())
        availability[label] = {"training_row_count": n_reference, "all_row_count": n_all,
            "global_importance_source": "training_rows" if n_reference else "all_rows" if n_all else "unavailable"}
    from autobehaver.atomic_io import atomic_write_json
    atomic_write_json(availability, os.path.join(paths["analysis_folder"],
        f'SHAP_SEX_AVAILABILITY{paths.get("file_suffix", "")}.json'))
    
    # Validation Check: Check global importance
    print(f"  - Male global importance: min={male_global_importance.min():.6f}, max={male_global_importance.max():.6f}, non-zero={np.count_nonzero(male_global_importance)}")
    print(f"  - Female global importance: min={female_global_importance.min():.6f}, max={female_global_importance.max():.6f}, non-zero={np.count_nonzero(female_global_importance)}")
    
    # Add per-group summaries
    all_groups_in_summaries = sorted(set(list(male_summaries.keys()) + list(female_summaries.keys())))
    print(f"  - Adding columns for groups: {all_groups_in_summaries}")
    
    for group_name in all_groups_in_summaries:
        if group_name in male_summaries:
            male_df[f'MeanSHAP_{group_name}'] = male_summaries[group_name]['mean']
            male_df[f'MeanAbsSHAP_{group_name}'] = male_summaries[group_name]['mean_abs']
            male_df[f'StdSHAP_{group_name}'] = male_summaries[group_name]['std']
        else:
            # Fill with NaN if missing
            male_df[f'MeanSHAP_{group_name}'] = np.nan
            male_df[f'MeanAbsSHAP_{group_name}'] = np.nan
            male_df[f'StdSHAP_{group_name}'] = np.nan
            print(f"    WARNING: Group '{group_name}' missing from male_summaries")
        
        if group_name in female_summaries:
            female_df[f'MeanSHAP_{group_name}'] = female_summaries[group_name]['mean']
            female_df[f'MeanAbsSHAP_{group_name}'] = female_summaries[group_name]['mean_abs']
            female_df[f'StdSHAP_{group_name}'] = female_summaries[group_name]['std']
        else:
            # Fill with NaN if missing
            female_df[f'MeanSHAP_{group_name}'] = np.nan
            female_df[f'MeanAbsSHAP_{group_name}'] = np.nan
            female_df[f'StdSHAP_{group_name}'] = np.nan
            print(f"    WARNING: Group '{group_name}' missing from female_summaries")
    
    # Add Normalized Columns (Post-Aggregation)
    # Normalize Global_Importance for male and female separately
    total_male_global = male_df['Global_Importance'].sum()
    male_df['Global_Importance_Normalized'] = _normalized_or_zero(
        male_df['Global_Importance'],
        float(total_male_global),
    )
    
    total_female_global = female_df['Global_Importance'].sum()
    female_df['Global_Importance_Normalized'] = _normalized_or_zero(
        female_df['Global_Importance'],
        float(total_female_global),
    )
    
    # Normalize MeanSHAP and MeanAbsSHAP for each group (separately for male and female)
    for group_name in all_groups_in_summaries:
        # Male normalization
        mean_shap_col_male = f'MeanSHAP_{group_name}'
        if mean_shap_col_male in male_df.columns:
            total_abs_mean_shap_male = male_df[mean_shap_col_male].abs().sum()
            male_df[f'MeanSHAP_Normalized_{group_name}'] = _normalized_or_zero(
                male_df[mean_shap_col_male],
                float(total_abs_mean_shap_male),
            )
        
        mean_abs_shap_col_male = f'MeanAbsSHAP_{group_name}'
        if mean_abs_shap_col_male in male_df.columns:
            total_mean_abs_shap_male = male_df[mean_abs_shap_col_male].sum()
            male_df[f'MeanAbsSHAP_Normalized_{group_name}'] = _normalized_or_zero(
                male_df[mean_abs_shap_col_male],
                float(total_mean_abs_shap_male),
            )
        
        # Female normalization
        mean_shap_col_female = f'MeanSHAP_{group_name}'
        if mean_shap_col_female in female_df.columns:
            total_abs_mean_shap_female = female_df[mean_shap_col_female].abs().sum()
            female_df[f'MeanSHAP_Normalized_{group_name}'] = _normalized_or_zero(
                female_df[mean_shap_col_female],
                float(total_abs_mean_shap_female),
            )
        
        mean_abs_shap_col_female = f'MeanAbsSHAP_{group_name}'
        if mean_abs_shap_col_female in female_df.columns:
            total_mean_abs_shap_female = female_df[mean_abs_shap_col_female].sum()
            female_df[f'MeanAbsSHAP_Normalized_{group_name}'] = _normalized_or_zero(
                female_df[mean_abs_shap_col_female],
                float(total_mean_abs_shap_female),
            )
    
    # Reorder columns: place normalized columns immediately after their raw counterparts
    # For male_df
    male_column_order = ['feature_name', 'Global_Importance', 'Global_Importance_Normalized']
    for group_name in all_groups_in_summaries:
        if f'MeanSHAP_{group_name}' in male_df.columns:
            male_column_order.append(f'MeanSHAP_{group_name}')
            if f'MeanSHAP_Normalized_{group_name}' in male_df.columns:
                male_column_order.append(f'MeanSHAP_Normalized_{group_name}')
        if f'MeanAbsSHAP_{group_name}' in male_df.columns:
            male_column_order.append(f'MeanAbsSHAP_{group_name}')
            if f'MeanAbsSHAP_Normalized_{group_name}' in male_df.columns:
                male_column_order.append(f'MeanAbsSHAP_Normalized_{group_name}')
        if f'StdSHAP_{group_name}' in male_df.columns:
            male_column_order.append(f'StdSHAP_{group_name}')
    male_remaining = [col for col in male_df.columns if col not in male_column_order]
    male_column_order.extend(male_remaining)
    male_df = male_df[male_column_order]
    
    # For female_df
    female_column_order = ['feature_name', 'Global_Importance', 'Global_Importance_Normalized']
    for group_name in all_groups_in_summaries:
        if f'MeanSHAP_{group_name}' in female_df.columns:
            female_column_order.append(f'MeanSHAP_{group_name}')
            if f'MeanSHAP_Normalized_{group_name}' in female_df.columns:
                female_column_order.append(f'MeanSHAP_Normalized_{group_name}')
        if f'MeanAbsSHAP_{group_name}' in female_df.columns:
            female_column_order.append(f'MeanAbsSHAP_{group_name}')
            if f'MeanAbsSHAP_Normalized_{group_name}' in female_df.columns:
                female_column_order.append(f'MeanAbsSHAP_Normalized_{group_name}')
        if f'StdSHAP_{group_name}' in female_df.columns:
            female_column_order.append(f'StdSHAP_{group_name}')
    female_remaining = [col for col in female_df.columns if col not in female_column_order]
    female_column_order.extend(female_remaining)
    female_df = female_df[female_column_order]
    
    # Validation Check: Check DataFrame shapes before sorting
    print(f"\n[Validation Check - Male vs Female] Df shapes before sorting:")
    print(f"  - male_df shape: {male_df.shape}, columns: {list(male_df.columns)[:5]}...")
    print(f"  - female_df shape: {female_df.shape}, columns: {list(female_df.columns)[:5]}...")
    
    # Match the main consolidated-summary ordering: training rows only when
    # available, otherwise fall back to all rows.
    combined_global_importance = _mean_abs_shap_for_mask(train_mask)
    if combined_global_importance is None:
        combined_global_importance = np.mean(np.abs(avg_shap_values), axis=0)
        print("  - WARNING: No training rows found for combined Global_Importance ordering; using all rows.")
    
    # Create a DataFrame with feature names and combined Global_Importance for sorting
    combined_global_df = pd.DataFrame({
        'feature_name': feature_cols_list,
        'Global_Importance_Combined': combined_global_importance
    })
    
    # Sort by the combined Global_Importance ordering used by the main analysis
    combined_global_df = combined_global_df.sort_values(by='Global_Importance_Combined', ascending=False)
    feature_order = combined_global_df['feature_name'].values
    
    # Reorder both male_df and female_df to match the combined Global_Importance order
    male_df = male_df.set_index('feature_name').reindex(feature_order).reset_index()
    female_df = female_df.set_index('feature_name').reindex(feature_order).reset_index()
    
    # Validation Check: Check DataFrame shapes after sorting
    print(f"\n[Validation Check - Male vs Female] Df shapes after sorting:")
    print(f"  - male_df shape: {male_df.shape}")
    print(f"  - female_df shape: {female_df.shape}")
    print(f"  - Checking for NaN values in female_df:")
    for col in female_df.columns:
        if col != 'feature_name':
            nan_count = female_df[col].isna().sum()
            if nan_count > 0:
                print(f"    - {col}: {nan_count} NaN values")
    
    _combine_male_female_summaries(male_df, female_df, paths)



def _magnitude_rank_correlation(first, second):
    """Spearman magnitude correlation with average ties; constants are undefined."""
    a, b = np.abs(np.asarray(first, dtype=float)), np.abs(np.asarray(second, dtype=float))
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)

def _combine_male_female_summaries(male_df, female_df, paths):
    """Combine male and female consolidated summaries into a single file
    for easier comparison. Similar structure to in-sample vs out-of-sample.

    Args:
        male_df: DataFrame with male consolidated summary
        female_df: DataFrame with female consolidated summary
        paths: dictionary of paths
    """
    if male_df is None or female_df is None:
        print("Warning: Could not combine summaries - one or both DataFrames are None.")
        return
    
    if 'feature_name' not in male_df.columns or 'feature_name' not in female_df.columns:
        print("Warning: 'feature_name' column not found in summaries.")
        return
    
    # Reorder female to match male order
    male_feature_order = male_df['feature_name'].values
    female_df_reordered = female_df.set_index('feature_name').reindex(male_feature_order).reset_index()
    
    # Remove 'feature_name' from female (we'll use male's)
    female_cols_to_keep = [col for col in female_df_reordered.columns if col != 'feature_name']
    female_df_subset = female_df_reordered[female_cols_to_keep]
    
    # Add suffix to female column names
    female_df_subset = female_df_subset.copy()
    female_df_subset.columns = [f'{col}_FEMALE' if col != '' else col for col in female_df_subset.columns]
    
    # Add suffix to male column names (except feature_name, Global_Importance, and Global_Importance_Normalized get special handling)
    male_df_labeled = male_df.copy()
    new_male_cols = []
    for col in male_df_labeled.columns:
        if col == 'feature_name':
            new_male_cols.append(col)
        elif col == 'Global_Importance':
            new_male_cols.append('Global_Importance_MALE')
        elif col == 'Global_Importance_Normalized':
            new_male_cols.append('Global_Importance_Normalized_MALE')
        else:
            new_male_cols.append(f'{col}_MALE' if col != '' else col)
    male_df_labeled.columns = new_male_cols
    
    combined_df = male_df_labeled.copy()
    
    # Calculate comparison metrics between male and female    
    mean_shap_cols_male = [col for col in combined_df.columns if col.startswith('MeanSHAP_') and col.endswith('_MALE') and not col.startswith('MeanSHAP_Normalized_')]
    mean_abs_shap_cols_male = [col for col in combined_df.columns if col.startswith('MeanAbsSHAP_') and col.endswith('_MALE') and not col.startswith('MeanAbsSHAP_Normalized_')]
    
    # Check magnitude differences between male and female
    print(f"\n[Male vs Female] Magnitude analysis:")
    male_global = combined_df['Global_Importance_MALE'].values
    female_global = female_df_subset['Global_Importance_FEMALE'].values
    mask_valid = ~(np.isnan(male_global) | np.isnan(female_global))
    male_valid = male_global[mask_valid]
    female_valid = female_global[mask_valid]
    
    if len(male_valid) > 0 and len(female_valid) > 0:
        print(f"  - Features with both male and female data: {len(male_valid)}")
        print(f"  - Male Global_Importance: min={male_valid.min():.6f}, max={male_valid.max():.6f}, mean={male_valid.mean():.6f}")
        print(f"  - Female Global_Importance: min={female_valid.min():.6f}, max={female_valid.max():.6f}, mean={female_valid.mean():.6f}")
        print(f"  - Ratio (female/male): min={(female_valid / (male_valid + 1e-10)).min():.3f}, max={(female_valid / (male_valid + 1e-10)).max():.3f}, mean={(female_valid / (male_valid + 1e-10)).mean():.3f}")
        
        # Check features where one is much larger than the other
        ratio = female_valid / (male_valid + 1e-10)
        male_dominant = np.sum(ratio < 0.1)  # Female < 10% of male
        female_dominant = np.sum(ratio > 10)  # Female > 10x male
        print(f"  - Features with male >> female (ratio < 0.1): {male_dominant}")
        print(f"  - Features with female >> male (ratio > 10): {female_dominant}")
        
        # Check top features for each sex
        top_male_idx = np.argsort(-male_valid)[:5]
        top_female_idx = np.argsort(-female_valid)[:5]
        print(f"  - Top 5 male features (by Global_Importance):")
        for idx in top_male_idx:
            feat_idx_in_combined = np.where(mask_valid)[0][idx]
            feat_name = combined_df.iloc[feat_idx_in_combined]['feature_name']
            print(f"      '{feat_name[:60]}': male={male_valid[idx]:.6f}, female={female_valid[idx]:.6f}, ratio={ratio[idx]:.3f}")
        print(f"  - Top 5 female features (by Global_Importance):")
        for idx in top_female_idx:
            feat_idx_in_combined = np.where(mask_valid)[0][idx]
            feat_name = combined_df.iloc[feat_idx_in_combined]['feature_name']
            print(f"      '{feat_name[:60]}': male={male_valid[idx]:.6f}, female={female_valid[idx]:.6f}, ratio={ratio[idx]:.3f}")
    
    comparison_metrics = {}
    
    for male_col in mean_shap_cols_male:
        # Extract group name (remove '_MALE' suffix, keep 'MeanSHAP_GroupName')
        group_col_base = male_col[:-5]  # Remove '_MALE'
        female_col = f'{group_col_base}_FEMALE'
        
        if female_col not in female_df_subset.columns:
            continue
        
        group_name = group_col_base.replace('MeanSHAP_', '')
        
        # Get the values
        male_vals = combined_df[male_col].values
        female_vals = female_df_subset[female_col].values
        
        # Remove NaN values for calculations
        mask = ~(np.isnan(male_vals) | np.isnan(female_vals))
        male_clean = male_vals[mask]
        female_clean = female_vals[mask]
        
        if len(male_clean) == 0:
            continue
        
        # 1. MeanSHAP_Diff (absolute difference)
        diff = male_clean - female_clean
        abs_diff = np.abs(diff)
        abs_diff_full = np.full(len(combined_df), np.nan)
        abs_diff_full[mask] = abs_diff
        comparison_metrics[f'MeanSHAP_Diff_{group_name}'] = abs_diff_full
        
        # 2. MeanSHAP_Diff_Abs_Normalized
        max_abs_shap = np.max([np.abs(male_clean).max() if len(male_clean) > 0 else 0,
                               np.abs(female_clean).max() if len(female_clean) > 0 else 0])
        if max_abs_shap > 0:
            abs_diff_norm = abs_diff / max_abs_shap
        else:
            abs_diff_norm = abs_diff
        
        abs_diff_norm_full = np.full(len(combined_df), np.nan)
        abs_diff_norm_full[mask] = abs_diff_norm
        comparison_metrics[f'MeanSHAP_Diff_Abs_Norm_{group_name}'] = abs_diff_norm_full
        
        # 3. Stability_Ratio (female / male)
        epsilon = 1e-10
        stability_ratio = np.full(len(male_clean), np.nan)
        for i in range(len(male_clean)):
            if abs(male_clean[i]) < epsilon and abs(female_clean[i]) < epsilon:
                stability_ratio[i] = 1.0
            elif abs(male_clean[i]) < epsilon:
                stability_ratio[i] = np.nan
            else:
                stability_ratio[i] = female_clean[i] / male_clean[i]
        
        stability_ratio_full = np.full(len(combined_df), np.nan)
        stability_ratio_full[mask] = stability_ratio
        comparison_metrics[f'Stability_Ratio_{group_name}'] = stability_ratio_full
        
        # 4. Rank_Correlation
        if len(male_clean) > 1:
            corr = _magnitude_rank_correlation(male_clean, female_clean)
            rank_corr_full = np.full(len(combined_df), corr)
            comparison_metrics[f'Rank_Correlation_{group_name}'] = rank_corr_full
        else:
            rank_corr_full = np.full(len(combined_df), np.nan)
            comparison_metrics[f'Rank_Correlation_{group_name}'] = rank_corr_full
    
    # Reorganize columns by class/group instead of by sex
    # Extract all groups/classes from column names
    all_groups = set()
    for col in mean_shap_cols_male:
        # Format: MeanSHAP_GroupName_MALE
        group_name = col.replace('MeanSHAP_', '').replace('_MALE', '')
        all_groups.add(group_name)
    
    # First, combine all data into one DataFrame
    combined_with_female = pd.concat([combined_df, female_df_subset], axis=1)
    comparison_metrics_df = pd.DataFrame(comparison_metrics)
    combined_with_all = pd.concat([combined_with_female, comparison_metrics_df], axis=1)
    
    # Order identifiers and global importance before per-group blocks.
    new_column_order = ['feature_name']
    
    # Add Global_Importance columns (MALE and FEMALE)
    if 'Global_Importance_MALE' in combined_with_all.columns:
        new_column_order.append('Global_Importance_MALE')
    if 'Global_Importance_Normalized_MALE' in combined_with_all.columns:
        new_column_order.append('Global_Importance_Normalized_MALE')
    if 'Global_Importance_FEMALE' in combined_with_all.columns:
        new_column_order.append('Global_Importance_FEMALE')
    if 'Global_Importance_Normalized_FEMALE' in combined_with_all.columns:
        new_column_order.append('Global_Importance_Normalized_FEMALE')
    
    # For each group/class, add: Male columns, Female columns, Stats columns
    for group_name in sorted(all_groups):
        # Collect male columns for this group (all columns ending with _GroupName_MALE)
        group_male_cols = [col for col in combined_with_all.columns 
                          if col.endswith(f'_{group_name}_MALE') or 
                          (f'_{group_name}_MALE' in col and col.startswith(('MeanSHAP_', 'MeanAbsSHAP_', 'StdSHAP_')))]
        # Sort to maintain consistent order
        group_male_cols.sort()
        new_column_order.extend(group_male_cols)
        
        # Collect female columns for this group
        group_female_cols = [col for col in combined_with_all.columns 
                            if col.endswith(f'_{group_name}_FEMALE') or 
                            (f'_{group_name}_FEMALE' in col and col.startswith(('MeanSHAP_', 'MeanAbsSHAP_', 'StdSHAP_')))]
        group_female_cols.sort()
        new_column_order.extend(group_female_cols)
        
        # Collect stats columns for this group
        group_stats_cols = [col for col in comparison_metrics.keys() if col.endswith(f'_{group_name}')]
        group_stats_cols.sort()
        new_column_order.extend(group_stats_cols)
        
        # Add blank separator between groups (except after last group)
        if group_name != sorted(all_groups)[-1]:
            new_column_order.append('_SEP_')  # Use unique separator name
    
    # Filter to only include columns that actually exist
    final_column_order = [col for col in new_column_order if col == 'feature_name' or col == '_SEP_' or col in combined_with_all.columns]
    
    # Create reorganized DataFrame by selecting columns in order
    reorganized_dfs = []
    sep_counter = 0
    for col in final_column_order:
        if col == '_SEP_':
            # Add blank separator column with unique name to avoid duplicate column names
            # Use incrementing spaces - pandas will handle if there are duplicates
            sep_col_name = ' ' * (sep_counter + 1)  # ' ', '  ', '   ', etc.
            reorganized_dfs.append(pd.DataFrame({sep_col_name: [''] * len(combined_with_all)}))
            sep_counter += 1
        elif col in combined_with_all.columns:
            reorganized_dfs.append(combined_with_all[[col]])
    
    # Concatenate all parts
    reordered_df = pd.concat(reorganized_dfs, axis=1)
    
    # Save with formatting (similar to in-sample vs out-of-sample)
    combined_path = os.path.join(paths['analysis_folder'], 'SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx')
    save_consolidated_summary_with_formatting(reordered_df, combined_path, is_combined_insample_outofsample=False)
    print(f"Male vs Female consolidated summary saved to {combined_path}")


def _combine_insample_outofsample_summaries(insample_df, outofsample_df, paths):
    """Combine in-sample and out-of-sample consolidated summaries into a single file
    for easier comparison. Features are ordered by in-sample global importance.

    Args:
        insample_df: DataFrame with in-sample consolidated summary
        outofsample_df: DataFrame with out-of-sample consolidated summary
        paths: dictionary of paths
    """
    if insample_df is None or outofsample_df is None:
        print("Warning: Could not combine summaries - one or both DataFrames are None.")
        return
    
    # Use in-sample feature order (sorted by Global_Importance)
    # Get feature_name column from in-sample
    if 'feature_name' not in insample_df.columns:
        print("Warning: 'feature_name' column not found in in-sample summary.")
        return
    
    # Reorder out-of-sample to match in-sample order
    insample_feature_order = insample_df['feature_name'].values
    outofsample_df_reordered = outofsample_df.set_index('feature_name').reindex(insample_feature_order).reset_index()
    
    # Remove 'feature_name' from out-of-sample (we'll use in-sample's)
    outofsample_cols_to_keep = [col for col in outofsample_df_reordered.columns if col != 'feature_name']
    outofsample_df_subset = outofsample_df_reordered[outofsample_cols_to_keep]
    
    # Add suffix to out-of-sample column names to make them clearly identifiable
    outofsample_df_subset = outofsample_df_subset.copy()
    outofsample_df_subset.columns = [f'{col}_OUTOFSAMPLE' if col != '' else col for col in outofsample_df_subset.columns]
    
    # Add suffix to in-sample column names (except feature_name, Global_Importance, and Global_Importance_Normalized get special handling)
    insample_df_labeled = insample_df.copy()
    new_insample_cols = []
    for col in insample_df_labeled.columns:
        if col == 'feature_name':
            new_insample_cols.append(col)
        elif col == 'Global_Importance':
            new_insample_cols.append('Global_Importance_INSAMPLE')
        elif col == 'Global_Importance_Normalized':
            new_insample_cols.append('Global_Importance_Normalized_INSAMPLE')
        else:
            new_insample_cols.append(f'{col}_INSAMPLE' if col != '' else col)
    insample_df_labeled.columns = new_insample_cols
    
    # Create combined DataFrame: in-sample cols, comparison metrics, blank column, out-of-sample cols
    combined_df = insample_df_labeled.copy()
    
    # Find all MeanSHAP columns for each group in in-sample data (exclude normalized columns)
    mean_shap_cols_insample = [col for col in combined_df.columns if col.startswith('MeanSHAP_') and col.endswith('_INSAMPLE') and not col.startswith('MeanSHAP_Normalized_')]
    mean_abs_shap_cols_insample = [col for col in combined_df.columns if col.startswith('MeanAbsSHAP_') and col.endswith('_INSAMPLE') and not col.startswith('MeanAbsSHAP_Normalized_')]
    
    comparison_metrics = {}
    
    for insample_col in mean_shap_cols_insample:
        # Extract group name (remove '_INSAMPLE' suffix, keep 'MeanSHAP_GroupName')
        group_col_base = insample_col[:-9]  # Remove '_INSAMPLE'
        outofsample_col = f'{group_col_base}_OUTOFSAMPLE'
        
        # Check if out-of-sample column exists in the outofsample_df_subset (before it's added to combined_df)
        if outofsample_col not in outofsample_df_subset.columns:
            continue
        
        group_name = group_col_base.replace('MeanSHAP_', '')
        
        # Get the values (in-sample from combined_df, out-of-sample from outofsample_df_subset)
        insample_vals = combined_df[insample_col].values
        outofsample_vals = outofsample_df_subset[outofsample_col].values
        
        # Remove NaN values for calculations
        mask = ~(np.isnan(insample_vals) | np.isnan(outofsample_vals))
        insample_clean = insample_vals[mask]
        outofsample_clean = outofsample_vals[mask]
        
        if len(insample_clean) == 0:
            continue
        
        # Validation Check: Check for suspiciously identical values
        if len(insample_clean) > 0:
            identical_count = np.sum(np.abs(insample_clean - outofsample_clean) < 1e-10)
            if identical_count > len(insample_clean) * 0.1:  # More than 10% identical
                print(f"\n[WARNING] Group '{group_name}': {identical_count}/{len(insample_clean)} features have identical in-sample and out-of-sample values (diff < 1e-10)")
                # Check if they're all near zero (which would be expected)
                near_zero_both = np.sum((np.abs(insample_clean) < 1e-6) & (np.abs(outofsample_clean) < 1e-6))
                if near_zero_both < identical_count:
                    print(f"  - {identical_count - near_zero_both} features have identical non-zero values (this is suspicious!)")
                    # Show a few examples
                    suspicious_mask = (np.abs(insample_clean - outofsample_clean) < 1e-10) & (np.abs(insample_clean) > 1e-6)
                    if np.any(suspicious_mask):
                        suspicious_indices = np.where(suspicious_mask)[0][:3]
                        print(f"  - Example suspicious features (first 3):")
                        for idx in suspicious_indices:
                            feat_idx_in_combined = np.where(mask)[0][idx]
                            feat_name = combined_df.iloc[feat_idx_in_combined]['feature_name']
                            print(f"      '{feat_name[:60]}': in-sample={insample_clean[idx]:.8f}, out-of-sample={outofsample_clean[idx]:.8f}")
                else:
                    print(f"  - All {identical_count} identical values are near zero (expected for unused features)")
        
        # 1. MeanSHAP_Diff (absolute difference)
        diff = insample_clean - outofsample_clean
        abs_diff = np.abs(diff)
        abs_diff_full = np.full(len(combined_df), np.nan)
        abs_diff_full[mask] = abs_diff
        comparison_metrics[f'MeanSHAP_Diff_{group_name}'] = abs_diff_full
        
        # 2. MeanSHAP_Diff_Abs_Normalized (absolute difference, normalized by max absolute MeanSHAP)
        abs_diff = np.abs(diff)
        # Normalize by max absolute MeanSHAP across both in-sample and out-of-sample
        max_abs_shap = np.max([np.abs(insample_clean).max() if len(insample_clean) > 0 else 0,
                               np.abs(outofsample_clean).max() if len(outofsample_clean) > 0 else 0])
        if max_abs_shap > 0:
            abs_diff_norm = abs_diff / max_abs_shap
        else:
            abs_diff_norm = abs_diff
        
        abs_diff_norm_full = np.full(len(combined_df), np.nan)
        abs_diff_norm_full[mask] = abs_diff_norm
        comparison_metrics[f'MeanSHAP_Diff_Abs_Norm_{group_name}'] = abs_diff_norm_full
        
        # 3. Stability_Ratio (out-of-sample / in-sample)
        # Handle division by zero - use small epsilon or set ratio to 1 if both are near zero
        epsilon = 1e-10
        stability_ratio = np.full(len(insample_clean), np.nan)
        for i in range(len(insample_clean)):
            if abs(insample_clean[i]) < epsilon and abs(outofsample_clean[i]) < epsilon:
                stability_ratio[i] = 1.0  # Both near zero, consider stable
            elif abs(insample_clean[i]) < epsilon:
                stability_ratio[i] = np.nan  # Can't compute ratio
            else:
                stability_ratio[i] = outofsample_clean[i] / insample_clean[i]
        
        stability_ratio_full = np.full(len(combined_df), np.nan)
        stability_ratio_full[mask] = stability_ratio
        comparison_metrics[f'Stability_Ratio_{group_name}'] = stability_ratio_full
        
        # 4. Rank_Correlation (Spearman correlation of feature importance rankings)
        if len(insample_clean) > 1:
            corr = _magnitude_rank_correlation(insample_clean, outofsample_clean)

            # Store as constant value for all features (same correlation for the group)
            rank_corr_full = np.full(len(combined_df), corr)
            comparison_metrics[f'Rank_Correlation_{group_name}'] = rank_corr_full
        else:
            rank_corr_full = np.full(len(combined_df), np.nan)
            comparison_metrics[f'Rank_Correlation_{group_name}'] = rank_corr_full
    
    # Add blank separator column (between in-sample and out-of-sample)
    blank_sep1 = pd.DataFrame({'' : [''] * len(combined_df)})
    combined_df = pd.concat([combined_df, blank_sep1], axis=1)
    
    # Add out-of-sample columns (rename Global_Importance if it exists)
    if 'Global_Importance_OUTOFSAMPLE' in outofsample_df_subset.columns:
        outofsample_df_subset = outofsample_df_subset.rename(columns={'Global_Importance_OUTOFSAMPLE': 'Global_Importance_OUTOFSAMPLE'})
    
    combined_df = pd.concat([combined_df, outofsample_df_subset], axis=1)
    
    # Add blank separator column (between out-of-sample and stats)
    blank_sep2 = pd.DataFrame({' ' : [''] * len(combined_df)})  # Use space to make it unique
    combined_df = pd.concat([combined_df, blank_sep2], axis=1)
    
    # Add comparison metrics to combined DataFrame (at the end, after out-of-sample)
    comparison_metrics_df = pd.DataFrame(comparison_metrics)
    combined_df = pd.concat([combined_df, comparison_metrics_df], axis=1)
    
    # Save combined summary with special handling for the separator
    combined_path = os.path.join(paths['analysis_folder'], f'SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE{paths.get("file_suffix", "")}.xlsx')
    save_consolidated_summary_with_formatting(combined_df, combined_path, is_combined_insample_outofsample=True)
    print(f"Combined in-sample vs out-of-sample consolidated summary saved to {combined_path}")
