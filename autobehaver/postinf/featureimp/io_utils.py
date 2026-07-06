"""Feature-importance I/O utilities.

This module contains functions that read/write reports (logs, Excel sheets) for the
Feature Importance ("featureimp") post-inference analysis.

It is kept separate from the staged planner/shard/finalize modules so reporting
and spreadsheet helpers stay lightweight and reusable.
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

def write_analysis_log(analysis_folder_path,
                       normalized_matrix_path,
                       results_dir,
                       analysis_run_name,
                       control_group_tag,
                       training_group_a,
                       training_group_b,
                       include_feature_categories,
                       n_splits=5,
                       variance_filtering_stats=None,
                       min_variance_threshold=1e-6,
                       **kwargs):
    """
    Write a plain-text log capturing analysis settings and provenance.
    
    The log is saved as ``ANALYSIS_SETTINGS_LOG*.txt`` in the analysis folder and is intended to make
    runs reproducible (data paths, key hyperparameters, filtering settings, and design notes).
    
    Args:
        analysis_folder_path: Analysis folder where the log is written.
        normalized_matrix_path: Path to the normalized feature matrix used.
        results_dir: Base results folder for the run.
        analysis_run_name: Run name used in the analysis folder naming.
        control_group_tag: Control group tag used during normalization.
        training_group_a: Training group A (class 0).
        training_group_b: Training group B (class 1).
        include_feature_categories: Optional list of included feature category prefixes (``None`` = all recognized feature categories).
        n_splits: Number of StratifiedKFold folds in the committee design.
        variance_filtering_stats: Optional dict summarizing variance filtering results.
        min_variance_threshold: Threshold used for near-zero variance filtering.
        **kwargs: Additional optional values. Currently supports:
            - file_suffix: Suffix appended to the log filename.
    
    Returns:
        None
    """
    file_suffix = kwargs.get('file_suffix', '')
    log_path = os.path.join(analysis_folder_path, f'ANALYSIS_SETTINGS_LOG{file_suffix}.txt')
    
    with open(log_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS SETTINGS LOG\n")
        f.write("="*80 + "\n\n")
        
        # Timestamp
        f.write(f"Analysis Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Analysis identification
        f.write("-"*80 + "\n")
        f.write("ANALYSIS IDENTIFICATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Analysis Run Name: {analysis_run_name if analysis_run_name else '(default)'}\n")
        f.write(f"Analysis Folder: {os.path.basename(analysis_folder_path)}\n")
        f.write(f"Full Analysis Path: {analysis_folder_path}\n")
        f.write(f"Results Folder: {results_dir}\n\n")
        
        # Data paths
        f.write("-"*80 + "\n")
        f.write("DATA PATHS\n")
        f.write("-"*80 + "\n")
        f.write(f"Normalized Feature Matrix: {normalized_matrix_path}\n")
        if normalized_matrix_path and os.path.exists(normalized_matrix_path):
            file_size = os.path.getsize(normalized_matrix_path) / (1024 * 1024)  # MB
            f.write(f"  File Size: {file_size:.2f} MB\n")
        f.write("\n")
        
        # Normalization settings
        f.write("-"*80 + "\n")
        f.write("NORMALIZATION SETTINGS\n")
        f.write("-"*80 + "\n")
        f.write(f"Control Group Tag: {control_group_tag if control_group_tag else '(default: CONTROL)'}\n")
        f.write(f"Normalization Method: CLR (compositional) + Z-score (all features)\n")
        f.write(f"Z-score Reference: Control group only\n\n")
        
        sex_filter = kwargs.get('sex_filter', 'all')

        # Analysis type
        f.write("-"*80 + "\n")
        f.write("ANALYSIS TYPE\n")
        f.write("-"*80 + "\n")
        f.write(f"Analysis Type: BINARY (A vs B)\n")
        f.write(f"Training Group A: {training_group_a}\n")
        f.write(f"Training Group B: {training_group_b}\n")
        f.write(f"Sex Filter: {sex_filter}\n")
        if str(sex_filter).lower() in {'male', 'female'}:
            f.write("Note: Analysis rows were filtered by sex before model training.\n")
        f.write(f"Note: SHAP analysis performed on ALL groups using binary model\n")
        f.write("\n")
        
        # Feature filtering
        f.write("-"*80 + "\n")
        f.write("FEATURE FILTERING\n")
        f.write("-"*80 + "\n")
        f.write(f"Univariate Filtering: DISABLED (all features tested via specialist models)\n")
        f.write(f"Univariate Baseline Output: ENABLED (training-group Welch t-test + FDR summary written separately)\n")
        if include_feature_categories:
            f.write(f"Included Feature Categories: {', '.join(include_feature_categories)}\n")
        else:
            f.write(f"Included Feature Categories: ALL recognized feature categories (default)\n")
        
        # Variance filtering information
        if variance_filtering_stats is not None:
            f.write(f"\nVariance Filtering: ENABLED\n")
            f.write(f"  Zero Variance Filtering: ENABLED (removed {variance_filtering_stats['zero_variance_count']} features)\n")
            f.write(f"  Near-Zero Variance Filtering: ENABLED (threshold: {min_variance_threshold}, removed {variance_filtering_stats['near_zero_variance_count']} features)\n")
            f.write(f"  Note: Filtering performed on imputed data (NaN → 0)\n")
            
            # Examples
            if variance_filtering_stats['zero_variance_examples']:
                f.write(f"\n  Zero Variance Examples (first {len(variance_filtering_stats['zero_variance_examples'])}):\n")
                for feat in variance_filtering_stats['zero_variance_examples']:
                    f.write(f"    - {feat}\n")
            if variance_filtering_stats['near_zero_variance_examples']:
                f.write(f"\n  Near-Zero Variance Examples (first {len(variance_filtering_stats['near_zero_variance_examples'])}):\n")
                for feat in variance_filtering_stats['near_zero_variance_examples']:
                    f.write(f"    - {feat}\n")
        else:
            f.write(f"\nVariance Filtering: DISABLED\n")
        f.write("\n")
        
        # K-fold cross-validation
        f.write("-"*80 + "\n")
        f.write("K-FOLD CROSS-VALIDATION SETTINGS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of Folds (n_splits): {n_splits}\n")
        f.write(f"Shuffle: True\n")
        f.write(f"Random State: 42\n")
        f.write(f"Stratified: True (maintains class proportions)\n\n")
        
        # Committee/Specialist Model Settings
        f.write("-"*80 + "\n")
        f.write("COMMITTEE/SPECIALIST MODEL SETTINGS\n")
        f.write("-"*80 + "\n")
        f.write(f"Pipeline Design: K-Fold Committee (replaces filter-then-bootstrap)\n")
        f.write(f"Note: Specialist model settings (H_hits_per_feature, n_features_per_model, etc.)\n")
        f.write(f"      are defined in global settings at top of script.\n\n")
        
        # SHAP settings
        f.write("-"*80 + "\n")
        f.write("SHAP ANALYSIS SETTINGS\n")
        f.write("-"*80 + "\n")
        f.write("\n")
        
        # Prediction aggregation
        f.write("-"*80 + "\n")
        f.write("PREDICTION AGGREGATION\n")
        f.write("-"*80 + "\n")
        f.write(f"In-Sample Predictions: Average across (n_splits - 1) = {n_splits - 1} folds\n")
        f.write(f"Out-of-Sample Predictions: Average across 1 fold (held-out)\n")
        f.write(f"Combined Predictions: Weighted average (in-sample * {n_splits - 1} + out-of-sample) / {n_splits}\n")
        f.write(f"Non-Training Groups: Average across all {n_splits} folds\n\n")
        
        # Output files
        f.write("-"*80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-"*80 + "\n")
        f.write("Generated in this analysis folder:\n")
        f.write("  - AVERAGED_PREDICTIONS_FULL.xlsx (predictions with in-sample/out-of-sample breakdown)\n")
        f.write("  - SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx (SHAP summaries)\n")
        f.write("  - SHAP summary plots (no suffix, _insample, _outofsample)\n")
        f.write("  - NORMALIZED_FEATURE_MATRIX.csv (normalized matrix used)\n")
        f.write("  - NORMALIZATION_PARAMS.json (normalization parameters)\n")
        f.write("  - FEATUREIMP_REFERENCE_MANIFEST.json (reference/provenance manifest)\n")
        f.write("  - FEATUREIMP_REFERENCE/ (frozen feature lists, model inventory, run provenance)\n")
        f.write("  - SAVED_PIPELINE/ (models, SHAP values, etc.)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF LOG\n")
        f.write("="*80 + "\n")
    
    print(f"Analysis settings log saved to: {log_path}")

def format_summary_df(pivot_df, desired_order, id_cols=['individual', 'group']):
    """
    Format a summary DataFrame with group means and separators.
    
    This utility is used when exporting Excel-ready summaries. It:
    - ensures identifier columns are present,
    - enforces a desired column order,
    - computes group means (ignoring NaNs), and
    - inserts blank separator rows between groups.
    
    Args:
        pivot_df: Input DataFrame (identifier columns may be in the index or columns).
        desired_order: List of feature columns in the desired output order.
        id_cols: Identifier columns to place at the left of the output.
    
    Returns:
        A DataFrame containing the group-mean block (top), a blank separator row, and the per-individual
        rows with separators inserted between groups.
    """
    # Ensure all id_cols are present as columns, not index
    if any(col in pivot_df.index.names for col in id_cols):
        pivot_df = pivot_df.reset_index()

    # Include non-feature spacer columns in the final set
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

def _sanitize_excel_sheet_name(name, max_length=31):
    """
    Sanitizes a sheet name to be Excel-compatible.
    
    Excel restrictions:
    - Maximum 31 characters
    - Cannot contain: : \\ / ? * [ ]
    - Cannot be empty
    
    Args:
        name: Original sheet name
        max_length: Maximum length (default: 31 for Excel)
    
    Returns:
        Sanitized sheet name
    """
    if not name or pd.isna(name):
        return "Sheet1"
    
    # Convert to string
    name_str = str(name)
    
    # Remove invalid characters
    invalid_chars = [':', '\\', '/', '?', '*', '[', ']']
    for char in invalid_chars:
        name_str = name_str.replace(char, '_')
    
    # Truncate to max length
    if len(name_str) > max_length:
        name_str = name_str[:max_length]
    
    # Ensure not empty
    if not name_str.strip():
        name_str = "Sheet1"
    
    return name_str

def _write_dataframe_with_autofilter(df, output_path, sheet_name='Sheet1', index=False):
    """
    Writes a DataFrame to Excel with AutoFilter enabled for easy sorting/filtering.
    
    Args:
        df: DataFrame to write
        output_path: Path to output Excel file
        sheet_name: Name of the sheet (default: 'Sheet1')
        index: Whether to include index (default: False)
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        worksheet = writer.sheets[sheet_name]
        
        # Apply AutoFilter if there's data
        if worksheet.max_row > 1 and worksheet.max_column > 0:
            try:
                last_col_letter = get_column_letter(worksheet.max_column)
                auto_filter_range = f"A1:{last_col_letter}{worksheet.max_row}"
                worksheet.auto_filter.ref = auto_filter_range
            except Exception as e:
                print(f"  WARNING: Could not apply AutoFilter to {output_path}: {e}")

def save_to_excel_sheets(sheets_dict, output_path, sheet_order=None):
    """
    Write multiple DataFrames into a single Excel workbook.
    
    This helper sanitizes sheet names, enforces Excel row/column limits, and applies left alignment to
    header cells. It includes defensive checks to reduce the chance of producing a corrupted workbook.
    
    Args:
        sheets_dict: Mapping of sheet name -> DataFrame.
        output_path: Destination ``.xlsx`` path.
        sheet_order: Optional list specifying the sheet order. If ``None``, uses ``sheets_dict`` order.
    
    Returns:
        None
    """
    # Excel limits
    MAX_ROWS = 1048576  # Excel 2007+ limit
    MAX_COLS = 16384    # Excel 2007+ limit
    
    try:
        # If a specific order is provided, use it. Otherwise, use the dict's order.
        sheets_to_process = sheet_order if sheet_order else list(sheets_dict.keys())
    
        # Sanitize and validate all sheet names
        sanitized_sheet_names = {}
        for original_name in sheets_to_process:
            sanitized = _sanitize_excel_sheet_name(original_name)
            # Handle duplicate sanitized names
            if sanitized in sanitized_sheet_names.values():
                counter = 1
                base_name = sanitized[:28]  # Leave room for counter
                while f"{base_name}_{counter}" in sanitized_sheet_names.values():
                    counter += 1
                sanitized = f"{base_name}_{counter}"
            sanitized_sheet_names[original_name] = sanitized
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for original_name in sheets_to_process:
                if original_name not in sheets_dict:
                    continue
                
                df = sheets_dict[original_name]
                sanitized_name = sanitized_sheet_names[original_name]
                
                # Validate DataFrame
                if df is None:
                    print(f"  WARNING: Skipping sheet '{original_name}' - DataFrame is None")
                    continue
                
                if df.empty:
                    print(f"  WARNING: Skipping sheet '{original_name}' - DataFrame is empty")
                    continue
                
                # Check Excel limits
                if len(df) > MAX_ROWS:
                    print(f"  WARNING: Sheet '{original_name}' has {len(df)} rows (max: {MAX_ROWS}). Truncating...")
                    df = df.head(MAX_ROWS)
                
                if len(df.columns) > MAX_COLS:
                    print(f"  WARNING: Sheet '{original_name}' has {len(df.columns)} columns (max: {MAX_COLS}). Truncating...")
                    df = df.iloc[:, :MAX_COLS]
                
                # Convert data types to Excel-compatible types
                df_cleaned = df.copy()
                for col in df_cleaned.columns:
                    # Convert object dtype columns to string (if they're not numeric)
                    if df_cleaned[col].dtype == 'object':
                        try:
                            # Try to convert to numeric first
                            pd.to_numeric(df_cleaned[col], errors='raise')
                        except (ValueError, TypeError):
                            # If not numeric, convert to string
                            df_cleaned[col] = df_cleaned[col].astype(str)
                            # Replace NaN strings with empty string
                            df_cleaned[col] = df_cleaned[col].replace('nan', '')
                    
                    # Handle numpy array columns (shouldn't happen, but defensive)
                    if len(df_cleaned) > 0:
                        first_val = df_cleaned[col].iloc[0]
                        if hasattr(first_val, '__iter__') and not isinstance(first_val, str):
                            df_cleaned[col] = df_cleaned[col].astype(str)
                
                # Replace inf and -inf with NaN (Excel can't handle inf)
                df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
                
                try:
                    df_cleaned.to_excel(writer, sheet_name=sanitized_name, index=False)
                except Exception as e:
                    print(f"  ERROR: Failed to write sheet '{original_name}' (sanitized: '{sanitized_name}'): {e}")
                    traceback.print_exc()
                    continue
        
            # Apply left alignment to headers in all sheets
            workbook = writer.book
            num_sheets_written = len(workbook.sheetnames)  # Capture count before context manager exits
            for original_name, sanitized_name in sanitized_sheet_names.items():
                if sanitized_name in workbook.sheetnames:
                    try:
                        worksheet = workbook[sanitized_name]
                        # Check if worksheet has at least one row before accessing
                        if worksheet.max_row > 0:
                            for cell in worksheet[1]:
                                cell.alignment = Alignment(horizontal='left')
                    except Exception as e:
                        print(f"  WARNING: Could not apply alignment to sheet '{sanitized_name}': {e}")
        
        print(f"Saved Excel file with {num_sheets_written} sheets to {output_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save Excel file to {output_path}: {e}")
        traceback.print_exc()
        # Try to remove corrupted file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"  Removed potentially corrupted file: {output_path}")
            except:
                pass
        raise



