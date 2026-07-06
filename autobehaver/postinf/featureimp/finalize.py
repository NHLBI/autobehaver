"""Finalization / aggregation helpers for the featureimp pipeline.

This module contains the import-heavy "end of pipeline" logic:

- Aggregating committee outputs into final SHAP summaries and reports
- Generating consolidated SHAP summary spreadsheets
- Writing averaged prediction spreadsheets
- Handling analogous outputs for permutation-test runs

The goal is to keep the staged orchestration modules focused on planning,
shard execution, and merge/finalize control flow while isolating
import-heavy plotting/spreadsheet code.
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .committee import POSITIVE_CLASS_INDEX
from .compat import (
    _clean_performance_dict_for_json,
    _generate_kfold_shap_summaries,
    _combine_insample_outofsample_summaries,
    save_predictions_with_formatting,
    save_consolidated_summary_with_formatting,
    save_consolidated_summary_sheets_with_formatting,
)
from .io_utils import _sanitize_excel_sheet_name, save_to_excel_sheets

_UNWANTED_CONSOLIDATED_GI_COLUMNS = frozenset({
    'Global_Importance_WeightedRaw',
    'Global_Importance_PermBaseline',
})


def _sheet_display_name_for_consolidated_view(view_name: str | None, *, primary_is_weighted: bool) -> str | None:
    """Return the user-facing consolidated-summary sheet label."""

    if view_name is None:
        return 'Weighted SHAP' if primary_is_weighted else 'Unweighted SHAP'

    normalized = str(view_name or '').strip()
    if not normalized:
        return None
    if normalized in {'WeightedEnsembleConsistent', 'weightedensembleconsistent'}:
        return None
    if normalized in {'WeightedEquitable', 'Weighted SHAP'}:
        return 'Weighted SHAP'
    if normalized in {'UnweightedEquitable', 'Unweighted SHAP'}:
        return 'Unweighted SHAP'
    if normalized in {'SHAP_Summary', 'SHAP Summary'}:
        return 'Weighted SHAP' if primary_is_weighted else 'Unweighted SHAP'
    return normalized


def _clean_consolidated_shap_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal diagnostic columns from user-facing SHAP summaries."""

    if df is None:
        return df
    drop_cols = [col for col in _UNWANTED_CONSOLIDATED_GI_COLUMNS if col in df.columns]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)

def _build_consolidated_shap_summary_df(
    *,
    avg_shap_values: np.ndarray,
    df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols_list: list,
    le,
    paths: dict,
    feature_stats_df: pd.DataFrame | None,
    global_importance_overrides: dict | None = None,
    extra_feature_annotations_df: pd.DataFrame | None = None,
    generate_nontraining_plots: bool = True,
    plot_tag: str = "",
) -> pd.DataFrame:
    """Build the consolidated SHAP summary DataFrame for one SHAP view.

    This keeps aggregate summary formatting consistent and is
    factored out so we can generate multiple comparison tabs (e.g. weighted vs
    unweighted) without duplicating downstream file generation.

    Args:
        avg_shap_values: (n_samples, n_features) SHAP values (single dimension: push to class 1).
        df: master dataframe with at least ['group'].
        X: feature matrix (same row order as df).
        feature_cols_list: list of feature names corresponding to columns.
        le: LabelEncoder trained on the training groups.
        paths: pipeline paths dict (analysis_folder, file_suffix, ...).
        feature_stats_df: feature stats dataframe (group-level mean/std per feature).
        global_importance_overrides: optional dict of column_name -> vector, used to override Global_Importance.
        generate_nontraining_plots: whether to generate beeswarm plots for non-training groups.
        plot_tag: optional string appended to plot filenames.

    Returns:
        consolidated_df sorted by Global_Importance.
    """

    # Validate
    if avg_shap_values is None or getattr(avg_shap_values, 'size', 0) == 0:
        raise ValueError("avg_shap_values is empty")

    if not isinstance(avg_shap_values, np.ndarray):
        avg_shap_values = np.array(avg_shap_values)

    # Replace inf -> NaN (best-effort)
    if np.any(np.isinf(avg_shap_values)):
        avg_shap_values = np.where(np.isinf(avg_shap_values), np.nan, avg_shap_values)

    all_groups_in_data = df['group'].unique()
    train_groups = set(le.classes_)

    # Per-group SHAP summary stats
    group_summary_data: dict = {}

    for group_name in all_groups_in_data:
        group_mask = (df['group'] == group_name).values
        group_shap_values = avg_shap_values[group_mask, :]
        if group_shap_values.shape[0] == 0:
            continue

        trained_class_index = POSITIVE_CLASS_INDEX
        trained_class_name = le.classes_[trained_class_index]

        group_summary_data[group_name] = {
            'feature_name': feature_cols_list,
            f'mean_shap_for_{trained_class_name}': np.nanmean(group_shap_values, axis=0),
            f'mean_abs_shap_for_{trained_class_name}': np.nanmean(np.abs(group_shap_values), axis=0),
            f'std_shap_for_{trained_class_name}': np.nanstd(group_shap_values, axis=0),
        }

        # Optional: non-training group plots only
        if generate_nontraining_plots and group_name not in train_groups:
            try:
                if group_shap_values.shape[0] == 0 or X[group_mask].shape[0] == 0:
                    continue
                if np.any(np.isnan(group_shap_values)) or np.any(np.isinf(group_shap_values)):
                    continue
                plt.figure()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    plt.clf()
                    shap.summary_plot(group_shap_values, X[group_mask], show=False, plot_size=(10, 12))
                plt.title(f'SHAP Summary for Group: {group_name}\n(Push towards {trained_class_name})')
                plt.tight_layout()
                plot_path = os.path.join(
                    paths['analysis_folder'],
                    f'SHAP_SUMMARY_PLOT_{group_name}_vs_{trained_class_name}{plot_tag}{paths.get("file_suffix", "")}.png',
                )
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close('all')
            except Exception:
                plt.close('all')

    # Global importance from training rows only
    train_mask = df['group'].isin(train_groups).values
    if np.sum(train_mask) == 0:
        global_mean_abs_shap = np.nanmean(np.abs(avg_shap_values), axis=0)
    else:
        global_mean_abs_shap = np.nanmean(np.abs(avg_shap_values[train_mask, :]), axis=0)

    global_shap_df = pd.DataFrame({
        'feature_name': feature_cols_list,
        'mean_abs_shap_global': global_mean_abs_shap,
    }).sort_values(by='mean_abs_shap_global', ascending=False)

    consolidated_df = global_shap_df.copy()
    consolidated_df.rename(columns={'mean_abs_shap_global': 'Global_Importance'}, inplace=True)

    # Apply overrides (phase4 weighted-excess global importance)
    if global_importance_overrides:
        try:
            overrides_dict = dict(global_importance_overrides)
            for col_name, vec in overrides_dict.items():
                if col_name == 'feature_name' or vec is None:
                    continue
                arr = np.asarray(vec).reshape(-1)
                if arr.shape[0] != len(feature_cols_list):
                    continue
                mapping = dict(zip(feature_cols_list, arr))
                consolidated_df[col_name] = consolidated_df['feature_name'].map(mapping)

            if 'Global_Importance' in consolidated_df.columns:
                consolidated_df['Global_Importance'] = (
                    consolidated_df['Global_Importance'].fillna(0).clip(lower=0)
                )

            for extra_gi_col in [
                'Global_Importance_Unweighted',
                'Global_Importance_WeightedRaw',
                'Global_Importance_PermBaseline',
            ]:
                if extra_gi_col in consolidated_df.columns:
                    consolidated_df[extra_gi_col] = consolidated_df[extra_gi_col].fillna(0)
        except Exception as e:
            print(f"WARNING: Failed applying global_importance_overrides: {e}")

    # Prepare raw feature stats for merging
    stats_pivoted = pd.DataFrame({'feature_name': []})
    if feature_stats_df is not None:
        stats_long = feature_stats_df.melt(id_vars='group', var_name='temp_col', value_name='value')
        stats_long[['feature_name', 'stat_type']] = stats_long['temp_col'].str.rsplit('_', n=1, expand=True)
        mask_invalid_split = ~stats_long['stat_type'].isin(['mean', 'std'])
        if mask_invalid_split.any():
            stats_long.loc[mask_invalid_split, 'feature_name'] = stats_long.loc[mask_invalid_split, 'temp_col']
            stats_long.loc[mask_invalid_split, 'stat_type'] = 'mean'
        stats_long['new_col_name'] = 'Feature' + stats_long['stat_type'].str.capitalize() + '_' + stats_long['group']
        stats_pivoted = stats_long.pivot_table(
            index='feature_name',
            columns='new_col_name',
            values='value',
            observed=True,
        ).reset_index()

    # Merge per-group blocks
    for group_name in all_groups_in_data:
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
                    if std_col in class_block_df.columns:
                        rename_dict_shap[std_col] = f'StdSHAP_{group_name}'

            class_block_df.rename(columns=rename_dict_shap, inplace=True)
            cols_to_keep = ['feature_name'] + list(rename_dict_shap.values())
            class_block_df = class_block_df[cols_to_keep]
        else:
            class_block_df = pd.DataFrame({'feature_name': feature_cols_list})

        feature_mean_col = f'FeatureMean_{group_name}'
        feature_std_col = f'FeatureStd_{group_name}'

        cols_to_get_from_stats = ['feature_name']
        if feature_mean_col in stats_pivoted.columns:
            cols_to_get_from_stats.append(feature_mean_col)
        if feature_std_col in stats_pivoted.columns:
            cols_to_get_from_stats.append(feature_std_col)

        if len(cols_to_get_from_stats) > 1:
            class_block_df = pd.merge(
                class_block_df,
                stats_pivoted[cols_to_get_from_stats],
                on='feature_name',
                how='left',
            )

        consolidated_df = pd.merge(consolidated_df, class_block_df, on='feature_name', how='left')

    # Optional extra feature-level annotations (e.g. across-subset-model stability).
    if feature_cols_list and extra_feature_annotations_df is not None:
        try:
            extra_df = pd.DataFrame(extra_feature_annotations_df).copy()
            if not extra_df.empty and 'feature_name' in extra_df.columns:
                extra_df['feature_name'] = extra_df['feature_name'].astype(str)
                extra_df = extra_df.drop_duplicates(subset=['feature_name'], keep='first')
                annotation_cols = [c for c in extra_df.columns if c != 'feature_name' and c not in consolidated_df.columns]
                if annotation_cols:
                    consolidated_df = pd.merge(
                        consolidated_df,
                        extra_df[['feature_name'] + annotation_cols],
                        on='feature_name',
                        how='left',
                    )
        except Exception as e:
            print(f"WARNING: Failed merging extra feature annotations into consolidated SHAP summary: {e}")

    # Normalized columns
    total_global_importance = consolidated_df['Global_Importance'].sum()
    if total_global_importance > 1e-10:
        consolidated_df['Global_Importance_Normalized'] = consolidated_df['Global_Importance'] / total_global_importance
    else:
        consolidated_df['Global_Importance_Normalized'] = np.nan

    for group_name in all_groups_in_data:
        mean_shap_col = f'MeanSHAP_{group_name}'
        if mean_shap_col in consolidated_df.columns:
            total_abs_mean_shap = consolidated_df[mean_shap_col].abs().sum()
            if total_abs_mean_shap > 1e-10:
                consolidated_df[f'MeanSHAP_Normalized_{group_name}'] = consolidated_df[mean_shap_col] / total_abs_mean_shap
            else:
                consolidated_df[f'MeanSHAP_Normalized_{group_name}'] = np.nan

        mean_abs_shap_col = f'MeanAbsSHAP_{group_name}'
        if mean_abs_shap_col in consolidated_df.columns:
            total_mean_abs_shap = consolidated_df[mean_abs_shap_col].sum()
            if total_mean_abs_shap > 1e-10:
                consolidated_df[f'MeanAbsSHAP_Normalized_{group_name}'] = consolidated_df[mean_abs_shap_col] / total_mean_abs_shap
            else:
                consolidated_df[f'MeanAbsSHAP_Normalized_{group_name}'] = np.nan

    # Reorder columns
    column_order = ['feature_name', 'Global_Importance']
    for extra_gi_col in [
        'Global_Importance_Unweighted',
        'Global_Importance_WeightedRaw',
        'Global_Importance_PermBaseline',
    ]:
        if extra_gi_col in consolidated_df.columns:
            column_order.append(extra_gi_col)

    column_order.append('Global_Importance_Normalized')

    for group_name in all_groups_in_data:
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

    consolidated_df = consolidated_df.sort_values(by='Global_Importance', ascending=False)

    if 'Global_Importance_Normalized' in consolidated_df.columns:
        consolidated_df['Global_Importance_Cumulative'] = consolidated_df['Global_Importance_Normalized'].cumsum()
        if len(consolidated_df) > 0 and not pd.isna(consolidated_df['Global_Importance_Cumulative'].iloc[-1]):
            consolidated_df.iloc[-1, consolidated_df.columns.get_loc('Global_Importance_Cumulative')] = 1.0
        try:
            cum_col = consolidated_df.pop('Global_Importance_Cumulative')
            insert_idx = consolidated_df.columns.get_loc('Global_Importance_Normalized') + 1
            consolidated_df.insert(insert_idx, 'Global_Importance_Cumulative', cum_col)
        except Exception:
            pass
    else:
        consolidated_df['Global_Importance_Cumulative'] = np.nan

    return consolidated_df




def _aggregate_and_summarize_permuted_results(df, selected_features, all_permuted_performance,
                                              all_permuted_shap_values, le, paths, permutation_idx,
                                              permuted_insample_shap, permuted_oof_shap, df_for_training,
                                              train_groups_list, n_splits, permuted_all_groups_avg_probas=None,
                                              permuted_all_groups_avg_shap_values=None, df_for_shap_and_agg=None,
                                              permuted_insample_probas=None, permuted_oof_probas=None,
                                              all_fold_indices=None):
    """
    Aggregates results from permuted models and generates output files with PERMUTED_N_ prefix.
    Similar to _aggregate_and_summarize_results but for permuted models.
    """
    print(f"\n--- Aggregating Permuted Results (Permutation {permutation_idx + 1}) ---")
    print("Note: Probabilities have been flipped to be relative to true labels for direct comparison with real models.")
    
    # Modify paths to include permutation prefix
    permuted_paths = paths.copy()
    permuted_paths['file_suffix'] = f'_PERMUTED_{permutation_idx}'
    permuted_analysis_folder = os.path.join(paths['analysis_folder'], f'PERMUTED_{permutation_idx}_RESULTS')
    os.makedirs(permuted_analysis_folder, exist_ok=True)
    permuted_paths['analysis_folder'] = permuted_analysis_folder
    
    # Performance results are expected to be a flat list of dicts
    flattened_perf = all_permuted_performance if all_permuted_performance is not None else []
    if len(flattened_perf) > 0 and isinstance(flattened_perf[0], list):
        raise ValueError(
            "Unexpected nested-list structure for all_permuted_performance. "
            "This script now expects a flat list of performance dicts."
        )
    
    # Save performance report
    # Clean performance results for JSON serialization
    cleaned_flattened_perf = [_clean_performance_dict_for_json(p) for p in flattened_perf]
    perf_path = os.path.join(permuted_analysis_folder, f'PERMUTED_{permutation_idx}_BOOTSTRAP_PERFORMANCE_DETAILS.json')
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_flattened_perf, f, indent=2, ensure_ascii=False)
    print(f"Permuted performance report saved to {perf_path}")
    
    # Generate in-sample vs out-of-sample summaries
    print(f"Generating in-sample vs out-of-sample summaries for permutation {permutation_idx + 1}...")
    
    # Generate summaries (sequential; plotting is not thread-safe)
    def generate_permuted_outofsample():
        try:
            if matplotlib.get_backend().lower() != 'agg':
                plt.switch_backend('Agg')
        except Exception:
            pass
        return _generate_kfold_shap_summaries(
            shap_values=permuted_oof_shap,
            df=df_for_training,
            selected_features=selected_features,
            le=le,
            paths=permuted_paths,
            suffix='outofsample',
            train_groups=train_groups_list
        )
    
    def generate_permuted_insample():
        try:
            if matplotlib.get_backend().lower() != 'agg':
                plt.switch_backend('Agg')
        except Exception:
            pass
        return _generate_kfold_shap_summaries(
            shap_values=permuted_insample_shap,
            df=df_for_training,
            selected_features=selected_features,
            le=le,
            paths=permuted_paths,
            suffix='insample',
            train_groups=train_groups_list
        )
    
    # Run sequentially for deterministic plotting
    permuted_outofsample_result = generate_permuted_outofsample()
    permuted_insample_result = generate_permuted_insample()
    
    # Extract individual summary sheets and consolidated DataFrames
    if permuted_outofsample_result is not None and permuted_insample_result is not None:
        permuted_outofsample_sheets, permuted_outofsample_df = permuted_outofsample_result
        permuted_insample_sheets, permuted_insample_df = permuted_insample_result
        
        # Combine individual summary sheets into a single file with two tabs
        if permuted_outofsample_sheets is not None and permuted_insample_sheets is not None:
            print(f"\nCombining per-individual SHAP summaries for permutation {permutation_idx + 1}...")
            combined_permuted_individual_sheets = {}
            
            # Add out-of-sample sheets with "OutOfSample_" prefix
            for sheet_name, sheet_df in permuted_outofsample_sheets.items():
                combined_sheet_name = _sanitize_excel_sheet_name(f'OutOfSample_{sheet_name}')
                combined_permuted_individual_sheets[combined_sheet_name] = sheet_df
            
            # Add in-sample sheets with "InSample_" prefix
            for sheet_name, sheet_df in permuted_insample_sheets.items():
                combined_sheet_name = _sanitize_excel_sheet_name(f'InSample_{sheet_name}')
                combined_permuted_individual_sheets[combined_sheet_name] = sheet_df
            
            # Save combined individual summary file for permuted data
            permuted_individual_path = os.path.join(permuted_analysis_folder, 
                                                   f'SHAP_PER_INDIVIDUAL_SUMMARY_PERMUTED_{permutation_idx}.xlsx')
            save_to_excel_sheets(combined_permuted_individual_sheets, permuted_individual_path)
            print(f"Per-individual SHAP summary (permuted {permutation_idx + 1}, combined) saved to {permuted_individual_path}")
        else:
            print(f"Warning: Could not combine individual summary sheets for permutation {permutation_idx + 1} (missing data).")
    else:
        print(f"Warning: Could not extract results from parallel execution for permutation {permutation_idx + 1}.")
        permuted_outofsample_df = None
        permuted_insample_df = None
    
    # Combine summaries
    if permuted_outofsample_df is not None and permuted_insample_df is not None:
        _combine_insample_outofsample_summaries(permuted_insample_df, permuted_outofsample_df, permuted_paths)
    
    # --- Generate AVERAGED_PREDICTIONS_FULL.xlsx for all groups (similar to real models) ---
    if permuted_all_groups_avg_probas is not None and df_for_shap_and_agg is not None:
        print(f"\n--- Saving Averaged Predictions for All Groups (Permutation {permutation_idx + 1}) ---")
        num_classes = len(le.classes_)
        prob_cols = [f'Prob_{le.classes_[c]}' for c in range(num_classes)]
        
        # Create DataFrames for in-sample and out-of-sample predictions (training groups only)
        if permuted_insample_probas is not None:
            permuted_insample_probas_df = pd.DataFrame(permuted_insample_probas, columns=[f'{col}_insample' for col in prob_cols])
        else:
            permuted_insample_probas_df = pd.DataFrame(
                np.nan, 
                index=range(len(df_for_training)), 
                columns=[f'{col}_insample' for col in prob_cols]
            )
        
        if permuted_oof_probas is not None:
            permuted_outofsample_probas_df = pd.DataFrame(permuted_oof_probas, columns=[f'{col}_outofsample' for col in prob_cols])
        else:
            permuted_outofsample_probas_df = pd.DataFrame(
                np.nan, 
                index=range(len(df_for_training)), 
                columns=[f'{col}_outofsample' for col in prob_cols]
            )

        # Main (unsuffixed) probabilities for training rows = OOF predictions
        if permuted_oof_probas is not None:
            permuted_oof_main_probas_df = pd.DataFrame(permuted_oof_probas, columns=prob_cols)
        else:
            permuted_oof_main_probas_df = pd.DataFrame(
                np.nan,
                index=range(len(df_for_training)),
                columns=prob_cols
            )

        # Create DataFrame for training groups: base info + insample + outofsample + main(OOF)
        base_info_df = df_for_training[['individual', 'group']].reset_index(drop=True)
        training_predictions_df = pd.concat(
            [base_info_df, permuted_insample_probas_df, permuted_outofsample_probas_df, permuted_oof_main_probas_df],
            axis=1
        )

        max_diff = np.max(np.abs(
            training_predictions_df[prob_cols].values -
            training_predictions_df[[f"{c}_outofsample" for c in prob_cols]].values
        ))
        print(f"[Validation permuted] max |Prob - Prob_outofsample| (training rows) = {max_diff:.6g}")

        
        # Get averaged predictions for ALL groups (including non-training groups)
        all_groups_probas_df = pd.DataFrame(permuted_all_groups_avg_probas, columns=prob_cols)
        all_groups_base_info = df_for_shap_and_agg[['individual', 'group']].reset_index(drop=True)
        
        # Identify which samples are training vs non-training
        training_individuals = set(df_for_training['individual'].values)
        non_training_mask = ~all_groups_base_info['individual'].isin(training_individuals)
        non_training_base_info = all_groups_base_info[non_training_mask].reset_index(drop=True)
        non_training_probas_df = all_groups_probas_df[non_training_mask].reset_index(drop=True)
        
        # Create empty DataFrames for in-sample and out-of-sample columns for non-training groups
        non_training_insample_df = pd.DataFrame(
            np.nan, 
            index=non_training_probas_df.index, 
            columns=[f'{col}_insample' for col in prob_cols]
        )
        non_training_outofsample_df = pd.DataFrame(
            np.nan, 
            index=non_training_probas_df.index, 
            columns=[f'{col}_outofsample' for col in prob_cols]
        )
        
        # Combine non-training groups
        non_training_predictions_df = pd.concat([
            non_training_base_info, 
            non_training_insample_df, 
            non_training_outofsample_df, 
            non_training_probas_df
        ], axis=1)
        
        # Combine training and non-training DataFrames
        if training_predictions_df is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_predictions_df = pd.concat([training_predictions_df, non_training_predictions_df], axis=0, ignore_index=True)
        else:
            all_predictions_df = non_training_predictions_df
        
        # Save to Excel with formatting
        all_predictions_path = os.path.join(permuted_analysis_folder, f'AVERAGED_PREDICTIONS_PERMUTED_{permutation_idx}.xlsx')
        save_predictions_with_formatting(all_predictions_df, all_predictions_path)
        print(f"Averaged predictions for all groups saved to {all_predictions_path}")
        print(f"  - Training groups: {len(training_predictions_df)} samples with in-sample, out-of-sample, and OOF (main) Prob_* columns")
        print(f"  - Non-training groups: {len(non_training_predictions_df)} samples with committee-average Prob_* columns only")

    
    # --- Generate SHAP_CONSOLIDATED_SUMMARY.xlsx for all groups ---
    if permuted_all_groups_avg_shap_values is not None and df_for_shap_and_agg is not None:
        print(f"\n--- Generating Full Analysis for All Groups (Permutation {permutation_idx + 1}) ---")
        
        # Save averaged SHAP values to memmap file (similar to real models)
        permuted_shap_values_folder = os.path.join(permuted_analysis_folder, 'temp_shap_values')
        os.makedirs(permuted_shap_values_folder, exist_ok=True)
        permuted_paths['shap_values_folder'] = permuted_shap_values_folder
        
        avg_shap_file_path = os.path.join(permuted_shap_values_folder, f'all_groups_avg_shap_values_PERMUTED_{permutation_idx}.mmap')
        all_groups_avg_shap_values_f32 = permuted_all_groups_avg_shap_values.astype(np.float32)
        
        print(f"  - Saving to memmap: {avg_shap_file_path}")
        print(f"  - Memmap shape: {all_groups_avg_shap_values_f32.shape}, dtype: float32")
        
        # Save to memmap
        memmap_shape = all_groups_avg_shap_values_f32.shape
        memmap_array = np.memmap(avg_shap_file_path, dtype=np.float32, mode='w+', shape=memmap_shape)
        memmap_array[:] = all_groups_avg_shap_values_f32[:]
        del memmap_array
        
        # Call _aggregate_and_summarize_results to generate full SHAP summary
        _aggregate_and_summarize_results(
            df=df_for_shap_and_agg,
            selected_features=selected_features,
            performance_results=flattened_perf,
            all_shap_values=[avg_shap_file_path],
            le=le,
            paths=permuted_paths,
            avg_shap_values_override=all_groups_avg_shap_values_f32,
        )


def _aggregate_and_summarize_results(
    df,
    selected_features,
    performance_results,
    all_shap_values,
    le,
    paths,
    global_importance_overrides=None,
    extra_consolidated_shap_views=None,
    avg_shap_values_override=None,
    extra_consolidated_shap_arrays=None,
    extra_feature_annotations_df=None,
):
    """
    Aggregate per-specialist outputs into final SHAP summaries and reports.
    
    This function runs after all specialist models have completed. It aggregates SHAP values and
    probabilities, generates consolidated summary spreadsheets/plots, and writes performance reports.
    
    Notes:
        Some output filenames use the ``BOOTSTRAP_`` prefix expected by downstream reports.
    
    Args:
        df: Full normalized feature matrix containing all groups.
        selected_features: List of feature columns used in the analysis.
        performance_results: List of per-specialist performance dictionaries.
        all_shap_values: List of SHAP file paths produced by specialists.
        le: FixedBinaryLabelEncoder fit to the two training groups.
        paths: Dict of output paths for the analysis.
    
    Returns:
        None
    """
    print("\n--- Aggregating and Saving Final Results ---")
    X = df[selected_features]
    feature_cols_list = list(X.columns)  # Convert to list to avoid pandas Index printing issues
    # num_classes is determined by the LabelEncoder fit on the training data
    num_classes = len(le.classes_)

    feature_stats_df = None

    # --- Performance Report ---
    # Clean performance results
    cleaned_performance_results = [_clean_performance_dict_for_json(p) for p in performance_results]

    perf_path = os.path.join(paths['analysis_folder'], 'BOOTSTRAP_PERFORMANCE_DETAILS.json')
    combined_perf_payload = {
        'main_models': cleaned_performance_results,
        'permuted_models': []  # Filled later if permutation testing is enabled
    }
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(combined_perf_payload, f, indent=2, ensure_ascii=False)
    print(f"Detailed performance report saved to {perf_path}")

    if len(performance_results) > 0:
        test_accuracies = []
        train_accuracies = []
        for p in performance_results:
            a_test = p.get('accuracy_test', np.nan)
            a_train = p.get('accuracy_train', np.nan)
            if a_test is not None and not np.isnan(a_test):
                test_accuracies.append(a_test)
            if a_train is not None and not np.isnan(a_train):
                train_accuracies.append(a_train)

        if test_accuracies:
            print(f"Mean Specialist TEST (OOF) Accuracy: {np.mean(test_accuracies):.4f} +/- {np.std(test_accuracies):.4f}")
            print(f"  (Based on {len(test_accuracies)} specialist models)")

        if train_accuracies:
            print(f"Mean Specialist TRAIN Accuracy:      {np.mean(train_accuracies):.4f} +/- {np.std(train_accuracies):.4f}")
            print(f"  (Based on {len(train_accuracies)} specialist models)")


    # --- Aggregate Standard SHAP Values ---
    avg_shap_values = None
    if avg_shap_values_override is not None:
        avg_shap_values = np.asarray(avg_shap_values_override, dtype=np.float32)
    elif all_shap_values:
        # Check if we have a pre-aggregated file (from K-Fold committee design)
        # This handles both main analysis files and permuted analysis files
        if (len(all_shap_values) == 1 and 
            all_shap_values[0].endswith('.mmap') and 
            'all_groups_avg_shap_values' in all_shap_values[0] and
            os.path.exists(all_shap_values[0])):
            # This is the pre-aggregated file from the K-Fold loop - load it directly
            print("\nLoading pre-aggregated SHAP values from K-Fold committee analysis...")
            n_shap_samples, n_features = X.shape
            output_shape = (n_shap_samples, n_features)
            
            # Load the pre-aggregated memmap file
            print(f"  - Loading from: {all_shap_values[0]}")
            print(f"  - Expected shape: {output_shape}")
            
            avg_shap_values = np.memmap(all_shap_values[0], dtype=np.float32, mode='r', shape=output_shape)
            # Convert to regular array (not memmap) for downstream processing
            avg_shap_values = np.array(avg_shap_values)
            
            # Quick verification after loading memmap
            if avg_shap_values.shape != output_shape:
                print(f"[ERROR] Shape mismatch after loading memmap! Expected {output_shape}, got {avg_shap_values.shape}")

    if avg_shap_values is None:
        print("Skipping downstream SHAP analysis as no SHAP values were generated.")
        return

    # Create base_info_with_probas_df for downstream SHAP analysis
    # Predictions are generated in the K-Fold loop and accumulated across all folds
    base_info_with_probas_df = df[['individual', 'group']].reset_index(drop=True)

    # Generate SHAP Summaries and Plots
    
    # Per-Individual Summary
    # Validation Check: Verify avg_shap_values before creating per-individual summary
    # Validate avg_shap_values before creating DataFrame
    if avg_shap_values is None or avg_shap_values.size == 0:
        print("WARNING: No SHAP values for per-individual summary. Skipping.")
        return
    
    # Ensure avg_shap_values is a numpy array
    if not isinstance(avg_shap_values, np.ndarray):
        avg_shap_values = np.array(avg_shap_values)
    
    # Check for inf values
    if np.any(np.isinf(avg_shap_values)):
        print("WARNING: Found inf values in SHAP data. Replacing with NaN.")
        avg_shap_values = np.where(np.isinf(avg_shap_values), np.nan, avg_shap_values)
    
    # Note: Per-individual summary is now generated in the main analysis section
    # by combining in-sample and out-of-sample sheets. This combined version
    # (from aggregated SHAP values) is no longer needed and has been removed.
    # The individual summary is now only created from the in-sample/out-of-sample
    # breakdown files for better clarity and consistency.

    # Build consolidated summary for the primary SHAP view
    consolidated_df = _build_consolidated_shap_summary_df(
        avg_shap_values=avg_shap_values,
        df=df,
        X=X,
        feature_cols_list=feature_cols_list,
        le=le,
        paths=paths,
        feature_stats_df=feature_stats_df,
        global_importance_overrides=global_importance_overrides,
        extra_feature_annotations_df=extra_feature_annotations_df,
        generate_nontraining_plots=True,
        plot_tag='',
    )

    # Optional: additional consolidated-summary tabs (e.g. weighted / unweighted).
    primary_is_weighted = global_importance_overrides is not None
    consolidated_df = _clean_consolidated_shap_output_df(consolidated_df)
    consolidated_sheets = {}
    sheet_order = []

    primary_sheet_name = _sheet_display_name_for_consolidated_view(
        None,
        primary_is_weighted=primary_is_weighted,
    )
    if primary_sheet_name:
        consolidated_sheets[primary_sheet_name] = consolidated_df
        sheet_order.append(primary_sheet_name)

    if isinstance(extra_consolidated_shap_views, dict) and extra_consolidated_shap_views:
        extra_arrays = extra_consolidated_shap_arrays if isinstance(extra_consolidated_shap_arrays, dict) else {}
        for view_name, mmap_path in extra_consolidated_shap_views.items():
            user_sheet_name = _sheet_display_name_for_consolidated_view(
                str(view_name),
                primary_is_weighted=primary_is_weighted,
            )
            if not user_sheet_name:
                continue
            if not mmap_path or not os.path.exists(mmap_path):
                continue

            try:
                if str(view_name) in extra_arrays and extra_arrays[str(view_name)] is not None:
                    view_shap = np.asarray(extra_arrays[str(view_name)], dtype=np.float32)
                else:
                    view_shape = (X.shape[0], X.shape[1])
                    mm = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=view_shape)
                    view_shap = np.array(mm)
                    del mm

                view_df = _build_consolidated_shap_summary_df(
                    avg_shap_values=view_shap,
                    df=df,
                    X=X,
                    feature_cols_list=feature_cols_list,
                    le=le,
                    paths=paths,
                    feature_stats_df=feature_stats_df,
                    global_importance_overrides=None,
                    extra_feature_annotations_df=extra_feature_annotations_df,
                    generate_nontraining_plots=False,
                    plot_tag=f'_{view_name}',
                )
                view_df = _clean_consolidated_shap_output_df(view_df)
                consolidated_sheets[user_sheet_name] = view_df
                if user_sheet_name not in sheet_order:
                    sheet_order.append(user_sheet_name)
            except Exception as e:
                print(f"WARNING: Failed to build consolidated SHAP view '{view_name}' from {mmap_path}: {e}")

    # Save consolidated summary workbook (multi-tab)
    consolidated_path = os.path.join(
        paths['analysis_folder'],
        f'SHAP_CONSOLIDATED_SUMMARY{paths.get("file_suffix", "")}.xlsx',
    )
    save_consolidated_summary_sheets_with_formatting(
        consolidated_sheets,
        consolidated_path,
        sheet_order=sheet_order,
    )
    print(f"Consolidated SHAP summary saved to: {consolidated_path}")

    # Save global importance ranking CSV (canonical = consolidated_df Global_Importance)
    try:
        ranking_cols = [
            'feature_name',
            'Global_Importance',
        ]

        for extra_gi_col in [
            'Global_Importance_Unweighted',
            'Global_Importance_WeightedRaw',
            'Global_Importance_PermBaseline',
        ]:
            if extra_gi_col in consolidated_df.columns:
                ranking_cols.append(extra_gi_col)

        ranking_cols.extend([
            'Global_Importance_Normalized',
            'Global_Importance_Cumulative',
        ])

        available_cols = [c for c in ranking_cols if c in consolidated_df.columns]
        ranking_df = consolidated_df[available_cols].copy()
        ranking_df.insert(0, 'rank', np.arange(1, len(ranking_df) + 1, dtype=int))
        ranking_path = os.path.join(
            paths['analysis_folder'],
            f"GLOBAL_IMPORTANCE_RANKING{paths.get('file_suffix', '')}.csv",
        )
        ranking_df.to_csv(ranking_path, index=False)
        print(f"Global importance ranking saved to: {ranking_path}")
    except Exception as e:
        print(f"WARNING: Could not write GLOBAL_IMPORTANCE_RANKING CSV: {e}")

