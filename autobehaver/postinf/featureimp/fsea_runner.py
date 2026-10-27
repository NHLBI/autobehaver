"""Run feature-set enrichment analysis on group MeanSHAP or global-importance rankings.

Enrichment uses the ranking metric. Signed-SHAP barcodes retain the original class
direction; global-importance barcodes use a sequential blue magnitude scale.
"""

from __future__ import annotations

import os
import re
import traceback
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from autobehaver.completion import code_digest, verified_run
from .fsea_cache import cached_prerank, dependency_identity
from autobehaver.atomic_io import atomic_write_json, sha256_file

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Optional dependency: required only when the standalone FSEA stage is run.
# gseapy is an optional dependency unless the user explicitly requests FSEA.
# Import it lazily inside the runner so feature importance can run without FSEA installed.

from .config import DEFAULT_SETTINGS
from .runtime_settings import FSEARuntimeSettings
from .fsea import (
    _generate_bout_timebudget_groups,
    _generate_kinematic_groups,
    _generate_transition_groups,
    _generate_postural_groups,
    _generate_spatial_groups,
    _generate_sleep_groups,
    _clean_fsea_results_df,
    _write_fsea_group_documentation,
)
from .io_utils import _write_dataframe_with_autofilter
from .mp_utils import iter_parallel_results

# Manually curated feature groups used by FSEA.
# These definitions are shared across feature-space postanalysis utilities.
from .feature_groups import fsea_feature_groups

# Module-level defaults used when explicit runtime settings are omitted.

_DEFAULT_FSEA_SETTINGS = FSEARuntimeSettings.from_mapping(DEFAULT_SETTINGS)


def _verified_fsea_artifacts(base_folder, artifacts):
    """Verify recorded outputs only inside their declared FSEA directory."""
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    root = Path(base_folder).resolve()
    try:
        for relative, digest in artifacts.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                return False
            relative_path = Path(relative)
            if relative_path.is_absolute() or relative_path.drive or '..' in relative_path.parts:
                return False
            path = (root / relative_path).resolve()
            path.relative_to(root)
            if not path.is_file() or sha256_file(path) != digest:
                return False
    except (OSError, ValueError):
        return False
    return True


def apply_settings(settings: dict) -> None:
    """Update module-level FSEA defaults used by direct callers."""
    global _DEFAULT_FSEA_SETTINGS
    _DEFAULT_FSEA_SETTINGS = FSEARuntimeSettings.from_mapping(settings)



def _warn_if_permissive_fsea_group_size_settings(settings: FSEARuntimeSettings) -> None:
    """Emit a lightweight warning when broad FSEA size settings are in use."""
    if settings.fsea_min_size < 10 or settings.fsea_max_size > 1000:
        print(
            "WARNING: FSEA is using permissive group-size settings "
            f"(min_size={settings.fsea_min_size}, max_size={settings.fsea_max_size}). "
            "This may admit very small or very large groups and can make enrichment "
            "results harder to interpret. Consider using the recommended "
            "range (for example min_size=15, max_size=500)."
        )


def _sanitize_fsea_label(label: str | None) -> str:
    raw = str(label or "").strip() or "GlobalImportance"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe or "GlobalImportance"


def _standardize_ranked_list_dataframe(
    ranked_df: pd.DataFrame,
    *,
    feature_col: str,
    rank_col: str,
    source_name: str,
) -> pd.DataFrame:
    """Validate unique feature names and finite ranking values, then sort descending."""
    if feature_col not in ranked_df.columns:
        raise ValueError(f"{source_name} is missing required column '{feature_col}'")
    if rank_col not in ranked_df.columns:
        raise ValueError(f"{source_name} is missing required column '{rank_col}'")

    out = ranked_df[[feature_col, rank_col]].copy()
    out = out[out[feature_col].notna()].copy()
    out[feature_col] = out[feature_col].astype(str).str.strip()
    out = out[out[feature_col] != ""].copy()
    if out.empty:
        raise ValueError(f"{source_name} does not contain any non-empty feature names in '{feature_col}'")

    dup_mask = out[feature_col].duplicated()
    if dup_mask.any():
        dup_examples = out.loc[dup_mask, feature_col].tolist()[:5]
        raise ValueError(
            f"{source_name} contains duplicate feature names in column '{feature_col}', "
            f"for example: {dup_examples}"
        )

    out = out.rename(columns={feature_col: "feature_name", rank_col: "rank_value"})
    out["rank_value"] = pd.to_numeric(out["rank_value"], errors="raise")
    if not np.isfinite(out["rank_value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source_name} contains nonfinite ranking values")
    out = out.sort_values("rank_value", ascending=False).reset_index(drop=True)
    return out


def _build_combined_fsea_groups(all_feature_names):
    """Construct all supported enrichment groups from the ordered feature list."""
    print("\nGenerating programmatic feature groups for BoutSummary/TimeBudget...")
    programmatic_groups = _generate_bout_timebudget_groups(all_feature_names)
    print(f"  Generated {len(programmatic_groups)} programmatic groups")

    print("\nGenerating programmatic feature groups for Kinematic features...")
    kinematic_groups = _generate_kinematic_groups(all_feature_names)
    print(f"  Generated {len(kinematic_groups)} programmatic kinematic groups")

    print("\nGenerating programmatic feature groups for Transition features...")
    transition_groups = _generate_transition_groups(all_feature_names)
    print(f"  Generated {len(transition_groups)} programmatic transition groups")

    print("\nGenerating programmatic feature groups for Postural features...")
    postural_groups = _generate_postural_groups(all_feature_names)
    print(f"  Generated {len(postural_groups)} programmatic postural groups")

    print("\nGenerating programmatic feature groups for Spatial features...")
    spatial_groups = _generate_spatial_groups(all_feature_names)
    print(f"  Generated {len(spatial_groups)} programmatic spatial groups")

    print("\nGenerating feature groups for Sleep features...")
    sleep_groups = _generate_sleep_groups(all_feature_names)
    print(f"  Generated {len(sleep_groups)} sleep groups")

    all_programmatic_groups = {
        **programmatic_groups,
        **kinematic_groups,
        **transition_groups,
        **postural_groups,
        **spatial_groups,
        **sleep_groups,
    }
    combined_groups = {**all_programmatic_groups, **fsea_feature_groups}
    if len(fsea_feature_groups) > 0:
        print(f"  Merged with {len(fsea_feature_groups)} manually defined groups")
    return combined_groups


def _prepare_validated_fsea_groups(
    all_feature_names,
    output_dir: str,
    settings: FSEARuntimeSettings,
):
    """Filter enrichment groups by present features and size, then document their
    membership.
    """
    combined_groups = _build_combined_fsea_groups(all_feature_names)
    if len(combined_groups) == 0:
        raise ValueError("No feature groups found (neither programmatic nor manual)")

    all_feature_names_set = set(all_feature_names)
    validated_groups = {}
    missing_features_by_group = {}
    skipped_groups = {}

    for group_name, feature_list in combined_groups.items():
        if not feature_list:
            skipped_groups[group_name] = {
                'reason': 'empty',
                'original_features': [],
                'existing_count': 0,
            }
            print(f"WARNING: Feature group '{group_name}' is empty. Skipping.")
            continue

        existing_features = [f for f in feature_list if f in all_feature_names_set]
        missing_features = [f for f in feature_list if f not in all_feature_names_set]

        if missing_features:
            print(f"WARNING: {len(missing_features)} features in group '{group_name}' not found in feature list:")
            if len(missing_features) <= 10:
                print(f"  Missing: {missing_features}")
            else:
                print(f"  Missing: {missing_features[:10]}... (and {len(missing_features) - 10} more)")

        if len(existing_features) < settings.fsea_min_size:
            skipped_groups[group_name] = {
                'reason': 'too_small',
                'original_features': feature_list,
                'existing_count': len(existing_features),
                'missing_features': missing_features if missing_features else [],
            }
            print(
                f"WARNING: Feature group '{group_name}' has only {len(existing_features)} valid features "
                f"(min: {settings.fsea_min_size}). Skipping."
            )
            continue

        validated_groups[group_name] = existing_features
        if missing_features:
            missing_features_by_group[group_name] = missing_features
        print(f"  Group '{group_name}': {len(existing_features)} features")

    if len(validated_groups) == 0:
        raise ValueError("No valid feature groups after validation")

    print(f"\nValidated {len(validated_groups)} feature groups for FSEA analysis.")
    _write_fsea_group_documentation(
        validated_groups,
        all_feature_names,
        output_dir,
        missing_features_by_group,
        skipped_groups,
        settings.fsea_min_size,
    )
    return validated_groups


def _sort_fsea_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    if 'FDR q-val' in results_df.columns:
        return results_df.sort_values('FDR q-val', ascending=True, na_position='last')
    if 'FDR' in results_df.columns:
        return results_df.sort_values('FDR', ascending=True, na_position='last')
    if 'fdr' in results_df.columns:
        return results_df.sort_values('fdr', ascending=True, na_position='last')
    if 'pval' in results_df.columns:
        return results_df.sort_values('pval', ascending=True, na_position='last')
    return results_df


def _significant_count(results_df, threshold):
    for column in ('FDR q-val', 'FDR', 'fdr'):
        if column in results_df.columns:
            return int((pd.to_numeric(results_df[column], errors='raise') < threshold).sum())
    raise ValueError('FSEA result has no empirical FDR column')


def _check_fsea_table(frame):
    """Require finite enrichment scores and valid false-discovery values in a native
    result table.
    """
    if not isinstance(frame, pd.DataFrame) or not any(c in frame.columns for c in ('Term', 'Feature_Set')) or 'ES' not in frame.columns:
        raise ValueError('FSEA result lacks required term/ES columns')
    _significant_count(frame, 0)
    if len(frame):
        scores = pd.to_numeric(frame['ES'], errors='raise').to_numpy(dtype=float)
        fdr_column = next(c for c in ('FDR q-val', 'FDR', 'fdr') if c in frame.columns)
        fdr = pd.to_numeric(frame[fdr_column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(fdr).all() or np.any((fdr < 0) | (fdr > 1)):
            raise ValueError('Required FSEA scores/FDR are nonfinite or outside their valid domain')


def _run_single_global_preranked_fsea(
    *,
    ranked_df: pd.DataFrame,
    validated_groups,
    output_dir: str,
    label: str,
    settings: FSEARuntimeSettings,
    visualization_failures=None,
) -> str:
    """Run one global-importance enrichment ranking and write its results and plots."""
    try:
        import gseapy as gsea  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "gseapy is required for the standalone FSEA stage. "
            "Install it (e.g., `pip install gseapy`) or skip the 'fsea' stage."
        ) from e

    safe_label = _sanitize_fsea_label(label)
    display_label = str(label or '').strip() or 'GlobalImportance'

    fsea_outdir = os.path.join(output_dir, f'FSEA_{safe_label}')
    os.makedirs(fsea_outdir, exist_ok=True)

    result = cached_prerank(gsea,
        rnk=ranked_df,
        gene_sets=validated_groups,
        threads=settings.num_workers,
        permutation_num=settings.fsea_permutations,
        min_size=settings.fsea_min_size,
        max_size=settings.fsea_max_size,
        seed=settings.global_random_seed,
        verbose=False,
        outdir=fsea_outdir,
        no_plot=False,
    )

    _check_fsea_table(result.res2d)
    sorted_results = _sort_fsea_results_table(result.res2d)
    sorted_results = _clean_fsea_results_df(sorted_results)

    output_path = os.path.join(output_dir, f'FSEA_RESULTS_{safe_label}.xlsx')
    _write_dataframe_with_autofilter(sorted_results, output_path)
    print(f"    Results saved to: {output_path}")
    print("    Interpret: Positive ES = feature groups with highest global importance")

    try:
        if len(result.res2d) > 0:
            n_terms_to_plot = min(5, len(result.res2d))
            col_name = 'Feature_Set' if 'Feature_Set' in sorted_results.columns else 'Term'
            terms_to_plot = sorted_results.head(n_terms_to_plot)[col_name].values
            result.plot(terms=terms_to_plot, ofname=os.path.join(fsea_outdir, 'enrichment_plots'))
            print(f"    Plots saved to: {fsea_outdir}")
        else:
            print("    WARNING: No enriched terms found, skipping plots")
    except Exception as e:
        if visualization_failures is not None: visualization_failures.append(str(e))
        print(f"    WARNING: Could not generate plots: {e}")

    n_sig = _significant_count(result.res2d, settings.fsea_fdr_threshold)
    print(f"    Significant groups (FDR < {settings.fsea_fdr_threshold}): {n_sig}")

    try:
        ranked_df_viz = ranked_df.copy()
        ranked_df_viz['shap_value'] = ranked_df_viz['rank_value']
        ranked_df_viz['metric_type'] = 'global_importance'
        _orchestrate_fsea_visualization(
            ranked_df=ranked_df_viz,
            fsea_results_df=sorted_results,
            output_dir=fsea_outdir,
            group_name=display_label,
            top_n=settings.fsea_viz_top_n,
            validated_groups=validated_groups,
            settings=settings,
        )
    except Exception as e:
        if visualization_failures is not None: visualization_failures.append(str(e))
        print(f"    WARNING: Could not generate visualizations: {e}")

    return output_path


def _load_global_importance_ranking_from_analysis_folder(
    *,
    analysis_folder: str,
    all_feature_names,
    file_suffix: str = '',
) -> pd.DataFrame:
    """Load and validate the saved global ranking for the selected analysis features."""
    ranking_csv_path = os.path.join(analysis_folder, f'GLOBAL_IMPORTANCE_RANKING{file_suffix}.csv')
    if os.path.exists(ranking_csv_path):
        print(f"  Loading Global_Importance from ranking CSV: {ranking_csv_path}")
        ranking_df = pd.read_csv(ranking_csv_path)
        ranked_df_global = _standardize_ranked_list_dataframe(
            ranking_df,
            feature_col='feature_name',
            rank_col='Global_Importance',
            source_name='GLOBAL_IMPORTANCE_RANKING.csv',
        )
        ranked_df_global = ranked_df_global[ranked_df_global['feature_name'].isin(all_feature_names)].copy()
        ranked_df_global = ranked_df_global.sort_values('rank_value', ascending=False).reset_index(drop=True)
        n_nonzero = int((ranked_df_global['rank_value'] > 0).sum())
        print(f"  Loaded Global_Importance from CSV for {n_nonzero} features")
        print("  Ranking: Descending (most important features at top)")
        return ranked_df_global

    print("  Ranking CSV not found; falling back to in-sample vs out-of-sample consolidated summary file...")
    insample_outofsample_path = os.path.join(
        analysis_folder,
        f'SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE{file_suffix}.xlsx',
    )
    if not os.path.exists(insample_outofsample_path):
        raise FileNotFoundError(
            f"Could not find either ranking CSV ({ranking_csv_path}) or fallback workbook ({insample_outofsample_path})"
        )

    consolidated_df = pd.read_excel(
        insample_outofsample_path,
        sheet_name='SHAP_Summary',
        header=[0, 1],
    )
    global_importance_col = ('OUT-OF-SAMPLE', 'Global_Importance')
    if global_importance_col not in consolidated_df.columns:
        raise ValueError("Could not find ('OUT-OF-SAMPLE', 'Global_Importance') column")

    feature_name_col = None
    for col in consolidated_df.columns:
        if isinstance(col, tuple) and len(col) >= 2 and str(col[1]).strip() == 'feature_name':
            feature_name_col = col
            break
    if feature_name_col is None:
        for col in consolidated_df.columns:
            if 'feature_name' in str(col).strip():
                feature_name_col = col
                break
    if feature_name_col is None:
        raise ValueError('Could not find feature_name column')

    global_importance_map = dict(zip(consolidated_df[feature_name_col], consolidated_df[global_importance_col]))
    ranked_df_global = pd.DataFrame({
        'feature_name': list(all_feature_names),
        'rank_value': [global_importance_map.get(fname, 0.0) for fname in all_feature_names],
    })
    ranked_df_global['rank_value'] = pd.to_numeric(ranked_df_global['rank_value'], errors='raise')
    if not np.isfinite(ranked_df_global['rank_value'].to_numpy(dtype=float)).all():
        raise ValueError('Global importance ranking contains nonfinite values')
    ranked_df_global = ranked_df_global.sort_values('rank_value', ascending=False).reset_index(drop=True)
    n_nonzero = int((ranked_df_global['rank_value'] > 0).sum())
    print(f"  Loaded Global_Importance_OUTOFSAMPLE for {n_nonzero} features")
    print("  Ranking: Descending (most important features at top)")
    return ranked_df_global


def run_global_importance_fsea_from_file(
    *,
    ranked_list_path: str,
    output_dir: str,
    feature_col: str = 'feature_name',
    rank_col: str = 'Global_Importance',
    label: str = '',
    settings: FSEARuntimeSettings | None = None,
) -> str:
    """Validate a ranking CSV and run one global-importance enrichment analysis."""
    settings = settings or _DEFAULT_FSEA_SETTINGS

    ranked_list_path = str(ranked_list_path or '').strip()
    output_dir = str(output_dir or '').strip()
    if not ranked_list_path:
        raise ValueError('ranked_list_path is required')
    if not output_dir:
        raise ValueError('output_dir is required')
    if not os.path.exists(ranked_list_path):
        raise FileNotFoundError(f'Ranking file not found: {ranked_list_path}')

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Performing Global-Importance FSEA from Ranking File")
    print("=" * 80)
    _warn_if_permissive_fsea_group_size_settings(settings)

    ranking_df = pd.read_csv(ranked_list_path)
    ranked_df = _standardize_ranked_list_dataframe(
        ranking_df,
        feature_col=feature_col,
        rank_col=rank_col,
        source_name=os.path.basename(ranked_list_path) or 'ranking CSV',
    )
    all_feature_names = ranked_df['feature_name'].tolist()
    print(f"  Loaded ranking file: {ranked_list_path}")
    print(f"  Features in ranked list: {len(all_feature_names)}")
    print("  Ranking: Descending (highest rank value at top)")

    validated_groups = _prepare_validated_fsea_groups(
        all_feature_names=all_feature_names,
        output_dir=output_dir,
        settings=settings,
    )

    label_final = str(label or '').strip() or 'GlobalImportance'
    print(f"\nRunning FSEA analysis for {label_final}...")
    visualization_failures = []
    _run_single_global_preranked_fsea(
        ranked_df=ranked_df,
        validated_groups=validated_groups,
        output_dir=output_dir,
        label=label_final,
        settings=settings,
        visualization_failures=visualization_failures,
    )
    status = {"status": "partial" if visualization_failures else "completed",
              "plots_complete": not visualization_failures,
              "visualization_failures": visualization_failures, "settings": asdict(settings)}
    atomic_write_json(status, os.path.join(output_dir, "FSEA_RANKING_STATUS.json"))
    if visualization_failures:
        raise RuntimeError("Requested FSEA plots failed: " + "; ".join(visualization_failures))
    return str(output_dir)


@verified_run
def _perform_fsea_analysis_owned(oof_shap_values, df_for_training, all_feature_names, 
                           training_group_a, training_group_b, df_for_shap_and_agg,
                           paths, le, num_classes, all_groups_avg_shap_values=None,
                           settings: FSEARuntimeSettings | None = None):
    """Run feature-set enrichment on group MeanSHAP and global-importance rankings.

    Training group A and non-training groups rank by -MeanSHAP, so positive ES points
    toward class 0. Training group B ranks by MeanSHAP, so positive ES points toward
    class 1. Barcode colors use the original signed SHAP values: class 0 green and class
    1 magenta.

    Use out-of-fold SHAP for training rows and all-model averaged SHAP for non-training
    rows; arrays must follow the supplied sample and feature order.
    """
    settings = settings or _DEFAULT_FSEA_SETTINGS

    print("\n" + "="*80)
    print("Performing FSEA Analysis on Out-of-Sample MeanSHAP Values")
    print("="*80)
    _warn_if_permissive_fsea_group_size_settings(settings)

    # Lazy import so that users can run feature importance without having gseapy installed,
    # unless they explicitly enable FSEA.
    try:
        import gseapy as gsea  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "gseapy is required for the standalone FSEA stage. "
            "Install it (e.g., `pip install gseapy`) or skip the 'fsea' stage."
        ) from e
    
    # Create base FSEA output folder to organize all FSEA results
    fsea_base_folder = os.path.join(paths['analysis_folder'], 'FSEA')
    os.makedirs(fsea_base_folder, exist_ok=True)
    completion_path = os.path.join(fsea_base_folder, 'FSEA_COMPLETE.json')
    written_outputs = []
    visualization_failures = []
    input_digest = hashlib.sha256()
    for values in (oof_shap_values, all_groups_avg_shap_values):
        if values is None:
            input_digest.update(b'none')
            continue
        array = np.ascontiguousarray(np.asarray(values))
        if not np.isfinite(array).all():
            raise ValueError('FSEA requires finite SHAP inputs')
        input_digest.update(str((array.shape, array.dtype.str)).encode())
        input_digest.update(memoryview(array).cast('B'))
    for frame in (df_for_training, df_for_shap_and_agg):
        identity_columns = [c for c in ('individual','group','sex') if c in frame]
        input_digest.update(frame[identity_columns].to_json(orient='split').encode())
    input_digest.update(json.dumps({'features':list(all_feature_names), 'training_groups':[training_group_a,training_group_b], 'settings':asdict(settings)},sort_keys=True,default=str).encode())
    source_files = [Path(__file__).with_name(name) for name in ('fsea_runner.py', 'fsea_cache.py', 'fsea.py', 'feature_groups.py', 'runtime_settings.py')]
    input_digest.update(json.dumps({'source': code_digest(source_files), 'dependency': dependency_identity(gsea),
                                   'manual_groups': fsea_feature_groups}, sort_keys=True, default=str).encode())
    if os.path.exists(completion_path):
        try:
            previous = json.loads(Path(completion_path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        ranked = _load_global_importance_ranking_from_analysis_folder(analysis_folder=paths['analysis_folder'],
                    all_feature_names=all_feature_names, file_suffix=paths.get('file_suffix', ''))
        candidate_digest = input_digest.copy()
        candidate_digest.update(pd.util.hash_pandas_object(ranked, index=True).to_numpy().tobytes())
        artifacts = previous.get('artifact_sha256', {})
        if (previous.get('receipt_version') == 4 and previous.get('status') == 'completed' and previous.get('plots_complete') is True and previous.get('input_sha256') == candidate_digest.hexdigest()
                and _verified_fsea_artifacts(fsea_base_folder, artifacts)):
            print('FSEA: reusing verified completed statistics and plots.')
            return previous
        print('FSEA: inputs or requested outputs changed; updating from verified numeric caches where applicable.')
    _archive_fsea_generation(fsea_base_folder)



    try:
        validated_groups = _prepare_validated_fsea_groups(
            all_feature_names=all_feature_names,
            output_dir=fsea_base_folder,
            settings=settings,
        )
    except ValueError as e:
        if str(e) not in {'No feature groups found (neither programmatic nor manual)', 'No valid feature groups after validation'}:
            raise
        status = {'status': 'inapplicable', 'reason': str(e)}
        atomic_write_json(status, os.path.join(fsea_base_folder, 'FSEA_STATUS.json'))
        return status

    # Calculate MeanSHAP for each group
    print("\nCalculating MeanSHAP values per group...")
    mean_shap_dict = {}
    all_groups = df_for_shap_and_agg['group'].unique()
    training_groups = set([training_group_a, training_group_b])
    
    # For training groups - use oof_shap_values
    for group_name in all_groups:
        if group_name in training_groups:
            # Training group - use df_for_training and oof_shap_values
            group_mask = df_for_training['group'] == group_name
            if group_mask.sum() == 0:
                print(f"WARNING: No samples found for training group '{group_name}'. Skipping.")
                continue
            
            group_shap = oof_shap_values[group_mask, :]
            mean_shap = np.mean(group_shap, axis=0)
            mean_shap_dict[group_name] = mean_shap
            print(f"  {group_name} (training): {group_mask.sum()} samples")
    
    # For non-training groups - use all_groups_avg_shap_values if available
    if all_groups_avg_shap_values is not None:
        for group_name in all_groups:
            if group_name not in training_groups:
                # Non-training group
                group_mask = df_for_shap_and_agg['group'] == group_name
                if group_mask.sum() == 0:
                    continue
                
                # Handle binary classification
                group_shap = all_groups_avg_shap_values[group_mask, :]
                
                mean_shap = np.mean(group_shap, axis=0)
                mean_shap_dict[group_name] = mean_shap
                print(f"  {group_name} (non-training): {group_mask.sum()} samples")
    
    # Create ranked lists and run FSEA for training groups
    print("\nRunning FSEA analysis for training groups...")
    
    # Training Group 1: Run FSEA (single analysis, no ascending/descending split)
    if training_group_a in mean_shap_dict:
        mean_shap_a = mean_shap_dict[training_group_a]
        
        # Flip signs: negative SHAP values become positive (class 0 direction)
        mean_shap_a_flipped = -mean_shap_a
        
        # Create ranked DataFrame (descending by default)
        ranked_df_a = pd.DataFrame({
            'feature_name': all_feature_names,
            'rank_value': mean_shap_a_flipped
        })
        ranked_df_a = ranked_df_a.sort_values('rank_value', ascending=False)
        
        print(f"\n  Training Group 1 ({training_group_a})")
        try:
            fsea_outdir_a = os.path.join(fsea_base_folder, f'FSEA_{training_group_a}')
            os.makedirs(fsea_outdir_a, exist_ok=True)
            
            result_a = cached_prerank(gsea,
                rnk=ranked_df_a,
                gene_sets=validated_groups,
                threads=settings.num_workers,
                permutation_num=settings.fsea_permutations,
                min_size=settings.fsea_min_size,
                max_size=settings.fsea_max_size,
                seed=settings.global_random_seed,
                verbose=False,
                outdir=fsea_outdir_a,
                no_plot=False
            )
            
            output_path_a = os.path.join(fsea_base_folder, 
                                        f'FSEA_RESULTS_{training_group_a}.xlsx')
            _check_fsea_table(result_a.res2d)
            sorted_results_a = _sort_fsea_results_table(result_a.res2d)
            
            # Clean up column names: remove 'Name', rename 'Term' to 'Feature_Set'
            sorted_results_a = _clean_fsea_results_df(sorted_results_a)
            
            _write_dataframe_with_autofilter(sorted_results_a, output_path_a)
            written_outputs.append(output_path_a)
            print(f"    Results saved to: {output_path_a}")
            
            try:
                if len(result_a.res2d) > 0:
                    n_terms_to_plot = min(5, len(result_a.res2d))
                    # Use 'Feature_Set' if available, otherwise 'Term' (for plotting before cleanup)
                    col_name = 'Feature_Set' if 'Feature_Set' in sorted_results_a.columns else 'Term'
                    terms_to_plot = sorted_results_a.head(n_terms_to_plot)[col_name].values
                    result_a.plot(terms=terms_to_plot, ofname=os.path.join(fsea_outdir_a, 'enrichment_plots'))
                    print(f"    Plots saved to: {fsea_outdir_a}")
            except Exception as e:
                visualization_failures.append(str(e))
                print(f"    WARNING: Could not generate plots: {e}")
            
            n_sig = _significant_count(sorted_results_a, settings.fsea_fdr_threshold)
            print(f"    Significant groups (FDR < {settings.fsea_fdr_threshold}): {n_sig}")
            
            try:
                ranked_df_a_viz = ranked_df_a.copy()

                # Barcode colors are based on the ORIGINAL (unflipped) mean SHAP,
                # so that A (class 0) direction is always GREEN and B (class 1) is always MAGENTA.
                shap_map_a = dict(zip(all_feature_names, mean_shap_a))
                ranked_df_a_viz['shap_value'] = ranked_df_a_viz['feature_name'].map(shap_map_a)

                # Tag metric type so Global Importance can use a blue palette
                ranked_df_a_viz['metric_type'] = 'signed_shap'

                _orchestrate_fsea_visualization(
                    ranked_df=ranked_df_a_viz,
                    fsea_results_df=sorted_results_a,
                    output_dir=fsea_outdir_a,
                    group_name=training_group_a,
                    top_n=settings.fsea_viz_top_n,
                    validated_groups=validated_groups,
                    settings=settings,
                )
            except Exception as e:
                visualization_failures.append(str(e))
                print(f"    WARNING: Could not generate visualizations: {e}")
                traceback.print_exc()

                
        except Exception as e:
            print(f"    ERROR running FSEA for {training_group_a}: {e}")
            traceback.print_exc()
            raise
    
    # Training Group 2: Run FSEA (single analysis, no ascending/descending split)
    if training_group_b in mean_shap_dict:
        mean_shap_b = mean_shap_dict[training_group_b]
        
        # Create ranked DataFrame (descending by default)
        ranked_df_b = pd.DataFrame({
            'feature_name': all_feature_names,
            'rank_value': mean_shap_b
        })
        ranked_df_b = ranked_df_b.sort_values('rank_value', ascending=False)
        
        print(f"\n  Training Group 2 ({training_group_b})")
        try:
            fsea_outdir_b = os.path.join(fsea_base_folder, f'FSEA_{training_group_b}')
            os.makedirs(fsea_outdir_b, exist_ok=True)
            
            result_b = cached_prerank(gsea,
                rnk=ranked_df_b,
                gene_sets=validated_groups,
                threads=settings.num_workers,
                permutation_num=settings.fsea_permutations,
                min_size=settings.fsea_min_size,
                max_size=settings.fsea_max_size,
                seed=settings.global_random_seed,
                verbose=False,
                outdir=fsea_outdir_b,
                no_plot=False
            )
            
            output_path_b = os.path.join(fsea_base_folder, 
                                        f'FSEA_RESULTS_{training_group_b}.xlsx')
            _check_fsea_table(result_b.res2d)
            sorted_results_b = _sort_fsea_results_table(result_b.res2d)
            
            # Clean up column names: remove 'Name', rename 'Term' to 'Feature_Set'
            sorted_results_b = _clean_fsea_results_df(sorted_results_b)
            
            _write_dataframe_with_autofilter(sorted_results_b, output_path_b)
            written_outputs.append(output_path_b)
            print(f"    Results saved to: {output_path_b}")
            
            try:
                if len(result_b.res2d) > 0:
                    n_terms_to_plot = min(5, len(result_b.res2d))
                    # Use 'Feature_Set' if available, otherwise 'Term' (for plotting before cleanup)
                    col_name = 'Feature_Set' if 'Feature_Set' in sorted_results_b.columns else 'Term'
                    terms_to_plot = sorted_results_b.head(n_terms_to_plot)[col_name].values
                    result_b.plot(terms=terms_to_plot, ofname=os.path.join(fsea_outdir_b, 'enrichment_plots'))
                    print(f"    Plots saved to: {fsea_outdir_b}")
            except Exception as e:
                visualization_failures.append(str(e))
                print(f"    WARNING: Could not generate plots: {e}")
            
            n_sig = _significant_count(sorted_results_b, settings.fsea_fdr_threshold)
            print(f"    Significant groups (FDR < {settings.fsea_fdr_threshold}): {n_sig}")
            
            try:
                ranked_df_b_viz = ranked_df_b.copy()

                # Barcode colors come from signed SHAP directly (negative=A green, positive=B magenta)
                ranked_df_b_viz['shap_value'] = ranked_df_b_viz['rank_value']

                # Tag metric type so Global Importance can use a blue palette
                ranked_df_b_viz['metric_type'] = 'signed_shap'

                _orchestrate_fsea_visualization(
                    ranked_df=ranked_df_b_viz,
                    fsea_results_df=sorted_results_b,
                    output_dir=fsea_outdir_b,
                    group_name=training_group_b,
                    top_n=settings.fsea_viz_top_n,
                    validated_groups=validated_groups,
                    settings=settings,
                )
            except Exception as e:
                visualization_failures.append(str(e))
                print(f"    WARNING: Could not generate visualizations: {e}")
                traceback.print_exc()

                
        except Exception as e:
            print(f"    ERROR running FSEA for {training_group_b}: {e}")
            traceback.print_exc()
            raise
    
    # Run FSEA for non-training groups
    if all_groups_avg_shap_values is not None:
        print("\nRunning FSEA analysis for non-training groups...")
        print("  Note: Ranking is flipped so Training Group A (class 0) direction is at the TOP (green).")
        print("    - Positive ES = enriched at top (pushing toward class 0 / training_group_a)")
        print("    - Negative ES = enriched at bottom (pushing toward class 1 / training_group_b)")
        for group_name in all_groups:
            if group_name not in training_groups and group_name in mean_shap_dict:
                print(f"\n  Non-Training Group ({group_name}):")
                mean_shap_nt = mean_shap_dict[group_name]

                ranked_df_nt = pd.DataFrame({
                    'feature_name': all_feature_names,
                    # Flip rank so A-direction (negative SHAP) features appear at the top of the list
                    'rank_value': -mean_shap_nt
                })
                ranked_df_nt = ranked_df_nt.sort_values('rank_value', ascending=False)

                
                try:
                    # Create output directory for this FSEA analysis
                    fsea_outdir_nt = os.path.join(fsea_base_folder, 
                                                  f'FSEA_{group_name}')
                    os.makedirs(fsea_outdir_nt, exist_ok=True)
                    
                    result_nt = cached_prerank(gsea,
                        rnk=ranked_df_nt,
                        gene_sets=validated_groups,
                        threads=settings.num_workers,
                        permutation_num=settings.fsea_permutations,
                        min_size=settings.fsea_min_size,
                        max_size=settings.fsea_max_size,
                        seed=settings.global_random_seed,
                        verbose=False,
                        outdir=fsea_outdir_nt,
                        no_plot=False  # Generate plots automatically
                    )
                    
                    output_path_nt = os.path.join(fsea_base_folder, 
                                                  f'FSEA_RESULTS_{group_name}.xlsx')
                    # Sort by FDR (ascending) - most significant results first
                    _check_fsea_table(result_nt.res2d)
                    sorted_results_nt = _sort_fsea_results_table(result_nt.res2d)
                    
                    # Clean up column names: remove 'Name', rename 'Term' to 'Feature_Set'
                    sorted_results_nt = _clean_fsea_results_df(sorted_results_nt)
                    
                    _write_dataframe_with_autofilter(sorted_results_nt, output_path_nt)
                    written_outputs.append(output_path_nt)
                    print(f"    Results saved to: {output_path_nt} (sorted by FDR, most significant first)")
                    print(f'    Interpret ES sign: Positive = class 0 direction, Negative = class 1 direction')
                    
                    # Generate enrichment plots manually
                    try:
                        if len(result_nt.res2d) > 0:
                            n_terms_to_plot = min(5, len(result_nt.res2d))
                            # Use 'Feature_Set' if available, otherwise 'Term' (for plotting before cleanup)
                            col_name = 'Feature_Set' if 'Feature_Set' in sorted_results_nt.columns else 'Term'
                            terms_to_plot = sorted_results_nt.head(n_terms_to_plot)[col_name].values
                            result_nt.plot(terms=terms_to_plot, ofname=os.path.join(fsea_outdir_nt, 'enrichment_plots'))
                            print(f"    Plots saved to: {fsea_outdir_nt}")
                        else:
                            print(f"    WARNING: No enriched terms found, skipping plots")
                    except Exception as e:
                        visualization_failures.append(str(e))
                        print(f"    WARNING: Could not generate plots: {e}")
                    
                    n_sig = _significant_count(result_nt.res2d, settings.fsea_fdr_threshold)
                    print(f"    Significant groups (FDR < {settings.fsea_fdr_threshold}): {n_sig}")
                    
                    try:
                        ranked_df_nt_viz = ranked_df_nt.copy()

                        # Barcode colors come from ORIGINAL signed SHAP (negative=A green, positive=B magenta)
                        shap_map_nt = dict(zip(all_feature_names, mean_shap_nt))
                        ranked_df_nt_viz['shap_value'] = ranked_df_nt_viz['feature_name'].map(shap_map_nt)

                        # Tag metric type so Global Importance can use a blue palette
                        ranked_df_nt_viz['metric_type'] = 'signed_shap'

                        _orchestrate_fsea_visualization(
                            ranked_df=ranked_df_nt_viz,
                            fsea_results_df=sorted_results_nt,
                            output_dir=fsea_outdir_nt,
                            group_name=group_name,
                            top_n=settings.fsea_viz_top_n,
                            validated_groups=validated_groups,
                            settings=settings,
                        )
                    except Exception as e:
                        visualization_failures.append(str(e))
                        print(f"    WARNING: Could not generate visualizations: {e}")

                        
                except Exception as e:
                    print(f"    ERROR running FSEA for {group_name}: {e}")
                    raise
    
    # Run FSEA for Global Importance (non-genotype-specific) - prefer ranking CSV
    print("\nRunning FSEA analysis for Global Importance (OUT-OF-SAMPLE)...")

    file_suffix = paths.get("file_suffix", "")

    try:
        ranked_df_global = _load_global_importance_ranking_from_analysis_folder(
            analysis_folder=paths["analysis_folder"],
            all_feature_names=all_feature_names,
            file_suffix=file_suffix,
        )
    except Exception as e:
        print(f"    ERROR loading global-importance ranking for FSEA: {e}")
        traceback.print_exc()
        raise

    try:
        _run_single_global_preranked_fsea(
            ranked_df=ranked_df_global,
            validated_groups=validated_groups,
            output_dir=fsea_base_folder,
            label="GlobalImportance_OutOfSample",
            settings=settings, visualization_failures=visualization_failures,
        )
    except Exception as e:
        print(f"    ERROR running FSEA for Global Importance (OUT-OF-SAMPLE): {e}")
        traceback.print_exc()
        raise
    written_outputs.append(os.path.join(fsea_base_folder, 'FSEA_RESULTS_GlobalImportance_OutOfSample.xlsx'))
    input_digest.update(pd.util.hash_pandas_object(ranked_df_global, index=True).to_numpy().tobytes())
    artifact_sha256 = {p.relative_to(Path(fsea_base_folder)).as_posix(): sha256_file(p)
                       for p in sorted(Path(fsea_base_folder).rglob('*')) if p.is_file() and p != Path(completion_path)}
    status = {'status': 'partial' if visualization_failures else 'completed', 'receipt_version':4, 'input_sha256':input_digest.hexdigest(),
              'artifact_sha256': artifact_sha256, 'plots_complete': not visualization_failures,
              'visualization_failures': visualization_failures,
              'settings':asdict(settings), 'artifact_paths':list(written_outputs),
              'artifacts': {os.path.basename(p): sha256_file(p) for p in written_outputs}}
    atomic_write_json(status, completion_path)
    return status



def _archive_fsea_generation(base_folder):
    """Preserve an old output generation; reuse only self-verified numerical caches."""
    import shutil
    from uuid import uuid4
    root = Path(base_folder).resolve()
    if not root.is_dir() or not any(root.iterdir()):
        return
    history = root.parent / "_featureimp_history"
    history.mkdir(exist_ok=True)
    archived = history / ("FSEA_" + uuid4().hex)
    # Both absolute paths are constrained to this analysis folder before the move.
    if not archived.resolve().is_relative_to(root.parent) or root.name != "FSEA":
        raise ValueError("Unsafe FSEA history destination")
    root.rename(archived)
    root.mkdir()
    for cache in archived.rglob(".numeric_cache"):
        if not cache.is_dir():
            continue
        target = root / cache.relative_to(archived)
        shutil.copytree(cache, target)
    return str(archived)


def _perform_fsea_analysis(oof_shap_values, df_for_training, all_feature_names,
                           training_group_a, training_group_b, df_for_shap_and_agg,
                           paths, le, num_classes, all_groups_avg_shap_values=None,
                           settings: FSEARuntimeSettings | None = None):
    """Own a complete output generation while retaining the public call contract."""
    from autobehaver.file_locks import FileLock
    root = Path(paths["analysis_folder"])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "_featureimp_fsea.lock"):
        return _perform_fsea_analysis_owned(
            oof_shap_values, df_for_training, all_feature_names, training_group_a,
            training_group_b, df_for_shap_and_agg, paths, le, num_classes,
            all_groups_avg_shap_values=all_groups_avg_shap_values, settings=settings)


# FSEA VISUALIZATION FUNCTIONS

def _calculate_running_enrichment_score(ranked_df, feature_list, score_col='rank_value', weight=1.0):
    """Calculate a weighted running enrichment curve from the ranking metric.

    Use absolute score raised to weight for hits and a uniform miss penalty; barcode
    colors do not affect enrichment.
    """
    if feature_list is None or len(feature_list) == 0:
        return None

    # Require expected columns
    if 'feature_name' not in ranked_df.columns or score_col not in ranked_df.columns:
        return None

    feature_set = set(feature_list)
    hits = ranked_df['feature_name'].isin(feature_set).values

    N = len(ranked_df)
    Nh = int(hits.sum())           # count hits actually present in ranked_df
    Nm = N - Nh

    if Nh == 0 or Nm <= 0:
        return np.zeros(N, dtype=float)

    scores = ranked_df[score_col].to_numpy(dtype=float)

    # Weighted hit contributions: |score|^weight (standard prerank GSEA)
    w = (np.abs(scores) ** float(weight))

    sum_hit_w = w[hits].sum()
    if sum_hit_w <= 0:
        # Fallback: unweighted hit walk
        p_hit = np.where(hits, 1.0 / Nh, 0.0)
    else:
        p_hit = np.where(hits, w / sum_hit_w, 0.0)

    # Miss penalty is uniform
    p_miss = np.where(hits, 0.0, 1.0 / Nm)

    running_es = np.cumsum(p_hit - p_miss)
    return running_es



def _generate_fsea_barcode_plot(args):
    """Render one barcode and optional enrichment curve from a worker payload.

    The tuple contains ranked data, feature membership, output path, title, figure size,
    DPI, curve toggle, and color percentile. Return the output path on success or None
    on a plotting error.
    """
    (ranked_df, feature_list, output_path, title, figsize, dpi, add_mountain_plot, color_percentile) = args

    try:
        # Safety: ensure required columns exist
        if 'feature_name' not in ranked_df.columns:
            raise ValueError("ranked_df is missing required column: 'feature_name'")

        # We color the barcode using `shap_value` (A=negative, B=positive).
        # If it doesn't exist, fall back to rank_value.
        if 'shap_value' not in ranked_df.columns:
            if 'rank_value' not in ranked_df.columns:
                raise ValueError("ranked_df must contain either 'shap_value' or 'rank_value'")
            ranked_df = ranked_df.copy()
            ranked_df['shap_value'] = ranked_df['rank_value']

        shap_values = ranked_df['shap_value'].to_numpy(dtype=float)
        n_features = len(shap_values)

        # Figure layout
        if add_mountain_plot and feature_list is not None:
            fig, axes = plt.subplots(
                1, 2, figsize=figsize, dpi=dpi,
                gridspec_kw={'width_ratios': [1, 0.8]}
            )
            ax_barcode = axes[0]
            ax_mountain = axes[1]
        else:
            fig, ax_barcode = plt.subplots(figsize=figsize, dpi=dpi)
            ax_mountain = None

        # Barcode colors
        abs_values = np.abs(shap_values)

        # Single-scale color range (no separate "clipped" output folder).
        # Set color_percentile = 100 for full range, or <100 for robust scaling.
        if color_percentile is not None and color_percentile < 100:
            robust_max = float(np.percentile(abs_values, color_percentile))
        else:
            robust_max = float(max(abs(shap_values.min()), abs(shap_values.max())))

        if robust_max < 1e-6:
            robust_max = 1.0

        # Decide palette: signed SHAP (A/B) vs global importance (blue)
        metric_type = None
        if 'metric_type' in ranked_df.columns and len(ranked_df) > 0:
            metric_type = str(ranked_df['metric_type'].iloc[0])

        # Colorblind-friendlier A/B colors (Okabe-Ito palette)
        COLOR_A_GREEN = '#009E73'
        COLOR_B_MAGENTA = '#CC79A7'
        COLOR_GLOBAL_BLUE = '#0072B2'

        if metric_type == 'global_importance':
            # Global importance: sequential white -> blue
            norm = mcolors.Normalize(vmin=0, vmax=robust_max)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "white_to_blue", ["white", COLOR_GLOBAL_BLUE]
            )
        else:
            # Signed SHAP: diverging green (A) -> white (0) -> magenta (B)
            norm = mcolors.TwoSlopeNorm(vmin=-robust_max, vcenter=0, vmax=robust_max)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "A_green_to_B_magenta", [COLOR_A_GREEN, "white", COLOR_B_MAGENTA]
            )

        ax_barcode.imshow(
            shap_values.reshape(-1, 1),
            cmap=cmap, norm=norm,
            aspect='auto', interpolation='nearest',
            extent=[-0.5, 0.5, n_features, 0]
        )

        # Hit marks (features in the set)
        if feature_list is not None:
            feature_set = set(feature_list)
            hits = ranked_df['feature_name'].isin(feature_set).values
            hit_indices = np.where(hits)[0]
            if hit_indices.size > 0:
                ax_barcode.hlines(hit_indices, -0.5, 0.5, colors='black', linewidth=1.5, alpha=0.8)

        # Style barcode
        ax_barcode.set_xticks([])
        ax_barcode.set_yticks([])
        for spine in ax_barcode.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('black')

        # Mountain plot (proper weighted running ES, oriented so bulge is right)
        if ax_mountain is not None:
            # ES computed from the *ranking metric*, not from barcode colors.
            es_curve = _calculate_running_enrichment_score(ranked_df, feature_list, score_col='rank_value', weight=1.0)

            if es_curve is not None:
                y_vals = np.arange(n_features)
                lavender_color = '#967BB6'

                # Orient so the strongest enrichment bulges to the RIGHT:
                # If the most extreme point is negative, flip the entire curve.
                peak_idx = int(np.argmax(np.abs(es_curve)))
                peak_es = float(es_curve[peak_idx])
                orientation = -1.0 if peak_es < 0 else 1.0
                es_plot = es_curve * orientation

                ax_mountain.plot(es_plot, y_vals, color=lavender_color, linewidth=2)
                ax_mountain.fill_betweenx(y_vals, es_plot, 0, color=lavender_color, alpha=0.3)
                ax_mountain.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

                ax_mountain.set_ylim(n_features, 0)
                ax_mountain.set_yticks([])
                ax_mountain.set_xlabel("Running ES (oriented â†’)")

                # Small label preserving the original ES sign
                ax_mountain.text(
                    0.02, 0.98, f"ES_peak={peak_es:.3f}",
                    transform=ax_mountain.transAxes,
                    ha='left', va='top', fontsize=8
                )

                ax_mountain.spines['top'].set_visible(False)
                ax_mountain.spines['right'].set_visible(False)
                ax_mountain.spines['left'].set_visible(False)

        # Title
        clean_title = title.replace("_", " ")
        if ax_mountain is not None:
            fig.suptitle(clean_title, fontsize=10, y=0.98)
        else:
            ax_barcode.set_title(clean_title, fontsize=10, pad=10)
        # Layout:
        # Matplotlib 3.7+ can emit a warning when calling tight_layout() on a figure whose
        # GridSpec was created with explicit wspace/hspace parameters. We keep layout
        # deterministic by:
        #   1) letting tight_layout() compute margins (esp. for titles/labels)
        #   2) then applying a small manual inter-axis wspace tweak.
        if ax_mountain is not None:
            # Leave a little headroom for the suptitle.
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(wspace=0.05)
        else:
            fig.tight_layout()

        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return output_path

    except Exception as e:
        print(f"    ERROR generating plot {output_path}: {e}")
        traceback.print_exc()
        return None


def _orchestrate_fsea_visualization(ranked_df, fsea_results_df,
                                   output_dir, group_name="", top_n=10, validated_groups=None,
                                   settings: FSEARuntimeSettings | None = None):
    """Render selected enrichment barcodes using one orientation and color scale.

    Signed SHAP uses green for class 0 and magenta for class 1; global importance uses a
    sequential blue palette.
    """
    settings = settings or _DEFAULT_FSEA_SETTINGS

    # Keep output to a few high-signal lines.

    if fsea_results_df is None or len(fsea_results_df) == 0:
        # Nothing to visualize.
        return

    # Ensure we have an FDR column to filter/sort by
    if 'FDR q-val' not in fsea_results_df.columns:
        fsea_results_df = fsea_results_df.copy()
        if 'FDR' in fsea_results_df.columns:
            fsea_results_df['FDR q-val'] = fsea_results_df['FDR']
        elif 'fdr' in fsea_results_df.columns:
            fsea_results_df['FDR q-val'] = fsea_results_df['fdr']
        else:
            print(f"    WARNING: Cannot find FDR/fdr/pval column in FSEA results. Skipping visualization.")
            return

    # Determine which groups to plot
    if settings.fsea_viz_plot_all_groups:
        df_to_plot = fsea_results_df.copy()
        df_to_plot = df_to_plot.sort_values('FDR q-val', ascending=True, na_position='last')
        print(f"    Plotting all {len(df_to_plot)} FSEA groups for '{group_name}' (plot_all_groups enabled)...")
    else:
        df_sig = fsea_results_df[fsea_results_df['FDR q-val'] <= settings.fsea_fdr_threshold].copy()
        if len(df_sig) == 0:
            print(f"    No significant groups (FDR <= {settings.fsea_fdr_threshold}). Skipping visualization.")
            return
        df_to_plot = df_sig.sort_values('FDR q-val', ascending=True).head(top_n)
        print(f"    Plotting top {len(df_to_plot)} significant FSEA groups for '{group_name}' (FDR <= {settings.fsea_fdr_threshold})...")

    # Extract groups - use validated_groups if available, otherwise fall back to Lead_genes
    groups = {}
    for _, row in df_to_plot.iterrows():
        term = row.get('Feature_Set', row.get('Term', ''))

        # Preferred source: validated_groups
        if validated_groups is not None and term in validated_groups:
            features = validated_groups[term]
            if len(features) > 0:
                groups[term] = features
                continue

        # Fallback: Lead_genes
        lead_genes_str = row.get('Lead_genes', '')
        if pd.isna(lead_genes_str) or not lead_genes_str:
            continue

        genes = [g.strip() for g in str(lead_genes_str).split(';') if g.strip()]
        if len(genes) > 0:
            groups[term] = genes

    if len(groups) == 0:
        print(f"    No valid feature groups found. Skipping visualization.")
        return

    print(f"    Generating barcode plots for {len(groups)} groups...")

    # Output folder (single set; no Normal/Clipped variants)
    current_out_dir = os.path.join(output_dir, "BarcodePlots")
    os.makedirs(current_out_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    plot_args = []

    # Master reference (all features, no hit marks, no mountain)
    plot_args.append((
        ranked_df, None,
        os.path.join(current_out_dir, "Master_Reference.png"),
        f"All Features ({group_name})",
        (6, 20), 100, False, settings.fsea_viz_color_percentile
    ))

    # Group plots
    for group_term, features in groups.items():
        safe_name = "".join([c if c.isalnum() or c in ('-','_') else '_' for c in group_term])
        out_path = os.path.join(current_out_dir, f"{safe_name}.png")

        plot_args.append((
            ranked_df, features, out_path,
            f"{group_term}",
            (6, 20), 100, True, settings.fsea_viz_color_percentile
        ))

    if len(plot_args) == 0:
        return

    results = list(
        iter_parallel_results(
            _generate_fsea_barcode_plot,
            plot_args,
            num_workers=settings.num_workers,
            desc="    Generating plots",
            stage_name=f"FSEA barcode plotting for {group_name}",
        )
    )

    successful = sum(1 for r in results if r is not None)
    print(f"    Generated {successful}/{len(plot_args)} barcode plots for '{group_name}'.")
    if successful != len(plot_args):
        raise RuntimeError(f'FSEA barcode plotting failed for {len(plot_args)-successful} requested plot(s)')



# RESULTS VALIDATION
