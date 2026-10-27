"""Preserve FeatureImp helper entry points by delegating to the analysis and reporting
modules.
"""

from __future__ import annotations

from .config import DEFAULT_SETTINGS

# Core I/O entry points delegated to `featureimp.matrix`.
from .matrix import (
    validate_feature_matrix as _validate_feature_matrix,
    normalize_feature_matrix as _normalize_feature_matrix,
)

def _get_current_settings() -> dict:
    """Return the active FeatureImp settings snapshot."""

    try:
        from . import core as _core
        return _core.get_current_settings()
    except Exception:
        return dict(DEFAULT_SETTINGS)


def _setting(key: str, default=None):
    settings = _get_current_settings()
    return settings.get(key, default)


def _int_setting(key: str, default: int) -> int:
    return int(_setting(key, default) or default)


def _float_setting(key: str, default: float) -> float:
    return float(_setting(key, default) or default)


def _bool_setting(key: str, default: bool) -> bool:
    return bool(_setting(key, default))


def apply_settings(settings_dict: dict) -> None:
    """Forward settings to the FeatureImp core snapshot."""

    from . import core as _core

    _core.apply_settings(settings_dict, propagate=False)



def _clean_performance_dict_for_json(*args, **kwargs):
    """Convert performance metrics to JSON-compatible values."""
    from . import summaries as _summaries
    return _summaries._clean_performance_dict_for_json(*args, **kwargs)

def validate_feature_matrix(matrix_path, required_id_cols=['individual', 'group', 'sex']):
    """Validate the feature matrix through the canonical matrix helper."""
    return _validate_feature_matrix(matrix_path, required_id_cols=required_id_cols)

def normalize_feature_matrix(
    results_dir,
    raw_matrix_path=None,
    skip_if_normalized_exists=False,
    analysis_folder_path=None,
):
    """Normalize the selected matrix through the canonical FeatureImp preprocessing
    helper.
    """
    return _normalize_feature_matrix(
        results_dir=results_dir,
        raw_matrix_path=raw_matrix_path,
        skip_if_normalized_exists=skip_if_normalized_exists,
        analysis_folder_path=analysis_folder_path,
        control_group_tag=_setting('control_group_tag', None),
        training_group_a=_setting('training_group_a', None),
        training_group_b=_setting('training_group_b', None),
    )

def get_analysis_folder_name(analysis_run_name='', sex_stratum=None):
    """Resolve the analysis folder name from the run name and sex stratum."""
    from . import prep as _prep
    return _prep.get_analysis_folder_name(analysis_run_name=analysis_run_name, sex_stratum=sex_stratum)

def filter_data_by_sex(df, sex_value):
    """Select the requested sex stratum for analysis."""
    from . import prep as _prep
    return _prep.filter_data_by_sex(df=df, sex_value=sex_value)

def _parse_feature_name(feature_name):
    """Parse a feature's source, category, smoothing, and measurement tokens."""
    from . import prep as _prep
    return _prep._parse_feature_name(feature_name)

def _filter_sparse_features_by_group(df, training_group_a, training_group_b, 
                                     min_completeness, feature_cols):
    """Retain features meeting completeness criteria in both training groups."""
    from . import prep as _prep
    return _prep._filter_sparse_features_by_group(
        df=df,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        min_completeness=min_completeness,
        feature_cols=feature_cols,
    )

def _filter_low_variance_features(X_train_full, y_train, training_group_a, training_group_b, 
                                  feature_cols, min_variance_threshold=1e-6):
    """Remove constant and near-constant imputed training features."""
    from . import prep as _prep
    return _prep._filter_low_variance_features(
        X_train_full=X_train_full,
        y_train=y_train,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        feature_cols=feature_cols,
        min_variance_threshold=min_variance_threshold,
    )

def _init_pool_worker(*args, **kwargs):
    """Initialize read-only matrices and labels in a committee worker."""
    from . import committee as _committee
    return _committee._init_pool_worker(*args, **kwargs)

def _train_subset_fold_worker(*args, **kwargs):
    """Train one specialist fold through the committee implementation."""
    from . import committee as _committee
    return _committee._train_subset_fold_worker(*args, **kwargs)


def _generate_sex_class_stratified_permutation(
    y_original,
    df_with_sex,
    train_indices,
    test_indices,
    permutation_seed=None,
):
    """Shuffle training-fold labels within sex strata."""
    
    from . import permutation as _permutation
    return _permutation._generate_sex_class_stratified_permutation(
        y_original,
        df_with_sex,
        train_indices,
        test_indices,
        permutation_seed,
    )

def _flip_probabilities_for_permuted_samples(probas, y_original_full, y_permuted_full, train_indices):
    """Restore permuted training predictions to the original class frame."""
    
    from . import permutation as _permutation
    return _permutation._flip_probabilities_for_permuted_samples(
        probas,
        y_original_full,
        y_permuted_full,
        train_indices,
    )


def _sanitize_group_name_for_filename(*args, **kwargs):
    """Return a portable filename token for a group label."""
    from . import summaries as _summaries
    return _summaries._sanitize_group_name_for_filename(*args, **kwargs)

def _create_base_frame_plot(ax_limits, xlabel, ylabel, title, legend_items=None, figsize=(12, 10), dpi=300):
    """Create a labeled, empty matplotlib frame for delegation callers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(ax_limits['xlim'])
    ax.set_ylim(ax_limits['ylim'])
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if legend_items:
        for label, color in legend_items:
            ax.scatter([], [], color=color, label=label, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    return fig, ax, ax.get_position()

def _create_group_only_plot(x_data, y_data, color, ax_limits, ax_position, figsize=(12, 10), dpi=300):
    """Create a transparent single-group overlay plot for delegation callers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='none')
    ax.set_position(ax_position)
    ax.set_xlim(ax_limits['xlim'])
    ax.set_ylim(ax_limits['ylim'])
    ax.scatter(x_data, y_data, c=[color], alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    return fig

def _generate_bout_timebudget_groups(*args, **kwargs):
    """Construct broad bout-summary and time-budget enrichment groups."""
    from . import fsea as _fsea
    return _fsea._generate_bout_timebudget_groups(*args, **kwargs)

def _generate_kinematic_groups(*args, **kwargs):
    """Construct supported behavior and body-part kinematic enrichment groups."""
    from . import fsea as _fsea
    return _fsea._generate_kinematic_groups(*args, **kwargs)

def _generate_transition_groups(*args, **kwargs):
    """Construct transition enrichment groups with the declared direction rules."""
    from . import fsea as _fsea
    return _fsea._generate_transition_groups(*args, **kwargs)

def _generate_postural_groups(*args, **kwargs):
    """Group postural features by orientation for enrichment."""
    from . import fsea as _fsea
    return _fsea._generate_postural_groups(*args, **kwargs)

def _generate_spatial_groups(*args, **kwargs):
    """Construct radial-distance, relative-angle, and occupancy-fraction groups."""
    from . import fsea as _fsea
    return _fsea._generate_spatial_groups(*args, **kwargs)

def _generate_sleep_groups(*args, **kwargs):
    """Collect sleep features in the declared enrichment group."""
    from . import fsea as _fsea
    return _fsea._generate_sleep_groups(*args, **kwargs)

def _write_fsea_group_documentation(*args, **kwargs):
    """Write enrichment membership and group-validation documentation."""
    from . import fsea as _fsea
    return _fsea._write_fsea_group_documentation(*args, **kwargs)


def _identify_ungrouped_features(*args, **kwargs):
    """Report features outside the theoretical enrichment groups."""
    from . import fsea as _fsea
    return _fsea._identify_ungrouped_features(*args, **kwargs)

def _compare_theoretical_vs_actual_groups(*args, **kwargs):
    """Report theoretical and validated group membership differences."""
    from . import fsea as _fsea
    return _fsea._compare_theoretical_vs_actual_groups(*args, **kwargs)

def _clean_fsea_results_df(*args, **kwargs):
    """Remove the native Name column and rename Term to Feature_Set."""
    from . import fsea as _fsea
    return _fsea._clean_fsea_results_df(*args, **kwargs)

def _print_validation_summary(*args, **kwargs):
    """Display the analysis-validation summary."""
    from . import reporting as _reporting
    return _reporting._print_validation_summary(*args, **kwargs)

def _save_validation_report(*args, **kwargs):
    """Write the detailed analysis-validation report."""
    from . import reporting as _reporting
    return _reporting._save_validation_report(*args, **kwargs)

def save_consolidated_summary_with_formatting(*args, **kwargs):
    """Write one formatted consolidated SHAP workbook."""
    from . import reporting as _reporting
    return _reporting.save_consolidated_summary_with_formatting(*args, **kwargs)


def save_consolidated_summary_sheets_with_formatting(*args, **kwargs):
    """Write formatted consolidated SHAP views in one workbook."""
    from . import reporting as _reporting
    return _reporting.save_consolidated_summary_sheets_with_formatting(*args, **kwargs)

def save_predictions_with_formatting(*args, **kwargs):
    """Write formatted weighted and optional unweighted prediction sheets."""
    from . import reporting as _reporting
    return _reporting.save_predictions_with_formatting(*args, **kwargs)

def create_formatted_predictions_for_graphpad(*args, **kwargs):
    """Write group-wise probability columns for GraphPad Prism."""
    from . import reporting as _reporting
    return _reporting.create_formatted_predictions_for_graphpad(*args, **kwargs)

def calculate_ensemble_metrics(*args, **kwargs):
    """Calculate in-sample and out-of-fold binary ensemble metrics."""
    from . import reporting as _reporting
    return _reporting.calculate_ensemble_metrics(*args, **kwargs)

def _generate_kfold_shap_summaries(*args, **kwargs):
    """Summarize one in-sample or out-of-fold SHAP view."""
    from . import summaries as _summaries
    return _summaries._generate_kfold_shap_summaries(*args, **kwargs)

def _combine_insample_outofsample_summaries(*args, **kwargs):
    """Combine matched in-sample and out-of-fold SHAP summaries."""
    from . import summaries as _summaries
    return _summaries._combine_insample_outofsample_summaries(*args, **kwargs)

def _aggregate_shap_from_files(*args, **kwargs):
    """Aggregate compatible legacy SHAP files using per-feature contribution counts."""
    from . import summaries as _summaries
    return _summaries._aggregate_shap_from_files(*args, **kwargs)

def _generate_male_female_comparison(*args, **kwargs):
    """Compare male and female SHAP summaries within groups."""
    from . import summaries as _summaries
    return _summaries._generate_male_female_comparison(*args, **kwargs)

