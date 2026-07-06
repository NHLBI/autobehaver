"""autobehaver.postinf.featureimp.compat

Delegation layer for the Feature Importance (featureimp) pipeline.

This module contains thin wrappers that route stable helper entry points to the
modular implementations in prep/committee/PCA/FSEA/reporting/summaries/
permutation/matrix. The wrappers are intentionally lightweight and should not
contain business logic.
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
    """Delegation wrapper (implementation moved to `featureimp.summaries`)."""
    from . import summaries as _summaries
    return _summaries._clean_performance_dict_for_json(*args, **kwargs)

def validate_feature_matrix(matrix_path, required_id_cols=['individual', 'group', 'sex']):
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.matrix.validate_feature_matrix`."""
    return _validate_feature_matrix(matrix_path, required_id_cols=required_id_cols)

def normalize_feature_matrix(
    results_dir,
    raw_matrix_path=None,
    skip_if_normalized_exists=False,
    analysis_folder_path=None,
):
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.matrix.normalize_feature_matrix`."""
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
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.prep.get_analysis_folder_name`."""
    from . import prep as _prep
    return _prep.get_analysis_folder_name(analysis_run_name=analysis_run_name, sex_stratum=sex_stratum)

def filter_data_by_sex(df, sex_value):
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.prep.filter_data_by_sex`."""
    from . import prep as _prep
    return _prep.filter_data_by_sex(df=df, sex_value=sex_value)

def _parse_feature_name(feature_name):
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.prep._parse_feature_name`."""
    from . import prep as _prep
    return _prep._parse_feature_name(feature_name)

def _filter_sparse_features_by_group(df, training_group_a, training_group_b, 
                                     min_completeness, feature_cols):
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.prep._filter_sparse_features_by_group`."""
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
    """Delegation wrapper for :func:`autobehaver.postinf.featureimp.prep._filter_low_variance_features`."""
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
    from . import committee as _committee
    return _committee._init_pool_worker(*args, **kwargs)

def _train_subset_fold_worker(*args, **kwargs):
    from . import committee as _committee
    return _committee._train_subset_fold_worker(*args, **kwargs)


def _generate_sex_class_stratified_permutation(
    y_original,
    df_with_sex,
    train_indices,
    test_indices,
    permutation_seed=None,
):
    """Wrapper for permutation helper (moved to featureimp.permutation)."""
    
    from . import permutation as _permutation
    return _permutation._generate_sex_class_stratified_permutation(
        y_original,
        df_with_sex,
        train_indices,
        test_indices,
        permutation_seed,
    )

def _flip_probabilities_for_permuted_samples(probas, y_original_full, y_permuted_full, train_indices):
    """Wrapper for probability flipping helper (moved to featureimp.permutation)."""
    
    from . import permutation as _permutation
    return _permutation._flip_probabilities_for_permuted_samples(
        probas,
        y_original_full,
        y_permuted_full,
        train_indices,
    )


def _sanitize_group_name_for_filename(*args, **kwargs):
    """Delegation wrapper (implementation moved to `featureimp.summaries`)."""
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
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_bout_timebudget_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_bout_timebudget_groups(*args, **kwargs)

def _generate_kinematic_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_kinematic_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_kinematic_groups(*args, **kwargs)

def _generate_transition_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_transition_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_transition_groups(*args, **kwargs)

def _generate_postural_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_postural_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_postural_groups(*args, **kwargs)

def _generate_spatial_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_spatial_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_spatial_groups(*args, **kwargs)

def _generate_sleep_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._generate_sleep_groups`."""
    from . import fsea as _fsea
    return _fsea._generate_sleep_groups(*args, **kwargs)

def _write_fsea_group_documentation(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._write_fsea_group_documentation`."""
    from . import fsea as _fsea
    return _fsea._write_fsea_group_documentation(*args, **kwargs)


def _identify_ungrouped_features(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._identify_ungrouped_features`."""
    from . import fsea as _fsea
    return _fsea._identify_ungrouped_features(*args, **kwargs)

def _compare_theoretical_vs_actual_groups(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._compare_theoretical_vs_actual_groups`."""
    from . import fsea as _fsea
    return _fsea._compare_theoretical_vs_actual_groups(*args, **kwargs)

def _clean_fsea_results_df(*args, **kwargs):
    """Wrapper for :func:`autobehaver.postinf.featureimp.fsea._clean_fsea_results_df`."""
    from . import fsea as _fsea
    return _fsea._clean_fsea_results_df(*args, **kwargs)

def _print_validation_summary(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting._print_validation_summary(*args, **kwargs)

def _save_validation_report(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting._save_validation_report(*args, **kwargs)

def save_consolidated_summary_with_formatting(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting.save_consolidated_summary_with_formatting(*args, **kwargs)


def save_consolidated_summary_sheets_with_formatting(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting.save_consolidated_summary_sheets_with_formatting(*args, **kwargs)

def save_predictions_with_formatting(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting.save_predictions_with_formatting(*args, **kwargs)

def create_formatted_predictions_for_graphpad(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting.create_formatted_predictions_for_graphpad(*args, **kwargs)

def calculate_ensemble_metrics(*args, **kwargs):
    from . import reporting as _reporting
    return _reporting.calculate_ensemble_metrics(*args, **kwargs)

def _generate_kfold_shap_summaries(*args, **kwargs):
    """Delegation wrapper (implementation in autobehaver.postinf.featureimp.summaries)."""
    from . import summaries as _summaries
    return _summaries._generate_kfold_shap_summaries(*args, **kwargs)

def _combine_insample_outofsample_summaries(*args, **kwargs):
    """Delegation wrapper (implementation in autobehaver.postinf.featureimp.summaries)."""
    from . import summaries as _summaries
    return _summaries._combine_insample_outofsample_summaries(*args, **kwargs)

def _aggregate_shap_from_files(*args, **kwargs):
    """Delegation wrapper (implementation in autobehaver.postinf.featureimp.summaries)."""
    from . import summaries as _summaries
    return _summaries._aggregate_shap_from_files(*args, **kwargs)

def _generate_male_female_comparison(*args, **kwargs):
    """Delegation wrapper (implementation in autobehaver.postinf.featureimp.summaries)."""
    from . import summaries as _summaries
    return _summaries._generate_male_female_comparison(*args, **kwargs)

