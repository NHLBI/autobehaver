import json
from pathlib import Path

import pandas as pd


def _write_legacy_master_matrix(matrix_path: Path) -> None:
    rows = []
    for i in range(12):
        group = 'A' if i < 6 else 'B'
        sex = 'M' if i % 2 == 0 else 'F'
        rows.append(
            {
                'individual': f'fly{i}',
                'group': group,
                'sex': sex,
                'BoutSummary_feat1': float(i + 1),
                'KinematicBB_Pose_Walking_inst_speed_mean_mean': float(i + 0.25),
                'LegacyFeature_speed_mean': float((i // 2) + 0.5),
                'Spatial_feat4': float(20 - i),
            }
        )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)


def _write_default_filter_guard_matrix(matrix_path: Path) -> None:
    rows = []
    for i in range(12):
        group = 'A' if i < 6 else 'B'
        sex = 'M' if i % 2 == 0 else 'F'
        rows.append(
            {
                'individual': f'fly{i}',
                'group': group,
                'sex': sex,
                'BoutSummary_feat1': float(i + 1),
                'LegacyFeature_speed_mean': float((i // 2) + 0.5),
                'Global_MeanSpeed_mm_s': float(2 * i + 0.75),
                'extra_meta_numeric': float(100 + i),
            }
        )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)



def test_featureimp_default_category_filter_excludes_legacy_family(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / 'matrix'
    results_dir = tmp_path / 'results'
    analysis_dir = results_dir / 'FEATURE_ANALYSIS_category_default'
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / 'MASTER_FEATURE_MATRIX.csv'
    _write_legacy_master_matrix(matrix_path)

    settings = {
        'analysis_run_name': 'category_default',
        'matrix_dir': str(matrix_dir),
        'results_dir': str(results_dir),
        'matrix_path': str(matrix_path),
        'control_group_tag': 'A',
        'training_group_a': 'A',
        'training_group_b': 'B',
        'num_workers': 1,
        'n_estimators': 5,
        'n_splits': 2,
        'H_hits_per_feature': 2,
        'n_features_per_model': 2,
        'run_permutation_test': False,
        'run_fsea_analysis': False,
    }
    core.apply_settings(settings, propagate=False)
    core.apply_warning_filters()

    assert core.include_feature_categories is None

    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_matrix_path is not None

    out = core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name='category_default',
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag='A',
        training_group_a='A',
        training_group_b='B',
        n_splits=2,
        H_hits_per_feature=2,
        n_features_per_model=2,
        sex_filter='all',
        num_shards=2,
    )
    assert out == str(analysis_dir)

    selected_features = json.loads((analysis_dir / '_featureimp_plan' / 'SELECTED_FEATURE_COLUMNS.json').read_text())
    assert 'LegacyFeature_speed_mean' not in selected_features



def test_featureimp_default_category_filter_excludes_unknown_numeric_metadata_and_includes_global(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / 'matrix'
    results_dir = tmp_path / 'results'
    analysis_dir = results_dir / 'FEATURE_ANALYSIS_default_filter_guard'
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / 'MASTER_FEATURE_MATRIX.csv'
    _write_default_filter_guard_matrix(matrix_path)

    settings = {
        'analysis_run_name': 'default_filter_guard',
        'matrix_dir': str(matrix_dir),
        'results_dir': str(results_dir),
        'matrix_path': str(matrix_path),
        'control_group_tag': 'A',
        'training_group_a': 'A',
        'training_group_b': 'B',
        'num_workers': 1,
        'n_estimators': 5,
        'n_splits': 2,
        'H_hits_per_feature': 2,
        'n_features_per_model': 2,
        'run_permutation_test': False,
        'run_fsea_analysis': False,
    }
    core.apply_settings(settings, propagate=False)
    core.apply_warning_filters()

    assert core.include_feature_categories is None

    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_matrix_path is not None

    out = core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name='default_filter_guard',
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag='A',
        training_group_a='A',
        training_group_b='B',
        n_splits=2,
        H_hits_per_feature=2,
        n_features_per_model=2,
        sex_filter='all',
        num_shards=2,
    )
    assert out == str(analysis_dir)

    selected_features = json.loads((analysis_dir / '_featureimp_plan' / 'SELECTED_FEATURE_COLUMNS.json').read_text())
    assert 'Global_MeanSpeed_mm_s' in selected_features
    assert 'extra_meta_numeric' not in selected_features




