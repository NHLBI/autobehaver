import json
from pathlib import Path

import pandas as pd


def _build_reference_run_for_reference_tests(tmp_path: Path) -> Path:
    from autobehaver.postinf.featureimp.reference import write_featureimp_reference_bundle

    analysis_dir = tmp_path / 'results' / 'FEATURE_ANALYSIS_ref_manual'
    models_dir = analysis_dir / 'SAVED_PIPELINE' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    normalized_df = pd.DataFrame([
        {'individual': 'fly0', 'group': 'A', 'sex': 'M', 'BoutSummary_feat1': -1.0, 'StdMeasure_feat2': 0.5},
        {'individual': 'fly1', 'group': 'A', 'sex': 'F', 'BoutSummary_feat1': 0.0, 'StdMeasure_feat2': -0.5},
        {'individual': 'fly2', 'group': 'B', 'sex': 'M', 'BoutSummary_feat1': 1.0, 'StdMeasure_feat2': 1.5},
        {'individual': 'fly3', 'group': 'B', 'sex': 'F', 'BoutSummary_feat1': 2.0, 'StdMeasure_feat2': -1.5},
    ])
    normalized_path = analysis_dir / 'NORMALIZED_FEATURE_MATRIX.csv'
    normalized_df.to_csv(normalized_path, index=False)

    norm_params = {
        'means': {'BoutSummary_feat1': 0.0, 'StdMeasure_feat2': 0.0},
        'stds': {'BoutSummary_feat1': 1.0, 'StdMeasure_feat2': 1.0},
        'id_columns': ['individual', 'group', 'sex'],
        'numeric_feature_columns': ['BoutSummary_feat1', 'StdMeasure_feat2'],
        'scalar_feature_columns': ['BoutSummary_feat1', 'StdMeasure_feat2'],
        'compositional_feature_columns': [],
        'compositional_sets': {},
        'normalization_method': 'clr_then_zscore_v2_transition_filtering',
    }
    (analysis_dir / 'NORMALIZATION_PARAMS.json').write_text(json.dumps(norm_params, indent=2))

    model_path = models_dir / 'model_h1_s0_f0.json'
    model_path.write_text('{"dummy": true}\n')

    write_featureimp_reference_bundle(
        analysis_folder=str(analysis_dir),
        normalized_matrix_path=str(normalized_path),
        selected_feature_columns=['BoutSummary_feat1', 'StdMeasure_feat2'],
        id_columns=['individual', 'group', 'sex'],
        control_group_tag='A',
        training_group_a='A',
        training_group_b='B',
        settings={'n_splits': 2, 'H_hits_per_feature': 1},
        performance_records=[{
            'h_hit': 1,
            'subset_idx': 0,
            'fold_idx': 0,
            'is_permuted': False,
            'permutation_idx': None,
            'n_features': 2,
            'accuracy_test': 1.0,
            'accuracy_train': 1.0,
            'model_path': str(model_path),
        }],
    )

    manifest_path = analysis_dir / 'FEATUREIMP_REFERENCE_MANIFEST.json'
    ref_dir = analysis_dir / 'FEATUREIMP_REFERENCE'
    assert manifest_path.exists()
    assert (ref_dir / 'FROZEN_FEATURE_COLUMNS.json').exists()
    assert (ref_dir / 'NORMALIZATION_FEATURE_COLUMNS.json').exists()
    assert (ref_dir / 'COMPOSITIONAL_SETS.json').exists()
    assert (ref_dir / 'MODEL_INVENTORY.csv').exists()

    inventory = pd.read_csv(ref_dir / 'MODEL_INVENTORY.csv')
    assert 'model_type' not in inventory.columns

    manifest = json.loads(manifest_path.read_text())
    assert manifest['reference_contract']['training_group_a'] == 'A'
    assert manifest['reference_contract']['training_group_b'] == 'B'
    assert manifest['models']['real_model_count'] == 1
    assert manifest['artifacts']['selected_feature_columns_json'] == 'FEATUREIMP_REFERENCE/FROZEN_FEATURE_COLUMNS.json'

    return analysis_dir


def test_featureimp_writes_reference_bundle(tmp_path: Path):
    analysis_dir = _build_reference_run_for_reference_tests(tmp_path)

    manifest_path = analysis_dir / 'FEATUREIMP_REFERENCE_MANIFEST.json'
    ref_dir = analysis_dir / 'FEATUREIMP_REFERENCE'
    assert manifest_path.exists()
    assert (ref_dir / 'FROZEN_FEATURE_COLUMNS.json').exists()
    assert (ref_dir / 'NORMALIZATION_FEATURE_COLUMNS.json').exists()
    assert (ref_dir / 'COMPOSITIONAL_SETS.json').exists()
    assert (ref_dir / 'MODEL_INVENTORY.csv').exists()


def test_write_reference_bundle_falls_back_to_parent_normalization_params(tmp_path: Path):
    from autobehaver.postinf.featureimp.reference import write_featureimp_reference_bundle

    parent_dir = tmp_path / 'parent_analysis'
    child_dir = parent_dir / 'FEATURE_ANALYSIS_MALE_ref'
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir.mkdir(parents=True, exist_ok=True)

    norm_params = {
        'means': {'BoutSummary_feat1': 0.0},
        'stds': {'BoutSummary_feat1': 1.0},
        'id_columns': ['individual', 'group', 'sex'],
        'numeric_feature_columns': ['BoutSummary_feat1'],
        'scalar_feature_columns': ['BoutSummary_feat1'],
        'compositional_feature_columns': [],
        'compositional_sets': {},
        'normalization_method': 'clr_then_zscore_v1',
    }
    (parent_dir / 'NORMALIZATION_PARAMS.json').write_text(json.dumps(norm_params, indent=2))

    normalized_df = pd.DataFrame([
        {'individual': 'fly1', 'group': 'A', 'sex': 'M', 'BoutSummary_feat1': 0.25},
        {'individual': 'fly2', 'group': 'B', 'sex': 'M', 'BoutSummary_feat1': -0.75},
    ])
    normalized_path = child_dir / 'NORMALIZED_FEATURE_MATRIX.csv'
    normalized_df.to_csv(normalized_path, index=False)

    manifest_path = write_featureimp_reference_bundle(
        analysis_folder=str(child_dir),
        normalized_matrix_path=str(normalized_path),
        selected_feature_columns=['BoutSummary_feat1'],
        id_columns=['individual', 'group', 'sex'],
        control_group_tag='A',
        training_group_a='A',
        training_group_b='B',
        settings={'n_splits': 2},
        performance_records=[],
    )
    manifest = json.loads(Path(manifest_path).read_text())
    assert Path(manifest_path).exists()
    assert manifest['artifacts']['normalization_params_json'] == '../NORMALIZATION_PARAMS.json'


