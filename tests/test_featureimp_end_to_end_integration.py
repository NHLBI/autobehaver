from pathlib import Path

import numpy as np
import pandas as pd


def test_featureimp_end_to_end_integration(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_integration_manual"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    # Each training fold needs enough observations to fit supported tree splits.
    # The previous five-row folds could only produce constant classifiers with
    # the default child-weight threshold, so the weighting guard correctly failed.
    for i in range(40):
        group = "A" if i < 20 else "B"
        group_signal = 0. if group == "A" else 10.
        sex = "M" if i % 2 == 0 else "F"
        rows.append(
            {
                "individual": f"fly{i}",
                "group": group,
                "sex": sex,
                "BoutSummary_feat1": float(group_signal + rng.normal(0, 0.1)),
                "StdMeasure_feat2": float(group_signal + rng.normal(0, 0.1)),
                "Kinematic_feat3": float(group_signal + rng.normal(0, 0.1)),
                "Spatial_feat4": float(-group_signal + rng.normal(0, 0.1)),
            }
        )
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)

    settings = {
        "analysis_run_name": "integration_manual",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 25,
        "n_splits": 2,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": False,
        "run_fsea_analysis": False,
    }
    core.apply_settings(settings)
    core.apply_warning_filters()

    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_matrix_path is not None

    out = core.run_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name="integration_manual",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
    )

    assert out == str(analysis_dir)
    assert (analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx").exists()
    assert (analysis_dir / "GLOBAL_IMPORTANCE_RANKING.csv").exists()
