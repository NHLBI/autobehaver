from pathlib import Path
import json

import numpy as np
import pandas as pd

def _make_matrix(tmp_path: Path) -> tuple[Path, Path, Path]:
    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_male_manual"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        group = "A" if i < 20 else "B"
        sex = "M" if i % 2 == 0 else "F"
        rows.append(
            {
                "individual": f"fly{i}",
                "group": group,
                "sex": sex,
                "BoutSummary_feat1": float(i + rng.normal(0, 0.1)),
                "StdMeasure_feat2": float((i % 3) + rng.normal(0, 0.1)),
                "Kinematic_feat3": float((i // 2) + rng.normal(0, 0.1)),
                "Spatial_feat4": float((12 - i) + rng.normal(0, 0.1)),
            }
        )
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    return matrix_dir, results_dir, analysis_dir



def test_featureimp_sex_filter_male_integration(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir, results_dir, analysis_dir = _make_matrix(tmp_path)
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"

    settings = {
        "analysis_run_name": "male_manual",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 12,
        "learning_rate": 0.3,
        "max_depth": 2,
        "n_splits": 2,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": False,
        "run_fsea_analysis": False,
        "sex_filter": "male",
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
        analysis_run_name="male_manual",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
        sex_filter="male",
    )

    assert out == str(analysis_dir)
    manifest = json.loads((analysis_dir / "FEATUREIMP_REFERENCE_MANIFEST.json").read_text())
    assert manifest["training_settings"]["sex_filter"] == "male"

    preds = pd.read_excel(analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx", sheet_name=0)
    observed_individuals = {str(x) for x in preds["individual"].dropna() if str(x).startswith("fly")}
    assert observed_individuals <= {f"fly{i}" for i in range(0, 40, 2)}
