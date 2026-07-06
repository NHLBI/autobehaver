from pathlib import Path

import pandas as pd

from autobehaver.postinf.featureimp.pca_axis import run_pca_analysis


def test_pca_axis_analysis_writes_reference_outputs(tmp_path: Path):
    X = pd.DataFrame(
        {
            "f1": [2.0, 2.1, -2.0, -2.2, 0.0, 0.1],
            "f2": [0.0, 0.2, 0.0, -0.2, 2.0, 2.1],
            "f3": [0.0, 0.1, 0.0, -0.1, 1.0, 1.1],
        }
    )
    df = pd.DataFrame(
        {
            "individual": [f"animal_{i}" for i in range(len(X))],
            "group": ["A", "A", "B", "B", "C", "C"],
            "sex": ["F", "M", "F", "M", "F", "M"],
        }
    )

    result = run_pca_analysis(
        X_train=X.iloc[:4].copy(),
        X_all_groups=X,
        df_train=df.iloc[:4].copy(),
        df_all_groups=df,
        all_feature_names=list(X.columns),
        paths={"analysis_folder": str(tmp_path)},
        GLOBAL_RANDOM_SEED=7,
        pca_fixed_k=2,
        training_group_a="A",
        training_group_b="B",
    )

    assert result["summary"] == "PCA reference-axis analysis completed."
    out_dir = tmp_path / "PCA"
    assert (out_dir / "PCA_EXPLAINED_VARIANCE.csv").exists()
    assert (out_dir / "PCA_PROJECTED_COORDINATES.csv").exists()
    assert (out_dir / "PCA_AXIS_SCORES.csv").exists()
    assert (out_dir / "PCA_AXIS_GROUP_SUMMARY.csv").exists()
    assert (out_dir / "PCA_AXIS_REFERENCE.json").exists()
    assert (out_dir / "MANIFEST_PCA.json").exists()

    scores = pd.read_csv(out_dir / "PCA_AXIS_SCORES.csv")
    assert {"axis_t", "axis_r", "axis_r_norm", "axis_training_sep", "axis_pca_k"}.issubset(scores.columns)
    assert set(scores["group"]) == {"A", "B", "C"}
    assert scores["axis_training_sep"].iloc[0] > 0
