from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings


def _make_small_matrix(tmp_path):
    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_cv_guard"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in ["A", "B"]:
        for sex in ["M", "F"]:
            for rep in range(2):
                rows.append(
                    {
                        "individual": f"{group}_{sex}_{rep}",
                        "group": group,
                        "sex": sex,
                        "BoutSummary_feat1": float(len(rows) + 1),
                        "StdMeasure_feat2": float((rep + 1) * (1 if group == "A" else 2)),
                        "Kinematic_feat3": float((rep + 2) * (1 if sex == "M" else 3)),
                    }
                )

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    return matrix_dir, results_dir, analysis_dir, matrix_path


def test_featureimp_training_fails_fast_when_sex_group_stratum_too_small(tmp_path):
    from autobehaver.postinf.featureimp import core

    matrix_dir, results_dir, analysis_dir, matrix_path = _make_small_matrix(tmp_path)

    settings = {
        "analysis_run_name": "cv_guard",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 5,
        "n_splits": 3,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": False,
        "run_fsea_analysis": False,
        "sex_filter": "all",
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

    with pytest.raises(ValueError, match=r"FeatureImp training cross-validation: n_splits=3") as exc:
        core.run_feature_importance_analysis(
            normalized_matrix_path=normalized_matrix_path,
            results_dir=str(results_dir),
            analysis_run_name="cv_guard",
            include_feature_categories=core.include_feature_categories,
            analysis_folder_path=str(analysis_dir),
            control_group_tag="A",
            training_group_a="A",
            training_group_b="B",
            n_splits=3,
            H_hits_per_feature=1,
            n_features_per_model=2,
            sex_filter="all",
        )

    msg = str(exc.value)
    assert "M_A=2" in msg
    assert "F_A=2" in msg
    assert "M_B=2" in msg
    assert "F_B=2" in msg


class _DummyBinaryEncoder:
    classes_ = np.array(["A", "B"], dtype=object)


def test_male_female_comparison_uses_training_mean_abs_shap_and_training_sort_order(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp import summaries

    df = pd.DataFrame(
        {
            "individual": [f"fly{i}" for i in range(10)],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "sex": ["M", "M", "F", "F", "M", "M", "F", "F", "M", "F"],
            "f1": np.linspace(0.0, 1.0, 10),
            "f2": np.linspace(1.0, 0.0, 10),
        }
    )
    selected_features = ["f1", "f2"]

    shap_values = np.array(
        [
            [10.0, 0.0],   # A male
            [-10.0, 0.0],  # A male
            [0.0, 1.0],    # A female
            [0.0, -1.0],   # A female
            [1.0, 0.0],    # B male
            [1.0, 0.0],    # B male
            [0.0, 5.0],    # B female
            [0.0, 5.0],    # B female
            [0.0, 100.0],  # C male (non-training; should not control ordering)
            [0.0, 100.0],  # C female (non-training; should not control ordering)
        ],
        dtype=np.float32,
    )

    shap_path = tmp_path / "avg_shap.bin"
    mm = np.memmap(shap_path, dtype=np.float32, mode="w+", shape=shap_values.shape)
    mm[:] = shap_values[:]
    mm.flush()
    del mm

    captured = {}

    def _capture(male_df, female_df, paths):
        captured["male_df"] = male_df.copy()
        captured["female_df"] = female_df.copy()
        captured["paths"] = dict(paths)

    monkeypatch.setattr(summaries, "_combine_male_female_summaries", _capture)

    summaries._generate_male_female_comparison(
        df=df,
        selected_features=selected_features,
        all_shap_file_paths=[str(shap_path)],
        le=_DummyBinaryEncoder(),
        paths={"analysis_folder": str(tmp_path)},
    )

    male_df = captured["male_df"]
    female_df = captured["female_df"]

    assert list(male_df["feature_name"]) == ["f1", "f2"]
    assert list(female_df["feature_name"]) == ["f1", "f2"]

    male_global = male_df.set_index("feature_name")["Global_Importance"]
    female_global = female_df.set_index("feature_name")["Global_Importance"]

    np.testing.assert_allclose(male_global["f1"], 5.5, rtol=0, atol=1e-6)
    np.testing.assert_allclose(male_global["f2"], 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(female_global["f1"], 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(female_global["f2"], 3.0, rtol=0, atol=1e-6)


def test_male_female_comparison_zero_shap_normalizes_to_zero_and_singletons_do_not_warn(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp import summaries

    df = pd.DataFrame(
        {
            "individual": ["a_m", "a_f", "b_m", "b_f"],
            "group": ["A", "A", "B", "B"],
            "sex": ["M", "F", "M", "F"],
            "f1": [0.0, 0.0, 0.0, 0.0],
            "f2": [0.0, 0.0, 0.0, 0.0],
        }
    )
    selected_features = ["f1", "f2"]
    shap_values = np.zeros((len(df), len(selected_features)), dtype=np.float32)

    shap_path = tmp_path / "avg_shap_zero.bin"
    mm = np.memmap(shap_path, dtype=np.float32, mode="w+", shape=shap_values.shape)
    mm[:] = shap_values[:]
    mm.flush()
    del mm

    captured = {}

    def _capture(male_df, female_df, paths):
        captured["male_df"] = male_df.copy()
        captured["female_df"] = female_df.copy()

    monkeypatch.setattr(summaries, "_combine_male_female_summaries", _capture)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summaries._generate_male_female_comparison(
            df=df,
            selected_features=selected_features,
            all_shap_file_paths=[str(shap_path)],
            le=_DummyBinaryEncoder(),
            paths={"analysis_folder": str(tmp_path)},
        )

    assert caught == []

    male_df = captured["male_df"]
    female_df = captured["female_df"]

    for out_df in (male_df, female_df):
        assert np.allclose(out_df["Global_Importance"].to_numpy(), 0.0)
        assert np.allclose(out_df["Global_Importance_Normalized"].to_numpy(), 0.0)
        for col in out_df.columns:
            if col.startswith("MeanSHAP_Normalized_") or col.startswith("MeanAbsSHAP_Normalized_"):
                assert np.allclose(out_df[col].to_numpy(), 0.0)
            if col.startswith("StdSHAP_"):
                assert out_df[col].isna().all()
