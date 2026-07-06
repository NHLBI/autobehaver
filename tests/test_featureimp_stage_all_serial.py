import json
import warnings
from pathlib import Path

import pandas as pd

from tests.test_featureimp_plan_stage import _write_small_master_matrix


def test_featureimp_stage_all_runs_serial_staged_wrapper(tmp_path: Path, monkeypatch):
    from autobehaver.postinf.featureimp import core
    import autobehaver.postinf.featureimp.prereqs as prereqs

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_all_serial"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "all_serial",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 4,
        "learning_rate": 0.2,
        "max_depth": 2,
        "n_splits": 2,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": False,
        "run_fsea_analysis": False,
        "run_pca_analysis": False,
        "num_shards": 3,
    }
    monkeypatch.setattr(prereqs, "check_shap_xgboost_treeexplainer_compatibility", lambda: None)

    core.apply_settings(settings, propagate=False)
    core.apply_warning_filters()

    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_matrix_path is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = core.run_feature_importance_analysis(
            normalized_matrix_path=normalized_matrix_path,
            results_dir=str(results_dir),
            analysis_run_name="all_serial",
            include_feature_categories=core.include_feature_categories,
            analysis_folder_path=str(analysis_dir),
            control_group_tag="A",
            training_group_a="A",
            training_group_b="B",
            n_splits=2,
            H_hits_per_feature=1,
            n_features_per_model=2,
            run_fsea_analysis=False,
            run_pca_analysis=False,
        )

    assert out == str(analysis_dir)
    assert any("forcing num_shards=1" in str(w.message) for w in caught)

    manifest = json.loads((analysis_dir / "_featureimp_plan" / "PLAN_MANIFEST.json").read_text())
    assert manifest["requested_num_shards"] == 1
    assert manifest["active_num_shards"] == 1

    shard_done_files = list((analysis_dir / "_featureimp_shards").glob("*/SHARD_DONE.json"))
    assert len(shard_done_files) == 1
    assert (analysis_dir / "_featureimp_merge" / "WEIGHTS_DONE.json").exists()
    assert (analysis_dir / "_featureimp_merge" / "FINAL_MERGE_DONE.json").exists()
    assert (analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx").exists()
    assert (analysis_dir / "GLOBAL_IMPORTANCE_RANKING.csv").exists()

    weighted_df = pd.read_excel(
        analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx",
        sheet_name="WeightedPredictions",
    )
    assert {"individual", "group"}.issubset(set(weighted_df.columns))


def test_featureimp_pipeline_module_removed() -> None:
    import importlib.util

    spec = importlib.util.find_spec("autobehaver.postinf.featureimp.pipeline")
    assert spec is None


def test_featureimp_stage_all_accepts_programmatic_runtime_flags(tmp_path: Path, monkeypatch):
    from autobehaver.postinf.featureimp import core

    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_programmatic_flags"
    results_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    called = {"plan": None, "process": None, "merge_shards": None}

    def _fake_plan(**kwargs):
        called["plan"] = kwargs
        return str(analysis_dir)

    def _fake_process(**kwargs):
        called["process"] = kwargs

    def _fake_merge_shards(**kwargs):
        called["merge_shards"] = kwargs

    monkeypatch.setattr(core, "plan_feature_importance_analysis", _fake_plan)
    monkeypatch.setattr(core, "process_featureimp_shard", _fake_process)
    monkeypatch.setattr(core, "merge_featureimp_shards", _fake_merge_shards)

    out = core.run_feature_importance_analysis(
        normalized_matrix_path=str(tmp_path / "dummy_normalized.csv"),
        results_dir=str(results_dir),
        analysis_run_name="programmatic_flags",
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
        run_fsea_analysis=False,
        run_pca_analysis=False,
    )

    assert out == str(analysis_dir)
    assert called["plan"] is not None
    assert called["process"] == {"analysis_folder_path": str(analysis_dir), "shard_index": 0}
    assert called["merge_shards"] == {"analysis_folder_path": str(analysis_dir)}

    settings = json.loads((analysis_dir / "FEATUREIMP_RUN_SETTINGS.json").read_text())
    assert settings["analysis_run_name"] == "programmatic_flags"
    assert settings["control_group_tag"] == "A"
    assert settings["run_fsea_analysis"] is False
    assert settings["run_pca_analysis"] is False
    assert settings["__settings_scope__"] == "analysis_folder_defining"
