import json
import shutil
from pathlib import Path

import pandas as pd

from tests.test_featureimp_plan_stage import _write_small_master_matrix
from autobehaver.postinf.featureimp.baseline_outputs import write_baseline_outputs_summary


def _prepare_small_finalized_analysis(tmp_path: Path) -> Path:
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_baseline_reruns"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "baseline_reruns",
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
        "n_permutations": 0,
        "run_fsea_analysis": False,
        "run_pca_analysis": False,
        "num_shards": 2,
    }
    core.apply_settings(settings, propagate=False)
    core.apply_warning_filters()

    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_matrix_path is not None

    core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name="baseline_reruns",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
        sex_filter="all",
        num_shards=2,
    )

    for shard_index in range(2):
        core.process_featureimp_shard(
            analysis_folder_path=str(analysis_dir),
            shard_index=shard_index,
        )

    out_merge = core.merge_featureimp_shards(analysis_folder_path=str(analysis_dir))
    assert out_merge == str(analysis_dir)
    return analysis_dir


def test_featureimp_baseline_standalone_reruns_refresh_outputs_and_index(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import plain_xgboost_baseline

    analysis_dir = _prepare_small_finalized_analysis(tmp_path)

    plain_dir = analysis_dir / "PLAIN_XGBOOST_BASELINE"
    summary_note = analysis_dir / "FEATUREIMP_BASELINE_OUTPUTS.txt"

    shutil.rmtree(plain_dir, ignore_errors=True)
    if summary_note.exists():
        summary_note.unlink()

    assert plain_xgboost_baseline.main(["--analysis_folder_path", str(analysis_dir)]) == 0

    assert (plain_dir / "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv").exists()
    assert (plain_dir / "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv").exists()
    assert (plain_dir / "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv").exists()
    assert (plain_dir / "PLAIN_XGBOOST_CONFIG.json").exists()
    assert (plain_dir / "AVERAGED_PREDICTIONS_FULL.xlsx").exists()
    assert (plain_dir / "AVERAGED_PREDICTIONS_FORMATTED_FOR_GRAPHPAD.xlsx").exists()
    assert (plain_dir / "SHAP_CONSOLIDATED_SUMMARY.xlsx").exists()
    assert (plain_dir / "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx").exists()
    assert summary_note.exists()

    summary_text = summary_note.read_text()
    assert "Univariate baseline" not in summary_text
    assert "PLAIN_XGBOOST_BASELINE [present]" in summary_text
    assert "PLAIN_XGBOOST_BASELINE/PLAIN_XGBOOST_POOLED_PERFORMANCE.csv [present]" in summary_text


    plain_fsea_dir = plain_dir / "FSEA"
    plain_fsea_dir.mkdir(parents=True, exist_ok=True)
    (plain_fsea_dir / "DUMMY_PLAIN_FSEA.txt").write_text("ok", encoding="utf-8")
    write_baseline_outputs_summary(str(analysis_dir))
    summary_text = summary_note.read_text()
    assert "PLAIN_XGBOOST_BASELINE/FSEA [present]" in summary_text
    assert "PLAIN_XGBOOST_BASELINE/FSEA/DUMMY_PLAIN_FSEA.txt [present]" in summary_text

    post_idx = json.loads((analysis_dir / "POSTANALYSIS_INDEX.json").read_text())
    stage_map = {row["stage"]: row for row in post_idx["stages"]}
    assert stage_map["plain_xgboost_baseline"]["status"] == "completed"
    assert stage_map["plain_xgboost_baseline"]["context"]["invoked_by"] == "standalone_rerun"
    assert any(path.endswith("PLAIN_XGBOOST_POOLED_PERFORMANCE.csv") for path in stage_map["plain_xgboost_baseline"].get("artifact_paths", []))

    plain_df = pd.read_csv(plain_dir / "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv")
    assert len(plain_df) == 1
