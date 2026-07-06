import json
import shutil
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from tests.test_featureimp_plan_stage import _write_small_master_matrix


def _prepare_small_planned_analysis(tmp_path: Path, *, h_hits: int = 2, run_permutation_test: bool = False, n_permutations: int = 0, num_shards: int = 2) -> Path:
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_plain_xgb_staged"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "plain_xgb_staged",
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
        "H_hits_per_feature": int(h_hits),
        "n_features_per_model": 2,
        "run_permutation_test": bool(run_permutation_test),
        "n_permutations": int(n_permutations),
        "num_shards": int(num_shards),
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

    out_plan = core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name="plain_xgb_staged",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=int(h_hits),
        n_features_per_model=2,
        sex_filter="all",
        num_shards=int(num_shards),
    )
    assert out_plan == str(analysis_dir)
    return analysis_dir


def test_plain_xgboost_staged_merge_matches_serial_outputs(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import rerun_plain_xgboost_baseline_from_analysis_folder

    analysis_dir = _prepare_small_planned_analysis(tmp_path, h_hits=2, run_permutation_test=False, num_shards=2)
    baseline_settings = {
        "n_estimators": 4,
        "learning_rate": 0.2,
        "max_depth": 2,
        "model_subsample": 1.0,
        "run_permutation_test": False,
        "n_permutations": 0,
    }

    serial_out = Path(
        rerun_plain_xgboost_baseline_from_analysis_folder(
            analysis_folder_path=str(analysis_dir),
            settings_override=baseline_settings,
            run_fsea_analysis=False,
        )
    )
    serial_reference = analysis_dir / "_serial_plain_xgb_reference"
    shutil.move(str(serial_out), str(serial_reference))

    core.apply_settings({"num_shards": 2, "shard_index": 0, "num_workers": 1}, propagate=False, explicit_keys={"num_shards", "shard_index", "num_workers"})
    plan_dir = core.plan_plain_xgboost_baseline_shards(
        analysis_folder_path=str(analysis_dir),
        num_shards=2,
        settings_override=baseline_settings,
    )
    assert Path(plan_dir).exists()

    shard0 = core.process_plain_xgboost_baseline_shard(analysis_folder_path=str(analysis_dir), shard_index=0)
    shard1 = core.process_plain_xgboost_baseline_shard(analysis_folder_path=str(analysis_dir), shard_index=1)
    assert Path(shard0).exists()
    assert Path(shard1).exists()

    staged_out = Path(
        core.merge_plain_xgboost_baseline_shards(
            analysis_folder_path=str(analysis_dir),
            settings_override=baseline_settings,
            run_fsea_analysis=False,
        )
    )
    assert staged_out.exists()

    for filename in [
        "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv",
        "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv",
        "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv",
        "GLOBAL_IMPORTANCE_RANKING.csv",
    ]:
        serial_df = pd.read_csv(serial_reference / filename)
        staged_df = pd.read_csv(staged_out / filename)
        if "repeat_idx" in serial_df.columns:
            serial_df = serial_df.sort_values(by=[c for c in ["repeat_idx", "fold_idx", "feature_name"] if c in serial_df.columns]).reset_index(drop=True)
            staged_df = staged_df.sort_values(by=[c for c in ["repeat_idx", "fold_idx", "feature_name"] if c in staged_df.columns]).reset_index(drop=True)
        elif "feature_name" in serial_df.columns:
            serial_df = serial_df.sort_values(by=["feature_name"]).reset_index(drop=True)
            staged_df = staged_df.sort_values(by=["feature_name"]).reset_index(drop=True)
        pdt.assert_frame_equal(serial_df, staged_df, check_dtype=False, rtol=1e-7, atol=1e-9)

    post_idx = json.loads((analysis_dir / "POSTANALYSIS_INDEX.json").read_text())
    stage_map = {row["stage"]: row for row in post_idx["stages"]}
    assert stage_map["plain_xgboost_baseline"]["status"] == "completed"
    assert stage_map["plain_xgboost_baseline"]["context"]["invoked_by"] == "staged_merge"



def test_plain_xgboost_staged_merge_requires_all_shards(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import core

    analysis_dir = _prepare_small_planned_analysis(tmp_path, h_hits=2, run_permutation_test=False, num_shards=2)
    baseline_settings = {
        "n_estimators": 4,
        "learning_rate": 0.2,
        "max_depth": 2,
        "model_subsample": 1.0,
        "run_permutation_test": False,
        "n_permutations": 0,
    }

    core.plan_plain_xgboost_baseline_shards(
        analysis_folder_path=str(analysis_dir),
        num_shards=2,
        settings_override=baseline_settings,
    )
    core.process_plain_xgboost_baseline_shard(analysis_folder_path=str(analysis_dir), shard_index=0)

    with pytest.raises(RuntimeError, match="expected shard completion marker"):
        core.merge_plain_xgboost_baseline_shards(
            analysis_folder_path=str(analysis_dir),
            settings_override=baseline_settings,
            run_fsea_analysis=False,
        )



def test_featureimp_cli_routes_plain_xgboost_staged_stages(monkeypatch, tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import cli
    import autobehaver.postinf.featureimp.core as core

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seen = []

    monkeypatch.setattr(core, "plan_plain_xgboost_baseline_shards", lambda **kwargs: seen.append(("plan", kwargs)) or str(analysis_dir / "PLAIN_XGBOOST_BASELINE" / "_STAGED_PLAN"))
    monkeypatch.setattr(core, "process_plain_xgboost_baseline_shard", lambda **kwargs: seen.append(("process", kwargs)) or str(analysis_dir / "PLAIN_XGBOOST_BASELINE" / "_STAGED_SHARDS" / "shard_000_of_002"))
    monkeypatch.setattr(core, "merge_plain_xgboost_baseline_shards", lambda **kwargs: seen.append(("merge", kwargs)) or str(analysis_dir / "PLAIN_XGBOOST_BASELINE"))

    assert cli.main(["--stage", "plain_xgboost_baseline_plan", "--analysis_folder_path", str(analysis_dir), "--num_shards", "2"]) is None
    assert cli.main(["--stage", "plain_xgboost_baseline_process_shard", "--analysis_folder_path", str(analysis_dir), "--shard_index", "1"]) is None
    assert cli.main(["--stage", "plain_xgboost_baseline_merge_shards", "--analysis_folder_path", str(analysis_dir)]) is None

    assert seen[0][0] == "plan"
    assert seen[0][1]["analysis_folder_path"] == str(analysis_dir)
    assert seen[0][1]["num_shards"] == 2
    assert seen[1][0] == "process"
    assert seen[1][1]["shard_index"] == 1
    assert seen[2][0] == "merge"
    assert seen[2][1]["analysis_folder_path"] == str(analysis_dir)


def test_plain_xgboost_staged_permutation_smoke(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import core

    analysis_dir = _prepare_small_planned_analysis(tmp_path, h_hits=1, run_permutation_test=True, n_permutations=1, num_shards=2)
    baseline_settings = {
        "n_estimators": 4,
        "learning_rate": 0.2,
        "max_depth": 2,
        "model_subsample": 1.0,
        "run_permutation_test": True,
        "n_permutations": 1,
    }

    core.plan_plain_xgboost_baseline_shards(
        analysis_folder_path=str(analysis_dir),
        num_shards=2,
        settings_override=baseline_settings,
    )
    core.process_plain_xgboost_baseline_shard(analysis_folder_path=str(analysis_dir), shard_index=0)
    core.process_plain_xgboost_baseline_shard(analysis_folder_path=str(analysis_dir), shard_index=1)
    out_dir = Path(
        core.merge_plain_xgboost_baseline_shards(
            analysis_folder_path=str(analysis_dir),
            settings_override=baseline_settings,
            run_fsea_analysis=False,
        )
    )

    assert (out_dir / "PERMUTED_0_RESULTS" / "SHAP_CONSOLIDATED_SUMMARY_PERMUTED_0.xlsx").exists()

