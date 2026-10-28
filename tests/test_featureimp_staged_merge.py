import json
from pathlib import Path

import pandas as pd

from autobehaver.postinf.featureimp.shap_tasks import load_shap_task_archive


def _write_small_master_matrix(matrix_path: Path) -> None:
    rows = []
    for i in range(20):
        group = "A" if i < 10 else "B"
        sex = "M" if i % 2 == 0 else "F"
        rows.append(
            {
                "individual": f"fly{i}",
                "group": group,
                "sex": sex,
                "BoutSummary_feat1": float(i + 1),
                "StdMeasure_feat2": float(i + 0.5),
                "Kinematic_turn_inst": float((i // 2) + 0.25),
                "Spatial_feat4": float(20 - i),
                "Constant_feat0": 1.0,
                "Postural_group": "ignored_text",
            }
        )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)



def test_featureimp_staged_core_outputs(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_staged_core"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "staged_core",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 4,
        "model_subsample": 1.0,
        "learning_rate": 0.2,
        "max_depth": 2,
        "n_splits": 2,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": True,
        "n_permutations": 1,
        "run_fsea_analysis": False,
        "run_pca_analysis": False,
        "subset_weight_tau_quantile": 0.5,
        "num_shards": 3,
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
        analysis_run_name="staged_core",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
        sex_filter="all",
        num_shards=3,
    )
    assert out_plan == str(analysis_dir)

    first_shard_done = None
    for shard_index in range(3):
        shard_out = Path(core.process_featureimp_shard(
            analysis_folder_path=str(analysis_dir),
            shard_index=shard_index,
        ))
        assert shard_out.exists()
        assert (shard_out / "SHARD_DONE.json").exists()
        if shard_index == 0:
            first_shard_done = json.loads((shard_out / "SHARD_DONE.json").read_text())

    assert first_shard_done is not None
    perm_shap_rel = first_shard_done["artifacts"]["perm_shap_tasks"]["0"]
    perm_shap_archive = analysis_dir / Path(perm_shap_rel)
    perm_archive = load_shap_task_archive(str(perm_shap_archive))
    assert perm_archive, "Expected permutation SHAP task archive payload"
    assert "insample_shap_flat" in perm_archive
    assert "all_groups_shap_flat" in perm_archive

    out_merge = core.merge_featureimp_shards(analysis_folder_path=str(analysis_dir))
    assert out_merge == str(analysis_dir)
    assert (analysis_dir / "_featureimp_merge" / "WEIGHTS_DONE.json").exists()
    assert (analysis_dir / "COMMITTEE_SUBSETS" / "SUBSET_WEIGHTS_REAL.csv").exists()
    for shard_index in range(3):
        shard_label = f"shard_{shard_index:03d}_of_003"
        assert (
            analysis_dir / "_featureimp_shards" / shard_label / "WEIGHTED_REDUCTION_DONE.json"
        ).exists()

    merge_done_path = analysis_dir / "_featureimp_merge" / "FINAL_MERGE_DONE.json"
    assert merge_done_path.exists()
    merge_done = json.loads(merge_done_path.read_text())
    assert merge_done["status"] == "completed"
    assert merge_done["real_weighted_reduction_count"] == 3

    assert (analysis_dir / "GLOBAL_IMPORTANCE_RANKING.csv").exists()
    assert (analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx").exists()
    assert (analysis_dir / "ENSEMBLE_PERFORMANCE_METRICS.xlsx").exists()
    assert (analysis_dir / "COMMITTEE_POOLED_PERFORMANCE.csv").exists()
    assert (analysis_dir / "FEATUREIMP_REFERENCE_MANIFEST.json").exists()
    assert (analysis_dir / "SHAP_CONSOLIDATED_SUMMARY.xlsx").exists()
    assert (analysis_dir / "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx").exists()
    assert (analysis_dir / "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx").exists()
    assert (analysis_dir / "SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx").exists()
    assert (analysis_dir / "COMMITTEE_SUBSETS" / "SHAP_WEIGHTING_MANIFEST.json").exists()
    assert (analysis_dir / "PERMUTED_0_RESULTS" / "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE_PERMUTED_0.xlsx").exists()
    assert (analysis_dir / "PERMUTED_RESULTS" / "AVERAGED_PREDICTIONS_PERMUTED_0.xlsx").exists()

    weighted_df = pd.read_excel(analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx", sheet_name="WeightedPredictions")
    unweighted_df = pd.read_excel(analysis_dir / "AVERAGED_PREDICTIONS_FULL.xlsx", sheet_name="UnweightedPredictions")
    assert set(["individual", "group"]).issubset(set(weighted_df.columns))
    assert set(["individual", "group"]).issubset(set(unweighted_df.columns))
    assert any(col.startswith("Prob_A") or col.startswith("Prob_B") for col in weighted_df.columns)

    # Every requested auxiliary family must remain verifiable after completion.
    import pytest
    from autobehaver.postinf.featureimp.integrity import validate_receipt
    validate_receipt(str(merge_done_path), str(analysis_dir))
    for relative in (
        "PERMUTED_0_RESULTS/GLOBAL_IMPORTANCE_RANKING_PERMUTED_0.csv",
        "FEATUREIMP_REFERENCE/FROZEN_FEATURE_COLUMNS.json",
    ):
        target = analysis_dir / relative
        original = target.read_bytes()
        target.unlink()
        with pytest.raises(ValueError, match="Missing scientific artifact"):
            validate_receipt(str(merge_done_path), str(analysis_dir))
        target.write_bytes(original)
        validate_receipt(str(merge_done_path), str(analysis_dir))


def test_featureimp_merge_shards_serial_tail(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_merge_shards"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "merge_shards",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 4,
        "model_subsample": 1.0,
        "learning_rate": 0.2,
        "max_depth": 2,
        "n_splits": 2,
        "H_hits_per_feature": 1,
        "n_features_per_model": 2,
        "run_permutation_test": True,
        "n_permutations": 1,
        "run_fsea_analysis": False,
        "run_pca_analysis": False,
        "subset_weight_tau_quantile": 0.5,
        "num_shards": 3,
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
        analysis_run_name="merge_shards",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=1,
        n_features_per_model=2,
        sex_filter="all",
        num_shards=3,
    )

    for shard_index in range(3):
        shard_out = Path(core.process_featureimp_shard(
            analysis_folder_path=str(analysis_dir),
            shard_index=shard_index,
        ))
        assert (shard_out / "SHARD_DONE.json").exists()

    out_merge = core.merge_featureimp_shards(analysis_folder_path=str(analysis_dir))
    assert out_merge == str(analysis_dir)
    assert (analysis_dir / "_featureimp_merge" / "WEIGHTS_DONE.json").exists()
    assert (analysis_dir / "_featureimp_merge" / "FINAL_MERGE_DONE.json").exists()
    assert (analysis_dir / "GLOBAL_IMPORTANCE_RANKING.csv").exists()

