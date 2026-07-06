import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_small_master_matrix(matrix_path: Path) -> None:
    rows = []
    for i in range(12):
        group = "A" if i < 6 else "B"
        sex = "M" if i % 2 == 0 else "F"
        rows.append(
            {
                "individual": f"fly{i}",
                "group": group,
                "sex": sex,
                "BoutSummary_feat1": float(i + 1),
                "StdMeasure_feat2": float((i % 4) + 0.5),
                "Kinematic_turn_inst": float((i // 2) + 0.25),
                "Spatial_feat4": float(20 - i),
                "Constant_feat0": 1.0,
                "Postural_group": "ignored_text",
            }
        )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)



def test_featureimp_process_shard_stage_core_writes_shard_outputs(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_process_shard_core"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "process_shard_core",
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
        "run_permutation_test": True,
        "n_permutations": 1,
        "num_shards": 3,
        "run_fsea_analysis": False,
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
        analysis_run_name="process_shard_core",
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

    shard_out = Path(
        core.process_featureimp_shard(
            analysis_folder_path=str(analysis_dir),
            shard_index=0,
        )
    )
    assert shard_out.exists()

    done_path = shard_out / "SHARD_DONE.json"
    assert done_path.exists()
    assert not (shard_out / "SHARD_RUNNING.json").exists()
    assert not (shard_out / "SHARD_FAILED.json").exists()

    done = json.loads(done_path.read_text())
    assert done["status"] == "completed"
    assert done["task_count"] == 2
    assert done["real_task_count"] == 1
    assert done["perm_task_count"] == 1
    assert done["real_performance_record_count"] == 2
    assert done["perm_performance_record_count"] == 2
    assert done["real_subset_row_count"] == 1
    assert done["perm_subset_row_count"] == 1
    assert done["written_perm_indices"] == [0]

    real_partials = np.load(shard_out / "REAL_PARTIALS.npz")
    assert real_partials["oof_probas_sum"].shape == (12, 2)
    assert real_partials["oof_shap_sum"].shape == (12, 4)
    assert int(real_partials["all_groups_probas_count"]) == 2

    perm_partials = np.load(shard_out / "PERM_PARTIALS_perm000.npz")
    assert perm_partials["oof_probas_sum"].shape == (12, 2)
    assert perm_partials["oof_shap_sum"].shape == (12, 4)
    assert int(perm_partials["all_groups_probas_count"]) == 2

    real_margin_tasks = np.load(shard_out / "REAL_MARGIN_TASKS.npz")
    assert real_margin_tasks["oof_margin_sum"].shape == (1, 12)
    assert real_margin_tasks["insample_margin_sum"].shape == (1, 12)
    assert real_margin_tasks["all_groups_margin_sum"].shape == (1, 12)

    perm_margin_tasks = np.load(shard_out / "PERM_MARGIN_TASKS_perm000.npz")
    assert perm_margin_tasks["oof_margin_sum"].shape == (1, 12)
    assert perm_margin_tasks["insample_margin_sum"].shape == (1, 12)
    assert perm_margin_tasks["all_groups_margin_sum"].shape == (1, 12)

    real_shap_tasks = np.load(shard_out / "REAL_SHAP_TASKS.npz")
    assert real_shap_tasks["h_hit"].shape == (1,)
    assert real_shap_tasks["feature_count"].shape == (1,)
    assert "insample_shap_flat" in real_shap_tasks.files
    assert "all_groups_shap_flat" in real_shap_tasks.files

    perm_shap_tasks = np.load(shard_out / "PERM_SHAP_TASKS_perm000.npz")
    assert perm_shap_tasks["h_hit"].shape == (1,)
    assert perm_shap_tasks["feature_count"].shape == (1,)
    # Permutation SHAP task archives retain the same full payload shape as real
    # archives so the weighted second wave can reuse one reducer and, when
    # requested, export per-permutation prediction/SHAP views.
    assert "insample_shap_flat" in perm_shap_tasks.files
    assert "all_groups_shap_flat" in perm_shap_tasks.files
    assert bool(perm_shap_tasks["meta__include_insample"].reshape(())) is True
    assert bool(perm_shap_tasks["meta__include_all_groups"].reshape(())) is True

    assert (shard_out / "PERFORMANCE_RECORDS_REAL.jsonl").exists()
    assert (shard_out / "PERFORMANCE_RECORDS_PERM.jsonl").exists()
    assert (shard_out / "SUBSET_PERFORMANCE_REAL_PART.csv").exists()
    assert (shard_out / "SUBSET_PERFORMANCE_PERM_PART.csv").exists()

    real_subset_df = pd.read_csv(shard_out / "SUBSET_PERFORMANCE_REAL_PART.csv")
    perm_subset_df = pd.read_csv(shard_out / "SUBSET_PERFORMANCE_PERM_PART.csv")
    assert real_subset_df["folds_seen_count"].tolist() == [2]
    assert real_subset_df["folds_with_metrics_count"].tolist() == [2]
    assert real_subset_df["n_test_seen_total"].tolist() == [12]
    assert real_subset_df["subset_complete"].tolist() == [True]
    assert perm_subset_df["folds_seen_count"].tolist() == [2]
    assert perm_subset_df["folds_with_metrics_count"].tolist() == [2]
    assert perm_subset_df["n_test_seen_total"].tolist() == [12]
    assert perm_subset_df["subset_complete"].tolist() == [True]

    model_root = analysis_dir / "SAVED_PIPELINE" / "models" / done["shard_label"]
    model_files = sorted(model_root.rglob("*.json"))
    assert len(model_files) == 4


def test_featureimp_cli_stage_process_shard_writes_outputs(tmp_path: Path):
    from autobehaver.postinf.featureimp import cli

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    cli.main(
        [
            "--stage",
            "plan",
            "--analysis_run_name",
            "process_shard_cli",
            "--matrix_dir",
            str(matrix_dir),
            "--results_dir",
            str(results_dir),
            "--matrix_path",
            str(matrix_path),
            "--control_group_tag",
            "A",
            "--training_group_a",
            "A",
            "--training_group_b",
            "B",
            "--n_splits",
            "2",
            "--H_hits_per_feature",
            "1",
            "--n_features_per_model",
            "2",
            "--n_estimators",
            "4",
            "--learning_rate",
            "0.2",
            "--max_depth",
            "2",
            "--num_workers",
            "1",
            "--run_permutation_test",
            "--n_permutations",
            "1",
            "--num_shards",
            "3",
        ]
    )

    analysis_dirs = sorted(results_dir.glob("FEATURE_ANALYSIS_process_shard_cli_*"))
    assert len(analysis_dirs) == 1
    analysis_dir = analysis_dirs[0]

    cli.main(
        [
            "--stage",
            "process_shard",
            "--analysis_folder_path",
            str(analysis_dir),
            "--shard_index",
            "0",
        ]
    )

    shard_out = analysis_dir / "_featureimp_shards" / "shard_000_of_003"
    assert shard_out.exists()
    assert (shard_out / "SHARD_DONE.json").exists()
    assert (shard_out / "REAL_PARTIALS.npz").exists()
    assert (shard_out / "REAL_MARGIN_TASKS.npz").exists()
    assert (shard_out / "REAL_SHAP_TASKS.npz").exists()
    assert (shard_out / "PERM_PARTIALS_perm000.npz").exists()
    assert (shard_out / "PERM_MARGIN_TASKS_perm000.npz").exists()
    assert (shard_out / "PERM_SHAP_TASKS_perm000.npz").exists()
