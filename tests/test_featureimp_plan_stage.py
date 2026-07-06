import csv
import json
from pathlib import Path

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



def test_featureimp_plan_stage_core_writes_frozen_plan(tmp_path: Path):
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_plan_core"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "plan_core",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
        "num_workers": 1,
        "n_estimators": 5,
        "n_splits": 2,
        "H_hits_per_feature": 2,
        "n_features_per_model": 2,
        "run_permutation_test": True,
        "n_permutations": 2,
        "num_shards": 5,
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

    out = core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name="plan_core",
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=str(analysis_dir),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        n_splits=2,
        H_hits_per_feature=2,
        n_features_per_model=2,
        sex_filter="all",
        num_shards=5,
    )

    assert out == str(analysis_dir)

    plan_dir = analysis_dir / "_featureimp_plan"
    manifest_path = plan_dir / "PLAN_MANIFEST.json"
    task_manifest_path = plan_dir / "TASK_MANIFEST.csv"
    subset_plan_path = plan_dir / "HHIT_SUBSET_PLAN.jsonl"
    fold_plan_path = plan_dir / "HHIT_FOLD_PLAN.jsonl"
    selected_features_path = plan_dir / "SELECTED_FEATURE_COLUMNS.json"

    assert manifest_path.exists()
    assert task_manifest_path.exists()
    assert subset_plan_path.exists()
    assert fold_plan_path.exists()
    assert selected_features_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "planned"
    assert manifest["requested_num_shards"] == 5
    assert manifest["active_num_shards"] == 5
    assert manifest["selected_feature_count"] == 4
    assert manifest["planned_family_counts"]["real"] == 4
    assert manifest["planned_family_counts"]["perm"] == 8
    assert manifest["planned_task_count"] == 12
    assert manifest["planned_total_model_count"] == 24

    selected_features = json.loads(selected_features_path.read_text())
    assert len(selected_features) == 4
    assert "Constant_feat0" not in selected_features
    assert "Postural_group" not in selected_features

    with open(task_manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["family"] for row in rows} == {"real", "perm"}
    assert {int(row["shard_index"]) for row in rows} == {0, 1, 2, 3, 4}

    shard_root = plan_dir / "shards"
    shard_dirs = sorted(p for p in shard_root.iterdir() if p.is_dir())
    assert len(shard_dirs) == 5
    for shard_dir in shard_dirs:
        assert (shard_dir / "SHARD_PLAN.json").exists()
        assert (shard_dir / "SHARD_TASKS.csv").exists()



def test_featureimp_cli_stage_plan_writes_plan_outputs(tmp_path: Path):
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
            "plan_cli",
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
            "2",
            "--n_features_per_model",
            "2",
            "--run_permutation_test",
            "--n_permutations",
            "1",
            "--num_shards",
            "3",
        ]
    )

    analysis_dirs = sorted(results_dir.glob("FEATURE_ANALYSIS_plan_cli_*"))
    assert len(analysis_dirs) == 1
    analysis_dir = analysis_dirs[0]

    assert (analysis_dir / "FEATUREIMP_RUN_SETTINGS.json").exists()
    assert (analysis_dir / "NORMALIZED_FEATURE_MATRIX.csv").exists()
    assert (analysis_dir / "NORMALIZATION_PARAMS.json").exists()
    assert (analysis_dir / "_featureimp_plan" / "PLAN_MANIFEST.json").exists()
    assert (analysis_dir / "_featureimp_plan" / "PLAN_DONE.json").exists()
    assert (analysis_dir / "_featureimp_plan" / "TASK_MANIFEST.csv").exists()

    manifest = json.loads((analysis_dir / "_featureimp_plan" / "PLAN_MANIFEST.json").read_text())
    assert manifest["requested_num_shards"] == 3
    assert manifest["active_num_shards"] == 3
    assert manifest["planned_family_counts"]["real"] == 4
    assert manifest["planned_family_counts"]["perm"] == 4
