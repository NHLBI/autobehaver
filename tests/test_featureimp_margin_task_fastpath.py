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


def _jsonl_load(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_margin_task_fastpath_matches_saved_model_logit_rescoring(tmp_path: Path):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.logit_ensemble import (
        compute_logit_ensemble_predictions_from_saved_models,
    )
    from autobehaver.postinf.featureimp.margin_tasks import (
        compute_logit_ensemble_predictions_from_margin_task_archives,
    )
    from autobehaver.postinf.featureimp.plan import load_featureimp_plan_manifest
    from autobehaver.postinf.featureimp.shard_runner import _load_featureimp_context

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_margin_fastpath"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "margin_fastpath",
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

    core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=str(results_dir),
        analysis_run_name="margin_fastpath",
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
        core.process_featureimp_shard(
            analysis_folder_path=str(analysis_dir),
            shard_index=shard_index,
        )

    manifest = load_featureimp_plan_manifest(str(analysis_dir))
    context = _load_featureimp_context(str(analysis_dir), manifest)
    X_train_full = context["X_train_full"].reset_index(drop=True)
    X_all_groups = context["X_shap_full"].reset_index(drop=True)
    y_encoded_train = np.asarray(context["y_encoded_train"], dtype=np.int32)
    sex_group_encoded = np.asarray(context["sex_group_encoded"], dtype=np.int32)
    df_for_training = context["df_for_training"].reset_index(drop=True)
    num_classes = int(context["num_classes"])
    resolved_global_seed = int(manifest.get("resolved_global_random_seed", 0) or 0)

    shard_dirs = sorted((analysis_dir / "_featureimp_shards").glob("shard_*"))
    real_perf_rows = []
    perm_perf_rows = []
    real_margin_paths = []
    perm_margin_paths = []
    for shard_dir in shard_dirs:
        done = json.loads((shard_dir / "SHARD_DONE.json").read_text())
        artifacts = done["artifacts"]
        real_perf_rows.extend(_jsonl_load(analysis_dir / artifacts["performance_records_real_jsonl"]))
        perm_perf_rows.extend(_jsonl_load(analysis_dir / artifacts["performance_records_perm_jsonl"]))
        real_margin_paths.append(str(analysis_dir / artifacts["real_margin_tasks_npz"]))
        perm_margin_artifacts = dict(artifacts.get("perm_margin_tasks") or {})
        if "0" in perm_margin_artifacts:
            perm_margin_paths.append(str(analysis_dir / perm_margin_artifacts["0"]))

    subset_weights = {(1, 0): 0.25, (1, 1): 0.75}

    real_fast = compute_logit_ensemble_predictions_from_margin_task_archives(
        archive_paths=real_margin_paths,
        performance_records=real_perf_rows,
        n_train=len(X_train_full),
        n_all=len(X_all_groups),
        num_classes=num_classes,
        n_splits=2,
        subset_weights=subset_weights,
        include_permuted=False,
        verbose=False,
    )
    real_saved = compute_logit_ensemble_predictions_from_saved_models(
        performance_records=real_perf_rows,
        X_train_full=X_train_full,
        X_all_groups=X_all_groups,
        y_encoded_train=y_encoded_train,
        sex_group_encoded=sex_group_encoded,
        num_classes=num_classes,
        n_splits=2,
        GLOBAL_RANDOM_SEED=resolved_global_seed,
        subset_weights=subset_weights,
        include_permuted=False,
        verbose=False,
    )

    assert np.allclose(real_fast.oof_probas, real_saved.oof_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(real_fast.insample_probas, real_saved.insample_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(real_fast.all_groups_probas, real_saved.all_groups_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(real_fast.oof_probas_weighted, real_saved.oof_probas_weighted, atol=1e-8, rtol=1e-8)
    assert np.allclose(real_fast.insample_probas_weighted, real_saved.insample_probas_weighted, atol=1e-8, rtol=1e-8)
    assert np.allclose(real_fast.all_groups_probas_weighted, real_saved.all_groups_probas_weighted, atol=1e-8, rtol=1e-8)

    perm_fast = compute_logit_ensemble_predictions_from_margin_task_archives(
        archive_paths=perm_margin_paths,
        performance_records=perm_perf_rows,
        n_train=len(X_train_full),
        n_all=len(X_all_groups),
        num_classes=num_classes,
        n_splits=2,
        subset_weights=subset_weights,
        include_permuted=True,
        permutation_idx=0,
        verbose=False,
    )
    perm_saved = compute_logit_ensemble_predictions_from_saved_models(
        performance_records=perm_perf_rows,
        X_train_full=X_train_full,
        X_all_groups=X_all_groups,
        y_encoded_train=y_encoded_train,
        sex_group_encoded=sex_group_encoded,
        num_classes=num_classes,
        n_splits=2,
        GLOBAL_RANDOM_SEED=resolved_global_seed,
        subset_weights=subset_weights,
        include_permuted=True,
        permutation_idx=0,
        y_original_labels_train=df_for_training["group"].to_numpy(),
        sex_labels_train=df_for_training["sex"].to_numpy(),
        verbose=False,
    )

    assert np.allclose(perm_fast.oof_probas, perm_saved.oof_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(perm_fast.insample_probas, perm_saved.insample_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(perm_fast.all_groups_probas, perm_saved.all_groups_probas, atol=1e-8, rtol=1e-8)
    assert np.allclose(perm_fast.oof_probas_weighted, perm_saved.oof_probas_weighted, atol=1e-8, rtol=1e-8)
    assert np.allclose(perm_fast.insample_probas_weighted, perm_saved.insample_probas_weighted, atol=1e-8, rtol=1e-8)
    assert np.allclose(perm_fast.all_groups_probas_weighted, perm_saved.all_groups_probas_weighted, atol=1e-8, rtol=1e-8)
