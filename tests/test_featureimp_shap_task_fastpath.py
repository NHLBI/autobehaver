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


def _read_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_shap_task_fastpath_matches_saved_model_weighted_shap(tmp_path: Path):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.hhit import make_hhit_folds
    from autobehaver.postinf.featureimp.plan import load_featureimp_plan_manifest
    from autobehaver.postinf.featureimp.shap_ensemble import compute_weighted_shap_from_saved_models
    from autobehaver.postinf.featureimp.shap_tasks import compute_weighted_shap_from_task_archives
    from autobehaver.postinf.featureimp.shard_runner import _build_real_fold_indices, _load_featureimp_context

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_shap_fastpath"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "shap_fastpath",
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
        analysis_run_name="shap_fastpath",
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
    sex_group_encoded = np.asarray(context["sex_group_encoded"], dtype=np.int32)
    selected_features = list(context["selected_features"])
    fold_plan_records = dict(context["fold_plan_records"])
    n_train = len(X_train_full)
    resolved_global_seed = int(manifest.get("resolved_global_random_seed", 0) or 0)

    shard_dirs = sorted((analysis_dir / "_featureimp_shards").glob("shard_*"))
    real_perf_rows = []
    perm_perf_rows = []
    real_shap_paths = []
    perm_shap_paths = []
    for shard_dir in shard_dirs:
        done = json.loads((shard_dir / "SHARD_DONE.json").read_text())
        artifacts = done["artifacts"]
        real_perf_rows.extend(_read_jsonl(analysis_dir / artifacts["performance_records_real_jsonl"]))
        perm_perf_rows.extend(_read_jsonl(analysis_dir / artifacts["performance_records_perm_jsonl"]))
        real_shap_paths.append(str(analysis_dir / artifacts["real_shap_tasks_npz"]))
        perm_shap_artifacts = dict(artifacts.get("perm_shap_tasks") or {})
        if "0" in perm_shap_artifacts:
            perm_shap_paths.append(str(analysis_dir / perm_shap_artifacts["0"]))

    subset_weights = {(1, 0): 0.25, (1, 1): 0.75}
    fold_indices_by_h_hit = {int(h_hit): _build_real_fold_indices(rec) for h_hit, rec in sorted(fold_plan_records.items())}

    real_fast = compute_weighted_shap_from_task_archives(
        archive_paths=real_shap_paths,
        performance_records=real_perf_rows,
        subset_weights=subset_weights,
        all_feature_names=selected_features,
        n_train=n_train,
        n_all=len(X_all_groups),
        n_splits=2,
        include_permuted=False,
        require_full=True,
        verbose=False,
    )
    real_saved = compute_weighted_shap_from_saved_models(
        performance_records=real_perf_rows,
        subset_weights=subset_weights,
        fold_indices_by_h_hit=fold_indices_by_h_hit,
        X_all_groups=X_all_groups,
        all_feature_names=selected_features,
        n_train=n_train,
        n_splits=2,
        skip_permuted=True,
        positive_class_index=1,
        verbose=False,
    )
    for fast_arr, saved_arr in zip(real_fast[:4], real_saved[:4]):
        assert np.allclose(np.asarray(fast_arr), np.asarray(saved_arr), atol=1e-6, rtol=1e-6)

    perm_perf = [r for r in perm_perf_rows if int(r.get("permutation_idx", -1)) == 0]
    perm_fold_indices_by_h_hit = {
        1: make_hhit_folds(
            X_train_full,
            sex_group_encoded,
            n_splits=2,
            random_state=int(resolved_global_seed + 0 * 100000 + 1 * 1000),
        )
    }
    perm_fast = compute_weighted_shap_from_task_archives(
        archive_paths=perm_shap_paths,
        performance_records=perm_perf_rows,
        subset_weights=subset_weights,
        all_feature_names=selected_features,
        n_train=n_train,
        n_all=n_train,
        n_splits=2,
        include_permuted=True,
        permutation_idx=0,
        require_full=False,
        verbose=False,
    )
    perm_saved = compute_weighted_shap_from_saved_models(
        performance_records=perm_perf,
        subset_weights=subset_weights,
        fold_indices_by_h_hit=perm_fold_indices_by_h_hit,
        X_all_groups=X_train_full,
        all_feature_names=selected_features,
        n_train=n_train,
        n_splits=2,
        skip_permuted=False,
        positive_class_index=1,
        verbose=False,
    )
    assert np.allclose(np.asarray(perm_fast[0]), np.asarray(perm_saved[0]), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(perm_fast[3]), np.asarray(perm_saved[3]), atol=1e-6, rtol=1e-6)


def test_shap_task_fastpath_handles_all_zero_subset_weights_without_fallback(tmp_path: Path):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.plan import load_featureimp_plan_manifest
    from autobehaver.postinf.featureimp.shap_tasks import compute_weighted_shap_from_task_archives
    from autobehaver.postinf.featureimp.shard_runner import _load_featureimp_context

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_shap_zero_weight"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "shap_zero_weight",
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
        "num_shards": 2,
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
        analysis_run_name="shap_zero_weight",
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

    manifest = load_featureimp_plan_manifest(str(analysis_dir))
    context = _load_featureimp_context(str(analysis_dir), manifest)
    selected_features = list(context["selected_features"])
    n_train = int(context["X_train_full"].shape[0])
    n_all = int(context["X_shap_full"].shape[0])

    shard_dirs = sorted((analysis_dir / "_featureimp_shards").glob("shard_*"))
    real_perf_rows = []
    perm_perf_rows = []
    real_shap_paths = []
    perm_shap_paths = []
    for shard_dir in shard_dirs:
        done = json.loads((shard_dir / "SHARD_DONE.json").read_text())
        artifacts = done["artifacts"]
        real_perf_rows.extend(_read_jsonl(analysis_dir / artifacts["performance_records_real_jsonl"]))
        perm_perf_rows.extend(_read_jsonl(analysis_dir / artifacts["performance_records_perm_jsonl"]))
        real_shap_paths.append(str(analysis_dir / artifacts["real_shap_tasks_npz"]))
        perm_shap_artifacts = dict(artifacts.get("perm_shap_tasks") or {})
        if "0" in perm_shap_artifacts:
            perm_shap_paths.append(str(analysis_dir / perm_shap_artifacts["0"]))

    subset_weights = {(1, 0): 0.0, (1, 1): 0.0}
    real_fast = compute_weighted_shap_from_task_archives(
        archive_paths=real_shap_paths,
        performance_records=real_perf_rows,
        subset_weights=subset_weights,
        all_feature_names=selected_features,
        n_train=n_train,
        n_all=n_all,
        n_splits=2,
        include_permuted=False,
        require_full=True,
        verbose=False,
    )
    assert np.allclose(np.asarray(real_fast[0]), 0.0)
    assert np.allclose(np.asarray(real_fast[1]), 0.0)
    assert np.allclose(np.asarray(real_fast[2]), 0.0)
    assert np.allclose(np.asarray(real_fast[3]), 0.0)
    assert real_fast[4]["n_subsets_used"] == 0
    assert real_fast[4]["n_subsets_skipped_zero_weight"] == 2

    perm_fast = compute_weighted_shap_from_task_archives(
        archive_paths=perm_shap_paths,
        performance_records=perm_perf_rows,
        subset_weights=subset_weights,
        all_feature_names=selected_features,
        n_train=n_train,
        n_all=n_train,
        n_splits=2,
        include_permuted=True,
        permutation_idx=0,
        require_full=False,
        verbose=False,
    )
    assert np.allclose(np.asarray(perm_fast[0]), 0.0)
    assert np.allclose(np.asarray(perm_fast[3]), 0.0)
    assert perm_fast[4]["n_subsets_used"] == 0
    assert perm_fast[4]["n_subsets_skipped_zero_weight"] == 2
