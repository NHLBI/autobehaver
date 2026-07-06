import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_featureimp_plan_stage import _write_small_master_matrix


def _build_small_staged_analysis(tmp_path: Path) -> Path:
    from autobehaver.postinf.featureimp import core

    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_postanalysis"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    _write_small_master_matrix(matrix_path)

    settings = {
        "analysis_run_name": "postanalysis",
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
        "num_shards": 2,
        "run_plain_xgboost_baseline": False,
        "run_fsea_analysis": False,
        "run_pca_analysis": False,
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
        analysis_run_name="postanalysis",
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
        core.process_featureimp_shard(analysis_folder_path=str(analysis_dir), shard_index=shard_index)
    core.merge_featureimp_shards(analysis_folder_path=str(analysis_dir))
    return analysis_dir


def test_merge_shards_writes_final_postanalysis_bundle(tmp_path: Path):
    analysis_dir = _build_small_staged_analysis(tmp_path)
    bundle_path = analysis_dir / "_featureimp_merge" / "FINAL_POSTANALYSIS_BUNDLE.json"
    assert bundle_path.exists()
    payload = json.loads(bundle_path.read_text())
    assert payload["status"] == "completed"
    assert payload["n_train"] > 0
    assert payload["n_all"] >= payload["n_train"]
    rel = str(payload["hybrid_shap_mmap"])
    assert (analysis_dir / rel).exists()




def test_postanalysis_runner_fsea_stage_uses_core_bundle(monkeypatch, tmp_path: Path):
    analysis_dir = _build_small_staged_analysis(tmp_path)

    calls = []

    def _fake_fsea(*, oof_shap_values, df_for_training, all_feature_names, training_group_a, training_group_b, df_for_shap_and_agg, paths, le, num_classes, all_groups_avg_shap_values=None, settings=None):
        calls.append({
            "oof_shape": tuple(oof_shap_values.shape),
            "all_shape": tuple(all_groups_avg_shap_values.shape),
            "training_groups": (training_group_a, training_group_b),
            "feature_count": len(all_feature_names),
        })
        out_dir = Path(paths["analysis_folder"]) / "FSEA"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dummy.txt").write_text("ok")
        return None

    monkeypatch.setattr(
        "autobehaver.postinf.featureimp.postanalysis_runner._perform_fsea_analysis",
        _fake_fsea,
    )

    from autobehaver.postinf.featureimp.postanalysis_runner import run_featureimp_postanalysis_stage

    out = run_featureimp_postanalysis_stage(
        analysis_folder_path=str(analysis_dir),
        stage_name="fsea",
    )
    assert out == str(analysis_dir)
    assert len(calls) == 1
    assert calls[0]["oof_shape"][0] == 12
    assert calls[0]["all_shape"][0] == 12

    post_idx = json.loads((analysis_dir / "POSTANALYSIS_INDEX.json").read_text())
    stage_map = {row["stage"]: row for row in post_idx["stages"]}
    assert stage_map["fsea"]["status"] == "completed"


def test_plain_xgboost_standalone_stage_respects_saved_fsea_toggle(monkeypatch, tmp_path: Path):
    analysis_dir = _build_small_staged_analysis(tmp_path / 'plain_fsea')
    import autobehaver.postinf.featureimp.plain_xgboost_baseline as px
    from autobehaver.postinf.featureimp import core

    core.apply_settings({"analysis_run_name": "plain_fsea_stage"}, propagate=False)
    seen = {}
    monkeypatch.setattr(px, 'run_plain_xgboost_baseline', lambda **kwargs: seen.update(kwargs) or str(Path(kwargs['analysis_folder_path']) / 'PLAIN_XGBOOST_BASELINE'))

    from autobehaver.postinf.featureimp.postanalysis_runner import run_featureimp_postanalysis_stage
    out = run_featureimp_postanalysis_stage(analysis_folder_path=str(analysis_dir), stage_name='plain_xgboost_baseline')
    assert out == str(analysis_dir)
    assert seen['run_fsea_analysis'] is False


def test_baseline_standalone_stage_respects_explicit_runtime_fsea_disable(monkeypatch, tmp_path: Path):
    analysis_dir = _build_small_staged_analysis(tmp_path / 'baseline_explicit_disable')
    import autobehaver.postinf.featureimp.plain_xgboost_baseline as px
    from autobehaver.postinf.featureimp import core

    core.apply_settings({"analysis_run_name": "baseline_disable", "run_fsea_analysis": False}, propagate=False)
    seen_plain = {}
    monkeypatch.setattr(px, 'run_plain_xgboost_baseline', lambda **kwargs: seen_plain.update(kwargs) or str(Path(kwargs['analysis_folder_path']) / 'PLAIN_XGBOOST_BASELINE'))

    from autobehaver.postinf.featureimp.postanalysis_runner import run_featureimp_postanalysis_stage
    run_featureimp_postanalysis_stage(analysis_folder_path=str(analysis_dir), stage_name='plain_xgboost_baseline')
    assert seen_plain['run_fsea_analysis'] is False


def test_postanalysis_saved_settings_use_only_curated_snapshot_and_explicit_overrides(tmp_path: Path):
    analysis_dir = tmp_path / 'postanalysis_saved_settings'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / 'FEATUREIMP_RUN_SETTINGS.json').write_text(json.dumps({
        'training_group_a': 'SAVED_A',
        'training_group_b': 'SAVED_B',
        'control_group_tag': 'SAVED_CTRL',
        'GLOBAL_RANDOM_SEED': 123,
        'run_fsea_analysis': False,
        'run_pca_analysis': False,
        'fsea_permutations': 25,
        'num_workers': 99,
    }))

    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.postanalysis_runner import _apply_saved_settings_snapshot

    core.apply_settings({'analysis_run_name': 'stage_context_only'}, propagate=False)
    effective = _apply_saved_settings_snapshot(str(analysis_dir))
    assert effective['training_group_a'] == 'SAVED_A'
    assert effective['training_group_b'] == 'SAVED_B'
    assert effective['control_group_tag'] == 'SAVED_CTRL'
    assert int(effective['GLOBAL_RANDOM_SEED']) == 123
    assert effective['run_fsea_analysis'] is False
    assert effective['run_pca_analysis'] is False
    assert int(effective['fsea_permutations']) == 25
    assert effective['analysis_run_name'] == 'stage_context_only'
    assert 'num_workers' not in effective

    core.apply_settings({'GLOBAL_RANDOM_SEED': 777, 'run_fsea_analysis': True}, propagate=False)
    overridden = _apply_saved_settings_snapshot(str(analysis_dir))
    assert int(overridden['GLOBAL_RANDOM_SEED']) == 777
    assert overridden['run_fsea_analysis'] is True
    assert overridden['run_pca_analysis'] is False


def test_runtime_num_workers_no_longer_comes_from_saved_snapshot(tmp_path: Path):
    analysis_dir = tmp_path / 'runtime_num_workers'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / 'FEATUREIMP_RUN_SETTINGS.json').write_text(json.dumps({'num_workers': 99}))

    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.shard_runner import _get_runtime_num_workers, _load_settings_snapshot

    core.apply_settings({'num_workers': 3}, propagate=False)
    assert 'num_workers' not in _load_settings_snapshot(str(analysis_dir))
    assert _get_runtime_num_workers() == 3


def test_merge_shards_postanalysis_respects_saved_toggles_and_explicit_overrides(monkeypatch, tmp_path: Path):
    analysis_dir = tmp_path / 'merge_saved_toggles'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / 'FEATUREIMP_RUN_SETTINGS.json').write_text(json.dumps({
        'run_fsea_analysis': False,
        'run_pca_analysis': False,
        'postanalysis_strict': True,
    }))

    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.merge_shards import _run_default_reference_postanalysis
    import autobehaver.postinf.featureimp.postanalysis_runner as postanalysis_runner

    calls = []

    def _fake_run_featureimp_postanalysis_stage(*, analysis_folder_path, stage_name):
        calls.append(stage_name)
        return analysis_folder_path

    monkeypatch.setattr(postanalysis_runner, 'run_featureimp_postanalysis_stage', _fake_run_featureimp_postanalysis_stage)

    core.apply_settings(
        {'stage': 'merge_shards', 'analysis_folder_path': str(analysis_dir)},
        propagate=False,
        explicit_keys={'stage', 'analysis_folder_path'},
    )
    _run_default_reference_postanalysis(str(analysis_dir))
    assert calls == []

    core.apply_settings(
        {'stage': 'merge_shards', 'analysis_folder_path': str(analysis_dir), 'run_pca_analysis': True},
        propagate=False,
        explicit_keys={'stage', 'analysis_folder_path', 'run_pca_analysis'},
    )
    _run_default_reference_postanalysis(str(analysis_dir))
    assert calls == ['pca']
