"""Regression cases for the 2026-10-19 FeatureImp audit corrections."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_identity_reader_keeps_leading_zero_and_na_text(tmp_path):
    from autobehaver.postinf.featureimp.matrix import validate_feature_matrix
    path = tmp_path / "matrix.csv"
    path.write_text("individual,group,sex,source_files,StdMeasure_x\n001,NA,M,001,1\n002,A,F,NA,\n")
    valid, error, frame = validate_feature_matrix(str(path))
    assert valid, error
    assert frame["individual"].tolist() == ["001", "002"]
    assert frame["group"].tolist() == ["NA", "A"]
    assert frame["source_files"].tolist() == ["001", "NA"]
    assert np.isnan(frame["StdMeasure_x"].iloc[1])


def test_frame_metadata_survives_normalization(tmp_path):
    from autobehaver.postinf.featureimp import core
    from tests.test_featureimp_plan_stage import _write_small_master_matrix
    raw = tmp_path / "raw.csv"
    _write_small_master_matrix(raw)
    frame = pd.read_csv(raw)
    frame["total_frames_analyzed"] = np.arange(len(frame)) + 1000
    frame.to_csv(raw, index=False)
    analysis = tmp_path / "a"
    analysis.mkdir()
    core.apply_settings({"control_group_tag": "A"}, propagate=False)
    result = core.normalize_feature_matrix(results_dir=str(tmp_path), raw_matrix_path=str(raw),
        analysis_folder_path=str(analysis), skip_if_normalized_exists=False)
    assert result
    actual = pd.read_csv(result)
    np.testing.assert_array_equal(actual["total_frames_analyzed"], frame["total_frames_analyzed"])
    params = json.loads((analysis / "NORMALIZATION_PARAMS.json").read_text())
    assert "total_frames_analyzed" not in params["numeric_feature_columns"]


@pytest.mark.parametrize("seed", [None, 0, 19])
def test_applied_settings_plan_freezes_effective_values(tmp_path, seed):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.baseline_outputs import load_featureimp_baseline_context
    from autobehaver.postinf.featureimp.shard_runner import _load_settings_snapshot
    from tests.test_featureimp_plan_stage import _write_small_master_matrix
    analysis = tmp_path / "a"
    analysis.mkdir()
    raw = tmp_path / "raw.csv"
    _write_small_master_matrix(raw)
    core.apply_settings({"analysis_run_name":"applied_only","control_group_tag":"A",
        "training_group_a":"A","training_group_b":"B","n_splits":2,"H_hits_per_feature":2,
        "n_features_per_model":2,"run_permutation_test":False,"run_fsea_analysis":False,
        "GLOBAL_RANDOM_SEED":seed}, propagate=False)
    normalized = core.normalize_feature_matrix(results_dir=str(tmp_path),raw_matrix_path=str(raw),
        analysis_folder_path=str(analysis),skip_if_normalized_exists=False)
    core.plan_feature_importance_analysis(normalized_matrix_path=normalized,
        results_dir=str(tmp_path),analysis_folder_path=str(analysis))
    manifest = json.loads((analysis/"_featureimp_plan/PLAN_MANIFEST.json").read_text())
    snapshot = json.loads((analysis/"FEATUREIMP_RUN_SETTINGS.json").read_text())
    assert manifest["n_splits"] == 2
    assert manifest["H_hits_per_feature"] == 2
    assert manifest["training_groups"] == ["A","B"]
    assert snapshot["GLOBAL_RANDOM_SEED"] == manifest["resolved_global_random_seed"]
    assert snapshot["__requested_global_random_seed__"] == seed
    if seed is not None:
        assert snapshot["GLOBAL_RANDOM_SEED"] == seed
    loaded = load_featureimp_baseline_context(str(analysis))
    assert loaded["global_random_seed"] == snapshot["GLOBAL_RANDOM_SEED"]
    before = (analysis/"FEATUREIMP_RUN_SETTINGS.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        core.plan_feature_importance_analysis(normalized_matrix_path=normalized,
            results_dir=str(tmp_path),analysis_folder_path=str(analysis))
    assert (analysis/"FEATUREIMP_RUN_SETTINGS.json").read_bytes() == before
    log = (analysis/"ANALYSIS_SETTINGS_LOG.txt").read_text()
    assert f'Resolved Global Random Seed: {snapshot["GLOBAL_RANDOM_SEED"]}' in log
    assert "Main training Prob_*: held-out" in log
    (analysis/"FEATUREIMP_RUN_SETTINGS.json").unlink()
    with pytest.raises(FileNotFoundError, match="Required frozen"):
        _load_settings_snapshot(str(analysis))


def test_collision_allocation_is_exclusive(tmp_path):
    from autobehaver.postinf.featureimp.core import allocate_analysis_folder
    a, b = [allocate_analysis_folder(str(tmp_path), "same_second") for _ in range(2)]
    assert a != b
    assert Path(a).is_dir() and Path(b).is_dir()


@pytest.mark.parametrize("kind", ["shard", "weights", "final"])
def test_contender_cannot_mutate_owner_status(tmp_path, monkeypatch, kind):
    from autobehaver.postinf.featureimp import shard_runner as shard
    from autobehaver.postinf.featureimp import subset_weight_merge as weights
    from autobehaver.postinf.featureimp import final_merge as final
    if kind == "shard":
        monkeypatch.setattr(shard, "_load_shard_plan_and_tasks", lambda *a, **k: ({}, [], "s0"))
        directory = tmp_path / shard.SHARD_OUTPUT_DIRNAME / "s0"
        lock_name, invoke = shard.LOCK_DIRNAME, lambda: shard.process_featureimp_shard_run(str(tmp_path), shard_index=0)
    elif kind == "weights":
        directory = tmp_path / weights.MERGE_DIRNAME
        lock_name, invoke = weights.WEIGHTS_LOCK_DIRNAME, lambda: weights.calculate_subset_weights_run(str(tmp_path))
    else:
        directory = tmp_path / final.MERGE_DIRNAME
        lock_name, invoke = final.FINAL_MERGE_LOCK_DIRNAME, lambda: final.finalize_featureimp_outputs_run(str(tmp_path))
    directory.mkdir(parents=True)
    markers = [directory / name for name in ("RUNNING.json", "FAILED.json", "DONE.json")]
    for marker in markers:
        marker.write_bytes(b"owner state")
    with shard._directory_lock(str(directory / lock_name)):
        with pytest.raises(RuntimeError, match="holds"):
            invoke()
    assert all(marker.read_bytes() == b"owner state" for marker in markers)
    with shard._directory_lock(str(directory / lock_name)):
        pass


def test_os_lock_is_released_by_process_exit(tmp_path):
    from autobehaver.file_locks import file_lock_active
    path = tmp_path / "exit.lock"
    code = "import os,sys;from autobehaver.file_locks import FileLock;x=FileLock(sys.argv[1]);x.acquire(metadata={'test':1});os._exit(0)"
    result = subprocess.run([sys.executable,"-c",code,str(path)], timeout=20, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert path.is_file() and not file_lock_active(path)


def test_baseline_admission_allows_independent_shards_only(tmp_path):
    from autobehaver.postinf.featureimp.plain_xgboost_staged import _baseline_lifecycle_lock
    with _baseline_lifecycle_lock(str(tmp_path), "shard", 0):
        with _baseline_lifecycle_lock(str(tmp_path), "shard", 1):
            pass
        for operation, index in (("shard",0),("merge",None),("plan",None)):
            with pytest.raises(RuntimeError, match="active"):
                with _baseline_lifecycle_lock(str(tmp_path),operation,index):
                    pass
    with _baseline_lifecycle_lock(str(tmp_path),"merge"):
        with pytest.raises(RuntimeError, match="active"):
            with _baseline_lifecycle_lock(str(tmp_path),"shard",2):
                pass


@pytest.mark.parametrize("valid", [True, False])
def test_completed_baseline_shard_is_preserved(tmp_path, monkeypatch, valid):
    from autobehaver.postinf.featureimp import plain_xgboost_staged as baseline
    monkeypatch.setattr(baseline, "_load_staged_manifest", lambda *args: {"active_num_shards":1})
    directory = Path(baseline._shard_output_dir(str(tmp_path), 0, 1))
    directory.mkdir(parents=True)
    done = directory / baseline.SHARD_DONE_FILENAME
    done.write_text("saved completion")
    output = directory / "science.npz"
    output.write_bytes(b"saved science")
    def validate(*args):
        if not valid:
            raise ValueError("injected invalid receipt")
    monkeypatch.setattr(baseline, "validate_baseline_receipt", validate)
    if valid:
        assert baseline.process_plain_xgboost_baseline_shard(analysis_folder_path=str(tmp_path), shard_index=0) == str(directory)
    else:
        with pytest.raises(ValueError, match="invalid receipt"):
            baseline.process_plain_xgboost_baseline_shard(analysis_folder_path=str(tmp_path), shard_index=0)
    assert done.read_text() == "saved completion"
    assert output.read_bytes() == b"saved science"


def test_hybrid_report_uses_oof_rows_by_identity():
    from autobehaver.postinf.featureimp.finalize import _hybrid_report_shap
    all_values = np.array([[20,30],[1,2],[40,50]],dtype=np.float32)
    actual = _hybrid_report_shap(all_values,[[4,5],[6,7]],
        pd.DataFrame({"individual":["b","other","a"]}),
        pd.DataFrame({"individual":["a","b"]}))
    np.testing.assert_array_equal(actual, [[6,7],[1,2],[4,5]])
    np.testing.assert_array_equal(all_values, [[20,30],[1,2],[40,50]])


@pytest.mark.parametrize("weighted", [False, True])
def test_permutation_report_forwards_weighting_and_oof(tmp_path, monkeypatch, weighted):
    from autobehaver.postinf.featureimp import finalize
    captured = {}
    monkeypatch.setattr(finalize, "_generate_kfold_shap_summaries", lambda **kw: (None,None))
    monkeypatch.setattr(finalize, "_aggregate_and_summarize_results", lambda **kw: captured.update(kw))
    frame = pd.DataFrame({"individual":["a","b","c"],"group":["A","B","C"]})
    finalize._aggregate_and_summarize_permuted_results(frame,["x"],[],[],SimpleNamespace(classes_=["A","B"]),
        {"analysis_folder":str(tmp_path)},0,np.ones((2,1)),np.array([[2],[3]]),
        frame.iloc[:2],["A","B"],2,permuted_all_groups_avg_shap_values=np.array([[90],[80],[7]]),
        df_for_shap_and_agg=frame,primary_is_weighted=weighted)
    assert captured["primary_is_weighted"] is weighted
    np.testing.assert_array_equal(captured["avg_shap_values_override"], [[2],[3],[7]])


def test_missing_sex_is_unavailable(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp.summaries import _generate_male_female_comparison
    frame = pd.DataFrame({"individual":["a","b"],"group":["A","B"],"sex":["M","M"],"x":[0,1],"y":[1,0]})
    from autobehaver.postinf.featureimp import summaries
    captured = []
    monkeypatch.setattr(summaries, "_combine_male_female_summaries", lambda male, female, paths: captured.extend([male, female]))
    _generate_male_female_comparison(frame,["x","y"],[],
        SimpleNamespace(classes_=["A","B"]),{"analysis_folder":str(tmp_path)},
        avg_shap_values_override=np.array([[1.,2.],[3.,4.]]))
    male, female = captured
    assert female["Global_Importance"].isna().all()
    assert female["Global_Importance_Normalized"].isna().all()
    status = json.loads((tmp_path/"SHAP_SEX_AVAILABILITY.json").read_text())
    assert status["female"]["global_importance_source"] == "unavailable"
    assert status["male"]["training_row_count"] == 2
    np.testing.assert_allclose(male["Global_Importance"].sort_values(), [2,3])


def test_rank_correlation_preserves_ties_and_constants():
    from autobehaver.postinf.featureimp.summaries import _magnitude_rank_correlation
    from scipy.stats import spearmanr
    a,b=np.array([2,2,1,0]),np.array([3,1,1,0])
    assert _magnitude_rank_correlation(a,b) == pytest.approx(spearmanr(a,b).statistic)
    assert np.isnan(_magnitude_rank_correlation(np.ones(4),b))


@pytest.mark.parametrize("column,expected", [
    ("MeanSHAP_wild_type",("wild_type","MeanSHAP")),
    ("MeanSHAP_Diff_Abs_Norm_wild_type",("wild_type","MeanSHAP_Diff_Abs_Norm")),
    ("MeanAbsSHAP_GroupA_Normalized",("GroupA","MeanAbsSHAP_Normalized")),
    ("MeanSHAP_Normalized_wild_type_MALE",("wild_type (MALE)","MeanSHAP_Normalized")),
    ("MeanSHAP_wild_type_OUTOFSAMPLE",("wild_type (OUT-OF-SAMPLE)","MeanSHAP")),
    ("Global_Importance_Normalized",("","Global_Importance_Normalized"))])
def test_consolidated_headers(column,expected):
    from autobehaver.postinf.featureimp.reporting import _consolidated_header
    assert _consolidated_header(column) == expected


def test_graphpad_collisions_retain_every_original_group(tmp_path):
    from autobehaver.postinf.featureimp.reporting import create_formatted_predictions_for_graphpad
    frame = pd.DataFrame({"individual":["1","2"],"group":["A/B","A?B"],"Prob_control":[.2,.8]})
    output = create_formatted_predictions_for_graphpad(frame,"control","mutant",str(tmp_path))
    actual = pd.read_excel(output)
    assert actual.shape == (1,2)
    assert sorted(actual.iloc[0].tolist()) == [.2,.8]
    assert len(set(actual.columns)) == 2


def test_centered_pca_does_not_include_extra_sample_component():
    from autobehaver.postinf.featureimp.pca_utils import _fit_project_reference_pca
    ref=np.array([[0.,1,2,3],[1,0,1,4],[2,3,0,1]],dtype=np.float64)
    scaler,pca,xr,xa,k,variance = _fit_project_reference_pca(ref,ref,pca_fixed_k=50,random_state=0)
    assert k == 2 and xr.shape == (3,2)
    assert np.all(variance > 1e-12)


@pytest.mark.parametrize("value", [pd.Index(["x","y"]), np.array(["x","y"]), None])
def test_saved_model_features_accept_sequence_and_none(value):
    from autobehaver.postinf.featureimp.shap_ensemble import _record_features
    assert _record_features({"specialist_features":value,"features":["x","y"]}) == ["x","y"]


def test_fsea_generation_preserves_history_and_only_caches(tmp_path):
    from autobehaver.postinf.featureimp.fsea_runner import _archive_fsea_generation
    root=tmp_path/"FSEA"
    cache=root/"FSEA_A/.numeric_cache"
    cache.mkdir(parents=True)
    (cache/"prerank.json").write_text("numeric")
    (root/"old_plot.png").write_text("old")
    (root/"user_notes.txt").write_text("preserve")
    archived=Path(_archive_fsea_generation(root))
    assert (archived/"old_plot.png").is_file()
    assert (archived/"user_notes.txt").read_text() == "preserve"
    assert not (root/"old_plot.png").exists() and not (root/"user_notes.txt").exists()
    assert (root/"FSEA_A/.numeric_cache/prerank.json").read_text() == "numeric"


def test_requested_fsea_plot_failure_is_partial_and_logged(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp import fsea_runner as runner
    from autobehaver.postinf.featureimp import postanalysis_runner as post
    from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings
    names=["f1","f2"]
    ranking=pd.DataFrame({"feature_name":names,"Global_Importance":[2.,1.]})
    ranking.to_csv(tmp_path/"GLOBAL_IMPORTANCE_RANKING.csv",index=False)
    class Result:
        res2d=pd.DataFrame({"Term":["all"],"ES":[.5],"NES":[1.2],"FDR":[.01]})
        def plot(self, **kwargs):
            raise RuntimeError("injected plot failure")
    monkeypatch.setitem(sys.modules,"gseapy",SimpleNamespace(prerank=lambda **kw:Result()))
    monkeypatch.setattr(runner,"_prepare_validated_fsea_groups",lambda **kw:{"all":names})
    monkeypatch.setattr(runner,"_orchestrate_fsea_visualization",lambda **kw:None)
    frame=pd.DataFrame({"individual":["a","b"],"group":["A","B"],"sex":["M","F"]})
    kwargs=dict(oof_shap_values=np.array([[1.,2.],[2.,3.]]),df_for_training=frame,
        all_feature_names=names,training_group_a="A",training_group_b="B",df_for_shap_and_agg=frame,
        paths={"analysis_folder":str(tmp_path)},le=None,num_classes=2,
        settings=FSEARuntimeSettings(fsea_min_size=1,fsea_max_size=5,fsea_permutations=1))
    result=runner._perform_fsea_analysis(**kwargs)
    assert result["status"] == "partial" and not result["plots_complete"]
    assert any("injected plot failure" in msg for msg in result["visualization_failures"])
    assert json.loads((tmp_path/"FSEA/FSEA_COMPLETE.json").read_text())["status"] == "partial"
    captured=[]
    monkeypatch.setattr(post,"upsert_postanalysis_stage_record",lambda *args:captured.append(args[1]))
    with pytest.raises(RuntimeError,match="partial"):
        post._run_logged_stage("fsea",lambda:result,analysis_folder_path=str(tmp_path),settings_snapshot={})
    assert captured[-1]["context"]["result"]["visualization_failures"]
    with pytest.raises(RuntimeError,match="injected plot failure"):
        runner.run_global_importance_fsea_from_file(ranked_list_path=str(tmp_path/"GLOBAL_IMPORTANCE_RANKING.csv"),
            output_dir=str(tmp_path/"standalone"),settings=kwargs["settings"])
    status=json.loads((tmp_path/"standalone/FSEA_RANKING_STATUS.json").read_text())
    assert status["status"] == "partial"
    obsolete = tmp_path / "FSEA" / "obsolete.png"
    obsolete.write_bytes(b"previous request")
    monkeypatch.setattr(Result, "plot", lambda self, **kw: Path(kw["ofname"]).write_text("current plot"))
    complete = runner._perform_fsea_analysis(**kwargs)
    assert complete["status"] == "completed"
    assert "obsolete.png" not in complete["artifact_sha256"] and not obsolete.exists()
    assert list((tmp_path/"_featureimp_history").glob("FSEA_*/obsolete.png"))
    monkeypatch.setitem(sys.modules,"gseapy",SimpleNamespace(prerank=lambda **kw: (_ for _ in ()).throw(AssertionError("should reuse"))))
    assert runner._perform_fsea_analysis(**kwargs) == complete


def test_auxiliary_receipt_inventory_requires_reference_and_permutations(tmp_path):
    from autobehaver.postinf.featureimp.final_merge import _auxiliary_report_artifacts
    with pytest.raises(FileNotFoundError,match="FEATUREIMP_REFERENCE_MANIFEST"):
        _auxiliary_report_artifacts(tmp_path,1)
    names=["FEATUREIMP_REFERENCE_MANIFEST.json","SHAP_SEX_AVAILABILITY.json"]
    names += ["FEATUREIMP_REFERENCE/"+name for name in ("FROZEN_FEATURE_COLUMNS.json",
        "NORMALIZATION_FEATURE_COLUMNS.json","COMPOSITIONAL_SETS.json","MODEL_INVENTORY.csv")]
    for name in names:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text("fixture")
    assert set(_auxiliary_report_artifacts(tmp_path,0)) == set(names)
    with pytest.raises(FileNotFoundError,match="PERMUTED_0_RESULTS"):
        _auxiliary_report_artifacts(tmp_path,1)


def test_baseline_merge_observes_saved_fsea_false(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp import plain_xgboost_staged as baseline
    monkeypatch.setattr(baseline, "_load_staged_manifest", lambda *a: {"settings":{"run_fsea_analysis":False}})
    monkeypatch.setattr(baseline, "load_featureimp_baseline_context", lambda *a: {})
    monkeypatch.setattr(baseline, "_iter_explicit_settings", lambda *a: {})
    seen = {}
    class StopAfterResolution(Exception):
        pass
    def prepare(**kwargs):
        seen.update(kwargs)
        raise StopAfterResolution
    monkeypatch.setattr(baseline, "_prepare_context", prepare)
    with pytest.raises(StopAfterResolution):
        baseline.merge_plain_xgboost_baseline_shards(analysis_folder_path=str(tmp_path))
    assert seen["run_fsea_analysis"] is False


def test_optional_baseline_fsea_propagates_partial(tmp_path, monkeypatch):
    from autobehaver.postinf.featureimp import fsea_runner, plain_xgboost_baseline as baseline
    monkeypatch.setattr(fsea_runner, "_perform_fsea_analysis", lambda **kw: {"status":"partial", "visualization_failures":["plot failed"]})
    with pytest.raises(RuntimeError, match="plot failed"):
        baseline._run_optional_plain_xgboost_fsea(out_dir=str(tmp_path), settings_snapshot={},
            oof_shap_values=np.ones((2,1)), all_groups_avg_shap_values=np.ones((2,1)),
            df_for_training=pd.DataFrame(), df_for_shap_and_agg=pd.DataFrame(), feature_names=["x"],
            training_group_a="A", training_group_b="B", le=None, num_classes=2)

