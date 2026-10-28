"""Scientific arithmetic, missingness, persistence and reuse regressions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def conditional_headers():
    poses = ['Standing', 'Walking', 'Groom-Head', 'Groom-Forelegs', 'Groom-Midlegs', 'Groom-Hindlegs', 'Groom-Abdomen', 'Groom-Wings']
    orientations = ['Floor', 'Wall', 'Ceiling']
    return [f'TimeBudget_PoseGivenOrient_{p}_{o}' for p in poses for o in orientations] + [f'TimeBudget_OrientGivenPose_{o}_{p}' for p in poses for o in orientations]


def test_conditional_groups_and_independent_clr():
    from autobehaver.postinf.featureimp.matrix import infer_compositional_sets, _clr_transform
    sets = infer_compositional_sets(conditional_headers())
    assert len(sets) == 11 and sorted(map(len, sets.values())) == [3]*8+[8]*3
    values = np.array([.1, .2, .7])
    np.testing.assert_allclose(_clr_transform(values), np.log(values)-np.log(values).mean(), atol=1e-14)


@pytest.mark.parametrize('values', [[.2, -.1, .9], [.2, np.inf, .8]])
def test_invalid_composition_rejected(values):
    from autobehaver.postinf.featureimp.matrix import _clr_transform
    with pytest.raises(ValueError):
        _clr_transform(values)


def test_partial_composition_remains_unavailable():
    from autobehaver.postinf.featureimp.matrix import _apply_clr_to_dataframe
    frame = pd.DataFrame([[.2, np.nan, .8], [0, .4, .6], [0, 0, 0]], columns=['a','b','c'])
    _apply_clr_to_dataframe(frame, {'composition':['a','b','c']})
    assert frame.iloc[0].isna().all()
    assert np.isfinite(frame.iloc[1:]).all().all()
    np.testing.assert_allclose(frame.iloc[1:].sum(axis=1), 0, atol=1e-12)


def test_insufficient_control_support_recorded(tmp_path):
    from autobehaver.postinf.featureimp.matrix import normalize_feature_matrix
    frame = pd.DataFrame({'individual':['a','b','c'], 'group':['A','A','B'], 'sex':['M']*3, 'good':[1.,2.,3.], 'unsupported':[1.,np.nan,5.]})
    source = tmp_path/'raw.csv';frame.to_csv(source,index=False)
    result = normalize_feature_matrix(str(tmp_path),str(source),control_group_tag='A')
    assert 'unsupported' not in pd.read_csv(result)
    params = json.loads((tmp_path/'NORMALIZATION_PARAMS.json').read_text())
    assert params['control_finite_counts']['unsupported'] == 1
    assert params['unsupported_control_reference_features'] == ['unsupported']
    np.testing.assert_allclose(pd.read_csv(result)['good'], [-2**-.5,2**-.5,3*2**-.5],atol=1e-11)


def test_pca_changed_values_and_order_do_not_reuse(tmp_path):
    from autobehaver.postinf.featureimp.pca_utils import get_cached_pca_projection
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(24,5)), columns=list('abcde'))
    paths={'analysis_folder':str(tmp_path)}
    def project(x, p):
        return get_cached_pca_projection(x,x,paths=p,cache_key='regression',pca_fixed_k=2,random_state=0)[3]
    original=project(frame,paths)
    changed=frame.copy();changed.iloc[:,0] += changed.iloc[:,1]*3
    actual=project(changed,paths);fresh=project(changed,{'analysis_folder':str(tmp_path/'fresh')})
    np.testing.assert_allclose(actual,fresh,atol=1e-6)
    assert not np.allclose(original,actual)
    permuted=frame.iloc[::-1]
    np.testing.assert_allclose(project(permuted,paths), project(permuted,{'analysis_folder':str(tmp_path/'ordered')}), atol=1e-6)
    np.testing.assert_allclose(project(frame,{'analysis_folder':str(tmp_path)}),original,atol=1e-5)


def test_weighting_valid_negative_null_and_false_flags():
    from autobehaver.postinf.featureimp.subset_weighting import compute_tau_from_permuted, compute_subset_weights, SubsetWeightingConfig
    config=SubsetWeightingConfig()
    tau,_=compute_tau_from_permuted(pd.DataFrame({'delta_norm':[-.2,-.1], 'subset_complete':['True','False']}),config)
    assert tau == 0
    result=compute_subset_weights(pd.DataFrame({'delta_norm':[.2,.4], 'subset_complete':['True','False']}),tau,config)
    np.testing.assert_allclose(result['subset_weight'],[.04,0])
    with pytest.raises(ValueError):
        compute_tau_from_permuted(pd.DataFrame({'delta_norm':[np.nan]}),config)


@pytest.mark.parametrize('gamma,tau', [(0,0),(-1,0),(np.inf,0),(2,-1),(2,np.nan)])
def test_invalid_weight_domains(gamma,tau):
    from autobehaver.postinf.featureimp.subset_weighting import compute_subset_weights, SubsetWeightingConfig
    with pytest.raises(ValueError):
        compute_subset_weights(pd.DataFrame({'delta_norm':[.1]}),tau,SubsetWeightingConfig(gamma=gamma))


def test_margin_support_and_unequal_weight_arithmetic():
    from autobehaver.postinf.featureimp.margin_tasks import finalize_weighted_margin_partial
    partial={'oof_margin_num':np.array([2.,-2.]), 'oof_weight_den':np.array([2.,4.]), 'insample_margin_num':np.array([3.,6.]), 'insample_weight_den':np.array([3.,3.]), 'all_margin_num':np.array([3.]), 'all_weight_den_scalar':3.}
    result=finalize_weighted_margin_partial(partial=partial,num_classes=2)
    np.testing.assert_allclose(result.oof_probas[:,1],1/(1+np.exp(-np.array([1.,-.5]))))
    partial['all_weight_den_scalar']=0
    with pytest.raises(ValueError):
        finalize_weighted_margin_partial(partial=partial,num_classes=2)


def test_shap_support_and_inconsistent_zero_denominator():
    from autobehaver.postinf.featureimp.shap_tasks import finalize_weighted_shap_partial
    partial={'feature_weight_den':np.array([2.,0.]), 'oof_shap_num':np.array([[4.,0.]]), 'insample_shap_num':np.array([[8.,0.]]), 'all_shap_num':np.array([[6.,0.]]), 'total_weight_sum':2.}
    oof,ins,all_,_,diagnostics=finalize_weighted_shap_partial(partial=partial,n_splits=3,require_full=True)
    np.testing.assert_allclose(oof,[[2.,0.]])
    np.testing.assert_allclose(ins,[[2.,0.]])
    np.testing.assert_allclose(all_,[[3.,0.]])
    assert diagnostics['unsupported_feature_indices']==[1]
    partial['oof_shap_num'][0,1]=1
    with pytest.raises(ValueError):
        finalize_weighted_shap_partial(partial=partial,n_splits=3,require_full=True)


def test_receipt_rejects_changed_data_output_and_legacy(tmp_path):
    from autobehaver.postinf.featureimp.integrity import seal_receipt,validate_receipt
    plan=tmp_path/'_featureimp_plan';plan.mkdir()
    (plan/'PLAN_MANIFEST.json').write_text('{}')
    source=tmp_path/'NORMALIZED_FEATURE_MATRIX.csv';source.write_text('data')
    output=tmp_path/'out.npz';output.write_text('output')
    destination=tmp_path/'SHARD_DONE.json'
    payload=seal_receipt({'status':'completed','analysis_folder':str(tmp_path),'artifacts':{'result':'out.npz'}},str(destination))
    destination.write_text(json.dumps(payload))
    validate_receipt(str(destination),str(tmp_path))
    output.write_text('changed')
    with pytest.raises(ValueError):validate_receipt(str(destination),str(tmp_path))
    output.write_text('output');source.write_text('changed')
    with pytest.raises(ValueError):validate_receipt(str(destination),str(tmp_path))
    destination.write_text(json.dumps({'status':'completed'}))
    with pytest.raises(ValueError):validate_receipt(str(destination),str(tmp_path))


def test_required_workbook_atomic_failure_and_empty_sheet(tmp_path,monkeypatch):
    from autobehaver.postinf.featureimp.io_utils import save_to_excel_sheets
    destination=tmp_path/'report.xlsx';destination.write_bytes(b'original')
    original=pd.DataFrame.to_excel
    def fail(frame,*args,**kwargs):
        if kwargs.get('sheet_name')=='failed':raise OSError('injected write failure')
        return original(frame,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,'to_excel',fail)
    with pytest.raises(OSError):
        save_to_excel_sheets({'good':pd.DataFrame({'x':[1]}),'failed':pd.DataFrame({'x':[2]})},destination)
    assert destination.read_bytes()==b'original'
    save_to_excel_sheets({'good':pd.DataFrame({'x':[1]}),'empty':pd.DataFrame(columns=['x'])},destination)
    assert pd.ExcelFile(destination).sheet_names==['good','empty']


def test_required_native_model_persistence_reload_predictions(tmp_path):
    import xgboost as xgb
    from autobehaver.postinf.featureimp.committee_fast import _save_model_path
    from autobehaver.postinf.featureimp.logit_ensemble import load_classifier_model
    features=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]*5)
    labels=np.array([0,0,1,1]*5)
    model=xgb.XGBClassifier(n_estimators=8,max_depth=2,n_jobs=1,random_state=0).fit(features,labels)
    saved=_save_model_path(model=model,paths={'models_folder':str(tmp_path)},h_hit=1,subset_idx=0,fold_idx=0,is_permuted=False,permutation_idx=None)
    reloaded=load_classifier_model(saved)
    np.testing.assert_array_equal(model.predict_proba(features),reloaded.predict_proba(features))
    assert json.loads(model.get_booster().attr('scikit_learn'))['_estimator_type']=='classifier'


def test_strict_settings_and_seed_zero():
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings
    original=core.get_current_settings()
    try:
        core.apply_settings({'run_fsea_analysis':'False','GLOBAL_RANDOM_SEED':0,'MIN_VARIANCE_THRESHOLD':0})
        assert core.run_fsea_analysis is False and core.GLOBAL_RANDOM_SEED==0
        settings=FSEARuntimeSettings.from_mapping({'GLOBAL_RANDOM_SEED':0,'fsea_fdr_threshold':0,'fsea_viz_top_n':0,'fsea_plot_sig_only':'False'})
        assert settings.global_random_seed==0 and settings.fsea_fdr_threshold==0 and settings.fsea_viz_top_n==0 and settings.fsea_plot_sig_only is False
        with pytest.raises(ValueError):core.apply_settings({'run_fsea_analysis':'garbage'})
    finally:core.apply_settings(original)


def test_baseline_preserves_requested_fsea_settings():
    from autobehaver.postinf.featureimp.plain_xgboost_staged import _collect_baseline_settings
    request={'GLOBAL_RANDOM_SEED':0,'fsea_permutations':123,'fsea_fdr_threshold':.10,'fsea_min_size':3,'fsea_max_size':50,'fsea_plot_sig_only':'False'}
    actual=_collect_baseline_settings({'settings_snapshot':{},'global_random_seed':7},settings_override=request)
    assert all(actual[name]==value for name,value in request.items())


def test_fsea_native_statistics_and_required_failure(tmp_path,monkeypatch):
    from autobehaver.postinf.featureimp import fsea_runner as runner
    from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings
    import gseapy
    names=[f'f{i}' for i in range(30)]
    ranking=pd.DataFrame({'feature_name':names,'rank_value':np.linspace(3,-3,30)})
    settings=FSEARuntimeSettings(fsea_min_size=3,fsea_max_size=20,fsea_permutations=25,fsea_fdr_threshold=0,fsea_viz_top_n=0)
    path=runner._run_single_global_preranked_fsea(ranked_df=ranking,validated_groups={'top':names[:6],'bottom':names[-6:]},output_dir=str(tmp_path),label='native',settings=settings)
    table=pd.read_excel(path)
    assert len(table)==2 and 'FDR q-val' in table and runner._significant_count(table,0)==0
    assert table['ES'].max()>0 and table['ES'].min()<0
    monkeypatch.setattr(runner,'_prepare_validated_fsea_groups',lambda **kwargs:{'top':names[:6]})
    monkeypatch.setattr(gseapy,'prerank',lambda **kwargs:(_ for _ in ()).throw(RuntimeError('injected required failure')))
    frame=pd.DataFrame({'group':['A','B']})
    with pytest.raises(RuntimeError,match='injected'):
        runner._perform_fsea_analysis(np.ones((2,30)),frame,names,'A','B',frame,{'analysis_folder':str(tmp_path)},None,2,settings=settings)
    assert not (tmp_path/'FSEA'/'FSEA_COMPLETE.json').exists()


@pytest.mark.parametrize('column,value', [('ES',np.nan),('FDR q-val',np.inf),('FDR q-val',-0.1),('FDR q-val',1.1)])
def test_fsea_required_statistics_domains(column,value):
    from autobehaver.postinf.featureimp.fsea_runner import _check_fsea_table
    frame=pd.DataFrame({'Term':['A'],'ES':[.4],'FDR q-val':[.5]})
    frame.loc[0,column]=value
    with pytest.raises(ValueError):_check_fsea_table(frame)


@pytest.mark.parametrize('value', [np.nan,np.inf,'invalid'])
def test_fsea_rank_values_cannot_become_measured_zeros(value):
    from autobehaver.postinf.featureimp.fsea_runner import _standardize_ranked_list_dataframe
    with pytest.raises(ValueError):
        _standardize_ranked_list_dataframe(pd.DataFrame({'name':['a'],'score':[value]}),feature_col='name',rank_col='score',source_name='test')


def _small_analysis(path,workers,shards):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.plan import plan_featureimp_run
    from autobehaver.postinf.featureimp.matrix import normalize_feature_matrix
    from autobehaver.postinf.featureimp.shard_runner import process_featureimp_shard_run
    from autobehaver.postinf.featureimp.merge_shards import merge_featureimp_shards_run
    path.mkdir();source=path/'raw.csv'
    rng=np.random.default_rng(18)
    n=24
    frame=pd.DataFrame({'individual':[f'fly{i}' for i in range(n)],'group':['A']*12+['B']*12,'sex':['M','F']*12})
    for j in range(4):frame[f'KinematicBB_feature{j}']=np.r_[rng.normal(-3,.5,12),rng.normal(3,.5,12)]
    frame.to_csv(source,index=False)
    core.apply_settings({'training_group_a':'A','training_group_b':'B','control_group_tag':'A','n_estimators':12,'learning_rate':.3,'max_depth':2,'num_workers':workers,'run_fsea_analysis':False,'run_pca_analysis':False,'run_permutation_test':False,'GLOBAL_RANDOM_SEED':0})
    (path/'FEATUREIMP_RUN_SETTINGS.json').write_text(json.dumps(core.get_current_settings(),default=str))
    normalized=normalize_feature_matrix(str(path),str(source),analysis_folder_path=str(path),control_group_tag='A')
    plan_featureimp_run(normalized,str(path),analysis_folder_path=str(path),control_group_tag='A',training_group_a='A',training_group_b='B',n_splits=2,H_hits_per_feature=2,n_features_per_model=2,run_permutation_test=False,num_shards=shards,GLOBAL_RANDOM_SEED=0)
    for index in range(shards):process_featureimp_shard_run(str(path),shard_index=index)
    merge_featureimp_shards_run(str(path))
    return pd.read_excel(path/'AVERAGED_PREDICTIONS_FULL.xlsx'),pd.read_csv(path/'GLOBAL_IMPORTANCE_RANKING.csv')


@pytest.mark.slow
def test_native_serial_parallel_staged_equivalence(tmp_path):
    from autobehaver.postinf.featureimp import core
    original=core.get_current_settings()
    try:
        a,rank_a=_small_analysis(tmp_path/'serial',1,1)
        b,rank_b=_small_analysis(tmp_path/'parallel',2,1)
        c,rank_c=_small_analysis(tmp_path/'staged',1,2)
        for result in (b,c):
            assert a['individual'].fillna('').tolist()==result['individual'].fillna('').tolist()
            for column in [x for x in a if x.startswith('Prob_')]:
                np.testing.assert_allclose(a[column],result[column],atol=2e-6,equal_nan=True)
        for result in (rank_b,rank_c):
            pd.testing.assert_frame_equal(rank_a,result,check_exact=False,atol=2e-6,rtol=2e-6)
    finally:core.apply_settings(original)


def test_native_baseline_staged_equivalence_and_tamper_rejection(tmp_path):
    from autobehaver.postinf.featureimp import core
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import rerun_plain_xgboost_baseline_from_analysis_folder
    from autobehaver.postinf.featureimp.plain_xgboost_staged import plan_plain_xgboost_baseline_shards, process_plain_xgboost_baseline_shard, merge_plain_xgboost_baseline_shards
    original=core.get_current_settings()
    try:
        path=tmp_path/'baseline'
        _small_analysis(path,1,1)
        rerun_plain_xgboost_baseline_from_analysis_folder(analysis_folder_path=str(path),run_fsea_analysis=False)
        baseline=path/'PLAIN_XGBOOST_BASELINE'
        expected=pd.read_excel(baseline/'AVERAGED_PREDICTIONS_FULL.xlsx')
        plan_plain_xgboost_baseline_shards(analysis_folder_path=str(path),num_shards=2)
        for index in range(2):process_plain_xgboost_baseline_shard(analysis_folder_path=str(path),shard_index=index)
        merge_plain_xgboost_baseline_shards(analysis_folder_path=str(path),run_fsea_analysis=False)
        observed=pd.read_excel(baseline/'AVERAGED_PREDICTIONS_FULL.xlsx')
        for column in [c for c in expected if c.startswith('Prob_')]:
            np.testing.assert_allclose(expected[column],observed[column],atol=2e-6,equal_nan=True)
        from autobehaver.postinf.featureimp.integrity import validate_baseline_receipt
        marker=baseline/'_STAGED_PLAN'/'PLAIN_XGBOOST_STAGED_MERGE_DONE.json'
        validate_baseline_receipt(str(marker),str(path))
        from autobehaver.postinf.featureimp.shard_runner import _directory_lock
        with _directory_lock(str(path/'_plain_xgboost_merge.lock')):
            with pytest.raises(RuntimeError):
                merge_plain_xgboost_baseline_shards(analysis_folder_path=str(path),run_fsea_analysis=False)
        partial=next((baseline/'_STAGED_SHARDS').glob('*/REAL_PARTIALS.npz'))
        partial.write_bytes(partial.read_bytes()+b'tamper')
        with pytest.raises(ValueError,match='changed'):
            merge_plain_xgboost_baseline_shards(analysis_folder_path=str(path),run_fsea_analysis=False)
        pd.testing.assert_frame_equal(pd.read_excel(baseline/'AVERAGED_PREDICTIONS_FULL.xlsx'),observed)
    finally:core.apply_settings(original)
