from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _pid(value):
    return os.getpid(), value


def _fail_worker(value):
    raise ValueError('deliberate worker failure')


def _task_result(h, subset, features, offset):
    n_train, n_all = 4, 6
    return {'h_hit': h, 'subset_idx': subset, 'features': features,
            'oof_shap_sum': np.arange(n_train * len(features), dtype=np.float32).reshape(n_train, -1) + offset,
            'insample_shap_sum': np.ones((n_train, len(features)), np.float32) * offset,
            'all_groups_shap_sum': np.ones((n_all, len(features)), np.float32) * offset,
            'oof_margin_sum': np.arange(n_train, dtype=float) + offset,
            'insample_margin_sum': np.arange(n_train, dtype=float) - offset,
            'all_groups_margin_sum': np.arange(n_all, dtype=float) + offset,
            'test_pred_count': np.ones(n_train, np.int32),
            'train_pred_count': np.ones(n_train, np.int32) * 2, 'all_groups_model_count': 3}


def test_streamed_archive_matches_existing_layout_and_values(tmp_path):
    from autobehaver.postinf.featureimp.archive_io import TaskArchiveWriter
    from autobehaver.postinf.featureimp.shard_runner import _shap_task_record_from_result, _margin_task_record_from_result
    from autobehaver.postinf.featureimp.shap_tasks import save_shap_task_archive, load_shap_task_archive
    from autobehaver.postinf.featureimp.margin_tasks import save_margin_task_archive, load_margin_task_archive
    results = [_task_result(2, 1, ('b',), 2), _task_result(1, 0, ('a', 'b'), 1)]
    lookup = {'a': 0, 'b': 1}
    writer = TaskArchiveWriter(tmp_path / 'scratch', {(r['h_hit'], r['subset_idx']): r['features'] for r in results}, n_train=4, n_all=6)
    try:
        for result in results:writer.append(result, lookup)
        writer.publish(margin_path=tmp_path / 'margin.npz', shap_path=tmp_path / 'shap.npz', metadata={'family': 'real_tasks', 'permutation_idx': -1, 'shard_index': 0})
    finally:writer.close()
    save_shap_task_archive(str(tmp_path / 'old_shap.npz'), task_records=[_shap_task_record_from_result(r,feature_to_idx=lookup,include_insample=True,include_all_groups=True) for r in results], n_train=4,n_all=6,include_insample=True,include_all_groups=True)
    save_margin_task_archive(str(tmp_path / 'old_margin.npz'), task_records=[_margin_task_record_from_result(r) for r in results], n_train=4,n_all=6)
    for loader, name in [(load_shap_task_archive, 'shap'), (load_margin_task_archive, 'margin')]:
        new, old = loader(str(tmp_path / f'{name}.npz')), loader(str(tmp_path / f'old_{name}.npz'))
        for key in old:np.testing.assert_array_equal(new[key],old[key])
        assert isinstance(next(v for v in new.values() if v.ndim > 0),np.memmap)


def test_archive_rejects_duplicate_or_incomplete_tasks(tmp_path):
    from autobehaver.postinf.featureimp.archive_io import TaskArchiveWriter
    result = _task_result(1,0,('a',),1)
    writer = TaskArchiveWriter(tmp_path, {(1,0):('a',),(1,1):('a',)},n_train=4,n_all=6)
    try:
        writer.append(result,{'a':0})
        with pytest.raises(ValueError,match='identity'):writer.append(result,{'a':0})
        with pytest.raises(ValueError,match='coverage'):writer.publish(margin_path=tmp_path/'m.npz',shap_path=tmp_path/'s.npz',metadata={'family':'real_tasks'})
    finally:writer.close()


def test_compressed_legacy_archive_read_and_malformed_size(tmp_path):
    from autobehaver.postinf.featureimp.archive_io import load_npz_arrays
    path=tmp_path/'compressed.npz';expected=np.arange(12).reshape(3,4)
    np.savez_compressed(path,values=expected)
    np.testing.assert_array_equal(load_npz_arrays(path,['values'])['values'],expected)
    with pytest.raises(ValueError,match='missing'):load_npz_arrays(path,['absent'])


def test_persistent_pool_recycles_at_finished_group_and_surfaces_failure():
    from autobehaver.postinf.featureimp.bounded_pool import BoundedProcessRunner
    with BoundedProcessRunner(num_workers=2,recycle_after_tasks=2) as runner:
        first=list(runner.run(_pid,range(2),desc='tiny',stage_name='tiny',show_progress=False))
        executor=runner.executor
        second=list(runner.run(_pid,range(2),desc='tiny',stage_name='tiny',show_progress=False))
        assert sorted(v for _,v in first)==[0,1] and sorted(v for _,v in second)==[0,1]
        if runner.workers>1:
            assert executor is not None and runner.executor is None
        with pytest.raises(RuntimeError if runner.workers>1 else ValueError,match='deliberate worker failure'):
            list(runner.run(_fail_worker,[1],desc='tiny',stage_name='tiny',show_progress=False))


def test_prerank_numerical_cache_replots_without_permutations(tmp_path,monkeypatch):
    import gseapy
    from autobehaver.postinf.featureimp.fsea_cache import cached_prerank
    names=[f'f{i}' for i in range(30)]
    ranking=pd.DataFrame({'feature_name':names,'rank_value':np.linspace(3,-3,30)})
    kwargs={'outdir':str(tmp_path/'fsea'),'threads':1,'permutation_num':25,'min_size':3,'max_size':20,'seed':7,'no_plot':False,'verbose':False}
    first=cached_prerank(gseapy,rnk=ranking.copy(),gene_sets={'top':names[:6],'bottom':names[-6:]},**kwargs)
    monkeypatch.setattr(gseapy,'prerank',lambda **kwargs:(_ for _ in ()).throw(AssertionError('permutations reran')))
    second=cached_prerank(gseapy,rnk=ranking.copy(),gene_sets={'top':names[:6],'bottom':names[-6:]},**{**kwargs,'graph_num':1})
    pd.testing.assert_frame_equal(first.res2d,second.res2d,check_dtype=False)
    pd.testing.assert_series_equal(first.ranking,second.ranking)
    assert second.results==first.results
    with pytest.raises(AssertionError,match='reran'):
        cached_prerank(gseapy,rnk=ranking.assign(rank_value=ranking.rank_value+0.1),gene_sets={'top':names[:6],'bottom':names[-6:]},**kwargs)


def test_native_small_permutation_stream_and_resume(tmp_path,monkeypatch):
    from autobehaver.postinf.featureimp import core, shard_runner
    from autobehaver.postinf.featureimp.plan import plan_featureimp_run
    from autobehaver.postinf.featureimp.matrix import normalize_feature_matrix
    from autobehaver.postinf.featureimp.shap_tasks import load_shap_task_archive
    old=core.get_current_settings()
    try:
        rng=np.random.default_rng(18);n=24
        raw=tmp_path/'raw.csv'
        frame=pd.DataFrame({'individual':[f'fly{i}' for i in range(n)],'group':['A']*12+['B']*12,'sex':['M','F']*12})
        for j in range(4):frame[f'KinematicBB_feature{j}']=np.r_[rng.normal(-3,.5,12),rng.normal(3,.5,12)]
        frame.to_csv(raw,index=False)
        settings={'training_group_a':'A','training_group_b':'B','control_group_tag':'A','n_estimators':12,
                  'learning_rate':.3,'max_depth':2,'num_workers':2,'run_fsea_analysis':False,'run_pca_analysis':False,
                  'run_permutation_test':True,'n_permutations':1,'GLOBAL_RANDOM_SEED':0}
        core.apply_settings(settings)
        (tmp_path/'FEATUREIMP_RUN_SETTINGS.json').write_text(json.dumps(core.get_current_settings(),default=str))
        normalized=normalize_feature_matrix(str(tmp_path),str(raw),analysis_folder_path=str(tmp_path),control_group_tag='A')
        plan_featureimp_run(normalized,str(tmp_path),analysis_folder_path=str(tmp_path),control_group_tag='A',
            training_group_a='A',training_group_b='B',n_splits=2,H_hits_per_feature=2,n_features_per_model=2,
            run_permutation_test=True,n_permutations=1,num_shards=1,GLOBAL_RANDOM_SEED=0)
        output=Path(shard_runner.process_featureimp_shard_run(str(tmp_path),shard_index=0))
        archives={name:(output/name).read_bytes() for name in ('REAL_SHAP_TASKS.npz','PERM_SHAP_TASKS_perm000.npz')}
        for name in archives:
            payload=load_shap_task_archive(str(output/name))
            assert len(payload['h_hit'])==4
            assert np.isfinite(payload['oof_shap_flat']).all()
        monkeypatch.setattr(shard_runner,'_train_subset_worker',lambda task:(_ for _ in ()).throw(AssertionError('training reran')))
        shard_runner.process_featureimp_shard_run(str(tmp_path),shard_index=0)
        assert all((output/name).read_bytes()==data for name,data in archives.items())
    finally:core.apply_settings(old)


def test_full_fsea_reuse_tamper_and_failed_plot_retry(tmp_path,monkeypatch):
    import gseapy
    from autobehaver.postinf.featureimp import fsea_runner as runner
    from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings
    names=[f'f{i}' for i in range(30)]
    groups={'top':names[:6],'bottom':names[-6:]}
    monkeypatch.setattr(runner,'_prepare_validated_fsea_groups',lambda **kwargs:groups)
    pd.DataFrame({'feature_name':names,'Global_Importance':np.linspace(3,.1,30)}).to_csv(tmp_path/'GLOBAL_IMPORTANCE_RANKING.csv',index=False)
    frame=pd.DataFrame({'individual':['a1','a2','b1','b2'],'group':['A','A','B','B'],'sex':['M','F','M','F']})
    values=np.tile(np.linspace(3,-3,30),(4,1));values[:2]*=-1
    settings=FSEARuntimeSettings(num_workers=1,fsea_min_size=3,fsea_max_size=20,fsea_permutations=25,
                                 fsea_fdr_threshold=0,fsea_viz_top_n=0)
    def run():
        return runner._perform_fsea_analysis(values,frame,names,'A','B',frame,{'analysis_folder':str(tmp_path)},None,2,
                                            all_groups_avg_shap_values=values,settings=settings)
    first=run();assert first['plots_complete']
    original_viz=runner._orchestrate_fsea_visualization
    monkeypatch.setattr(gseapy,'prerank',lambda **kwargs:(_ for _ in ()).throw(AssertionError('permutations reran')))
    outside=tmp_path/'outside.txt';outside.write_text('outside the FSEA outputs')
    unsafe={**first,'artifact_sha256':{**first['artifact_sha256'],'../outside.txt':runner.sha256_file(outside)}}
    (tmp_path/'FSEA'/'FSEA_COMPLETE.json').write_text(json.dumps(unsafe))
    recovered=run();assert recovered['plots_complete']
    assert '../outside.txt' not in recovered['artifact_sha256']
    (tmp_path/'FSEA'/'FSEA_COMPLETE.json').write_text('malformed completion receipt')
    recovered=run();assert recovered['plots_complete']
    monkeypatch.setattr(runner,'_orchestrate_fsea_visualization',lambda **kwargs:(_ for _ in ()).throw(RuntimeError('requested plot failed')))
    assert run()==recovered
    table=tmp_path/'FSEA'/'FSEA_RESULTS_A.xlsx';table.write_bytes(table.read_bytes()+b'changed')
    retry=run();assert not retry['plots_complete'] and retry['visualization_failures']
    monkeypatch.setattr(runner,'_orchestrate_fsea_visualization',original_viz)
    assert run()['plots_complete']
