"""Independent conservation, interrupted-write and native pipeline checks."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file, atomic_write_json
from autobehaver.postinf.mmgen.hdf_store import HDF_KEYS, hdf_safe_append, open_store, purge_file_from_store
from autobehaver.postinf.mmgen.integrity import (empty_inventory, validate_commits, content_hash,
    calculation_identity, bind_run_identity, source_identities, frame_fingerprint, commit_fingerprint, IDENTITY_FILENAME)
from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings


def identity():
    calc={'reference':'independent-test'}
    sources={f'{n}.csv':{'inference_sha256':str(n),'poscalc_sha256':str(n), 'original_name':f'{n}.csv'} for n in 'ab'}
    return {'calculation':calc,'calculation_sha256':content_hash(calc),'sources':sources,'sources_sha256':content_hash(sources)}


def write_store(path, files=('a','b')):
    payload=identity()
    with open_store(str(path),'w') as store:
        for n in files:
            frame=pd.DataFrame({'source_file':[n+'.csv']*3,'value':[1.,2.,3.]})
            hdf_safe_append(store,HDF_KEYS['postural'],frame)
            inventory=empty_inventory();inventory[HDF_KEYS['postural']]=3
            row = {
                'source_file':n+'.csv','group':'G','individual':n,'sex':'FEMALE',
                'n_frames':3,'calculation_sha256':payload['calculation_sha256'],
                'input_sha256':content_hash(payload['sources'][n+'.csv']),
                'output_inventory':json.dumps(inventory,sort_keys=True),
                'output_fingerprints':json.dumps({HDF_KEYS['postural']:frame_fingerprint(frame)},sort_keys=True)}
            row['commit_sha256'] = commit_fingerprint(row)
            hdf_safe_append(store,HDF_KEYS['video_lengths_tmp'],pd.DataFrame([row]))
    return payload


def test_commit_inventory_detects_deleted_rows_and_content_change(tmp_path):
    p=tmp_path/'x.h5';payload=write_store(p)
    assert validate_commits(p,payload)=={'a.csv','b.csv'}
    changed=dict(payload);changed['calculation_sha256']='different'
    with pytest.raises(RuntimeError,match='Changed calculation|identity contents changed'):validate_commits(p,changed)
    with open_store(str(p),'a') as store:
        store.remove(HDF_KEYS['postural'],where='source_file == "a.csv"',start=0,stop=1)
    with pytest.raises(RuntimeError,match='changed'):validate_commits(p,payload)


def test_commit_detects_same_count_value_edit(tmp_path):
    p=tmp_path/'x.h5';payload=write_store(p)
    with open_store(str(p),'a') as store:
        frame=store.select(HDF_KEYS['postural']);frame.iloc[0,1]=999.
        store.remove(HDF_KEYS['postural']);hdf_safe_append(store,HDF_KEYS['postural'],frame)
    with pytest.raises(RuntimeError,match='scientific cells changed'):validate_commits(p,payload)


@pytest.mark.parametrize('column,value', [('n_frames',999), ('individual','wrong-animal'),
                                         ('output_fingerprints','{}')])
def test_commit_metadata_and_fingerprint_omission_rejected(tmp_path,column,value):
    p=tmp_path/'x.h5';payload=write_store(p)
    with open_store(str(p),'a') as store:
        frame=store.select(HDF_KEYS['video_lengths_tmp']);frame.loc[0,column]=value
        store.remove(HDF_KEYS['video_lengths_tmp']);hdf_safe_append(store,HDF_KEYS['video_lengths_tmp'],frame)
    with pytest.raises(RuntimeError,match='metadata changed|Incomplete committed'):validate_commits(p,payload)


def test_finalization_rejects_orphans_but_resume_allows_selected_pending_rows(tmp_path):
    p=tmp_path/'x.h5';payload=write_store(p,('a',))
    with open_store(str(p),'a') as store:
        hdf_safe_append(store,HDF_KEYS['postural'],pd.DataFrame({'source_file':['b.csv'],'value':[999.]}))
    with pytest.raises(RuntimeError,match='Uncommitted'):validate_commits(p,payload)
    assert validate_commits(p,payload,allow_uncommitted=True)=={'a.csv'}
    with pytest.raises(RuntimeError,match='unselected'):validate_commits(p,payload,selected_files=['a.csv'],allow_uncommitted=True)


def test_empty_shard_has_same_calculation_contract(tmp_path):
    from autobehaver.postinf.mmgen import pipeline
    from autobehaver.postinf.mmgen.matrix_parts import finalize_shard_temp_store, _matrix_merge_consistency_signature
    inf,pos=synthetic_inputs(tmp_path)
    settings=MMGenRuntimeSettings(num_workers=1,make_measurements=False,calculate_occupancy=False,
        calculate_postural_features=False,keep_temp_files=True,emit_full_outputs=False)
    out=tmp_path/'empty'
    pipeline.run_post_inference_analysis(str(inf),str(pos),str(out),5,0,'_inference',runtime_settings=settings,
        selected_files=[],num_shards=2,shard_index=0,shard_by='row_identity',process_only=True)
    assert finalize_shard_temp_store(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(out),
        min_bout=5,edge_skip=0,inf_tag='_inference',runtime_settings=settings,file_filter=None,
        num_shards=2,shard_index=0,shard_by='row_identity') is None
    work=out/'_mmgen_shards/shard_000_of_002/_mmgen_work'
    marker=json.loads((work/'SHARD_MATRIX_DONE.json').read_text())
    assert marker['n_matrix_rows']==0 and marker['artifacts']=={}
    assert marker['calculation_sha256']==content_hash(calculation_identity(settings,min_bout=5,edge_skip=0,inf_tag='_inference'))


def test_purge_failure_preserves_entire_original_store(tmp_path):
    from autobehaver.postinf.mmgen import hdf_store
    p=tmp_path/'x.h5';write_store(p);before=p.read_bytes()
    original=hdf_store._rewrite_key_excluding_rows
    calls=[]
    def fail(*args,**kwargs):
        calls.append(kwargs['key'])
        if len(calls)==2:raise OSError('injected second-table write failure')
        return original(*args,**kwargs)
    with patch.object(hdf_store,'_rewrite_key_excluding_rows',side_effect=fail):
        with pytest.raises(OSError):purge_file_from_store(str(p),source_file='a.csv')
    assert p.read_bytes()==before
    purge_file_from_store(str(p),source_file='a.csv')
    with open_store(str(p),'r') as store:
        assert store.select(HDF_KEYS['postural']).source_file.tolist()==['b.csv']*3
        assert store.select(HDF_KEYS['video_lengths_tmp']).source_file.tolist()==['b.csv']


def test_shard_merge_failure_preserves_published_store(tmp_path):
    from autobehaver.postinf.mmgen import shard_merge
    from autobehaver.postinf.mmgen.cohort import bind_shard_cohort
    payload=identity()
    cohort = bind_shard_cohort(tmp_path / '_mmgen_shards', ['a.csv', 'b.csv'],
        inference_csv_dir=tmp_path, num_shards=2, shard_by='row_identity')
    for i,n in enumerate('ab'):
        directory=tmp_path/'_mmgen_shards'/f'shard_{i:03d}_of_002'
        work=directory/'_mmgen_work';work.mkdir(parents=True)
        store=directory/'temp_store.h5';write_store(store,(n,))
        item=dict(payload);item['sources']={n+'.csv':payload['sources'][n+'.csv']};item['sources_sha256']=content_hash(item['sources'])
        atomic_write_json(item,work/IDENTITY_FILENAME)
        atomic_write_json({'num_shards':2,'shard_index':i,'shard_by':'row_identity',
            'cohort_sha256':cohort['cohort_sha256'], 'selected_files':[n+'.csv'], 'n_selected_files':1},
            work/'SHARD_MANIFEST.json')
        atomic_write_json({'num_shards':2,'shard_index':i,'shard_by':'row_identity',
            'cohort_sha256':cohort['cohort_sha256'],'n_inference_files_selected':1,'n_committed_files':1,
            'calculation_sha256':item['calculation_sha256'],'sources_sha256':item['sources_sha256'],
            'temp_store_sha256':sha256_file(store)},work/'SHARD_DONE.json')
    published=tmp_path/'temp_store.h5';published.write_bytes(b'prior published generation')
    before=published.read_bytes()
    def failure(*args,**kwargs):
        raise OSError('injected completion-record append failure after scientific rows')
    with patch.object(shard_merge,'append_recording_commits',side_effect=failure):
        with pytest.raises(OSError):shard_merge.merge_shard_temp_stores(output_folder=str(tmp_path),num_shards=2)
    assert published.read_bytes()==before
    result=shard_merge.merge_shard_temp_stores(output_folder=str(tmp_path),num_shards=2)
    assert validate_commits(result,payload)=={'a.csv','b.csv'}


def synthetic_inputs(base):
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    inf=base/'inf';pos=base/'pos';inf.mkdir();pos.mkdir()
    n=200
    for movie in [0,1]:
        name=f'G_FEMALE_20260101_rpi-1_trial1_movie{movie}_chamber1_positional_calcs.csv'
        values={c:np.full(n,.99 if c.endswith('_likelihood') else 1.) for c in required_keypre_columns(include_postural=False,include_spatial=False,include_measurements=False)}
        pd.DataFrame(values).to_csv(pos/name,index=False)
        labels=(['Standing']*50+['Walking']*10+['Groom-Head']*50+['Walking']*90) if movie==0 else ['Standing']*n
        pd.DataFrame({'Frame Number':np.arange(1,n+1),'Smoothed Pose Label':labels,'Smoothed Orient Label':['Floor']*n}).to_csv(inf/name.replace('.csv','_inference.csv'),index=False)
    return inf,pos


def run_native(inf,pos,out,workers=1,**kwargs):
    from autobehaver.postinf.mmgen import pipeline
    settings=MMGenRuntimeSettings(num_workers=workers,input_batch_size=1,keep_temp_files=True,
        make_measurements=False,calculate_postural_features=False,calculate_occupancy=False,
        emit_full_outputs=False,make_detailed_transition_outputs=True,min_bout_count_for_features=1,
        min_bout_count_for_transitions=1,source_frame_offset=0)
    pipeline.run_post_inference_analysis(str(inf),str(pos),str(out),5,0,'_inference',runtime_settings=settings,**kwargs)
    return pd.read_csv(out/'MASTER_FEATURE_MATRIX.csv')


def test_native_serial_parallel_and_resume_preserve_matrix(tmp_path):
    inf,pos=synthetic_inputs(tmp_path)
    a=run_native(inf,pos,tmp_path/'serial',1)
    b=run_native(inf,pos,tmp_path/'parallel',2)
    pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-10,rtol=1e-10)
    c=run_native(inf,pos,tmp_path/'serial',1,finalize_only=True)
    pd.testing.assert_frame_equal(a,c,check_exact=False,atol=1e-10,rtol=1e-10)
    assert len(a)==1
    p=next(inf.glob('*.csv'));df=pd.read_csv(p);df.iloc[0,1]='Walking';df.to_csv(p,index=False)
    with pytest.raises(RuntimeError,match='changed'):run_native(inf,pos,tmp_path/'serial',1,resume=True)


def test_matrix_first_shards_equal_serial_and_detect_changed_part(tmp_path):
    from autobehaver.postinf.mmgen.matrix_parts import finalize_shard_temp_store, merge_matrix_part_files
    from autobehaver.postinf.mmgen.sharding import select_files_for_shard
    inf,pos=synthetic_inputs(tmp_path)
    # Give the second recording its own animal so row-identity shards partition.
    for directory in [inf,pos]:
        old=next(directory.glob('*movie1*'))
        old.rename(old.with_name(old.name.replace('trial1','trial2')))
    serial=run_native(inf,pos,tmp_path/'serial',1)
    out=tmp_path/'parts'
    settings=MMGenRuntimeSettings(num_workers=1,input_batch_size=1,keep_temp_files=True,
        make_measurements=False,calculate_postural_features=False,calculate_occupancy=False,
        emit_full_outputs=False,make_detailed_transition_outputs=True,min_bout_count_for_features=1,
        min_bout_count_for_transitions=1,source_frame_offset=0)
    from autobehaver.postinf.mmgen import pipeline
    for index in range(2):
        pipeline.run_post_inference_analysis(str(inf),str(pos),str(out),5,0,'_inference',runtime_settings=settings,
            process_only=True,num_shards=2,shard_index=index,shard_by='row_identity')
        finalize_shard_temp_store(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(out),
            min_bout=5,edge_skip=0,inf_tag='_inference',runtime_settings=settings,file_filter=None,
            num_shards=2,shard_index=index,shard_by='row_identity')
    merged=pd.read_csv(merge_matrix_part_files(output_dir=str(out),num_shards=2))
    pd.testing.assert_frame_equal(serial,merged,check_exact=False,atol=1e-10,rtol=1e-10)
    # A changed selected-count marker cannot hide a completed nonempty part.
    marker=next(out.glob('_mmgen_shards/*/_mmgen_work/SHARD_MATRIX_DONE.json'))
    original=marker.read_bytes();payload=json.loads(original);payload['n_selected_files']=0
    marker.write_text(json.dumps(payload))
    before=(out/'MASTER_FEATURE_MATRIX.csv').read_bytes()
    with pytest.raises(RuntimeError,match='selected-source count'):merge_matrix_part_files(output_dir=str(out),num_shards=2)
    assert (out/'MASTER_FEATURE_MATRIX.csv').read_bytes()==before
    marker.write_bytes(original)
    part=next(out.glob('_mmgen_shards/*/MASTER_FEATURE_MATRIX.csv'))
    before=(out/'MASTER_FEATURE_MATRIX.csv').read_bytes()
    part.write_text(part.read_text()+'\n',encoding='utf-8')
    with pytest.raises(RuntimeError,match='changed'):merge_matrix_part_files(output_dir=str(out),num_shards=2)
    assert (out/'MASTER_FEATURE_MATRIX.csv').read_bytes()==before
