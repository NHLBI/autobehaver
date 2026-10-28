"""Tiny native contracts for complete masters, identity and current inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file, atomic_write_json
from autobehaver.postinf.mmgen.feature_schema import complete_master_schema, feature_columns, schema_metadata
from autobehaver.postinf.mmgen.feature_matrix import MMGenFinalizationSources, create_feature_matrix_from_sources
from autobehaver.postinf.mmgen.worker_io import load_inference_and_poscalc
from autobehaver.postinf.mmgen.integrity import source_identities
from autobehaver.postinf.featureimp.matrix import validate_feature_matrix, normalize_feature_matrix


def test_full_master_schema_keeps_missing_and_actual_zero(tmp_path):
    bouts=pd.DataFrame([dict(individual='fly1',group='G',sex='FEMALE',category='Pose',
        bout_type='Walking',n_bouts=0,total_length_frames=0,bout_frequency=0.,bout_duration_mean_s=np.nan)])
    result=create_feature_matrix_from_sources(str(tmp_path),MMGenFinalizationSources(bout_summary_df=bouts),filter_features_by_bout_count=False)
    assert list(result.columns)==['individual','group','sex']+list(feature_columns())
    assert result.loc[0,'BoutSummary_Pose_Walking_bout_frequency']==0
    assert pd.isna(result.loc[0,'BoutSummary_Pose_Walking_bout_duration_mean_s'])
    assert schema_metadata()['feature_count']==9406
    assert json.loads((tmp_path/'MASTER_FEATURE_MATRIX_SCHEMA.json').read_text())['feature_count']==9406


def test_historical_alignment_preserves_values_and_rejects_unknown():
    old=pd.DataFrame(dict(individual=['a'],group=['G'],sex=['FEMALE'],TimeBudget_Pose_Walking=[.3]))
    aligned=complete_master_schema(old)
    assert aligned.loc[0,'TimeBudget_Pose_Walking']==.3
    assert pd.isna(aligned.loc[0,'TimeBudget_Pose_Standing'])
    with pytest.raises(ValueError,match='undeclared'):complete_master_schema(old.assign(misspelled_feature=1.))


def pair(tmp_path, explicit=False):
    inf=tmp_path/'inf';pos=tmp_path/'pos';inf.mkdir();pos.mkdir()
    name='G_FEMALE_trial1_rpi-1_movie0_chamber1_positional_calcs_normalized_inference.csv'
    ip=inf/name;pp=pos/name.replace('_normalized_inference','')
    data={'Frame Number':[1,2,3],'Smoothed Pose Label':['Walking']*3}
    if explicit:data['Source Frame Index']=[0,1,2]
    pd.DataFrame(data).to_csv(ip,index=False)
    pd.DataFrame({'centroid_x':[10.,20.,30.,40.,50.]}).to_csv(pp,index=False)
    return ip,pp


def test_predictions_reject_changed_positional_source(tmp_path):
    ip,pp=pair(tmp_path,True)
    atomic_write_json({'source_positional_sha256':sha256_file(pp)},Path(str(ip)+'.provenance.json'))
    assert len(load_inference_and_poscalc(str(ip),str(pp.parent),edge_skip=0,inf_tag='_normalized_inference')[0])==3
    changed=pd.read_csv(pp);changed.loc[0,'centroid_x']=999.;changed.to_csv(pp,index=False)
    with pytest.raises(ValueError,match='different processing'):load_inference_and_poscalc(str(ip),str(pp.parent),edge_skip=0,inf_tag='_normalized_inference')


@pytest.mark.parametrize('resmoothed',[False,True])
def test_native_prediction_receipt_checks_selected_csv(tmp_path,resmoothed):
    from autobehaver.postinf.mmgen.worker_io import prediction_provenance
    ip,pp=pair(tmp_path,True)
    output={'path':ip.name,'sha256':sha256_file(ip)}
    info={'status':'complete','source_positional_sha256':sha256_file(pp)}
    if resmoothed:
        info.update(operation='corrected_label_smoothing',output=output)
    else:
        info.update(schema='autobehaver.posepred_inference.provenance.v1',outputs=[output])
    side=Path(str(ip)+'.provenance.json')
    atomic_write_json(info,side)
    assert prediction_provenance(ip)==info
    assert len(load_inference_and_poscalc(str(ip),str(pp.parent),edge_skip=0,inf_tag='_normalized_inference')[0])==3
    info['status']='incomplete';atomic_write_json(info,side)
    with pytest.raises(ValueError,match='no completed'):prediction_provenance(ip)
    info['status']='complete';output['path']='another.csv';atomic_write_json(info,side)
    with pytest.raises(ValueError,match='exactly once'):prediction_provenance(ip)
    output['path']=ip.name;atomic_write_json(info,side)
    ip.write_text(ip.read_text().replace('Walking','Standing'))
    with pytest.raises(ValueError,match='does not match'):prediction_provenance(ip)


def test_legacy_offset_changes_reuse_identity(tmp_path):
    ip,pp=pair(tmp_path)
    side=Path(str(ip)+'.provenance.json')
    def identity():return source_identities([ip.name],inference_csv_dir=ip.parent,poscalc_dir=pp.parent,inf_tag='_normalized_inference')
    atomic_write_json({'source_frame_offset':0},side);first=identity()
    atomic_write_json({'source_frame_offset':1},side);second=identity()
    assert first!=second
    assert second[ip.name]['effective_source_frame_offset']==1
    assert load_inference_and_poscalc(str(ip),str(pp.parent),edge_skip=0,inf_tag='_normalized_inference')[2].centroid_x.tolist()==[20.,30.,40.]


def test_compact_names_keep_identity_shards_and_append_partner(tmp_path):
    from autobehaver.portable_paths import record_original_name
    from autobehaver.postinf.mmgen.worker_identity import resolve_group_individual_sex
    from autobehaver.postinf.mmgen.sharding import select_files_for_shard
    from autobehaver.postinf.mmgen.mapping import create_animal_mapping
    from autobehaver.postinf.mmgen.append_mode import _build_current_inventory, _filter_inventory_by_substring
    inf=tmp_path/'i';pos=tmp_path/'p';out=tmp_path/'o'
    for folder in (inf,pos,out):folder.mkdir()
    names=[]
    for chamber in (5,6):
        ip=inf/f'short{chamber}_inference.csv';pp=pos/f'short{chamber}_positional_calcs.csv'
        original=f'FULL_GROUP_FEMALE_trial1_rpi-10_movie0_chamber{chamber}_positional_calcs_normalized_inference.csv'
        ip.write_text('Frame Number,Smoothed Pose Label\n1,Walking\n')
        pp.write_text('centroid_x\n1\n')
        record_original_name(ip,original)
        record_original_name(pp,original.replace('_normalized_inference',''))
        names.append(ip.name)
        group,individual,sex=resolve_group_individual_sex(str(ip),None)
        assert (group,sex)==('FULL_GROUP','FEMALE') and f'chamber{chamber}' in individual
    mapping,_=create_animal_mapping(str(inf),str(pos),str(out))
    assert len(mapping)==2
    shards=[select_files_for_shard(names,num_shards=2,shard_index=i,strategy='row_identity',inference_csv_dir=str(inf)) for i in (0,1)]
    assert [len(s) for s in shards]==[1,1]
    inv=_build_current_inventory(inference_csv_dir=str(inf),poscalc_dir=str(pos),inf_tag='_normalized_inference')
    assert inv.poscalc_exists.all() and set(inv.group)=={'FULL_GROUP'}
    assert len(_filter_inventory_by_substring(inv,'FULL_GROUP'))==2


@pytest.mark.parametrize('value',['bad',np.inf])
def test_featureimp_identifies_invalid_numeric_feature(tmp_path,value):
    matrix=tmp_path/'m.csv'
    pd.DataFrame(dict(individual=['a','b'],group=['CONTROL','B'],sex=['FEMALE']*2,
                     KinematicBB_PoseOrient_Walking_speed_mean=[1.,value])).to_csv(matrix,index=False)
    valid,error,_=validate_feature_matrix(str(matrix))
    assert not valid and 'KinematicBB_PoseOrient_Walking_speed_mean' in error


def test_featureimp_rejects_existing_bad_mmgen_receipt(tmp_path):
    matrix=tmp_path/'MASTER_FEATURE_MATRIX.csv'
    pd.DataFrame(dict(individual=['a','b'],group=['CONTROL','B'],sex=['FEMALE']*2,Global_MeanSpeed_mm_s=[1.,2.])).to_csv(matrix,index=False)
    assert validate_feature_matrix(str(matrix))[0]
    atomic_write_json({'artifacts':{matrix.name:'0'*64}},tmp_path/'MMGEN_OUTPUT_RECEIPT.json')
    assert not validate_feature_matrix(str(matrix))[0]
    atomic_write_json({'artifacts':{matrix.name:sha256_file(matrix)}},tmp_path/'MMGEN_OUTPUT_RECEIPT.json')
    assert validate_feature_matrix(str(matrix))[0]


def test_featureimp_accepts_separately_normalized_csv_beside_master_receipt(tmp_path):
    from autobehaver.postinf.featureimp.matrix_input import verify_upstream_receipt
    master=tmp_path/'MASTER_FEATURE_MATRIX.csv'
    normalized=tmp_path/'NORMALIZED_FEATURE_MATRIX.csv'
    data=pd.DataFrame(dict(individual=['a','b'],group=['CONTROL','B'],sex=['FEMALE']*2,
                           Global_MeanSpeed_mm_s=[1.,2.]))
    data.to_csv(master,index=False)
    data.to_csv(normalized,index=False)
    receipt=tmp_path/'MMGEN_OUTPUT_RECEIPT.json'
    atomic_write_json({'artifacts':{master.name:sha256_file(master)}},receipt)
    assert validate_feature_matrix(str(normalized))[0]
    assert verify_upstream_receipt(normalized)['input_kind']=='standalone_matrix'
    assert verify_upstream_receipt(master)['input_kind']=='verified_mmgen_matrix'
    atomic_write_json({'artifacts':{}},receipt)
    with pytest.raises(ValueError,match='does not certify'):
        verify_upstream_receipt(master)


def test_empty_placeholder_does_not_poison_composition_or_change_master(tmp_path):
    master=tmp_path/'m.csv';out=tmp_path/'out';out.mkdir()
    pd.DataFrame(dict(individual=['a','b','c','d'],group=['CONTROL']*3+['B'],sex=['FEMALE']*4,
        TimeBudget_Pose_Walking=[.1,.2,.4,.6],TimeBudget_Pose_Standing=[.9,.8,.6,.4],
        TimeBudget_Pose_Jump=[np.nan]*4,Global_MeanSpeed_mm_s=[1.,2.,3.,4.])).to_csv(master,index=False)
    digest=sha256_file(master)
    result=normalize_feature_matrix(str(out),str(master),control_group_tag='CONTROL')
    normalized=pd.read_csv(result)
    assert normalized.TimeBudget_Pose_Walking.notna().all()
    assert 'TimeBudget_Pose_Jump' not in normalized
    assert sha256_file(master)==digest
    parameters=json.loads((out/'NORMALIZATION_PARAMS.json').read_text())
    assert parameters['globally_empty_features_removed_before_normalization']==['TimeBudget_Pose_Jump']


def test_native_sharded_append_preserves_unselected_animal(tmp_path):
    from autobehaver.postinf.mmgen.core import MMGenConfig
    from autobehaver.postinf.mmgen import pipeline, append_mode
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    inf=tmp_path/'i';pos=tmp_path/'p';out=tmp_path/'o'
    for folder in (inf,pos,out):folder.mkdir()
    def add(chamber):
        name=f'G_FEMALE_trial1_rpi-1_movie0_chamber{chamber}_positional_calcs.csv'
        n=200
        values={c:np.full(n,.99 if c.endswith('_likelihood') else 1.)
                for c in required_keypre_columns(include_postural=False,include_spatial=False,include_measurements=False)}
        pd.DataFrame(values).to_csv(pos/name,index=False)
        pd.DataFrame({'Frame Number':np.arange(1,n+1),'Source Frame Index':np.arange(n),
                      'Smoothed Pose Label':['Standing']*n,'Smoothed Orient Label':['Floor']*n}).to_csv(inf/name.replace('.csv','_normalized_inference.csv'),index=False)
    cfg=MMGenConfig(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(out),edge_skip=0,
        num_workers=1,make_measurements=False,calculate_postural_features=False,calculate_occupancy=False,
        emit_full_outputs=False,keep_temp_files=True,input_batch_size=1)
    add(1)
    pipeline.run_post_inference_analysis(str(inf),str(pos),str(out),cfg.min_bout,cfg.edge_skip,cfg.inf_tag,runtime_settings=cfg.runtime_settings)
    before=pd.read_csv(out/'MASTER_FEATURE_MATRIX.csv')
    add(2);add(3)
    cfg.append_new_data=True;cfg.num_shards=2;cfg.shard_by='row_identity';cfg.append_batch_id='tiny_append'
    plan,selected,_,batch=append_mode._create_append_plan(cfg)
    assert len(selected)==2
    for index in (0,1):
        cfg.shard_index=index
        append_mode._run_append_shard_from_plan(cfg=cfg,plan=plan,selected_files_df=selected,batch_dir=batch)
    append_mode._merge_finalize_append_batch(cfg=cfg,plan=plan,batch_dir=batch)
    after=pd.read_csv(out/'MASTER_FEATURE_MATRIX.csv')
    assert len(after)==3 and list(after.columns)==list(before.columns)
    pd.testing.assert_frame_equal(after.loc[after.individual.eq(before.individual.iloc[0])].reset_index(drop=True),before)
    append_mode._verified_generation(out)
