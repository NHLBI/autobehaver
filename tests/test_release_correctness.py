"""Tiny end-to-end and corruption checks for the integrated release."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import pytest


def test_completion_distinguishes_data_but_ignores_comments_and_relocation(tmp_path):
    from autobehaver.completion import request_identity,receipt_matches,write_receipt
    source=tmp_path/'source.csv'; source.write_text('x\n1\n')
    code=tmp_path/'math.py'; code.write_text('"""documentation"""\nx=1 # original\n')
    output=tmp_path/'output.csv'; output.write_text('x\n2\n')
    identity=request_identity('test',[source],{'scale':2},[code])
    write_receipt(output,identity)
    moved=tmp_path/'moved.csv'; moved.write_bytes(source.read_bytes())
    code.write_text('"""updated documentation"""\nx = 1 # updated comment\n')
    assert receipt_matches(output,request_identity('test',[moved],{'scale':2},[code]))
    source.write_text('x\n3\n')
    assert not receipt_matches(output,request_identity('test',[source],{'scale':2},[code]))
    output.write_text('x\n999\n')
    assert not receipt_matches(output,identity)


def test_keypre_corrupt_existing_is_regenerated_then_reused(tmp_path):
    from tests.test_keypre_raw_only import _write_synthetic_dlc_csv
    from autobehaver.preinf.keypre.worker import set_run_config,process_csv_file
    from autobehaver.preinf.keypre.contract import output_path,inspect_keypre_output
    source=tmp_path/'G_MALE_exp_trial1_rpi-1_movie0_chamber1DLC_longscorer.csv'
    _write_synthetic_dlc_csv(str(source),n_frames=60)
    output=output_path(source,tmp_path)
    pd.DataFrame({'bogus':np.ones(60)}).to_csv(output,index=False)
    set_run_config({'output_dir':str(tmp_path),'clobber':False,'movie_dim':600,'raw_only':False})
    assert process_csv_file(str(source))['status']=='success'
    assert inspect_keypre_output(str(source),str(output),require_receipt=True)[0]
    assert list((tmp_path/'_old').glob('*.csv'))
    assert process_csv_file(str(source))['status']=='skipped'
    set_run_config({'output_dir':str(tmp_path),'clobber':False,'movie_dim':700,'raw_only':False})
    assert process_csv_file(str(source))['status']=='success'


def test_small_model_preserves_original_frames_through_outputs(tmp_path):
    from tests.test_posepred_checkpoint_inference_compat import _set_training_globals,_make_supervised_checkpoint
    from autobehaver.checkpoint import build_checkpoint_global_vars
    from autobehaver.preinf.frame_identity import retained_frame_indices,write_h5_frame_timeline
    from autobehaver.preinf.posepred_inference.core import run_posepred_inference
    from autobehaver.preinf.posepred_inference.config import PosePredInferenceConfig
    from autobehaver.postinf.mmgen.worker_io import _align_poscalc_to_inference
    from autobehaver.postinf.infmovie.frame_map import load_source_frames
    source_dir=tmp_path/'norm';source_dir.mkdir()
    source=source_dir/'G_MALE_trial1_rpi-1_movie0_chamber1_normalized.h5'
    indices=retained_frame_indices(60,5)
    with h5py.File(source,'w') as hf:
        hf['data']=np.column_stack((indices,indices/2)).astype(np.float32)
        hf.attrs['columns']=np.array(['x','y'],dtype='S')
        write_h5_frame_timeline(hf,frame_count=60,indices=indices)
    _set_training_globals(canonical=['x','y'],included=['x','y'],columns_to_noise=[])
    variables=build_checkpoint_global_vars(pose_pred=True,orient_pred=True,recon=False)
    model=tmp_path/'tiny.pth'
    _make_supervised_checkpoint(model,global_vars=variables,pose_dict={'Rest':0,'Walk':1},orient_dict={'Floor':0,'Wall':1})
    output=tmp_path/'pred'
    cfg=PosePredInferenceConfig(input_dir=str(source_dir),output_dir=str(output),inference_model=[str(model)],
        num_workers=0,batch=16,inf_average=1,min_pose_bout=1,min_orient_bout=1)
    run_posepred_inference(cfg)
    csv=next(output.glob('*_inference.csv'))
    frame=pd.read_csv(csv)
    np.testing.assert_array_equal(frame['Source Frame Index'],np.arange(60))
    populated=np.flatnonzero(frame['Pose-Rest-Prob'].notna())
    np.testing.assert_array_equal(populated,np.arange(7,53))
    _,pos=_align_poscalc_to_inference(frame,pd.DataFrame({'marker':np.arange(60)}),edge_skip=5,source_name='test')
    np.testing.assert_array_equal(pos.marker,np.arange(60))
    np.testing.assert_array_equal(load_source_frames(csv),np.arange(60))
    context=next(output.rglob('*_cv.h5'))
    with h5py.File(context) as hf:
        assert hf['context_vector'].shape[0]==60
        assert np.isfinite(hf['context_vector'][7:53]).all()
    before=csv.stat().st_mtime_ns
    run_posepred_inference(cfg)
    assert csv.stat().st_mtime_ns==before


def test_prediction_heads_are_bound_to_requested_vocabulary(tmp_path):
    from autobehaver.preinf.posepred_inference.core import _validate_inference_csv
    source=tmp_path/'source.h5'
    with h5py.File(source,'w') as hf: hf['data']=np.ones((3,2))
    csv=tmp_path/'pred.csv'
    pd.DataFrame({'Frame Number':[1,2,3],'Pose-Walk-Prob':[1,1,1],'Predicted Pose Label':['Walk']*3}).to_csv(csv,index=False)
    assert not _validate_inference_csv(str(source),csv,{'Pose':['Rest','Walk']})[0]
    assert not _validate_inference_csv(str(source),csv,{'Pose':['Walk'],'Orient':['Floor']})[0]
    assert _validate_inference_csv(str(source),csv,{'Pose':['Walk']})[0]


def test_fractional_original_frame_metadata_is_rejected(tmp_path):
    from autobehaver.preinf.frame_identity import h5_frame_timeline
    source=tmp_path/'bad.h5'
    with h5py.File(source,'w') as hf:
        hf['data']=np.zeros((2,1))
        hf['source_frame_indices']=[5.5,6.5]
        hf.attrs['original_frame_count']=10
        with pytest.raises(ValueError,match='integer'):
            h5_frame_timeline(hf)


def test_portable_names_keep_biological_identity_and_pairing(tmp_path):
    from autobehaver.portable_paths import generated_path,original_name
    from autobehaver.postinf.mmgen.naming import parse_group_and_individual
    from autobehaver.postinf.mmgen.io import find_matching_poscalc_file
    name='Genotype'+('long'*20)+'_MALE_exp_trial12_rpi-10_movie2_chamber6_positional_calcs.csv'
    posdir=tmp_path/'pos';predir=tmp_path/'pred'
    posdir.mkdir();predir.mkdir()
    pos=generated_path(posdir,name);pos.write_text('x\n1\n')
    predname=name.replace('.csv','_normalized_inference.csv')
    pred=generated_path(predir,predname,reserve=40);pred.write_text('x\n1\n')
    assert original_name(pos)==name
    assert parse_group_and_individual(str(pred))[1].endswith('rpi-10_trial12_movie2_chamber6')
    assert find_matching_poscalc_file(str(pred),str(posdir),'_normalized_inference')==str(pos)
    assert len(str(pred))+40<=240


def test_portable_export_long_tree_is_lossless(tmp_path):
    from autobehaver.portable_export import export_results
    from autobehaver.completion import native_path
    source=tmp_path/'source'
    folder=source/('nested'*15)/('deeper'*15)
    Path(native_path(folder)).mkdir(parents=True)
    file=folder/(('measurement_'*10)+'.csv')
    with open(native_path(file),'w') as stream: stream.write('x\n3\n')
    exported=tmp_path/'export'
    records=export_results(source,exported)
    assert len(records)==1
    assert (exported/records[0]['portable_path']).read_text()=='x\n3\n'
    assert records[0]['original_path']==file.relative_to(source).as_posix()


def test_shard_advice_has_nonempty_hard_limit_and_groups_fast_tasks():
    from autobehaver.shard_advice import recommend_shards
    advice=recommend_shards(100,workers_per_shard=4,observed_seconds_per_task=1,target_seconds_per_shard=900)
    assert advice['hard_maximum']==100
    assert advice['recommended_shards']==1


