"""Short names must work through optional registry and overlay entrypoints."""
from pathlib import Path
import json
import pandas as pd
import pytest


def test_keypre_rejects_two_scorers_for_one_recording_before_work(tmp_path):
    from autobehaver.preinf.keypre.core import run_keypre
    from autobehaver.preinf.keypre.config import KeypreConfig
    for name in ('recordDLC_first.csv','recordDLC_second.csv'):
        (tmp_path/name).write_text('filename fixture')
    with pytest.raises(ValueError,match='same recording output'):
        run_keypre(cfg=KeypreConfig(input_dir=str(tmp_path),output_dir=str(tmp_path/'out'),dry_run=True))


def test_compact_movie_and_prediction_pair_through_folder_entrypoint(tmp_path):
    from autobehaver.portable_paths import generated_path,prediction_path
    from autobehaver.postinf.infmovie.pipeline import discover_pairs_from_folders
    movies=tmp_path/'movies';pred=tmp_path/'pred'
    movies.mkdir();pred.mkdir()
    name='GENOTYPE'+('long'*20)+'_MALE_exp_trial2_rpi-10_movie0_20261009115942AM_chamber1.mp4'
    movie=generated_path(movies,name);movie.write_bytes(b'filename fixture')
    norm=tmp_path/(Path(name).stem+'_positional_calcs_normalized.h5')
    output=prediction_path(norm,pred)
    output.write_text('Frame Number\n1\n')
    pairs=discover_pairs_from_folders(inference_csv_dir=pred,movie_dir=movies,file_filter=None)
    assert len(pairs)==1 and pairs[0]['movie_path']==str(movie.resolve())


def test_registry_updates_follow_compact_stage_names(tmp_path):
    from autobehaver.portable_paths import generated_path,normalized_path,prediction_path
    from autobehaver.registry import upsert_registry_rows,read_registry,relpath_str
    from autobehaver.preinf.keypre.contract import output_path
    from autobehaver.preinf.dlc.registry_update import _best_csv_for_video
    from autobehaver.preinf.keypre.registry_update import update_registry_from_keypre_outputs
    from autobehaver.preinf.keyprenorm.registry_update import update_registry_from_keyprenorm_outputs
    from autobehaver.preinf.posepred_inference.registry_update import update_registry_from_posepred_inference_outputs
    folders=[tmp_path/name for name in ('movies','dlc','keypre','norm','pred')]
    for p in folders:p.mkdir()
    movie_name='GENOTYPE'+('long'*20)+'_MALE_exp_trial2_rpi-10_movie0_20261009115942AM_chamber1.mp4'
    movie=generated_path(folders[0],movie_name);movie.write_bytes(b'fixture')
    dlc_original=Path(movie_name).stem+'DLC_very_long_original_scorer.csv'
    dlc=generated_path(folders[1],Path(movie).stem+'DLC_short.csv',original=dlc_original)
    dlc.write_text('x\n1\n')
    assert _best_csv_for_video(video_path=movie,search_dir=folders[1])==dlc
    keypre=output_path(dlc,folders[2]);keypre.write_text('x\n1\n')
    norm=normalized_path(keypre,folders[3]);norm.write_bytes(b'fixture')
    pred=prediction_path(norm,folders[4]);pred.write_text('Frame Number\n1\n')
    registry=tmp_path/'registry.csv'
    upsert_registry_rows(csv_path=registry,key='recording_id',rows=[{'recording_id':'record1','active':True,
        'cropped_movie_relpath':relpath_str(movie,tmp_path),'dlc_csv_relpath':relpath_str(dlc,tmp_path)}])
    update_registry_from_keypre_outputs(registry_csv=registry,project_root=tmp_path,dlc_csv_files=[dlc],output_dir=folders[2],keypre_version='fixture')
    params=tmp_path/'norm_params.csv';params.write_text('Header,Mean,StdDev\nx,0,1\n')
    update_registry_from_keyprenorm_outputs(registry_csv=registry,project_root=tmp_path,keypre_csv_files=[keypre],normalized_h5_dir=folders[3],norm_params_path=params)
    update_registry_from_posepred_inference_outputs(registry_csv=registry,project_root=tmp_path,normalized_h5_files=[norm],output_root=folders[4],use_inferences_subdir=False,model_paths=[])
    row=read_registry(registry).iloc[0]
    assert row['keypre_csv_relpath']==relpath_str(keypre,tmp_path)
    assert row['keyprenorm_h5_relpath']==relpath_str(norm,tmp_path)
    assert row['posepred_inference_csv_relpath']==relpath_str(pred,tmp_path)
