"""Tiny native regressions for prediction, original-frame and overlay handoffs."""
from pathlib import Path
import csv
from dataclasses import replace
import inspect
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from autobehaver.atomic_io import sha256_file
from autobehaver.completion import receipt_matches
from autobehaver.inference import (
    _archive_selected_predictions, _build_inference_provenance,
    _discover_inference_input_files, run_inference,
)
from autobehaver.portable_paths import normalized_path, prediction_path, record_original_name
from autobehaver.preinf.keyprenorm import normalize
from autobehaver.preinf.posepred_inference.config import PosePredInferenceConfig
from autobehaver.preinf.posepred_inference.core import run_posepred_inference
from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows
from autobehaver.resmooth import resmooth_file
from autobehaver.postinf.infmovie.frame_map import load_source_frames
from autobehaver.postinf.infmovie.config import InfMoviePipelineConfig
from autobehaver.postinf.infmovie import pipeline
from tests.test_posepred_inference_audit import _make_supervised_checkpoint


def _native_inputs(base):
    base.mkdir(exist_ok=True)
    source = base / 'sample_positional_calcs.csv'
    pd.DataFrame({'x': np.arange(17), 'y': np.arange(17)*2}).to_csv(source, index=False)
    normalize.init_normalize_worker(['x', 'y'], np.zeros(2, np.float32), np.ones(2, np.float32),
                                   5, str(base), None, None, None, 'WARNING')
    result = normalize.normalize_one_csv(str(source))
    assert result['status'] == 'success', result
    normalized = normalized_path(source, base)
    checkpoint = base / 'model.pth'
    torch.manual_seed(3)
    torch.set_num_threads(1)
    _make_supervised_checkpoint(checkpoint, columns=['x', 'y'],
                               pose_dict={'Walking':0, 'Standing':1},
                               orient_dict={'Floor':0, 'Wall':1})
    from autobehaver.feature_selection import compute_schema_hash
    saved = torch.load(checkpoint, weights_only=False)
    saved['global_vars']['feature_columns_included'] = ['x', 'y']
    saved['global_vars']['feature_schema_hash'] = compute_schema_hash(['x', 'y'])
    torch.save(saved, checkpoint)
    return source, normalized, checkpoint


def test_native_predictions_forward_source_and_keep_unsmoothed_heads(tmp_path):
    tmp_path = tmp_path.parent / 'native'
    tmp_path.mkdir()
    source, normalized, checkpoint = _native_inputs(tmp_path)
    registry = tmp_path / 'registry.csv'
    upsert_registry_rows(csv_path=registry, rows=[{
        'recording_id':'one', 'keyprenorm_h5_relpath':relpath_str(normalized, tmp_path),
        'posepred_models':'previous.pth',
    }])
    output = tmp_path / 'p'
    cfg = PosePredInferenceConfig(input_dir=str(tmp_path), output_dir=str(output),
        inference_model=[str(checkpoint)], min_pose_bout=1, min_orient_bout=1,
        inf_average=1, use_tta=False, num_workers=0, batch=4,
        flat_output=True, clobber=False, registry_enabled=True,
        registry_csv=str(registry), write_run_log=False)
    run_posepred_inference(cfg)
    prediction = prediction_path(normalized, output)
    frame = pd.read_csv(prediction)
    for head in ('Pose', 'Orient'):
        pd.testing.assert_series_equal(frame[f'Smoothed {head} Label'],
            frame[f'Predicted {head} Label'], check_names=False)
    assert len(frame) == 17
    assert frame['Source Frame Index'].tolist() == list(range(17))
    metadata = json.loads(Path(str(prediction)+'.provenance.json').read_text())
    assert metadata['source_positional_sha256'] == sha256_file(source)
    assert metadata['positional_source']['sha256'] == sha256_file(source)
    # Replacing an already-existing prediction must update the registry model too.
    replacement = tmp_path / 'replacement.pth'
    replacement.write_bytes(checkpoint.read_bytes())
    loaded = torch.load(replacement, weights_only=False)
    loaded['run_name'] = 'replacement'
    torch.save(loaded, replacement)
    cfg = replace(cfg, inference_model=[str(replacement)])
    run_posepred_inference(cfg)
    assert read_registry(registry).iloc[0]['posepred_models'] == replacement.name
    assert json.loads(Path(str(prediction)+'.provenance.json').read_text())['inputs']['checkpoints'][0]['path'] == str(replacement)
    from autobehaver.preinf.posepred_inference.registry_update import update_registry_from_posepred_inference_outputs
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id':'one', 'posepred_models':'outdated.pth'}])
    update_registry_from_posepred_inference_outputs(registry_csv=registry, project_root=tmp_path,
        normalized_h5_files=[normalized], output_root=output, use_inferences_subdir=False,
        model_paths=['requested_but_not_actual.pth'], only_set_models_if_new=True,
        existing_outputs_before={str(prediction.resolve())})
    assert read_registry(registry).iloc[0]['posepred_models'] == replacement.name


def test_inference_rejects_modified_completed_normalization(tmp_path):
    tmp_path = tmp_path.parent / 'changed'
    tmp_path.mkdir()
    _, normalized, checkpoint = _native_inputs(tmp_path)
    with h5py.File(normalized, 'r+') as handle:
        handle['data'][0, 0] += 1
    with pytest.raises(ValueError, match='completed processing record'):
        _build_inference_provenance(str(normalized), [str(checkpoint)], {})


def test_selected_archive_preserves_unrelated_predictions_and_details(tmp_path):
    assert inspect.signature(run_inference).parameters['archive_existing'].default is False
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    selected = inputs/'a.h5'; selected.touch()
    unselected = inputs/'b.h5'; unselected.touch()
    output = tmp_path/'out'; output.mkdir()
    details = output/'inference_details'; details.mkdir()
    context = details/'cv'/'m01'; context.mkdir(parents=True)
    for name in ('a', 'b'):
        (output/f'{name}_inference.csv').write_text(name)
        (output/f'{name}_inference.csv.provenance.json').write_text(json.dumps({'outputs':[
            {'path':f'inference_details/{name}_logits.csv'},
            # A malformed record must not expand the selection to another recording.
            {'path':'b_inference.csv'},
        ]}))
        (details/f'{name}_logits.csv').write_text(name)
        with h5py.File(context/f'{name}_cv.h5', 'w') as handle:
            handle.attrs['source_filename'] = f'{name}.h5'
    before = {path:sha256_file(path) for path in output.rglob('*') if path.is_file() and path.name.startswith('b')}
    _archive_selected_predictions([selected], output, output/'past_inferences')
    assert not (output/'a_inference.csv').exists()
    assert not (details/'a_logits.csv').exists()
    assert not (context/'a_cv.h5').exists()
    assert all(path.exists() and sha256_file(path)==digest for path,digest in before.items())
    assert len(list((output/'past_inferences').glob('*/archive.json'))) == 1


def test_original_name_filter_and_current_exclusions(tmp_path):
    from autobehaver.preinf.autocropper.selection import write_selection
    path = tmp_path/'short.h5'; path.touch()
    record_original_name(path, 'group_FEMALE_trial1_rpi1_movie0_chamber2_normalized.h5')
    assert _discover_inference_input_files(str(tmp_path), file_filter='FEMALE') == [str(path)]
    write_selection(tmp_path, {'trial1_rpi1_movie0_chamber2':{'included':False}})
    assert _discover_inference_input_files(str(tmp_path), file_filter='FEMALE') == []
    csvs = tmp_path/'csvs'; csvs.mkdir()
    movies = tmp_path/'movies'; movies.mkdir()
    prediction = csvs/'short_inference.csv'; prediction.touch()
    movie = movies/'short.mp4'; movie.touch()
    recording = 'group_FEMALE_trial1_rpi1_movie0_chamber2'
    record_original_name(prediction, recording+'_positional_calcs_normalized_inference.csv')
    record_original_name(movie, recording+'.mp4')
    assert len(pipeline.discover_pairs_from_folders(
        inference_csv_dir=csvs, movie_dir=movies, file_filter='FEMALE')) == 1
    registry = tmp_path/'registry.csv'
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id':'one',
        'cropped_movie_relpath':relpath_str(movie, tmp_path),
        'posepred_inference_csv_relpath':relpath_str(prediction, tmp_path)}])
    assert len(pipeline.discover_pairs_from_registry(
        registry_csv=registry, project_root=tmp_path, file_filter='FEMALE')) == 1
    write_selection(csvs, {'trial1_rpi1_movie0_chamber2':{'included':False}})
    assert pipeline.discover_pairs_from_folders(
        inference_csv_dir=csvs, movie_dir=movies, file_filter='FEMALE') == []


def _write_prediction(path, *, modern=False, count=4):
    frame = pd.DataFrame({'Frame Number':range(1,count+1),
                          'Predicted Pose Label':['A']*count, 'Pose-A-Prob':[1.0]*count})
    if modern:
        frame['Source Frame Index'] = range(count)
    frame.to_csv(path, index=False)


def test_resmoothing_preserves_modern_frame_identity_and_upstream_source(tmp_path):
    source, output = tmp_path/'source.csv', tmp_path/'output.csv'
    _write_prediction(source, modern=True)
    Path(str(source)+'.provenance.json').write_text(json.dumps({
        'source_frame_offset':0, 'source_positional_sha256':'a'*64,
        'positional_source':{'sha256':'a'*64},
    }))
    receipt = resmooth_file(source, output, min_pose_bout=1)
    assert receipt['source_frame_offset'] == 0
    assert receipt['source_positional_sha256'] == 'a'*64
    assert load_source_frames(output).tolist() == [0,1,2,3]
    old, new = list(csv.DictReader(source.open())), list(csv.DictReader(output.open()))
    assert all(all(n[key]==value for key,value in o.items()) for o,n in zip(old,new))
    with pytest.raises(ValueError, match='cannot be overridden'):
        resmooth_file(source, tmp_path/'bad.csv', source_frame_offset=5)


def test_resmoothing_uses_known_legacy_offset_and_requires_unknown_one(tmp_path):
    source = tmp_path/'legacy.csv'; _write_prediction(source)
    with pytest.raises(ValueError, match='explicit source_frame_offset'):
        resmooth_file(source, tmp_path/'unknown.csv')
    Path(str(source)+'.provenance.json').write_text(json.dumps({'source_frame_offset':3}))
    output = tmp_path/'known.csv'
    receipt = resmooth_file(source, output, min_pose_bout=1)
    assert receipt['source_frame_offset'] == 3
    assert load_source_frames(output).tolist() == [3,4,5,6]


def test_overlay_reuse_binds_sources_options_range_and_output(tmp_path, monkeypatch):
    cv2 = pytest.importorskip('cv2')
    source = tmp_path/'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 40.0, (120,100))
    assert writer.isOpened()
    for i in range(17):
        writer.write(np.full((100,120,3), 20+i, np.uint8))
    writer.release()
    prediction = tmp_path/'prediction.csv'; _write_prediction(prediction, modern=True, count=17)
    output = tmp_path/'overlay.mp4'
    row = {'inf_csv_path':str(prediction), 'movie_path':str(source),
           'output_movie_path':str(output), 'start_frame':0, 'end_frame_exclusive':4}
    cfg = InfMoviePipelineConfig(output_dir=str(tmp_path), num_workers=1,
                                clobber=False, skip_first_n_frames=0)
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    original_hash = sha256_file(output)
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'reused'
    cfg = replace(cfg, timeline='pose', white_mode=True)
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    cap = cv2.VideoCapture(str(output)); assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 100; cap.release()
    assert sha256_file(output) != original_hash
    frame = pd.read_csv(prediction); frame['Predicted Pose Label'] = 'B'; frame.to_csv(prediction, index=False)
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    row['end_frame_exclusive'] = 6
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 40.0, (120,100))
    assert writer.isOpened()
    for i in range(17):
        writer.write(np.full((100,120,3), 90+i, np.uint8))
    writer.release()
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    with output.open('ab') as handle: handle.write(b'changed')
    assert pipeline.render_plan(cfg, plan_rows=[row]).outcomes[str(output)]['status'] == 'succeeded'
    payload = {**row, **{key:getattr(cfg,key) for key in (
        'source_frame_offset','skip_first_n_frames','timeline','timeline_compare','timeline_labels',
        'timeline_height','white_mode','speed_factor','max_output_fps','show_timestamp',
        'timestamp_position','timestamp_decimals','timestamp_reference')}}
    assert receipt_matches(output, pipeline._render_identity(payload))
    from autobehaver.validation import StageValidationError
    monkeypatch.setattr(pipeline, '_render_recording_group',
        lambda rows: [(str(output), False, 'injected renderer failure')])
    cfg = replace(cfg, white_mode=False)
    with pytest.raises(StageValidationError):
        pipeline.render_plan(cfg, plan_rows=[row])
    report = json.loads((tmp_path/'INFMOVIE_VALIDATION.json').read_text())
    assert report['status'] == 'failed'

