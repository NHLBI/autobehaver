"""Exercise DLC 2.3.9 native scorer lookup with tiny videos and HDF5 files."""

import json
import pickle
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from autobehaver.preinf.dlc import core
from autobehaver.portable_paths import original_name, record_original_name


SCORER = 'DLC_resnet50_fixtureOct18shuffle1_100'


def _video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (32, 32))
    assert writer.isOpened()
    for value in (20, 50, 80, 110):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()


def _predictions(directory, stem):
    columns = pd.MultiIndex.from_product([[SCORER], ['a'], ['x', 'y', 'likelihood']],
                                         names=['scorer', 'bodyparts', 'coords'])
    table = pd.DataFrame([[10., 11., .9]] * 4, columns=columns)
    h5, csv, meta = (directory / (stem + suffix) for suffix in ('.h5', '.csv', '_meta.pickle'))
    table.to_hdf(h5, key='df_with_missing', format='table')
    table.to_csv(csv)
    meta.write_bytes(pickle.dumps({'data': {'cropping': False, 'cropping_parameters': [0, 32, 0, 32]}}))
    return dict(h5=str(h5), csv=str(csv), metadata=str(meta), rows=4)


@pytest.fixture
def inputs(tmp_path):
    video = tmp_path / 'short_movie.mp4'
    _video(video)
    record_original_name(video, 'Complete_G_MALE_trial1_rpi-1_movie0_chamber1.mp4')
    model = tmp_path / 'project/dlc-models/iteration-0/fixture/train'
    model.mkdir(parents=True)
    weights = model / 'snapshot-100.index'
    weights.write_bytes(b'unchanged model fixture')
    config = tmp_path / 'project/config.yaml'
    config.write_text('bodyparts: [a]\nproject_path: .\ndotsize: 2\npcutoff: .1\ncolormap: viridis\n')
    output = tmp_path / 'published'
    output.mkdir()
    details = _predictions(output, 'portable_predictions')
    return video, config, weights, output, details, tmp_path / 'scratch'


def _render_backend(config, videos, *, destfolder, shuffle, videotype):
    cfg = yaml.safe_load(Path(config).read_text())
    assert Path(cfg['project_path']) == Path(config).parent
    assert (Path(cfg['project_path']) / 'dlc-models/iteration-0/fixture/train/snapshot-100.index').read_bytes() == b'unchanged model fixture'
    assert (cfg['dotsize'], cfg['pcutoff'], cfg['colormap']) == (24, .6, 'rainbow')
    directory = Path(destfolder)
    for video in videos:
        # Native DLC 2.3.9 find_analyzed_data requires this literal prefix;
        # published portable filenames alone do not satisfy it.
        prefix = Path(video).stem + SCORER
        matches = [path for path in directory.glob('*.h5') if path.name.startswith(prefix)]
        assert len(matches) == 1
        assert len(pd.read_hdf(matches[0])) == 4
        assert pickle.loads((directory / (prefix + '_meta.pickle')).read_bytes())['data']['cropping'] is False
        shutil.copyfile(video, directory / (prefix + '_labeled.mp4'))
    return [True] * len(videos)


def _label(inputs, backend, **options):
    video, config, weights, output, details, scratch = inputs
    return core.create_labeled_movies(config_path=str(config), videos=[str(video)], videotype='.mp4',
        shuffle=1, dotsize=24, pcutoff=.6, colormap='rainbow', exclude_bodyparts=[],
        output_dir=str(output), backend=SimpleNamespace(create_labeled_video=backend),
        analyzed_outputs={str(video): details}, scratch_dir=str(scratch), **options)


def test_native_names_render_publish_and_preserve_inputs(inputs):
    video, config, weights, output, details, scratch = inputs
    before = {path: Path(path).read_bytes() for path in [str(config), str(weights), *[details[k] for k in ('h5', 'csv', 'metadata')]]}
    published = _label(inputs, _render_backend)
    rendered = Path(published[str(video)])
    assert core.Path(rendered).is_file()
    assert original_name(rendered) == Path(original_name(video)).stem + SCORER + '_labeled.mp4'
    from autobehaver.preinf.autocropper.contract import video_shape
    assert video_shape(rendered) == video_shape(video) == (4, 32, 32)
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    assert not list(scratch.glob('ab_dlc_labels_*'))


@pytest.mark.parametrize('bad_output', [False, True])
def test_native_false_or_unreadable_output_is_reported_and_retained(inputs, bad_output):
    video, config, weights, output, details, scratch = inputs
    def fail(config, videos, **kwargs):
        if bad_output:
            (Path(kwargs['destfolder']) / (Path(videos[0]).stem + SCORER + '_labeled.mp4')).write_bytes(b'not a movie')
        return [False]
    with pytest.raises(RuntimeError, match='Labeled video generation failed'):
        _label(inputs, fail)
    assert not list(output.glob('*_labeled.mp4'))
    assert len(list(scratch.glob('ab_dlc_labels_*'))) == 1
    assert Path(details['h5']).is_file() and weights.read_bytes() == b'unchanged model fixture'


def test_no_overwrite_reuses_verified_overlay(inputs):
    first = _label(inputs, _render_backend)
    target = Path(next(iter(first.values())))
    before = target.stat().st_mtime_ns
    def unexpected(*args, **kwargs):
        pytest.fail('verified existing overlay was rendered again')
    assert _label(inputs, unexpected, overwrite=False) == first
    assert target.stat().st_mtime_ns == before


def test_requested_label_failure_marks_stage_failed_without_invalidating_predictions(inputs, monkeypatch):
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    video, config, weights, output, details, scratch = inputs
    def analyze(config, videos, **kwargs):
        for path in videos:
            _predictions(Path(kwargs['destfolder']), Path(path).stem + SCORER)
    backend = SimpleNamespace(analyze_videos=analyze, create_labeled_video=lambda *a, **k: [False])
    monkeypatch.setattr(core, '_require_deeplabcut', lambda: backend)
    monkeypatch.setattr(core, 'setup_gpu', lambda: None)
    monkeypatch.setattr(core, 'model_identity', lambda *a: {'settings': {'bodyparts': ['a']}, 'snapshot_files': ['fixture']})
    cfg = DLCInferencerConfig(input_dir=str(video.parent), output_dir=str(output), dlc_config=str(config),
                              make_movie=True, max_retries=1, scratch_dir=str(scratch))
    assert core.run_dlc(cfg) == 1
    report = json.loads((output / 'DLC_VALIDATION.json').read_text())
    assert any(item['code'] == 'labeled_video_failed' for item in report['issues'])
    valid, _, predicted = core.inspect_dlc_outputs(str(video), str(output))
    assert valid and Path(predicted['csv'] + '.done.json').is_file()
