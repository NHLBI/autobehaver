"""Tiny encodes and boundary checks for shared video and CSV work."""
from pathlib import Path
from types import SimpleNamespace
import csv
import subprocess
import numpy as np
import pytest


def _decoded(path):
    import cv2
    from autobehaver.preinf.autocropper.thread_limits import open_video_capture
    capture = open_video_capture(path)
    frames, times = [], []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        capture.release()
    return np.asarray(frames), np.asarray(times)


@pytest.mark.parametrize('cap', [1, 6])
def test_shared_crop_matches_single_encodes_and_preserves_drift(tmp_path, monkeypatch, cap):
    from autobehaver.preinf.autocropper import cropping
    source = tmp_path / 'source.mp4'
    subprocess.run([cropping._get_ffmpeg_executable(), '-v', 'error', '-nostdin', '-f', 'lavfi',
        '-i', 'testsrc2=size=128x96:rate=40:duration=1', '-vf', r'settb=1/1000000,setpts=N*24900+mod(N\,2)*5',
        '-c:v', 'libx264', '-threads', '1', '-bf', '0', '-vsync', '0', '-enc_time_base', '1:1000000',
        str(source)], check=True, timeout=20)
    paths = {f'chamber{i}': tmp_path / f'chamber{i}.mp4' for i in (1, 2)}
    coords = {'chamber1': (32, 32), 'chamber2': (96, 64), 'chamber3': (64, 48)}
    commands = []
    actual_run = cropping.subprocess.run
    def traced_run(command, **kwargs):
        commands.append(command)
        return actual_run(command, **kwargs)
    monkeypatch.setattr(cropping.subprocess, 'run', traced_run)
    args = SimpleNamespace(timeout=20, crop_crf=16, ffmpeg_threads=1, max_workers=cap)
    result = cropping.crop_chambers_in_parallel(source, coords, paths, 48, args, source_size=(128, 96))
    assert set(result) == {'chamber1', 'chamber2'}
    assert len(commands) == (2 if cap == 1 else 1)
    assert all(command.count('-i') == 1 for command in commands)
    before_times = _decoded(source)[1]
    for cid, output in paths.items():
        reference = tmp_path / (cid + '_single.mp4')
        assert cropping.crop_single_chamber(str(source), reference, *coords[cid], 48, timeout=20, crf=16)
        frames, times = _decoded(output)
        expected, _ = _decoded(reference)
        assert len(frames) == 40
        np.testing.assert_array_equal(frames, expected)
        np.testing.assert_allclose(times, before_times, rtol=0, atol=.002)
    assert not list(tmp_path.glob('*.partial'))
    commands.clear()
    assert cropping.crop_chambers_in_parallel(source, coords, paths, 48, args, source_size=(128, 96)) == result
    assert not commands


def test_failed_shared_crop_preserves_prior_movie(tmp_path, monkeypatch):
    from autobehaver.preinf.autocropper import cropping
    source = tmp_path / 'source.mp4'; source.write_bytes(b'source')
    output = tmp_path / 'chamber1.mp4'; output.write_bytes(b'prior movie')
    monkeypatch.setattr(cropping, 'video_shape', lambda *a: (10, 128, 96))
    monkeypatch.setattr(cropping, 'valid_crop', lambda *a, **k: False)
    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b'partial')
        raise subprocess.CalledProcessError(1, command, stderr='fixture failure')
    monkeypatch.setattr(cropping.subprocess, 'run', fail)
    result = cropping.crop_chambers_in_parallel(source, {'chamber1':(32,32)}, {'chamber1':output}, 48,
        SimpleNamespace(timeout=10, crop_crf=16, max_workers=6), source_size=(128,96))
    assert result['chamber1'].startswith('failed:')
    assert output.read_bytes() == b'prior movie'
    assert not list(tmp_path.glob('*.partial'))


def test_shared_crop_rejects_source_modified_during_encoding(tmp_path, monkeypatch):
    from autobehaver.preinf.autocropper import cropping
    source = tmp_path / 'source.mp4'; source.write_bytes(b'first source')
    output = tmp_path / 'chamber1.mp4'
    monkeypatch.setattr(cropping, 'video_shape', lambda *a: (10,128,96))
    monkeypatch.setattr(cropping, 'valid_crop', lambda *a, **k: True)
    def mutate(command, **kwargs):
        source.write_bytes(b'changed source during encoding')
        Path(command[-1]).write_bytes(b'encoded crop')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(cropping.subprocess, 'run', mutate)
    result = cropping.crop_chambers_in_parallel(source, {'chamber1':(32,32)}, {'chamber1':output}, 48,
        SimpleNamespace(timeout=10, crop_crf=16, max_workers=6), source_size=(128,96))
    assert result['chamber1'].startswith('failed:') and not output.exists()


def test_csv_index_seeks_checked_multiline_record_and_rejects_changed_source(tmp_path):
    from autobehaver.postinf.infmovie.frame_map import read_csv_summary
    path = tmp_path / 'inference.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Frame Number', 'Source Frame Index', 'Predicted Pose Label', 'Pose-A-Prob'])
        for i in range(6):
            writer.writerow([i+1, i, f'label,{i}\nnext', '.8' if i != 2 else ''])
    summary = read_csv_summary(path, include_selection=True)
    assert summary.frames.tolist() == list(range(6))
    assert np.isnan(summary.columns['Pose-A-Prob'][2])
    with summary.reader(path, 3) as reader:
        assert next(reader) == summary.headers
        rows = list(reader)
    assert len(rows) == 3 and rows[0][2] == 'label,3\nnext'
    path.write_text(path.read_text().replace('.8', '.7'))
    with pytest.raises(ValueError, match='changed'):
        with summary.reader(path):
            pass


def test_overlay_group_indexes_recording_once(tmp_path, monkeypatch):
    from autobehaver.postinf.infmovie import pipeline
    path = tmp_path / 'inference.csv'; path.write_text('Frame Number,Source Frame Index\n1,0\n2,1\n')
    calls = []
    original = pipeline.read_csv_summary
    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'read_csv_summary', counted)
    summaries = []
    def render(payload):
        summaries.append(payload['_csv_summary'])
        return payload['output_movie_path'], True, 'ok'
    monkeypatch.setattr(pipeline, '_render_worker', render)
    payloads = [{'inf_csv_path':str(path), 'movie_path':'movie', 'output_movie_path':f'clip{i}'} for i in (1,2)]
    assert len(pipeline._render_recording_group(payloads)) == 2
    assert len(calls) == 1 and summaries[0] is summaries[1]


def test_overlay_workers_respect_tasks_and_allocation(monkeypatch):
    from autobehaver.postinf.infmovie.io import worker_count
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    assert worker_count(8, 20) == 2
    assert worker_count(8, 1) == 1


def test_dlc_index_preserves_ambiguity_and_natural_dlc_in_recording_name(tmp_path):
    from autobehaver.preinf.dlc.core import build_dlc_output_index
    for suffix in ('DLC_model.csv', 'DLC_second.csv', 'DLC_model.h5', 'DLC_model_meta.pickle'):
        (tmp_path / ('naturalDLCrecording' + suffix)).touch()
    indexed = build_dlc_output_index(tmp_path)
    assert len(indexed['naturalDLCrecording']['csv']) == 2
    assert len(indexed['naturalDLCrecording']['h5']) == 1


def test_dlc_backend_failure_is_not_hidden_double_call():
    from autobehaver.preinf.dlc.core import _call_dlc_func
    calls = []
    def backend(arg, **kwargs):
        calls.append(arg)
        raise RuntimeError('backend failed')
    with pytest.raises(RuntimeError, match='backend failed'):
        _call_dlc_func(backend, 'one', unused=1)
    assert calls == ['one']


def test_dlc_failure_keeps_owned_scratch_and_original_output(tmp_path, monkeypatch):
    import cv2
    from autobehaver.preinf.dlc import core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    inputs = tmp_path / 'movies'; inputs.mkdir()
    video = inputs / 'one.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 40, (32,32))
    writer.write(np.zeros((32,32,3), dtype=np.uint8)); writer.release()
    output = tmp_path / 'dlc'; output.mkdir()
    original = output / 'oneDLC_old.csv'; original.write_text('original unfinished output')
    config = tmp_path / 'config.yaml'; config.write_text('bodyparts: [a]\n')
    scratch = tmp_path / 'scratch'
    calls = []
    def fail(config, videos, **kwargs):
        calls.append(videos)
        (Path(kwargs['destfolder']) / 'diagnostic.txt').write_text('failure details')
        raise RuntimeError('fixture backend failure')
    monkeypatch.setattr(core, '_require_deeplabcut', lambda: SimpleNamespace(analyze_videos=fail))
    monkeypatch.setattr(core, 'setup_gpu', lambda: None)
    monkeypatch.setattr(core, 'model_identity', lambda *a: {'settings':{'bodyparts':['a']}, 'snapshot_files':['fixture']})
    cfg = DLCInferencerConfig(input_dir=str(inputs), output_dir=str(output), dlc_config=str(config),
                              scratch_dir=str(scratch), max_retries=1)
    assert core.run_dlc(cfg) == 1
    assert len(calls) == 1 and original.read_text() == 'original unfinished output'
    retained = list(scratch.glob('ab_dlc_*'))
    assert len(retained) == 1 and (retained[0] / 'diagnostic.txt').is_file()


def test_tiny_overlay_encodes_only_requested_source_frames(tmp_path):
    import cv2
    from autobehaver.postinf.infmovie.movies import render_overlay_movie
    movie = tmp_path / 'movie.mp4'
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), 40, (120,100))
    for i in range(12):
        writer.write(np.full((100,120,3), i*15, dtype=np.uint8))
    writer.release()
    path = tmp_path / 'inference.csv'
    with path.open('w', newline='') as handle:
        rows = csv.writer(handle)
        rows.writerow(['Frame Number', 'Source Frame Index', 'Predicted Pose Label', 'Predicted Orient Label', 'Pose-A-Prob'])
        for i in range(12):
            rows.writerow([i+1, i, 'A', 'Floor', .9])
    output = tmp_path / 'overlay.mp4'
    render_overlay_movie(str(path), str(movie), str(output), start_frame=3, end_frame_exclusive=9,
                         skip_first_n_frames=0, timeline='none', atomic_write=True)
    frames, times = _decoded(output)
    assert len(frames) == 6 and frames.shape[1:3] == (100,420)
    # Compare untouched source pixels away from the bottom-label overlay.
    source, _ = _decoded(movie)
    np.testing.assert_allclose(frames[:,0:20,0:20].mean(axis=(1,2,3)),
                               source[3:9,0:20,0:20].mean(axis=(1,2,3)), rtol=0, atol=4)


def test_overlay_encoder_failure_preserves_existing_movie(tmp_path, monkeypatch):
    import cv2
    from autobehaver.postinf.infmovie.movies import render_overlay_movie
    movie = tmp_path / 'movie.mp4'
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), 40, (32,32))
    writer.write(np.zeros((32,32,3), dtype=np.uint8)); writer.release()
    path = tmp_path / 'inference.csv'; path.write_text('Frame Number,Source Frame Index\n1,0\n')
    output = tmp_path / 'overlay.mp4'; output.write_bytes(b'prior validated movie')
    class FailedWriter:
        def __init__(self, *args):
            pass
        def isOpened(self):
            return False
        def release(self):
            pass
    monkeypatch.setattr(cv2, 'VideoWriter', FailedWriter)
    with pytest.raises(RuntimeError, match='Could not open overlay encoder'):
        render_overlay_movie(str(path), str(movie), str(output), skip_first_n_frames=0, atomic_write=True)
    assert output.read_bytes() == b'prior validated movie'
