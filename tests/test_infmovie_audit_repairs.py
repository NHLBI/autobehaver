"""Tiny independent checks of InfMovie publication, identity and timing repairs."""
import csv
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest

from autobehaver.completion import receipt_path
from autobehaver.postinf.infmovie import pipeline, movies, io
from autobehaver.postinf.infmovie.config import InfMoviePipelineConfig
from autobehaver.postinf.infmovie.frame_map import read_csv_summary


def tiny_inputs(tmp_path, fps=40., *, blank=False):
    import cv2
    movie, labels = tmp_path / 'input.mp4', tmp_path / 'input.csv'
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), fps, (120, 100))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((100, 120, 3), index * 15, np.uint8))
    writer.release()
    with labels.open('w', newline='') as stream:
        rows = csv.writer(stream)
        rows.writerow(['Frame Number', 'Source Frame Index', 'Predicted Pose Label', 'Predicted Orient Label', 'Pose-A-Prob'])
        for index in range(12):
            if blank and index in (0, 3, 9):
                rows.writerow([])
            rows.writerow([index + 1, index, 'A\nquoted label' if blank and index == 4 else 'A', 'Floor', .9])
        if blank:
            rows.writerow([])
    return movie, labels


def decode(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return np.asarray(frames), fps


def test_existing_movie_and_receipt_survive_unproduced_request(tmp_path, monkeypatch):
    movie, labels = tiny_inputs(tmp_path)
    output = tmp_path / 'overlay.mp4'
    payload = dict(movie_path=str(movie), inf_csv_path=str(labels), output_movie_path=str(output),
                   skip_first_n_frames=0, fps_in=40., timeline='none')
    assert pipeline._render_worker(payload)[1]
    previous = output.read_bytes(), receipt_path(output).read_bytes()
    assert not pipeline._render_worker({**payload, 'skip_first_n_frames': 100})[1]
    assert (output.read_bytes(), receipt_path(output).read_bytes()) == previous
    monkeypatch.setattr(pipeline, 'render_overlay_movie', lambda *args, **kwargs: None)
    assert not pipeline._render_worker({**payload, 'white_mode': True})[1]
    assert (output.read_bytes(), receipt_path(output).read_bytes()) == previous


@pytest.mark.parametrize('start,end,skip,speed,expected', [
    (0, 12, 0, 1., list(range(12))),
    (1, 11, 3, 1.5, [3, 5, 6, 8, 9, 10]),
    (4, 5, 0, 1.5, [4]),
])
def test_blank_and_multiline_records_keep_exact_source_schedule(tmp_path, start, end, skip, speed, expected):
    movie, labels = tiny_inputs(tmp_path, blank=True)
    summary = read_csv_summary(labels, include_selection=True, video_frames=12)
    assert len(summary.frames) == 12
    output = tmp_path / 'overlay.mp4'
    produced = movies.render_overlay_movie(str(labels), str(movie), str(output),
        start_frame=start, end_frame_exclusive=end, skip_first_n_frames=skip,
        speed_factor=speed, max_output_fps=40.)
    rendered, fps = decode(output)
    original, _ = decode(movie)
    assert len(rendered) == len(expected) == produced['frames']
    assert fps == pytest.approx(40.)
    np.testing.assert_allclose(rendered[:, :20, :20].mean(axis=(1, 2, 3)),
                               original[expected, :20, :20].mean(axis=(1, 2, 3)), rtol=0, atol=4)


def test_manual_ambiguous_recording_rejected_before_input_read(tmp_path, monkeypatch):
    pairs = [{'recording_id': name, 'movie_path': name, 'inf_csv_path': name} for name in ('fly1_chamber1', 'fly2_chamber1')]
    monkeypatch.setattr(pipeline, '_get_video_info', lambda *args, **kwargs: pytest.fail('ambiguous selection reached input read'))
    with pytest.raises(ValueError, match='fly1_chamber1.*fly2_chamber1'):
        pipeline.build_plan_manual(InfMoviePipelineConfig(recording_id='chamber1'), pairs=pairs)


def test_manual_exact_and_unique_substring_selection(tmp_path):
    movie, labels = tiny_inputs(tmp_path)
    pairs = [dict(recording_id='fly_chamber1', movie_path=str(movie), inf_csv_path=str(labels)),
             dict(recording_id='fly_chamber12', movie_path='unselected', inf_csv_path='unselected')]
    config = InfMoviePipelineConfig(recording_id='fly_chamber1', duration_sec=.1, skip_first_n_frames=0)
    assert pipeline.build_plan_manual(config, pairs=pairs)[0]['recording_id'] == 'fly_chamber1'
    assert pipeline.build_plan_manual(replace(config, recording_id='chamber1'), pairs=pairs[:1])[0]['recording_id'] == 'fly_chamber1'


def test_spacing_uses_each_recording_fps_without_rounding():
    candidates = [dict(recording_id=rid, fps_in=fps, start_frame=start, score=10-start/1000)
                  for rid, fps in [('twenty', 20.), ('forty', 40.)] for start in (0, 500)]
    selected = pipeline._dedupe_and_select(candidates, n_select=10, min_gap_sec=20., max_per_recording=10, rng=random.Random(0))
    assert {(c['recording_id'], c['start_frame']) for c in selected} == {('twenty', 0), ('twenty', 500), ('forty', 0)}
    fractional = [dict(recording_id='fractional', fps_in=29.97, start_frame=start, score=3-start) for start in (0, 1, 2)]
    selected = pipeline._dedupe_and_select(fractional, n_select=10, min_gap_sec=.05, max_per_recording=10, rng=random.Random(0))
    assert [c['start_frame'] for c in selected] == [0, 2]


def test_missing_fps_uses_same_configured_timebase_through_render(tmp_path, monkeypatch):
    import cv2
    movie, labels = tiny_inputs(tmp_path, fps=29.97)
    native = io.open_video_capture
    class MissingFPS:
        def __init__(self, path):
            self.capture = native(path)
            self.source = Path(path) == movie
        def get(self, prop):
            return 0. if self.source and prop == cv2.CAP_PROP_FPS else self.capture.get(prop)
        def __getattr__(self, name):
            return getattr(self.capture, name)
    monkeypatch.setattr(io, 'open_video_capture', MissingFPS)
    monkeypatch.setattr(pipeline, 'open_video_capture', MissingFPS)
    cfg = InfMoviePipelineConfig(recording_id='one', duration_sec=.2, default_movie_fps=29.97)
    row = pipeline.build_plan_manual(cfg, pairs=[dict(recording_id='one', movie_path=str(movie), inf_csv_path=str(labels))])[0]
    assert row['fps_in'] == 29.97 and row['end_frame_exclusive'] == 6
    output = tmp_path / 'overlay.mp4'
    produced = movies.render_overlay_movie(str(labels), str(movie), str(output), start_frame=row['start_frame'],
        end_frame_exclusive=row['end_frame_exclusive'], skip_first_n_frames=0,
        resolved_fps=row['fps_in'], default_movie_fps=cfg.default_movie_fps)
    frames, fps = decode(output)
    assert len(frames) == 6 and fps == pytest.approx(29.97, abs=.001)
    assert produced['fps_in'] == 29.97
    payload = {**row, 'output_movie_path': str(output), 'default_movie_fps': 29.97}
    assert pipeline._render_identity(payload) != pipeline._render_identity({**payload, 'fps_in': 40.})


def test_exact_sources_and_frames_disambiguate_output_names(tmp_path):
    rows = [dict(recording_id=rid, movie_path=str(tmp_path/'in.mp4'), inf_csv_path=str(tmp_path/'in.csv'),
                 start_sec=.01, end_sec=1.01, start_frame=start, end_frame_exclusive=40+start)
            for rid, start in [('rec:1', 0), ('rec?1', 0), ('rec:1', 1)]]
    pipeline._assign_output_paths(InfMoviePipelineConfig(), output_dir=tmp_path, plan_rows=rows)
    assert len({row['output_movie_path'] for row in rows}) == 3
    relocated = [{**row, 'movie_path': str(tmp_path/'moved/in.mp4'), 'inf_csv_path': str(tmp_path/'moved/in.csv')} for row in rows]
    pipeline._assign_output_paths(InfMoviePipelineConfig(), output_dir=tmp_path/'moved', plan_rows=relocated)
    assert [Path(row['output_movie_path']).name for row in rows] == [Path(row['output_movie_path']).name for row in relocated]
    with pytest.raises(ValueError, match='Duplicate InfMovie output destination'):
        pipeline.render_plan(InfMoviePipelineConfig(output_dir=str(tmp_path)), plan_rows=[rows[0], rows[0]])


def test_missing_predictions_are_named_without_changing_qc_rank(tmp_path):
    labels = tmp_path/'labels.csv'
    labels.write_text('Frame Number,Pose-A-Prob,Orient-B-Prob\n1,nan,nan\n2,nan,nan\n3,0.9,0.8\n4,0.2,0.3\n')
    summary = read_csv_summary(labels, source_frame_offset=0, include_selection=True)
    simple = pipeline._load_confidence_for_qc(labels)
    indexed = pipeline._load_confidence_for_qc(labels, summary=summary)
    np.testing.assert_equal(simple, indexed)
    np.testing.assert_equal(indexed, [np.nan, np.nan, .8, .2])
    kwargs = dict(combined_labels=np.array(['', '', 'A', 'B']), n_total=4, fps=40., clip_len=2,
                  padding=0, uncertainty_threshold=.6, min_uncertain_frames=1,
                  switch_window_sec=1, switch_min_changes=1, use_uncertainty=True, use_switching=False)
    corrected = pipeline.build_candidates_qc(confidence=indexed, **kwargs)
    prior_rank = pipeline.build_candidates_qc(confidence=np.nan_to_num(indexed), **kwargs)
    assert [(r['start_frame'], r['end_frame_exclusive'], r['score']) for r in corrected] == [(r['start_frame'], r['end_frame_exclusive'], r['score']) for r in prior_rank]
    assert corrected[0]['reason'].startswith('qc_missing_predictions:')
    assert corrected[0]['missing_prediction_frames'] == 2
    assert corrected[1]['reason'].startswith('qc_uncertainty:')


def test_literal_tilde_report_uses_resolved_render_root(tmp_path, monkeypatch):
    original = Path.expanduser
    monkeypatch.setattr(Path, 'expanduser', lambda path: tmp_path/'reports' if path.parts == ('~', 'reports') else original(path))
    with pytest.raises(RuntimeError):
        pipeline.render_plan(InfMoviePipelineConfig(output_dir='~/reports'), plan_rows=[])
    assert (tmp_path/'reports/INFMOVIE_VALIDATION.json').is_file()


def test_missing_probability_renders_as_unavailable(tmp_path, monkeypatch):
    import cv2
    movie, labels = tiny_inputs(tmp_path)
    labels.write_text(labels.read_text().replace(',0.9', ',nan'))
    text = []
    native = cv2.putText
    def capture(image, label, *args, **kwargs):
        text.append(label)
        return native(image, label, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', capture)
    result = movies.render_overlay_movie(str(labels), str(movie), str(tmp_path/'overlay.mp4'), skip_first_n_frames=0)
    assert result['frames'] == 12
    assert text.count('N/A') == 12
