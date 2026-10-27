"""Recording occupancy must agree with review state and crop eligibility."""
import csv
from pathlib import Path

import cv2
import pytest

from autobehaver.preinf.autocropper.recording_db import load_recording_database_index
from autobehaver.preinf.moviechecker import cli
from autobehaver.preinf.moviechecker.core import MovieCheckerApp, compute_ui_layout, derive_pair_key
from autobehaver.preinf.moviechecker.exclusions import ExclusionsStore


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def metadata(tmp_path, *, legacy=False):
    rows = []
    mappings = []
    for camera in (1, 2):
        for chamber in range(1, 7):
            empty = camera == 2 or chamber == 3
            rows.append(dict(experiment='exp_with_underscores', subexperiment='', trial_number=1,
                             rpi_name=f'rpi-{camera}', chamber_number=chamber, label='A_',
                             occupied=int(not empty), use_for_crop=int(not empty)))
            mappings.append(f'rpi-{camera}_chamber{chamber}:' + ('EMPTY::' if empty else '') + 'A_')
    if legacy:
        return write_csv(tmp_path / 'recording_database_local.csv', [dict(
            experiment='exp_with_underscores', subexperiment='', trial_number=1,
            camera_mappings=','.join(mappings))])
    return write_csv(tmp_path / 'rec_data/trials/trial_0001/crop_occupancy_trial1.csv', rows)


def movies(tmp_path):
    folder = tmp_path / '1_full_movies'
    folder.mkdir(exist_ok=True)
    result = []
    for camera in (1, 2):
        for segment in (0, 1):
            path = folder / f'exp_with_underscores_trial1_rpi-{camera}_movie{segment}_20260925120000PM.mp4'
            path.touch()
            result.append(path)
    return result


@pytest.mark.parametrize('legacy', [False, True])
def test_recording_flags_merge_with_review_and_cannot_be_cleared(tmp_path, legacy):
    index = load_recording_database_index(metadata(tmp_path, legacy=legacy))
    paths = movies(tmp_path)
    review = paths[0].parent / 'exclusions.csv'
    key = derive_pair_key(str(paths[0]))
    store = ExclusionsStore(str(review))
    store.upsert(key, paths[0].name, [0, 1, 0, 0, 0, 0])
    store.write_atomic()
    app = MovieCheckerApp(str(paths[0].parent), str(review), review_all=True, recording_index=index)
    app.scan_movies()
    assert app.review_movies == [str(paths[0])]
    app._load_state_for_pair(key, str(paths[0]))
    assert app.excluded_flags == [0, 1, 1, 0, 0, 0]
    app.layout = compute_ui_layout(2016, 1520)
    for chamber in (3, 4):
        x1, y1, x2, y2 = app.layout.tile_rects[chamber]
        app._mouse_callback(cv2.EVENT_LBUTTONDOWN, (x1+x2)//2, (y1+y2)//2, 0, None)
    assert app.excluded_flags == [0, 1, 1, 1, 0, 0]
    app._save_current_state()
    store.load()
    assert store.get(key).exclude_ch == [0, 1, 1, 1, 0, 0]


def test_skipped_pairs_have_crop_coverage_without_marking_occupied_camera_reviewed(tmp_path):
    index = load_recording_database_index(metadata(tmp_path))
    paths = movies(tmp_path)
    review = paths[0].parent / 'exclusions.csv'
    app = MovieCheckerApp(str(paths[0].parent), str(review), recording_index=index)
    app.scan_movies()
    assert not review.exists()  # scanning and validation remain read-only
    assert len(app.recording_skipped_keys) == 1
    app._save_recording_only_exclusions()
    store = ExclusionsStore(str(review))
    store.load()
    assert len(store.records) == 1
    assert store.get(derive_pair_key(str(paths[2]))).exclude_ch == [1] * 6
    assert not store.has(derive_pair_key(str(paths[0])))
    app.scan_movies()
    assert app.review_movies == [str(paths[0])]


def test_completed_reviews_are_not_rewritten_or_reopened(tmp_path, monkeypatch):
    index = load_recording_database_index(metadata(tmp_path))
    paths = movies(tmp_path)
    review = paths[0].parent / 'exclusions.csv'
    store = ExclusionsStore(str(review))
    for path in (paths[0], paths[2]):
        store.upsert(derive_pair_key(str(path)), path.name, [0, 1, 0, 0, 0, 0])
    store.write_atomic()
    before = review.read_bytes()
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: pytest.fail('Completed review opened a window'))
    app = MovieCheckerApp(str(paths[0].parent), str(review), recording_index=index)
    app.run()
    assert review.read_bytes() == before
    assert not app.review_movies


def test_all_empty_run_saves_exclusions_without_opening_video(tmp_path, monkeypatch):
    index = load_recording_database_index(metadata(tmp_path))
    paths = movies(tmp_path)
    for path in paths[:2]:
        path.unlink()
    review = paths[2].parent / 'exclusions.csv'
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: pytest.fail('Empty camera opened a window'))
    app = MovieCheckerApp(str(paths[2].parent), str(review), recording_index=index)
    app.run()
    saved = list(csv.DictReader(review.open(newline='')))
    assert len(saved) == 1
    assert [saved[0][f'exclude_ch{i}'] for i in range(1, 7)] == ['1'] * 6


def test_incomplete_metadata_stops_before_any_automatic_write(tmp_path):
    path = metadata(tmp_path)
    rows = list(csv.DictReader(path.open(newline='')))
    write_csv(path, rows[:-1])
    index = load_recording_database_index(path)
    paths = movies(tmp_path)
    review = paths[0].parent / 'exclusions.csv'
    app = MovieCheckerApp(str(paths[0].parent), str(review), recording_index=index)
    with pytest.raises(KeyError, match='rpi-2_chamber6'):
        app.run()
    assert not review.exists()


@pytest.mark.parametrize('identity', ['unknown_trial1_rpi-1_movie0_x', 'exp_with_underscores_trial10_rpi-1_movie0_x', 'bad_name'])
def test_unknown_movie_identity_is_not_treated_as_empty(tmp_path, identity):
    index = load_recording_database_index(metadata(tmp_path))
    with pytest.raises((KeyError, ValueError)):
        index.get_movie_exclusion_flags(Path(identity + '.mp4'))


def test_cli_discovers_project_metadata_and_explicit_csv_takes_precedence(tmp_path, monkeypatch):
    metadata(tmp_path)
    override = metadata(tmp_path, legacy=True)
    seen = []
    class FakeApp:
        def __init__(self, **kwargs):
            seen.append(kwargs['recording_index'])
        def run(self):
            pass
    monkeypatch.setattr(cli, 'MovieCheckerApp', FakeApp)
    cli.main(['--main_dir', str(tmp_path)])
    assert seen[-1].csv_path == (tmp_path / 'rec_data/trials').resolve()
    cli.main(['--main_dir', str(tmp_path), '--recording_database_dir', str(tmp_path / 'absent'),
              '--recording_database_csv', str(override)])
    assert seen[-1].csv_path == override.resolve()
    cli.main(['--main-dir', str(tmp_path), '--recording-database-csv', str(override)])
    assert seen[-1].csv_path == override.resolve()


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('occupied', 'perhaps', 'occupied must be one of'),
    ('occupied', '0', 'use_for_crop is true but occupied is false'),
    ('use_for_crop', 'perhaps', 'use_for_crop must be one of'),
    ('chamber_number', '7', 'chamber_number must be 1..6'),
    ('chamber_number', '1.5', 'chamber_number must be an integer'),
    ('trial_number', 'NaN', 'trial_number must be an integer'),
    ('trial_number', '1.5', 'trial_number must be an integer'),
    ('rpi_name', 'camera-one', 'rpi_name must look like'),
])
def test_shared_metadata_reader_rejects_bad_occupancy(tmp_path, field, value, message):
    path = metadata(tmp_path)
    rows = list(csv.DictReader(path.open(newline='')))
    rows[0][field] = value
    write_csv(path, rows)
    with pytest.raises(ValueError, match=message):
        load_recording_database_index(path)


def test_csv_reader_supports_bom_blank_occupancy_defaults_and_decimal_integers(tmp_path):
    path = metadata(tmp_path)
    rows = list(csv.DictReader(path.open(newline='')))
    rows[0].update(trial_number='1.0', chamber_number='1.0', occupied='', use_for_crop='')
    write_csv(path, rows)
    path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
    index = load_recording_database_index(path)
    movie = Path('exp_with_underscores_trial1_rpi-1_movie0_x.mp4')
    assert index.get_movie_exclusion_flags(movie) == [0, 0, 1, 0, 0, 0]


@pytest.mark.parametrize('content, message', [
    ('experiment,experiment\nexp,exp\n', 'duplicate columns'),
    ('experiment,\nexp,\n', 'unnamed column'),
    ('experiment,subexperiment\nexp,,extra\n', 'incorrect field count'),
    ('', 'CSV is empty'),
])
def test_shared_reader_rejects_malformed_csv(tmp_path, content, message):
    path = tmp_path / 'recording_database_local.csv'
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_recording_database_index(path)
