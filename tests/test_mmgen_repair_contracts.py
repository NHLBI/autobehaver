"""Independent identity, selection and preflight checks for the MMGen repair pass."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import atomic_write_json, sha256_file
from autobehaver.postinf.mmgen import append_mode as app, integrity, ledger
from autobehaver.postinf.mmgen.core import MMGenConfig
from autobehaver.postinf.mmgen.feature_matrix import validate_feature_matrix
from autobehaver.postinf.mmgen.io import read_metadata_csv
from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings


def recording_name(trial=1, movie=0, group='G'):
    return f'{group}_FEMALE_demo_rpi-1_trial{trial}_movie{movie}_chamber1_positional_calcs_inference.csv'


def write_pair(inf, pos, name, *, paired=True):
    (inf / name).write_text('Frame Number,Smoothed Pose Label,Smoothed Orient Label\n1,Standing,Floor\n', encoding='utf-8')
    if paired:
        (pos / name.replace('_inference', '')).write_text('centroid_x\n1\n', encoding='utf-8')


def seal_generation(root, cfg, names, *, value=1.0):
    """Create a minimal self-consistent generation for planning/publication tests.

    Source and artifact hashes use production helpers; the simple scientific value
    is a deliberate fixture and is not offered as a pipeline numerical oracle.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    inventory = app._build_current_inventory(inference_csv_dir=cfg.inference_csv_dir,
        poscalc_dir=cfg.poscalc_dir, inf_tag=cfg.inf_tag)
    rows = inventory[inventory.source_file.isin(names)].copy()
    rows['n_frames'] = 1
    rows['animal_id'] = rows.row_identity
    rows['rpi'] = 'rpi-1'
    rows['chamber'] = 'chamber1'
    rows['trial'] = 1
    matrix = rows[['row_identity', 'group', 'sex']].drop_duplicates().rename(columns={'row_identity':'individual'})
    matrix['Global_MeanSpeed_mm_s'] = value
    matrix.to_csv(root / 'MASTER_FEATURE_MATRIX.csv', index=False)
    rows.to_csv(root / ledger.SOURCE_LEDGER_FILENAME, index=False)
    ledger.build_row_identity_summary_df(source_ledger_df=rows, master_matrix_df=matrix).to_csv(root / ledger.ROW_IDENTITY_SUMMARY_FILENAME, index=False)
    atomic_write_json({'processing': {'append_new_data': False}}, root / ledger.BATCH_MANIFEST_FILENAME)
    calculation = app._cfg_calculation(cfg)
    sources = integrity.source_identities(names, inference_csv_dir=cfg.inference_csv_dir,
        poscalc_dir=cfg.poscalc_dir, inf_tag=cfg.inf_tag, source_frame_offset=cfg.source_frame_offset)
    identity = {'calculation': calculation, 'calculation_sha256': integrity.content_hash(calculation),
                'sources': sources, 'sources_sha256': integrity.content_hash(sources)}
    atomic_write_json(identity, root / app.IDENTITY_RELATIVE_PATH)
    atomic_write_json({'calculation_sha256': identity['calculation_sha256'],
        'sources_sha256': identity['sources_sha256'],
        'artifacts': {name: sha256_file(root / name) for name in app.CORE_ARTIFACTS}}, root / app.OUTPUT_RECEIPT_FILENAME)
    return matrix, rows


@pytest.fixture
def append_case(tmp_path):
    inf, pos, root = [tmp_path / name for name in ('inf', 'pos', 'out')]
    for directory in (inf, pos, root):
        directory.mkdir()
    cfg = MMGenConfig(inference_csv_dir=str(inf), poscalc_dir=str(pos), output_dir=str(root),
                     append_new_data=True, inf_tag='_inference', num_workers=1, emit_full_outputs=False)
    first = recording_name()
    write_pair(inf, pos, first)
    seal_generation(root, cfg, [first])
    return inf, pos, root, cfg, first


def test_literal_metadata_survives_matrix_csv_without_imputing_values(tmp_path):
    path = tmp_path / 'matrix.csv'
    original = pd.DataFrame({'individual':['NA_FEMALE_trial1', 'None_FEMALE_trial2', '001_FEMALE_trial3'],
        'group':['NA', 'None', '001'], 'sex':['FEMALE'] * 3, 'Global_MeanSpeed_mm_s':[np.nan, 0., 2.]})
    original.to_csv(path, index=False)
    valid, error, readback = validate_feature_matrix(path)
    assert valid, error
    assert readback.group.tolist() == ['NA', 'None', '001']
    assert readback.individual.tolist() == original.individual.tolist()
    assert np.isnan(readback.Global_MeanSpeed_mm_s.iloc[0])
    assert readback.Global_MeanSpeed_mm_s.iloc[1:].tolist() == [0., 2.]
    path.write_text('source_file,row_identity,source_file_sha256,n_frames\n001,NA,000123,4\n', encoding='utf-8')
    row = read_metadata_csv(path).iloc[0]
    assert (row.source_file, row.row_identity, row.source_file_sha256, row.n_frames) == ('001', 'NA', '000123', 4)


@pytest.mark.parametrize('name', ['mm_per_px', 'video_fps'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_calibration_rejected_before_config_can_start_work(name, value, tmp_path):
    with pytest.raises(ValueError, match=name):
        MMGenConfig(inference_csv_dir='inf', poscalc_dir='pos', output_dir=str(tmp_path / 'not-created'), **{name:value})
    assert not (tmp_path / 'not-created').exists()
    with pytest.raises(ValueError, match=name):
        replace(MMGenRuntimeSettings(), **{name:value})


def test_calibration_retains_explicit_auto_modes_and_valid_nondefault_values():
    for diameter in (0, None, 600):
        settings = MMGenRuntimeSettings(video_fps=30, mm_per_px=.025, chamber_diameter_px=diameter)
        assert settings.video_fps == 30 and settings.mm_per_px == .025
    with pytest.raises(ValueError, match='chamber_diameter_px'):
        MMGenRuntimeSettings(chamber_diameter_px=-1)


def test_plan_only_requires_append_before_any_output_creation(tmp_path):
    with pytest.raises(ValueError, match='plan_only requires append_new_data'):
        MMGenConfig(inference_csv_dir='inf', poscalc_dir='pos', output_dir=str(tmp_path / 'out'), plan_only=True)
    assert not (tmp_path / 'out').exists()


def test_append_file_list_selects_one_animal_and_preserves_all_its_movies(append_case, tmp_path):
    inf, pos, root, cfg, first = append_case
    selected, excluded, unpaired = recording_name(movie=1), recording_name(trial=2), recording_name(trial=3)
    for name in (selected, excluded):
        write_pair(inf, pos, name)
    write_pair(inf, pos, unpaired, paired=False)
    listing = tmp_path / 'selected.txt'
    listing.write_text(selected.removesuffix('_positional_calcs_inference.csv') + '\n', encoding='utf-8')
    cfg.file_list = str(listing)
    plan, rows, affected, batch = app._create_append_plan(cfg)
    assert set(rows.source_file) == {first, selected}
    assert set(plan['selected_sources']) == {first, selected}
    assert len(affected) == 1
    write_pair(inf, pos, recording_name(trial=4), paired=False)
    app._validate_plan_against_cfg(plan, cfg)
    (pos / selected.replace('_inference', '')).unlink()
    with pytest.raises(FileNotFoundError, match='Missing positional-calcs partner'):
        app._validate_plan_against_cfg(plan, cfg)


def test_missing_selected_partner_still_prevents_append_plan(append_case):
    inf, pos, root, cfg, first = append_case
    unpaired = recording_name(movie=1)
    write_pair(inf, pos, unpaired, paired=False)
    cfg.file_filter = 'movie1'
    with pytest.raises(FileNotFoundError, match='paired positional_calcs'):
        app._create_append_plan(cfg)
    assert not (root / app.APPEND_BATCHES_DIRNAME).exists()
