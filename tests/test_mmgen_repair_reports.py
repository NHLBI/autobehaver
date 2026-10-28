"""Literal HDF identity, image-axis, report label and source-route regressions."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autobehaver.postinf.mmgen import feature_matrix as fm, measurements, occupancy as o
from autobehaver.postinf.mmgen.io import read_metadata_csv
from autobehaver.postinf.mmgen.naming import collapse_individual_id
from tests.test_mmgen_repair_science import time_budget_sources


@pytest.fixture
def quoted_positions(tmp_path):
    names = ["O'Brien", 'A"B', "Both'\"quotes", r'x\y', 'NA']
    rows = pd.DataFrame([{'individual': name, 'group': 'NA', 'source_file': name + '.csv',
        'frame_idx': i, 'centroid_x': x, 'centroid_y': y, 'angle': angle,
        'masked_pose': 'Standing', 'masked_orient': 'Floor', 'masked_pose_orient': 'Standing + Floor',
        'masked_pose_valid': 'Standing', 'masked_orient_valid': 'Floor',
        'masked_pose_orient_valid': 'Standing + Floor', 'speed_mm_s': 1.}
        for name in names for i, (x, y, angle) in enumerate(
            [(399.5, 299.5, 90.), (299.5, 399.5, 180.), (99.5, 299.5, 270.)])])
    path = tmp_path / 'positions.h5'
    rows.to_hdf(path, key='positional', format='table', data_columns=['individual'])
    return path, rows


def test_quoted_occupancy_ids_match_full_csv_memory_and_worker_routes(quoted_positions, tmp_path):
    path, original = quoted_positions
    full_path, serial_path, chunk_path = [tmp_path / name for name in ('full.csv', 'serial.csv', 'chunk.csv')]
    acc = o._stream_transform_write_and_accumulate(str(path), '/positional', str(tmp_path),
        chamber_diameter_px=400, generate_quiver_plots=True, transformed_output_path=str(full_path))
    full = read_metadata_csv(full_path)
    assert len(full) == len(original) and set(full.individual) == set(original.individual)
    assert set(full.group) == {'NA'}
    o._serial_transform_write_without_accumulate(str(path), '/positional', str(serial_path), fixed_arena_radius=200.)
    memory = o._serial_transform_collect_without_accumulate(str(path), '/positional', fixed_arena_radius=200.)
    args = (str(path), '/positional', sorted(original.individual.unique()),
            [column for column in full if column in original], 200.)
    collected = o._occupancy_transform_chunk_collect_worker(args)['dataframe']
    o._occupancy_transform_chunk_worker((*args, str(chunk_path)))
    for frame in [read_metadata_csv(serial_path), memory, collected, read_metadata_csv(chunk_path)]:
        pd.testing.assert_frame_equal(full, frame.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
    # Independently known geometry: right, below and left of arena center.
    one = full[full.individual == 'NA'].sort_values('frame_idx')
    np.testing.assert_allclose(one.r_norm, [.5, .5, 1.])
    np.testing.assert_allclose(one.positional_angle, [0., 90., 180.])
    np.testing.assert_allclose(one.relative_angle, 0., atol=1e-12)
    assert acc.get_heatmap(('masked_pose', 'Standing', 'NA'))[1] == len(original)


def test_occupancy_real_read_error_keeps_its_cause(quoted_positions, tmp_path, monkeypatch):
    path, _ = quoted_positions
    failure = OSError('unreadable HDF table')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(pd.HDFStore, 'select', fail)
    with pytest.raises(RuntimeError, match='Failed to read persisted positional data') as result:
        o._stream_transform_write_and_accumulate(str(path), '/positional', str(tmp_path))
    assert result.value.__cause__ is failure


def test_asymmetric_heatmap_places_small_image_y_above_large_y(tmp_path, monkeypatch):
    acc = o._OccupancyAccumulator(heatmap_bins=6, heatmap_range=((0, 600), (0, 600)),
        quiver_grid_size=6, kde_max_points=100, seed=0)
    acc.add_heatmap(acc.behavior_cols[0], 'Standing', 'G', np.array([50., 50., 50., 50.]),
        np.array([50., 50., 50., 550.]))
    seen = []
    def inspect(path, **kwargs):
        ax = o.plt.gcf().axes[0]
        bins = np.asarray(ax.collections[0].get_array()).reshape(6, 6)
        assert bins[0, 0] == .75 and bins[5, 0] == .25
        assert ax.transData.transform((.5, .5))[1] > ax.transData.transform((.5, 5.5))[1]
        seen.append(path)
    monkeypatch.setattr(o.plt, 'savefig', inspect)
    assert len(o._generate_2d_heatmaps_from_acc(acc, str(tmp_path))) == len(seen) == 1
    assert not o.plt.get_fignums()


def test_measurement_workbook_labels_mean_recording_counts_without_changing_estimators(tmp_path):
    names = ['length_top', 'length_side', 'length_bottom', 'headwidth_top', 'headwidth_bottom',
        'headheight_side', 'thoraxwidth_top', 'thoraxwidth_bottom', 'thoraxheight_side',
        'abdomenwidth_top', 'abdomenwidth_bottom', 'abdomenheight_side',
        'wingLeftlength_top', 'wingRightlength_top', 'body_area_top', 'body_area_side']
    records = []
    for movie, (mean, std, n) in enumerate([(2., 1., 1), (8., 3., 100)]):
        row = {'individual_id': f'G_FEMALE_rpi-1_trial1_movie{movie}_chamber1',
               'group_label': 'G', 'sex': 'FEMALE'}
        for name in names:
            row.update({name + '_mean': mean, name + '_std': std, name + '_n_samples': n})
        records.append(row)
    result = measurements.calculate_and_save_pooled_measurements(pd.DataFrame(records), str(tmp_path), mm_per_px=1.)
    assert result.loc[0, 'length_side_mean'] == 5.
    assert result.loc[0, 'length_side_std'] == 2.
    assert result.loc[0, 'length_side_n_samples'] == 50.5
    sheets = pd.read_excel(tmp_path / '3_Standardized_Measurements.xlsx', sheet_name=None)
    animal = collapse_individual_id(records[0]['individual_id'])
    combined = sheets['Combined Measurements']
    row = combined.loc[combined.individual == animal].iloc[0]
    assert row.length_side_mean_samples_per_recording == 50.5
    assert row.length_side_mean == 5. and row.length_side_std == 2.
    assert all(not any(str(column).endswith('_n_samples') for column in sheet.columns) for sheet in sheets.values())


@pytest.mark.parametrize('suffix', ['_n_samples', '_mean_samples_per_recording'])
def test_old_and_new_measurement_count_headers_are_excluded_from_matrix(tmp_path, suffix):
    sources = time_budget_sources()
    measurements_df = pd.DataFrame({'individual': ['A', 'B', 'C'], 'group': ['G'] * 3,
        'length_side_mean': [1., 2., 3.], 'length_side' + suffix: [50.5] * 3})
    result = fm.create_feature_matrix_from_sources(str(tmp_path), replace(sources, measurements_df=measurements_df),
        filter_features_by_bout_count=False)
    assert 'StdMeasure_length_side_mean' in result
    assert not any('samples' in column for column in result)
