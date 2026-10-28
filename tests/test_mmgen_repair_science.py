"""Independent expected values for MMGen missingness, eligibility and report fixes."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file
from autobehaver.postinf.mmgen import feature_matrix as fm, kinematics as kin, transitions as tr
from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings


@pytest.mark.parametrize('n', [0, 1, 2, 5, 6])
@pytest.mark.parametrize('threshold', [0, 1, 6])
def test_bb_zero_observations_and_singleton_deviation_are_missing(n, threshold):
    observations = np.arange(1, n + 1, dtype=float)
    aggregate = pd.DataFrame([{'individual': 'G_trial1', 'group': 'G', 'bout_type': 'Standing + Floor',
        'speed_mean__sum': observations.sum(), 'speed_mean__sum_sq': (observations ** 2).sum(),
        'speed_mean__count': n}])
    row = kin._reconstruct_kinematic_table_from_aggregates({'_inst': aggregate},
        min_bout_count_for_features=threshold)['_inst'].iloc[0]
    mean = observations.mean() if n >= max(1, threshold) else np.nan
    sd = observations.std(ddof=1) if n >= max(2, threshold) else np.nan
    np.testing.assert_allclose(float(row.speed_mean_mean), mean, equal_nan=True)
    np.testing.assert_allclose(float(row.speed_mean_std), sd, equal_nan=True)
    assert np.isscalar(row.speed_mean_mean) and np.isscalar(row.speed_mean_std)


def test_bb_scalar_moments_keep_numeric_group_averages_in_real_workbook(tmp_path):
    aggregate = pd.DataFrame([{'individual': animal, 'group': 'G', 'bout_type': behavior,
        'speed_mean__sum': 6 * value, 'speed_mean__sum_sq': 6 * value ** 2, 'speed_mean__count': 6}
        for animal, value in [('A', 2.), ('B', 4.)]
        for behavior in ['Standing + Floor', 'Walking + Floor']])
    tables = kin._reconstruct_kinematic_table_from_aggregates({'_inst': aggregate}, min_bout_count_for_features=6)
    kin.calculate_and_save_kinematic_summary(tables, str(tmp_path))
    sheet = pd.read_excel(tmp_path / '5_Kinematic_Features.xlsx', sheet_name='Pose + Orient (inst)')
    group = sheet.loc[sheet.individual == '--- AVERAGE ---'].iloc[0]
    assert group['Standing+Floor_speed_mean_mean'] == group['Walking+Floor_speed_mean_mean'] == 3.
    assert sheet.columns.get_loc('Walking+Floor_speed_mean_mean') < sheet.columns.get_loc('Standing+Floor_speed_mean_mean')










def test_fdr_preserves_untested_cells_and_corrects_only_finite_tests(tmp_path, monkeypatch):
    order = ['Walking', 'Standing', 'Groom-Head']
    raw = pd.DataFrame([[.01, .03, np.nan], [.2, .9, np.nan], [np.nan]*3], index=order, columns=order)
    a = pd.DataFrame([[0., .5, .5], [1., 0., 0.], [np.nan]*3], index=order, columns=order)
    b = a * .5
    monkeypatch.setattr(tr, '_calculate_p_values_for_comparison', lambda *args, **kwargs: raw.copy())
    monkeypatch.setattr(tr, '_generate_comparison_heatmap', lambda *args, **kwargs: None)
    tr._run_comparative_analysis({'A': [('a', a)], 'B': [('b', b)]}, {'A': a, 'B': b}, order, str(tmp_path))
    result = pd.read_csv(tmp_path / 'all_transition_stats.csv').pivot(index='from_behavior', columns='to_behavior', values='p_value_corrected').reindex(index=order, columns=order)
    # BH: sorted p * 4/rank, followed by reverse cumulative minima.
    expected = np.array([[.04, .06, np.nan], [4 * .2 / 3, .9, np.nan], [np.nan]*3])
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)


def time_budget_sources():
    individuals = ['A', 'B', 'C']
    bouts = pd.DataFrame([{'individual': animal, 'group': 'G', 'sex': 'FEMALE', 'category': 'Pose',
        'bout_type': behavior, 'n_bouts': 6, 'n_switches': 1, 'total_length_frames': 240,
        'bout_frequency': .1, 'bout_duration_mean_s': 1.} for animal in individuals for behavior in ('Walking', 'Standing')])
    budgets = pd.DataFrame({'individual': individuals, 'group': ['G'] * 3, 'Walking': [.5]*3, 'Standing': [.5]*3})
    return fm.MMGenFinalizationSources(bout_summary_df=bouts, time_budget_sheets={'Pose (Overall)': budgets})


def test_repertoire_entropy_needs_only_time_budgets_and_keeps_all_animals(tmp_path):
    result = fm.create_feature_matrix_from_sources(str(tmp_path), time_budget_sources(), filter_features_by_bout_count=False)
    # Entropy is normalized by the declared ten-state pose alphabet, not just
    # the two observed states: log(2) / log(10).
    np.testing.assert_allclose(result['Transition_Repertoire_Pose_EntropyNorm'], math.log(2) / math.log(10))
    assert result.individual.tolist() == ['A', 'B', 'C']


def test_group_transition_workbook_retains_from_and_to_states(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, 'plt', __import__('unittest.mock', fromlist=['Mock']).Mock())
    monkeypatch.setattr(tr, 'sns', __import__('unittest.mock', fromlist=['Mock']).Mock())
    records = pd.DataFrame([dict(individual='A', group='G', from_behavior='Walking', to_behavior='Standing', count=3)])
    result = tr._run_transition_analysis_for_category(records, ['Walking', 'Standing'], '6', 'Pose', str(tmp_path),
        save_outputs=False, run_transition_statistics=False)
    path = tmp_path / 'transitions.xlsx'
    fm.save_to_excel_sheets(result['sheets_to_save'], str(path))
    readback = pd.read_excel(path, sheet_name='G_Means')
    assert readback.from_behavior.tolist() == ['Walking', 'Standing']
    assert readback.loc[0, 'Standing'] == 1.


def test_requested_comparison_failure_cannot_return_success(tmp_path, monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(tr, 'plt', Mock())
    monkeypatch.setattr(tr, 'sns', Mock())
    def fail(**kwargs):
        raise OSError('injected comparative failure')
    monkeypatch.setattr(tr, '_run_comparative_analysis', fail)
    records = pd.DataFrame([dict(individual=group, group=group, from_behavior='Walking', to_behavior='Standing', count=3) for group in ['A', 'B']])
    with pytest.raises((RuntimeError, OSError), match='comparative|comparison'):
        tr._run_transition_analysis_for_category(records, ['Walking', 'Standing'], '6', 'Pose', str(tmp_path),
            save_outputs=True, run_transition_statistics=True)




def test_native_edge_bouts_are_excluded_while_frame_budgets_remain(tmp_path, monkeypatch):
    from autobehaver.postinf.mmgen import pipeline
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    inf, pos = tmp_path / 'inf', tmp_path / 'pos'
    inf.mkdir(); pos.mkdir()
    sequences = [['Standing'] * 300, ['Standing'] * 600 + ['Walking'] * 10 + ['Standing'] * 600]
    for movie, labels in enumerate(sequences):
        name = f'G_FEMALE_demo_rpi-1_trial1_movie{movie}_chamber1_positional_calcs.csv'
        values = {column: np.ones(len(labels)) for column in required_keypre_columns(
            include_postural=False, include_spatial=False, include_measurements=False)}
        pd.DataFrame(values).to_csv(pos / name, index=False)
        pd.DataFrame({'Frame Number': np.arange(1, len(labels) + 1), 'Smoothed Pose Label': labels,
            'Smoothed Orient Label': ['Floor'] * len(labels)}).to_csv(inf / name.replace('.csv', '_inference.csv'), index=False)
    settings = MMGenRuntimeSettings(num_workers=1, input_batch_size=1, keep_temp_files=True,
        make_measurements=False, calculate_postural_features=False, calculate_occupancy=False,
        emit_full_outputs=False, make_detailed_transition_outputs=False, filter_features_by_bout_count=False,
        min_bout_count_for_features=6, source_frame_offset=0, video_fps=1, mm_per_px=1)
    captured = {}
    assemble = pipeline.create_feature_matrix_from_sources
    def capture(output, sources, **kwargs):
        captured['bouts'], captured['sleep'] = sources.bout_summary_df.copy(), sources.sleep_summary_df.copy()
        return assemble(output, sources, **kwargs)
    monkeypatch.setattr(pipeline, 'create_feature_matrix_from_sources', capture)
    out = tmp_path / 'out'
    pipeline.run_post_inference_analysis(str(inf), str(pos), str(out), 5, 0, '_inference', runtime_settings=settings)
    matrix = pd.read_csv(out / 'MASTER_FEATURE_MATRIX.csv')
    assert pd.isna(matrix.loc[0, 'KinematicBB_Pose+Orient_Standing+Floor_speed_mean_mean'])
    standing = captured['bouts'].query("category == 'Pose' and bout_type == 'Standing'").iloc[0]
    assert standing.bout_frequency == 0
    assert pd.isna(standing.bout_duration_mean_s)
    assert standing.n_bouts == 0
    assert standing.total_length_frames == 1500
    walking = captured['bouts'].query("category == 'Pose' and bout_type == 'Walking'").iloc[0]
    assert walking.n_bouts == 1
    np.testing.assert_allclose(walking.bout_frequency, 1 / 1510)
    sleep = captured['sleep'].query("bout_type == 'Standing'").iloc[0]
    assert pd.isna(sleep.sleep_bout_duration_mean_s)
    assert sleep.n_sleep_bouts == 0


def test_mixed_spatial_coverage_retains_entropy_with_real_workers(tmp_path):
    rows = pd.DataFrame([{'individual': animal, 'source_file': animal + '.csv', 'frame_idx': index,
        'r_norm': .5, 'relative_angle': 0., 'speed_mm_s': 1., 'masked_pose_valid': 'Walking',
        'masked_orient_valid': 'Floor', 'masked_pose_orient_valid': 'Walking + Floor'}
        for animal in ['A', 'C'] for index in range(120)])
    sources = replace(time_budget_sources(), spatial_df=rows)
    outputs = []
    for workers in (1, 2):
        out = tmp_path / str(workers)
        out.mkdir()
        outputs.append(fm.create_feature_matrix_from_sources(str(out), sources,
            filter_features_by_bout_count=False, spatial_num_workers=workers))
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_dtype=False, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outputs[1]['Transition_Repertoire_Pose_EntropyNorm'], math.log(2) / math.log(10))
