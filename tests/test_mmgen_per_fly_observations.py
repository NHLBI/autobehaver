"""Scientific regression cases for pooled, discontinuous per-fly observations."""

import numpy as np
import pandas as pd
import pytest

from autobehaver.label_gaps import fill_short_label_gaps
from autobehaver.postinf.mmgen.bouts import extract_bouts_from_label_sequence
from autobehaver.postinf.mmgen.worker_bouts import build_bouts_and_valid_arrays
from autobehaver.postinf.mmgen.worker_summaries import compute_bout_and_sleep_summaries
from autobehaver.postinf.mmgen.summary_pooling import pool_bout_summaries, pool_sleep_summaries
from autobehaver.postinf.mmgen.measurements import extract_measurement_candidates, summarize_measurement_candidates
from autobehaver.postinf.mmgen.grooming import complete_grooming_episodes
from autobehaver.postinf.mmgen.feature_schema import feature_columns


def recording_summary(labels, movie, fps=1):
    orient = np.array(['Floor'] * len(labels), dtype=object)
    result = build_bouts_and_valid_arrays(masked_poses=labels, masked_orients=orient,
        has_orient=True, min_bout=1, num_frames=len(labels))
    bouts, sleep = compute_bout_and_sleep_summaries(pose_bouts=result[0],
        pose_orient_bouts_for_boutsummary=result[3], orient_bouts_for_boutsummary=result[4],
        masked_poses=np.asarray(labels, dtype=object), masked_orients=orient, has_orient=True,
        num_frames=len(labels), fps=fps, sleep_threshold_frames=3, group='G',
        individual_id=f'G_FEMALE_rpi-1_trial1_movie{movie}_chamber1', sex='FEMALE',
        source_file=f'movie{movie}.csv')
    for table in (bouts, sleep):
        table['animal_id'] = 'fly'
    return bouts, sleep, result


def test_complete_bouts_pool_events_not_recording_means_and_keep_edge_time():
    a, _, _ = recording_summary(['Standing'] + ['Walking'] * 2 + ['Standing'] + ['Walking'] * 2 + ['Standing'], 0)
    b, _, _ = recording_summary(['Standing'] + ['Walking'] * 8 + ['Standing'], 1)
    pooled = pool_bout_summaries(pd.concat([a, b], ignore_index=True))
    walk = pooled.query("category == 'Pose' and bout_type == 'Walking'").iloc[0]
    assert walk.n_bouts == 3
    assert walk.bout_duration_mean_s == 4
    assert walk.bout_frequency == 3 / 17
    assert walk.bout_duration_coeff_of_var == pytest.approx(np.std([2, 2, 8]) / 4)
    assert walk.total_length_frames == 12
    # Absent behaviors still contribute recording exposure, not missing denominators.
    c, _, _ = recording_summary(['Standing'] * 20, 2)
    pooled = pool_bout_summaries(pd.concat([a, b, c], ignore_index=True))
    walk = pooled.query("category == 'Pose' and bout_type == 'Walking'").iloc[0]
    assert walk.bout_frequency == 3 / 37
    stand = pooled.query("category == 'Pose' and bout_type == 'Standing'").iloc[0]
    assert stand.total_length_frames == 25


def test_missing_gap_breaks_intervals_and_incomplete_bouts_keep_frame_labels():
    labels = ['Standing', 'Walking', 'Walking', 'Standing', None, None,
              'Standing', 'Walking', 'Walking', 'Standing']
    table, _, result = recording_summary(labels, 0)
    walk = table.query("category == 'Pose' and bout_type == 'Walking'").iloc[0]
    assert walk.n_bouts == 2
    assert walk.interval_count == 0
    assert walk.observed_frames == 8
    assert walk.bout_frequency == .25
    assert result[5][0] == 'Standing' and result[5][-1] == 'Standing'
    assert not result[0][0]['complete'] and not result[0][-1]['complete']


def test_sleep_pools_complete_event_variance_and_excludes_edge_sleep():
    _, a, _ = recording_summary(['Walking'] + ['Standing'] * 4 + ['Walking'], 0)
    _, b, _ = recording_summary(['Walking'] + ['Standing'] * 8 + ['Walking'], 1)
    _, c, _ = recording_summary(['Standing'] * 20, 2)
    result = pool_sleep_summaries(pd.concat([a, b, c], ignore_index=True))
    sleep = result.query("bout_type == 'Standing'").iloc[0]
    assert sleep.n_sleep_bouts == 2
    assert sleep.sleep_bout_duration_mean_s == 6
    assert sleep.sleep_bout_duration_std_s == 2
    assert sleep.sleep_time_budget == 12 / 36


@pytest.mark.parametrize('gap,expected', [(1, True), (4, True), (5, False)])
def test_gap_fill_only_matching_internal_flanks(gap, expected):
    labels = [None, 'Walking'] + [None] * gap + ['Walking', '', 'Standing', None]
    actual, filled = fill_short_label_gaps(labels)
    assert filled.sum() == (gap if expected else 0)
    assert pd.isna(actual[0]) and pd.isna(actual[-1])
    assert actual[-3] == ''


def test_nullable_labels_and_smoothing_are_stable_before_mmgen():
    from autobehaver.inference import smooth_bouts_vectorized
    values, filled = fill_short_label_gaps(['Walking', pd.NA, 'Walking', pd.NA])
    assert values[1] == 'Walking' and filled.tolist() == [False, True, False, False]
    labels = ['Walking'] * 10 + ['Standing', None] + ['Walking'] * 10
    actual = smooth_bouts_vectorized(labels, pd.DataFrame(index=range(len(labels))), 5, 'Pose')
    assert actual[11] == 'Walking'
    np.testing.assert_array_equal(fill_short_label_gaps(actual)[0], actual)
    build_bouts_and_valid_arrays(masked_poses=['Walking', pd.NA, 'Standing'],
        masked_orients=['Floor'] * 3, has_orient=True, min_bout=1, num_frames=3)


def test_observed_time_is_identical_in_overall_and_sex_specific_budgets(tmp_path):
    from autobehaver.postinf.mmgen.time_budgets import calculate_and_save_time_budgets
    table, _, _ = recording_summary(['Walking'] * 5 + [None] * 5, 0)
    sheets = calculate_and_save_time_budgets(table, {'fly': 10}, str(tmp_path), save_outputs=False)
    for prefix, feature in [('Pose', 'Walking'), ('Orientation', 'Floor'), ('Pose-Orient', 'Walking + Floor')]:
        values = [sheets[f'{prefix} ({kind})'].query("individual == 'fly'")[feature].iloc[0]
                  for kind in ('Overall', 'by Sex')]
        assert values == [1., 1.]


def test_grooming_blips_and_switch_rates_use_observed_exposure():
    from autobehaver.postinf.mmgen.feature_matrix import _compute_spatial_feature_summary_core
    labels = ['Standing'] * 3 + ['Groom-Head'] * 5 + ['Walking'] + ['Groom-Wings'] * 5 + ['Standing'] * 3
    valid = np.array(labels, dtype=object); valid[8] = np.nan
    spatial = pd.DataFrame({'individual': 'fly', 'source_file': 'a', 'frame_idx': range(17),
        'masked_pose': labels, 'masked_pose_valid': valid, 'masked_orient': 'Floor',
        'r_norm': .5, 'relative_angle': 0.})
    matrix = pd.DataFrame({'individual': ['fly']})
    result = _compute_spatial_feature_summary_core(spatial, None, matrix,
        video_fps=1., min_context_frames=1, groom_episode_max_gap_frames=2)
    assert result.loc[0, 'BoutSummary_GroomEpisode_All_MeanDuration_s'] == 11
    assert result.loc[0, 'BoutSummary_GroomEpisode_All_EpisodesPerMin'] == pytest.approx(60 / 17)
    spatial.loc[8, 'masked_pose'] = None
    result = _compute_spatial_feature_summary_core(spatial, None, matrix,
        video_fps=1., min_context_frames=1, groom_episode_max_gap_frames=2)
    assert 'BoutSummary_GroomEpisode_All_MeanDuration_s' not in result
    table, _, _ = recording_summary(['Walking'] * 25 + ['Standing'] * 25 + [None] * 50, 0)
    table['n_switches'] = ((table.category == 'Pose') & (table.bout_type == 'Walking')).astype(int)
    result = _compute_spatial_feature_summary_core(spatial, pool_bout_summaries(table), matrix,
        video_fps=1., min_context_frames=1, groom_episode_max_gap_frames=2)
    assert result.loc[0, 'Transition_Repertoire_Pose_SwitchRate_per_min'] == 1.2


def test_measurement_workbook_names_pooled_sample_counts(tmp_path, monkeypatch):
    from autobehaver.postinf.mmgen import measurements
    data = pd.DataFrame({'Antennae_x': [1., 2., 3.], 'Antennae_y': 0.,
                         'AbdomenTip_x': 0., 'AbdomenTip_y': 0.})
    selected = extract_measurement_candidates(['Standing'] * 3, ['Wall-Side'] * 3,
        data, 'G_FEMALE_rpi-1_trial1_movie0_chamber1', 'G', 'FEMALE')
    captured = {}
    monkeypatch.setattr(measurements, 'save_to_excel_sheets', lambda sheets, path: captured.update(sheets))
    measurements.calculate_and_save_pooled_measurements(selected, str(tmp_path), mm_per_px=1.)
    sheet = captured['Combined Measurements']
    animal = sheet.loc[sheet.individual.astype(str).str.startswith('G_FEMALE_')]
    assert animal['length_side_n_samples'].iloc[0] == 3
    assert not any('mean_samples_per_recording' in str(c) for c in sheet)


def test_edge_transitions_use_observed_departures_for_reliability_filter(tmp_path):
    from autobehaver.postinf.mmgen.categories import category_transition_tables
    from autobehaver.postinf.mmgen.transitions import calculate_and_save_transition_analysis
    from autobehaver.postinf.mmgen.feature_matrix import MMGenFinalizationSources, create_feature_matrix_from_sources
    summaries, observations = [], []
    for movie in range(6):
        summary, _, valid = recording_summary(['Walking'] * 5 + ['Standing'] * 5, movie)
        summaries.append(summary)
        counts, _ = category_transition_tables(pose_valid=valid[5], orient_valid=valid[6],
            combined_valid=valid[7], source_file=f'movie{movie}', individual='fly', group='G')
        observations.append(counts)
    pooled = pool_bout_summaries(pd.concat(summaries))
    assert pooled.n_bouts.sum() == 0
    transitions = calculate_and_save_transition_analysis(pd.concat(observations), str(tmp_path),
        save_outputs=False, run_transition_statistics=False)
    sources = MMGenFinalizationSources(bout_summary_df=pooled,
        transition_pose_tidy_df=transitions['pose_tidy_df'],
        transition_pose_orient_tidy_df=transitions['pose_orient_tidy_df'])
    column = 'Transition_Pose_Walking_to_Standing'
    for threshold, expected in [(6, 1.), (7, np.nan)]:
        matrix = create_feature_matrix_from_sources(str(tmp_path), sources,
            filter_features_by_bout_count=True, min_bout_count_for_transitions=threshold)
        np.testing.assert_allclose(matrix.loc[0, column], expected, equal_nan=True)


def candidates(lengths, poses, movie):
    frame = pd.DataFrame({'Antennae_x': lengths, 'Antennae_y': 0.,
                          'AbdomenTip_x': 0., 'AbdomenTip_y': 0.})
    return extract_measurement_candidates(poses, ['Floor'] * len(poses), frame,
        f'G_FEMALE_rpi-1_trial1_movie{movie}_chamber1', 'G', 'FEMALE')


def test_size_selection_uses_one_pose_and_one_sample_per_animal():
    a = candidates([1., 3.], ['Standing'] * 2, 0)
    b = candidates([100.] * 200, ['Walking'] * 200, 1)
    result = summarize_measurement_candidates(pd.concat([a, b]))
    assert result.loc[0, 'length_top_mean'] == 2
    assert result.loc[0, 'length_top_std'] == pytest.approx(np.std([1, 3], ddof=1))
    assert result.loc[0, 'length_top_n_samples'] == 2
    b = candidates([5., 7., 9.], ['Standing'] * 3, 1)
    result = summarize_measurement_candidates(pd.concat([b, a]))
    assert result.loc[0, 'length_top_mean'] == 5
    assert result.loc[0, 'length_top_std'] == pytest.approx(np.std([1, 3, 5, 7, 9], ddof=1))
    c = candidates(np.arange(400.), ['Standing'] * 400, 2)
    d = candidates(np.arange(700.), ['Standing'] * 700, 3)
    forward = summarize_measurement_candidates(pd.concat([c, d]))
    reverse = summarize_measurement_candidates(pd.concat([d, c]))
    pd.testing.assert_frame_equal(forward, reverse)
    assert forward.loc[0, 'length_top_n_samples'] == 100


def test_grooming_requires_observed_flanks_and_never_bridges_missing_data():
    seq = ['Groom-Head'] * 3 + ['Standing'] * 3 + ['Groom-Wings'] * 4 + ['Standing'] * 3
    assert complete_grooming_episodes(seq, np.arange(len(seq)), 2) == [(6, 9)]
    seq = ['Standing'] * 3 + ['Groom-Head'] * 4 + [None] + ['Groom-Wings'] * 4 + ['Standing'] * 3
    assert complete_grooming_episodes(seq, np.arange(len(seq)), 2) == []


def test_wall_side_stance_removed_without_changing_other_orientations():
    names = set(feature_columns())
    assert 'Postural_Wall-Side_stance_area_mean' not in names
    assert 'Postural_Wall-Side_stance_area_std' not in names
    assert {'Postural_Floor_stance_area_mean', 'Postural_Ceiling_stance_area_mean'} <= names


def test_native_measurement_candidates_pool_and_postural_output_excludes_side_stance(tmp_path, monkeypatch):
    from autobehaver.postinf.mmgen import pipeline
    from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    inf, pos = tmp_path / 'inf', tmp_path / 'pos'
    inf.mkdir(); pos.mkdir()
    labels = ['Standing'] * 20 + ['Walking'] * 20 + ['Standing'] * 20 + ['Walking'] * 20
    orients = ['Floor'] * 40 + ['Wall-Side'] * 20 + ['Ceiling'] * 20
    for movie, length in enumerate([2., 6.]):
        name = f'G_FEMALE_rpi-1_trial1_movie{movie}_chamber1_positional_calcs.csv'
        values = {c: np.full(80, .99 if c.endswith('_likelihood') else 1.)
                  for c in required_keypre_columns(include_postural=True, include_spatial=False, include_measurements=True)}
        values.update(Antennae_x=np.full(80, length), Antennae_y=np.zeros(80),
                      AbdomenTip_x=np.zeros(80), AbdomenTip_y=np.zeros(80),
                      ThoraxLeft_x=np.zeros(80), ThoraxRight_x=np.full(80, 2.))
        pd.DataFrame(values).to_csv(pos / name, index=False)
        pd.DataFrame({'Frame Number': np.arange(1, 81), 'Smoothed Pose Label': labels,
                      'Smoothed Orient Label': orients}).to_csv(inf / name.replace('.csv', '_inference.csv'), index=False)
    settings = MMGenRuntimeSettings(num_workers=1, input_batch_size=1, keep_temp_files=True,
        make_measurements=True, calculate_postural_features=True, calculate_occupancy=False,
        emit_full_outputs=False, make_detailed_transition_outputs=False,
        filter_features_by_bout_count=False, mm_per_px=1., video_fps=1.)
    captured = {}
    original = pipeline.calculate_and_save_pooled_measurements
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['measurements'] = result.copy()
        return result
    monkeypatch.setattr(pipeline, 'calculate_and_save_pooled_measurements', capture)
    out = tmp_path / 'out'
    pipeline.run_post_inference_analysis(str(inf), str(pos), str(out), 5, 0, '_inference', runtime_settings=settings)
    result = captured['measurements']
    assert len(result) == 1
    assert result.loc[0, 'length_top_mean'] == 4.
    assert result.loc[0, 'length_top_n_samples'] == 40
    assert result.loc[0, 'length_top_std'] == pytest.approx(np.std([2.] * 20 + [6.] * 20, ddof=1))
    matrix = pd.read_csv(out / 'MASTER_FEATURE_MATRIX.csv')
    assert len(matrix) == 1
    assert not any('_asym_' in column or column.endswith('_asym') for column in matrix)
    assert 'Postural_Floor_overall_asymmetry_LR_mean' in matrix
    assert len([c for c in matrix if c in feature_columns()]) == 9406
    assert 'Postural_Wall-Side_stance_area_mean' not in matrix
    assert np.isfinite(matrix.loc[0, 'Postural_Floor_stance_area_mean'])
    from dataclasses import replace
    from autobehaver.postinf.mmgen.matrix_parts import finalize_shard_temp_store, merge_matrix_part_files
    parallel = tmp_path / 'parallel'
    pipeline.run_post_inference_analysis(str(inf), str(pos), str(parallel), 5, 0, '_inference',
        runtime_settings=replace(settings, num_workers=2))
    pd.testing.assert_frame_equal(matrix, pd.read_csv(parallel / 'MASTER_FEATURE_MATRIX.csv'),
        check_exact=False, atol=1e-10, rtol=1e-10)
    parts = tmp_path / 'parts'
    for shard in range(2):
        pipeline.run_post_inference_analysis(str(inf), str(pos), str(parts), 5, 0, '_inference',
            runtime_settings=settings, process_only=True, num_shards=2, shard_index=shard, shard_by='row_identity')
        finalize_shard_temp_store(inference_csv_dir=str(inf), poscalc_dir=str(pos), output_dir=str(parts),
            min_bout=5, edge_skip=0, inf_tag='_inference', runtime_settings=settings, file_filter=None,
            num_shards=2, shard_index=shard, shard_by='row_identity')
    merged = pd.read_csv(merge_matrix_part_files(output_dir=str(parts), num_shards=2))
    pd.testing.assert_frame_equal(matrix, merged, check_exact=False, atol=1e-10, rtol=1e-10)
