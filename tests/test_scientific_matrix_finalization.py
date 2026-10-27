"""Present malformed scientific families must not disappear from a complete matrix."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from autobehaver.postinf.mmgen.feature_matrix import (
    MMGenFinalizationSources, create_feature_matrix_from_sources,
    load_finalization_sources_from_output_folder,
)


def base_sources():
    return MMGenFinalizationSources(bout_summary_df=pd.DataFrame([
        {'individual':'grp_trial1_fly1', 'group':'grp', 'sex':'FEMALE',
         'category':'Pose', 'bout_type':'Walking', 'n_bouts':6,
         'total_length_frames':240, 'bout_frequency':.1, 'bout_duration_mean_s':1.}
    ]))


def family_sources(kind, *, duplicate=False):
    row={'individual':'grp_trial1_fly1', 'group':'grp', 'value_mean':1.}
    if kind == 'time_budget':
        row={'individual':'grp_trial1_fly1', 'group':'grp', 'Walking':1.}
    elif kind in ['transitions','sleep']:
        row=({'individual':'grp_trial1_fly1', 'from_behavior':'Walking',
              'to_behavior':'Standing', 'probability':1.} if kind=='transitions' else
             {'individual':'grp_trial1_fly1', 'group':'grp', 'sex':'FEMALE',
              'bout_type':'Standing', 'n_sleep_bouts':1, 'sleep_time_budget':.5})
    elif kind=='spatial':
        row={'individual':'grp_trial1_fly1', 'r_norm':.5,
             'masked_pose_valid':'Walking', 'masked_orient_valid':'Floor',
             'masked_pose_orient_valid':'Walking + Floor'}
    if not duplicate:
        row.pop('individual')
    frame=pd.DataFrame([row,row] if duplicate else [row])
    kwargs={
        'time_budget':{'time_budget_sheets':{'Pose (Overall)':frame}},
        'measurements':{'measurements_df':frame},
        'postural':{'postural_summary_df':frame},
        'kinematics':{'kinematic_sheets':{'Pose (inst)':frame}},
        'transitions':{'transition_pose_tidy_df':frame},
        'sleep':{'sleep_summary_df':frame},
        'spatial':{'spatial_df':frame},
    }[kind]
    return replace(base_sources(),**kwargs)


@pytest.mark.parametrize('kind', ['time_budget','measurements','postural','kinematics','transitions','sleep','spatial'])
def test_present_malformed_family_does_not_publish_partial_matrix(tmp_path,kind):
    published=tmp_path/'MASTER_FEATURE_MATRIX.csv'
    published.write_bytes(b'previous complete scientific generation')
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        create_feature_matrix_from_sources(str(tmp_path),family_sources(kind),filter_features_by_bout_count=False)
    assert published.read_bytes()==b'previous complete scientific generation'


@pytest.mark.parametrize('kind', ['time_budget','measurements','postural','kinematics','transitions','sleep'])
def test_duplicate_finalized_family_keys_do_not_average_or_multiply_rows(tmp_path,kind):
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        create_feature_matrix_from_sources(str(tmp_path),family_sources(kind,duplicate=True),filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()


def test_duplicate_finalized_bout_keys_are_not_silently_averaged(tmp_path):
    sources=base_sources()
    sources.bout_summary_df=pd.concat([sources.bout_summary_df,sources.bout_summary_df],ignore_index=True)
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        create_feature_matrix_from_sources(str(tmp_path),sources,filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()


def test_present_corrupt_measurement_workbook_does_not_become_absent(tmp_path):
    (tmp_path/'3_Standardized_Measurements.xlsx').write_bytes(b'not a valid workbook')
    with pytest.raises((RuntimeError,ValueError)):
        load_finalization_sources_from_output_folder(str(tmp_path))


def test_requested_spatial_file_missing_is_not_treated_as_disabled(tmp_path):
    sources=replace(base_sources(),spatial_data_path=str(tmp_path/'missing.csv'))
    with pytest.raises((RuntimeError,FileNotFoundError)):
        create_feature_matrix_from_sources(str(tmp_path),sources,filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()


def test_valid_absent_and_empty_optional_families_are_unchanged(tmp_path):
    sources=replace(base_sources(),measurements_df=pd.DataFrame(),postural_summary_df=pd.DataFrame(),
                    spatial_df=pd.DataFrame(),sleep_summary_df=pd.DataFrame())
    matrix=create_feature_matrix_from_sources(str(tmp_path),sources,filter_features_by_bout_count=False)
    assert matrix.individual.tolist()==['grp_trial1_fly1']
    assert matrix.loc[0,'BoutSummary_Pose_Walking_bout_frequency']==.1


def test_nonnumeric_present_feature_does_not_silently_disappear_downstream(tmp_path):
    sources=replace(base_sources(),time_budget_sheets={'Pose (Overall)':pd.DataFrame([
        {'individual':'grp_trial1_fly1','group':'grp','Walking':'garbage'}])})
    with pytest.raises(ValueError,match='nonnumeric scientific feature'):
        create_feature_matrix_from_sources(str(tmp_path),sources,filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()


def test_display_averages_and_blank_separators_are_not_real_animals(tmp_path):
    frame=pd.DataFrame([
        {'individual':'--- AVERAGE ---','group':'grp','Walking':1.},
        {'individual':'','group':'','Walking':''},
        {'individual':'grp_trial1_fly1','group':'grp','Walking':1.},
    ])
    matrix=create_feature_matrix_from_sources(str(tmp_path),replace(base_sources(),
        time_budget_sheets={'Pose (Overall)':frame}),filter_features_by_bout_count=False)
    assert matrix.individual.tolist()==['grp_trial1_fly1']
    assert float(matrix.loc[0,'TimeBudget_Pose_Walking'])==1.


def test_blank_animal_with_real_values_is_not_silently_dropped(tmp_path):
    frame=pd.DataFrame([{'individual':None,'group':'grp','Walking':1.}])
    with pytest.raises(RuntimeError,match='time-budget'):
        create_feature_matrix_from_sources(str(tmp_path),replace(base_sources(),
            time_budget_sheets={'Pose (Overall)':frame}),filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()


def test_unselected_family_animal_is_not_silently_discarded(tmp_path):
    frame=pd.DataFrame([{'individual':'unexpected','group':'grp','Walking':1.}])
    with pytest.raises(RuntimeError,match='time-budget'):
        create_feature_matrix_from_sources(str(tmp_path),replace(base_sources(),
            time_budget_sheets={'Pose (Overall)':frame}),filter_features_by_bout_count=False)
    assert not (tmp_path/'MASTER_FEATURE_MATRIX.csv').exists()
