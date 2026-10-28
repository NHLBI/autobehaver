"""Check FSEA group membership and pruning rules."""

from __future__ import annotations


def test_generate_bout_timebudget_groups_cleaned_surface() -> None:
    from autobehaver.postinf.featureimp.fsea import _generate_bout_timebudget_groups

    feats = [
        'BoutSummary_Pose_Walking_bout_frequency',
        'TimeBudget_Pose_Walking',
        'TimeBudget_PoseGivenOrient_Walking_Floor',
        'BoutSummary_Orient_Floor_bout_frequency',
        'TimeBudget_Orient_Floor',
        'TimeBudget_OrientGivenPose_Floor_Walking',
        'BoutSummary_Pose+Orient_Walking+Floor_bout_frequency',
        'TimeBudget_PoseOrient_Walking + Floor',
        'BoutSummary_Pose_Scramble_bout_frequency',
        'TimeBudget_Pose_Scramble',
    ]
    groups = _generate_bout_timebudget_groups(feats)
    assert 'PoseStats_Walking' in groups
    assert 'Orient_Floor' in groups
    assert 'PoseStats_Scramble' not in groups
    assert all(not k.startswith('Pose_') for k in groups)
    assert all(not k.startswith('PoseOrient_') for k in groups)
    assert all(not k.startswith('Pose_All_') for k in groups)
    assert all(not k.startswith('Orient_All_') for k in groups)


def test_generate_kinematic_groups_prune_redundant_hybrids_and_fix_head_jerk() -> None:
    from autobehaver.postinf.featureimp.fsea import _generate_kinematic_groups

    feats = [
        'KinematicBB_Pose+Orient_Groom-Abdomen+Floor_backleg_velocity_max_mean',
        'KinematicBB_Pose+Orient_Groom-Abdomen+Floor_frontleg_velocity_max_mean',
        'KinematicBB_Pose+Orient_Standing+Floor_frontleg_velocity_max_mean',
        'KinematicBB_Pose+Orient_Standing+Floor_midleg_velocity_max_mean',
        'KinematicBB_Pose+Orient_Standing+Floor_backleg_velocity_max_mean',
        'KinematicBB_Pose+Orient_Groom-Head+Floor_head_jerk_mean_mean',
        'KinematicBB_Pose+Orient_Walking+Floor_head_jerk_mean_mean',
        'KinematicBB_Pose+Orient_Scramble_speed_mean_mean',
        'KinematicBB_Pose+Orient_Scramble_accel_mag_mean_mean',
        'KinematicBB_Pose+Orient_Scramble_jerk_mag_mean_mean',
        'KinematicBB_Pose+Orient_Scramble_frontleg_velocity_mean_mean',
        'KinematicBB_Pose+Orient_Jump_accel_mag_mean_mean',
        'BoutSummary_Pose_Scramble_bout_frequency',
        'TimeBudget_Pose_Scramble',
        'BoutSummary_Pose_Jump_bout_frequency',
    ]
    groups = _generate_kinematic_groups(feats)
    all_grouped = {f for vals in groups.values() for f in vals}
    # Implausible front-leg abdomen grooming pairing should be pruned.
    assert 'Kinematic_Groom-Abdomen_FrontLeg' not in groups
    # Standing legs should collapse into LegMove.
    assert 'Kinematic_Standing_LegMove' in groups
    assert 'Kinematic_Standing_FrontLeg' not in groups
    # Head jerk belongs to the Head family.
    assert 'Kinematic_Groom-Head_Head' in groups
    assert 'KinematicBB_Pose+Orient_Groom-Head+Floor_head_jerk_mean_mean' in groups['Kinematic_Groom-Head_Head']
    assert 'Kinematic_Walking_Head' in groups
    assert 'KinematicBB_Pose+Orient_Walking+Floor_head_jerk_mean_mean' in groups['Kinematic_Walking_Head']
    # Scramble/Jump use dedicated hybrid groups without redundant movement groups.
    assert 'PoseStatsKinematic_Scramble' in groups
    assert 'BoutSummary_Pose_Scramble_bout_frequency' in groups['PoseStatsKinematic_Scramble']
    assert 'KinematicBB_Pose+Orient_Scramble_speed_mean_mean' in groups['PoseStatsKinematic_Scramble']
    assert 'KinematicBB_Pose+Orient_Scramble_accel_mag_mean_mean' in groups['PoseStatsKinematic_Scramble']
    assert 'KinematicBB_Pose+Orient_Scramble_jerk_mag_mean_mean' in groups['PoseStatsKinematic_Scramble']
    assert 'PoseStatsKinematic_Jump' in groups
    assert 'BoutSummary_Pose_Jump_bout_frequency' in groups['PoseStatsKinematic_Jump']
    assert 'KinematicBB_Pose+Orient_Jump_accel_mag_mean_mean' in groups['PoseStatsKinematic_Jump']
    assert 'Kinematic_Scramble_Velocity' not in groups
    assert 'Kinematic_Scramble_Acceleration' not in groups
    assert 'Kinematic_Scramble_Jerk' not in groups
    assert 'Kinematic_Jump_Acceleration' not in groups


def test_generate_transition_groups_cleaned_surface() -> None:
    from autobehaver.postinf.featureimp.fsea import _generate_transition_groups

    feats = [
        'Transition_Pose_Walking_to_Standing',
        'Transition_Pose_Standing_to_Walking',
        'Transition_Pose_Groom-Head_to_Walking',
        'Transition_Pose_Walking_to_Groom-Head',
    ]
    groups = _generate_transition_groups(feats)
    assert 'Transition_Walking_From' in groups
    assert 'Transition_Walking_To' in groups
    assert 'Transition_Standing_From' in groups
    assert 'Transition_Standing_To' in groups
    assert 'Transition_Groom-Head' in groups
    assert 'Transition_Groom-Head_From' not in groups
    assert 'Transition_Groom-Head_To' not in groups


def test_generate_postural_groups_collapsed_by_orientation_and_spread() -> None:
    from autobehaver.postinf.featureimp.fsea import _generate_postural_groups

    feats = [
        'Postural_Floor_stance_area_mean',
        'Postural_Floor_extension_norm_front_mean',
        'Postural_Wall-Side_overall_asymmetry_TB_mean',
        'Postural_LegSpread_BL_BR_Standing_Ceiling_Mean_deg',
        'Postural_WingSpread_GroomWings_Floor_P95_deg',
    ]
    groups = _generate_postural_groups(feats)
    assert set(groups) == {'Postural_Ceiling', 'Postural_Floor', 'Postural_Wall-Side'}
    assert 'Postural_LegSpread_BL_BR_Standing_Ceiling_Mean_deg' in groups['Postural_Ceiling']
    assert 'Postural_WingSpread_GroomWings_Floor_P95_deg' in groups['Postural_Floor']


def test_generate_spatial_groups_include_fraction_family() -> None:
    from autobehaver.postinf.featureimp.fsea import _generate_spatial_groups

    feats = [
        'Spatial_Walking_Floor_r_norm',
        'Spatial_Standing_Floor_MeanAbsRelAngle',
        'Spatial_Frac_Overall_Center',
        'Spatial_Frac_Overall_Periphery',
    ]
    groups = _generate_spatial_groups(feats)
    assert groups['Spatial_RadialDistance'] == ['Spatial_Walking_Floor_r_norm']
    assert groups['Spatial_RelativeAngle'] == ['Spatial_Standing_Floor_MeanAbsRelAngle']
    assert groups['Spatial_Frac'] == ['Spatial_Frac_Overall_Center', 'Spatial_Frac_Overall_Periphery']


def test_write_fsea_group_documentation_classifies_renamed_pose_groups(tmp_path) -> None:
    from autobehaver.postinf.featureimp.fsea import _write_fsea_group_documentation

    validated_groups = {
        'PoseStats_Walking': ['BoutSummary_Pose_Walking_bout_frequency'],
        'Orient_Floor': ['TimeBudget_Orient_Floor'],
        'PoseStatsKinematic_Scramble': ['KinematicBB_Pose+Orient_Scramble_speed_mean_mean'],
        'Kinematic_Standing_LegMove': ['KinematicBB_Pose+Orient_Standing+Floor_frontleg_velocity_max_mean'],
    }
    _write_fsea_group_documentation(
        validated_groups=validated_groups,
        all_feature_names=[feat for feats in validated_groups.values() for feat in feats],
        fsea_base_folder=str(tmp_path),
        missing_features_by_group={},
        skipped_groups={},
        fsea_min_size=1,
    )
    text = (tmp_path / 'FSEA_GROUP_DEFINITIONS.txt').read_text()
    assert '=== BOUTSUMMARY/TIMEBUDGET GROUPS ===' in text
    assert 'PoseStats_Walking (1 features):' in text
    assert '=== KINEMATIC / HYBRID GROUPS ===' in text
    assert 'PoseStatsKinematic_Scramble (1 features):' in text


def test_manual_measurements_group_is_combined() -> None:
    from autobehaver.postinf.featureimp.feature_groups import fsea_feature_groups

    assert 'StdMeasurements_All' in fsea_feature_groups
    assert 'StdMeasurements_Raw' not in fsea_feature_groups
    assert 'StdMeasurements_Norm' not in fsea_feature_groups
