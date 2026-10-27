"""Curated feature groups and grouping rules for FeatureImp enrichment analysis."""

from __future__ import annotations

# Manually curated groups (small, hand-maintained)

fsea_feature_groups = {
    'StdMeasurements_All': [
        'StdMeasure_length_avg_mean',
        'StdMeasure_headwidth_avg_mean',
        'StdMeasure_thorax_width_avg_mean',
        'StdMeasure_abdomenwidth_avg_mean',
        'StdMeasure_winglength_avg_mean',
        'StdMeasure_length_side_mean',
        'StdMeasure_headheight_side_mean',
        'StdMeasure_thoraxheight_side_mean',
        'StdMeasure_abdomenheight_side_mean',
        'StdMeasure_body_area_top_mean',
        'StdMeasure_body_area_side_mean',
        # Also include _std versions
        'StdMeasure_abdomenheight_side_std',
        'StdMeasure_body_area_side_std',
        'StdMeasure_body_area_top_std',
        'StdMeasure_headheight_side_std',
        'StdMeasure_length_side_std',
        'StdMeasure_thoraxheight_side_std',
        'StdMeasure_winglength_avg_std',
        # normalized summary measurements
        'StdMeasure_length_norm_mean',
        'StdMeasure_headwidth_norm_mean',
        'StdMeasure_abdomenwidth_norm_mean',
        'StdMeasure_winglength_norm_mean',
        'StdMeasure_bodyarea_top_norm_mean',
        'StdMeasure_bodyarea_side_norm_mean',
    ],
}


# Kinematic grouping logic helpers

KINEMATIC_MEASUREMENT_GROUPS = {
    'Velocity': ['speed', 'velocity_fwd', 'velocity_str', 'angular_vel', 'curvature'],
    'Acceleration': ['accel_mag', 'accel_speed', 'accel_fwd', 'accel_str', 'angular_accel'],
    'Jerk': ['jerk_mag', 'jerk_speed', 'jerk_fwd', 'jerk_str', 'angular_jerk'],
    'FrontLeg': [
        'frontleg_velocity',
        'frontleg_acceleration',
        'frontleg_jerk',
        'frontleg_angvel',
        'frontleg_angaccel',
        'frontleg_angjerk',
    ],
    'MidLeg': [
        'midleg_velocity',
        'midleg_acceleration',
        'midleg_jerk',
        'midleg_angvel',
        'midleg_angaccel',
        'midleg_angjerk',
    ],
    'BackLeg': [
        'backleg_velocity',
        'backleg_acceleration',
        'backleg_jerk',
        'backleg_angvel',
        'backleg_angaccel',
        'backleg_angjerk',
    ],
    'Head': ['head_velocity', 'head_acceleration', 'head_jerk'],
}

KINEMATIC_STATISTICS = [
    'mean_mean',
    'mean_std',
    'std_mean',
    'std_std',
    'min_mean',
    'min_std',
    'max_mean',
    'max_std',
    'median_mean',
    'median_std',
    'skewness_mean',
    'skewness_std',
]

# For grooming poses, restrict which leg/head kinematic groups are created
# (to focus on the body parts that are actually active in each grooming type)
GROOMING_LEG_GROUP_FILTERS = {
    # head/frontleg grooming uses the head and front legs
    'Groom-Head': {'FrontLeg', 'Head'},
    'Groom-FrontLegs': {'FrontLeg', 'Head'},

    # side-leg grooming uses the front and middle legs
    'Groom-SideLeg': {'FrontLeg', 'MidLeg'},

    # abdomen / backleg / wing grooming are back-leg-driven
    'Groom-Abdomen': {'BackLeg'},
    'Groom-BackLegs': {'BackLeg'},
    'Groom-Wings': {'BackLeg'},
}

# Only these poses get pose-level body movement groups (Velocity/Acceleration/Jerk)
LOCOMOTOR_POSES_FOR_BODY_MOVEMENT = {'Walking', 'Scramble', 'Jump'}

KINEMATIC_BODY_MOVEMENT_GROUPS = {'Velocity', 'Acceleration', 'Jerk'}


KINEMATIC_CROSS_POSE_GROUPS = {
    'FrontLeg': [
        'frontleg_velocity',
        'frontleg_acceleration',
        'frontleg_jerk',
        'frontleg_angvel',
        'frontleg_angaccel',
        'frontleg_angjerk',
    ],
    'MidLeg': [
        'midleg_velocity',
        'midleg_acceleration',
        'midleg_jerk',
        'midleg_angvel',
        'midleg_angaccel',
        'midleg_angjerk',
    ],
    'BackLeg': [
        'backleg_velocity',
        'backleg_acceleration',
        'backleg_jerk',
        'backleg_angvel',
        'backleg_angaccel',
        'backleg_angjerk',
    ],
    'Head': ['head_velocity', 'head_acceleration', 'head_jerk'],
    'BodyMovement': [
        'speed',
        'velocity_fwd',
        'velocity_str',
        'accel_mag',
        'accel_speed',
        'accel_fwd',
        'accel_str',
        'jerk_mag',
        'jerk_speed',
        'jerk_fwd',
        'jerk_str',
    ],
}

POSTURAL_ORIENTATIONS = ['Floor', 'Ceiling', 'Wall-Side']
