"""
Feature-set enrichment analysis helpers.

This module contains the feature grouping logic and small utilities used by
the Feature Importance (featureimp) post-inference pipeline, primarily to
support the FSEA (feature set enrichment analysis) stage.

The code here is kept focused on FSEA grouping, result cleaning, and documentation.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import pandas as pd

from autobehaver.postinf.featureimp.feature_groups import (
    KINEMATIC_MEASUREMENT_GROUPS,
    KINEMATIC_STATISTICS,
    GROOMING_LEG_GROUP_FILTERS,
    KINEMATIC_BODY_MOVEMENT_GROUPS,
    POSTURAL_ORIENTATIONS,
    fsea_feature_groups,
)

POSE_STATS_GROUP_PREFIX = 'PoseStats_'
POSE_STATS_KINEMATIC_GROUP_PREFIX = 'PoseStatsKinematic_'
HYBRID_POSE_KINEMATIC_POSES = {'Jump', 'Scramble'}


def _generate_bout_timebudget_groups(all_feature_names, skip_existence_check=False):
    """Generate broad BoutSummary/TimeBudget groups.

    The cleaned grouping surface keeps only the broad pose-statistics and
    orientation-level groups used for enrichment. Tiny pose+orient-specific
    groups are omitted because they are redundant and often fail minimum-size
    criteria. Jump and Scramble suppress the tiny pure pose-statistics groups
    because those behaviors are represented by dedicated PoseStatsKinematic
    hybrid groups downstream.
    """
    groups: Dict[str, List[str]] = {}

    def _add(group_name: str | None, feat: str) -> None:
        if not group_name:
            return
        groups.setdefault(group_name, [])
        if feat not in groups[group_name]:
            groups[group_name].append(feat)

    def _clean_pose_orient_label(label: str) -> str:
        return str(label).replace(' ', '')

    def _pose_group_name(pose: str) -> str | None:
        if pose in HYBRID_POSE_KINEMATIC_POSES:
            return None
        return f'{POSE_STATS_GROUP_PREFIX}{pose}'

    for feat in all_feature_names:
        feat = str(feat)
        if feat.startswith('BoutSummary_Orient_'):
            orient = feat[len('BoutSummary_Orient_'):].split('_', 1)[0]
            _add(f'Orient_{orient}', feat)
            continue
        if feat.startswith('TimeBudget_OrientGivenPose_'):
            orient = feat[len('TimeBudget_OrientGivenPose_'):].split('_', 1)[0]
            _add(f'Orient_{orient}', feat)
            continue
        if feat.startswith('TimeBudget_Orient_') and not feat.startswith('TimeBudget_OrientGivenPose_'):
            orient = feat[len('TimeBudget_Orient_'):].split('_', 1)[0]
            _add(f'Orient_{orient}', feat)
            continue

        if feat.startswith('BoutSummary_Pose_'):
            pose = feat[len('BoutSummary_Pose_'):].split('_', 1)[0]
            _add(_pose_group_name(pose), feat)
            continue
        if feat.startswith('TimeBudget_PoseGivenOrient_'):
            pose = feat[len('TimeBudget_PoseGivenOrient_'):].split('_', 1)[0]
            _add(_pose_group_name(pose), feat)
            continue
        if feat.startswith('TimeBudget_Pose_') and not feat.startswith('TimeBudget_PoseGivenOrient_') and not feat.startswith('TimeBudget_PoseOrient_'):
            pose = feat[len('TimeBudget_Pose_'):].split('_', 1)[0]
            _add(_pose_group_name(pose), feat)
            continue

        if feat.startswith('BoutSummary_Pose+Orient_'):
            raw = feat[len('BoutSummary_Pose+Orient_'):].split('_', 1)[0]
            label = _clean_pose_orient_label(raw)
            if '+' in label:
                pose, orient = label.split('+', 1)
                _add(_pose_group_name(pose), feat)
                _add(f'Orient_{orient}', feat)
            continue
        if feat.startswith('TimeBudget_PoseOrient_'):
            raw = _clean_pose_orient_label(feat[len('TimeBudget_PoseOrient_'):])
            if '+' in raw:
                pose, orient = raw.split('+', 1)
                _add(_pose_group_name(pose), feat)
                _add(f'Orient_{orient}', feat)
            continue

    return {k: v for k, v in sorted(groups.items()) if v}


def _generate_kinematic_groups(all_feature_names, skip_existence_check=False):
    """Generate cleaned kinematic FSEA groups.

    This cleaned grouping surface:
      - treats KinematicBB and KinematicTW features together,
      - keeps broad body-movement groups for locomotion / all-grooming,
      - prunes biologically implausible grooming pose × body-part pairings,
      - keeps Standing as Head + LegMove + body movement,
      - keeps Scramble as Head + LegMove while routing its body-movement
        features into the hybrid group,
      - removes redundant Jump/Scramble body-movement-only subgroups because
        those behaviors are represented by dedicated pose/kinematic hybrids,
      - folds Jump/Scramble pose-timebudget signal into dedicated hybrid groups.
    """
    groups: Dict[str, List[str]] = {}

    normalized_to_actuals: Dict[str, List[str]] = {}
    normalized_features: List[str] = []
    for feat in all_feature_names:
        feat_str = str(feat)
        if feat_str.startswith('KinematicBB_'):
            normalized = 'Kinematic_' + feat_str[len('KinematicBB_'):]
        elif feat_str.startswith('KinematicTW_'):
            normalized = 'Kinematic_' + feat_str[len('KinematicTW_'):]
        else:
            normalized = feat_str
        normalized_features.append(normalized)
        normalized_to_actuals.setdefault(normalized, []).append(feat_str)

    def _expand_actuals(normalized_feat: str) -> List[str]:
        return normalized_to_actuals.get(normalized_feat, [normalized_feat])

    def _add(group_name: str, feat: str) -> None:
        groups.setdefault(group_name, [])
        if feat not in groups[group_name]:
            groups[group_name].append(feat)

    def _measurement_group(metric_part: str) -> str | None:
        for group_name, meas_list in KINEMATIC_MEASUREMENT_GROUPS.items():
            for meas in meas_list:
                if metric_part.startswith(meas + '_') or metric_part == meas:
                    return group_name
        return None

    pose_to_all_kin_feats: Dict[str, List[str]] = {}
    pose_set = set()

    for normalized_feat in normalized_features:
        if not normalized_feat.startswith('Kinematic_'):
            continue
        if normalized_feat.startswith('Kinematic_Pose+Orient_'):
            rest = normalized_feat[len('Kinematic_Pose+Orient_'):]
            if '_' not in rest:
                continue
            pose_orient, metric_part = rest.split('_', 1)
            pose = pose_orient.split('+', 1)[0]
        elif normalized_feat.startswith('Kinematic_Pose_'):
            rest = normalized_feat[len('Kinematic_Pose_'):]
            if '_' not in rest:
                continue
            pose, metric_part = rest.split('_', 1)
        else:
            continue

        pose_set.add(pose)
        actual_feats = _expand_actuals(normalized_feat)
        for actual in actual_feats:
            pose_to_all_kin_feats.setdefault(pose, [])
            if actual not in pose_to_all_kin_feats[pose]:
                pose_to_all_kin_feats[pose].append(actual)

        group_name = _measurement_group(metric_part)
        if group_name is None:
            continue

        is_grooming = pose.startswith('Groom')
        is_body_group = group_name in KINEMATIC_BODY_MOVEMENT_GROUPS
        is_bodypart_group = group_name in {'FrontLeg', 'MidLeg', 'BackLeg', 'Head'}

        if pose == 'Jump' and is_bodypart_group:
            continue

        if pose in HYBRID_POSE_KINEMATIC_POSES and is_body_group:
            continue

        if is_grooming:
            if is_body_group:
                for actual in actual_feats:
                    _add(f'Kinematic_AllGrooming_{group_name}', actual)
                continue
            if is_bodypart_group:
                allowed = GROOMING_LEG_GROUP_FILTERS.get(pose, set())
                if group_name not in allowed:
                    continue

        target_group = f'Kinematic_{pose}_{group_name}'
        for actual in actual_feats:
            _add(target_group, actual)

    # Standing and Scramble collapse individual leg groups into LegMove.
    for pose in ['Standing', 'Scramble']:
        combined: List[str] = []
        for leg_group in ['FrontLeg', 'MidLeg', 'BackLeg']:
            gname = f'Kinematic_{pose}_{leg_group}'
            if gname in groups:
                for feat in groups[gname]:
                    if feat not in combined:
                        combined.append(feat)
                del groups[gname]
        if combined:
            groups[f'Kinematic_{pose}_LegMove'] = combined

    # Fold Jump / Scramble pose-timebudget signal into dedicated hybrid groups so
    # these behaviors remain testable even when the pure pose/timebudget groups are tiny.
    for pose in sorted(HYBRID_POSE_KINEMATIC_POSES):
        combined: List[str] = []
        for feat in all_feature_names:
            feat = str(feat)
            if feat.startswith(f'BoutSummary_Pose_{pose}_'):
                combined.append(feat)
            elif feat.startswith(f'TimeBudget_Pose_{pose}'):
                combined.append(feat)
            elif feat.startswith(f'TimeBudget_PoseGivenOrient_{pose}_'):
                combined.append(feat)
        for feat in pose_to_all_kin_feats.get(pose, []):
            if feat not in combined:
                combined.append(feat)
        if combined:
            groups[f'{POSE_STATS_KINEMATIC_GROUP_PREFIX}{pose}'] = combined

    return {k: v for k, v in sorted(groups.items()) if v}


def _generate_transition_groups(all_feature_names, skip_existence_check=False):
    """Generate cleaned transition groups.

    Walking and Standing keep separate From/To groups. Rarer behaviors use a
    single combined transition group to avoid many tiny, direction-specific sets.
    """
    groups: Dict[str, List[str]] = {}
    by_from: Dict[str, List[str]] = {}
    by_to: Dict[str, List[str]] = {}

    def _add(bucket: Dict[str, List[str]], key: str, feat: str) -> None:
        bucket.setdefault(key, [])
        if feat not in bucket[key]:
            bucket[key].append(feat)

    for feat in all_feature_names:
        feat = str(feat)
        from_pose = None
        to_pose = None
        if feat.startswith('Transition_Pose_'):
            payload = feat[len('Transition_Pose_'):]
            if '_to_' in payload:
                from_pose, to_pose = payload.split('_to_', 1)
        elif feat.startswith('Transition_PoseOrient_'):
            payload = feat[len('Transition_PoseOrient_'):]
            if '_to_' in payload:
                from_raw, to_raw = payload.split('_to_', 1)
                from_pose = from_raw.split('+', 1)[0]
                to_pose = to_raw.split('+', 1)[0]
        if not from_pose or not to_pose:
            continue
        _add(by_from, from_pose, feat)
        _add(by_to, to_pose, feat)

    keep_directional = {'Walking', 'Standing'}
    poses = sorted(set(by_from) | set(by_to))
    for pose in poses:
        if pose in keep_directional:
            if by_from.get(pose):
                groups[f'Transition_{pose}_From'] = by_from[pose]
            if by_to.get(pose):
                groups[f'Transition_{pose}_To'] = by_to[pose]
        else:
            combined: List[str] = []
            for feat in by_from.get(pose, []) + by_to.get(pose, []):
                if feat not in combined:
                    combined.append(feat)
            if combined:
                groups[f'Transition_{pose}'] = combined

    return groups


def _generate_postural_groups(all_feature_names, skip_existence_check=False):
    """Generate broad postural groups by major orientation class.

    In addition to the direct Postural_<Orientation>_* families, this folds the
    LegSpread and WingSpread postural-angle features into the matching
    orientation bucket when an orientation token is embedded in the feature name.
    """
    groups: Dict[str, List[str]] = {}

    def _match_orientation(feat: str) -> str | None:
        for orient in POSTURAL_ORIENTATIONS:
            if feat.startswith(f'Postural_{orient}_') or f'_{orient}_' in feat:
                return orient
        return None

    for feat in all_feature_names:
        feat = str(feat)
        if not feat.startswith('Postural_'):
            continue
        orient = _match_orientation(feat)
        if orient is None:
            continue
        groups.setdefault(f'Postural_{orient}', [])
        if feat not in groups[f'Postural_{orient}']:
            groups[f'Postural_{orient}'].append(feat)

    return {k: sorted(v) for k, v in sorted(groups.items()) if v}


def _generate_spatial_groups(all_feature_names, skip_existence_check=False):
    """Generate aggregate Spatial feature groups.

    Creates three aggregate groups:
    - Spatial_RadialDistance: all *_r_norm features across contexts
    - Spatial_RelativeAngle: all MeanAbsRelAngle / StdAbsRelAngle features
    - Spatial_Frac: all center/periphery occupancy fraction features
    """
    groups: Dict[str, List[str]] = {}
    all_feature_names_set = set(all_feature_names)

    radial_features = []
    for feat in all_feature_names:
        if feat.startswith('Spatial_') and feat.endswith('_r_norm'):
            if skip_existence_check or feat in all_feature_names_set:
                radial_features.append(feat)
    if radial_features:
        groups['Spatial_RadialDistance'] = sorted(radial_features)

    angle_features = []
    for feat in all_feature_names:
        if feat.startswith('Spatial_') and ('MeanAbsRelAngle' in feat or 'StdAbsRelAngle' in feat):
            if skip_existence_check or feat in all_feature_names_set:
                angle_features.append(feat)
    if angle_features:
        groups['Spatial_RelativeAngle'] = sorted(angle_features)

    frac_features = []
    for feat in all_feature_names:
        if feat.startswith('Spatial_Frac_'):
            if skip_existence_check or feat in all_feature_names_set:
                frac_features.append(feat)
    if frac_features:
        groups['Spatial_Frac'] = sorted(frac_features)

    return groups


def _generate_sleep_groups(all_feature_names, skip_existence_check=False):
    """
    Programmatically generates FSEA feature groups for Sleep features.

    Creates a single group 'Sleep_All' containing all SleepFeat_* columns
    present in the feature matrix.
    
    Args:
        all_feature_names: List of all feature names in the feature matrix
        skip_existence_check: If True, includes all theoretically possible features without checking existence
        (Note: This function already works with all_feature_names, so skip_existence_check has no effect)
    """
    groups = {}

    sleep_features = [f for f in all_feature_names if f.startswith('SleepFeat_')]

    if sleep_features:
        groups['Sleep_All'] = sorted(sleep_features)

    return groups

def _write_fsea_group_documentation(validated_groups, all_feature_names, fsea_base_folder,
                                     missing_features_by_group=None, skipped_groups=None, fsea_min_size=1):
    """
    Creates two output files documenting the FSEA feature groups:
    1. CSV membership matrix: feature_name, group_count, then one column per group (0/1)
    2. Text log: detailed group definitions organized by source function
    
    Args:
        validated_groups: dict mapping group names to lists of feature names (actual groups used)
        all_feature_names: list of all feature names in the dataset
        fsea_base_folder: path to FSEA output folder
        missing_features_by_group: dict mapping group_name -> list of missing features
        skipped_groups: dict mapping group_name -> {'reason': str, 'original_features': list, ...}
        fsea_min_size: minimum group size threshold
    """
    if missing_features_by_group is None:
        missing_features_by_group = {}
    if skipped_groups is None:
        skipped_groups = {}
    print("\nCreating feature-to-group membership matrix...")
    membership_data = []
    all_feature_names_list = list(all_feature_names)
    
    for feat in all_feature_names_list:
        row = {'feature_name': feat}
        group_count = 0
        for group_name in sorted(validated_groups.keys()):
            if feat in validated_groups[group_name]:
                row[group_name] = 1
                group_count += 1
            else:
                row[group_name] = 0
        row['group_count'] = group_count
        membership_data.append(row)
    
    # Reorder columns: feature_name, group_count, then all groups
    membership_df = pd.DataFrame(membership_data)
    column_order = ['feature_name', 'group_count'] + sorted(validated_groups.keys())
    membership_df = membership_df[column_order]
    
    membership_csv_path = os.path.join(fsea_base_folder, 'FSEA_FEATURE_GROUP_MEMBERSHIP.csv')
    membership_df.to_csv(membership_csv_path, index=False)
    print(f"Feature-to-group membership matrix saved to: {membership_csv_path}")
    
    # Create detailed group definitions log file
    print("Creating detailed group definitions log...")
    log_path = os.path.join(fsea_base_folder, 'FSEA_GROUP_DEFINITIONS.txt')
    with open(log_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FSEA FEATURE GROUP DEFINITIONS\n")
        f.write("="*80 + "\n\n")
        
        # Organize groups by source function
        source_to_groups = {
            '_generate_bout_timebudget_groups': [],
            '_generate_kinematic_groups': [],
            '_generate_transition_groups': [],
            '_generate_postural_groups': [],
            '_generate_spatial_groups': [],
            '_generate_sleep_groups': [],
            'manual': []
        }
        
        for group_name in sorted(validated_groups.keys()):
            if group_name.startswith('PoseStats_') or group_name.startswith('Orient_') or group_name.startswith('PoseOrient_') or group_name.startswith('Pose_'):
                source_to_groups['_generate_bout_timebudget_groups'].append(group_name)
            elif group_name.startswith('Kinematic_') or group_name.startswith('PoseStatsKinematic_'):
                source_to_groups['_generate_kinematic_groups'].append(group_name)
            elif group_name.startswith('Transition_'):
                source_to_groups['_generate_transition_groups'].append(group_name)
            elif group_name.startswith('Postural_'):
                source_to_groups['_generate_postural_groups'].append(group_name)
            elif group_name.startswith('Spatial_'):
                source_to_groups['_generate_spatial_groups'].append(group_name)
            elif group_name.startswith('Sleep_'):
                source_to_groups['_generate_sleep_groups'].append(group_name)
            else:
                source_to_groups['manual'].append(group_name)
        
        # Write each section
        section_names = {
            '_generate_bout_timebudget_groups': 'BOUTSUMMARY/TIMEBUDGET GROUPS',
            '_generate_kinematic_groups': 'KINEMATIC / HYBRID GROUPS',
            '_generate_transition_groups': 'TRANSITION GROUPS',
            '_generate_postural_groups': 'POSTURAL GROUPS',
            '_generate_spatial_groups': 'SPATIAL GROUPS',
            '_generate_sleep_groups': 'SLEEP GROUPS',
            'manual': 'MANUAL GROUPS'
        }
        
        for source_func, group_list in source_to_groups.items():
            if not group_list:
                continue
            
            f.write(f"\n=== {section_names[source_func]} ===\n")
            if source_func != 'manual':
                f.write(f"Source: {source_func}()\n\n")
            else:
                f.write("Source: fsea_feature_groups (manual definition)\n\n")
            
            for group_name in sorted(group_list):
                features = validated_groups[group_name]
                f.write(f"{group_name} ({len(features)} features):\n")
                for feat in sorted(features):
                    f.write(f"  {feat}\n")
                f.write("\n")
        
        # Write missing features section
        if missing_features_by_group:
            f.write("\n" + "="*80 + "\n")
            f.write("MISSING FEATURES (excluded from groups)\n")
            f.write("="*80 + "\n\n")
            for group_name in sorted(missing_features_by_group.keys()):
                if group_name in validated_groups:  # Only show for groups that were validated
                    missing = missing_features_by_group[group_name]
                    f.write(f"{group_name}: {len(missing)} missing features\n")
                    for feat in sorted(missing):
                        f.write(f"  - {feat}\n")
                    f.write("\n")
        
        # Write skipped groups section
        if skipped_groups:
            f.write("\n" + "="*80 + "\n")
            f.write("SKIPPED GROUPS (not used in FSEA)\n")
            f.write("="*80 + "\n\n")
            for group_name in sorted(skipped_groups.keys()):
                info = skipped_groups[group_name]
                f.write(f"{group_name}: {info['reason']}\n")
                if info['reason'] == 'too_small':
                    f.write(f"  Existing features: {info['existing_count']} (min required: {fsea_min_size})\n")
                    if info.get('missing_features'):
                        f.write(f"  Missing features: {len(info['missing_features'])}\n")
                f.write(f"  Original feature count: {len(info['original_features'])}\n")
                f.write("\n")
    
    print(f"Group definitions log saved to: {log_path}")


def _identify_ungrouped_features(theoretical_groups, all_feature_names, fsea_base_folder):
    """
    Identifies features in the master feature matrix that are not part of any theoretical group.
    This helps identify features that might need additional groups.
    
    Args:
        theoretical_groups: dict of all theoretical groups
        all_feature_names: list of all features in the dataset
        fsea_base_folder: path to save output files
    """
    print("\nIdentifying features not in any theoretical group...")
    
    # Collect all features that are in at least one theoretical group
    all_grouped_features = set()
    for group_name, feature_list in theoretical_groups.items():
        all_grouped_features.update(feature_list)
    
    # Find features that are NOT in any group
    all_feature_names_set = set(all_feature_names)
    ungrouped_features = sorted(all_feature_names_set - all_grouped_features)
    
    print(f"  Total features in dataset: {len(all_feature_names)}")
    print(f"  Features in theoretical groups: {len(all_grouped_features)}")
    print(f"  Ungrouped features: {len(ungrouped_features)}")
    
    if not ungrouped_features:
        print("  All features are covered by at least one group!")
        return
    
    # Organize ungrouped features by prefix for easier review
    features_by_prefix = {}
    for feat in ungrouped_features:
        # Extract prefix (everything before first underscore or plus)
        if '_' in feat:
            prefix = feat.split('_')[0]
        elif '+' in feat:
            prefix = feat.split('+')[0]
        else:
            prefix = 'Other'
        
        if prefix not in features_by_prefix:
            features_by_prefix[prefix] = []
        features_by_prefix[prefix].append(feat)
    
    # Write report
    report_path = os.path.join(fsea_base_folder, 'FSEA_UNGROUPED_FEATURES.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UNGROUPED FEATURES ANALYSIS\n")
        f.write("Features in master matrix that are NOT in any theoretical group\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total features in dataset: {len(all_feature_names)}\n")
        f.write(f"  Features in theoretical groups: {len(all_grouped_features)}\n")
        f.write(f"  Ungrouped features: {len(ungrouped_features)}\n")
        f.write(f"  Coverage: {len(all_grouped_features) / len(all_feature_names) * 100:.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("UNGROUPED FEATURES BY PREFIX\n")
        f.write("="*80 + "\n\n")
        
        for prefix in sorted(features_by_prefix.keys()):
            features = sorted(features_by_prefix[prefix])
            f.write(f"{prefix} ({len(features)} features):\n")
            for feat in features:
                f.write(f"  - {feat}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("ALL UNGROUPED FEATURES (ALPHABETICAL)\n")
        f.write("="*80 + "\n\n")
        for feat in ungrouped_features:
            f.write(f"{feat}\n")
    
    print(f"Ungrouped features report saved to: {report_path}")
    
    # Also create a CSV for easier analysis
    if ungrouped_features:
        import pandas as pd
        csv_path = os.path.join(fsea_base_folder, 'FSEA_UNGROUPED_FEATURES.csv')
        rows = []
        for feat in ungrouped_features:
            # Extract prefix
            if '_' in feat:
                prefix = feat.split('_')[0]
            elif '+' in feat:
                prefix = feat.split('+')[0]
            else:
                prefix = 'Other'
            rows.append({
                'feature_name': feat,
                'prefix': prefix
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Ungrouped features CSV saved to: {csv_path}")

def _compare_theoretical_vs_actual_groups(theoretical_groups, validated_groups, skipped_groups,
                                          all_feature_names, fsea_base_folder):
    """
    Compares theoretical groups (what rules would create) vs actual groups (what passed validation).
    Identifies features that should be in groups but aren't.
    
    Args:
        theoretical_groups: dict of groups created by applying rules without filtering
        validated_groups: dict of groups that passed validation
        skipped_groups: dict of groups that were skipped (with reasons)
        all_feature_names: list of all features in the dataset
        fsea_base_folder: path to save output files
    """
    print("\nComparing theoretical vs actual groups...")
    
    all_feature_names_set = set(all_feature_names)
    comparison_results = {}
    
    # For each theoretical group, compare to actual group
    for group_name in theoretical_groups.keys():
        theoretical_features = set(theoretical_groups[group_name])
        
        if group_name in validated_groups:
            actual_features = set(validated_groups[group_name])
            missing_from_actual = theoretical_features - actual_features
            extra_in_actual = actual_features - theoretical_features
            
            comparison_results[group_name] = {
                'theoretical_count': len(theoretical_features),
                'actual_count': len(actual_features),
                'missing_from_actual': sorted(missing_from_actual),
                'missing_count': len(missing_from_actual),
                'extra_in_actual': sorted(extra_in_actual),
                'extra_count': len(extra_in_actual),
                'status': 'validated'
            }
        elif group_name in skipped_groups:
            # Group was skipped - check why
            skip_info = skipped_groups[group_name]
            existing_theoretical = [f for f in theoretical_features if f in all_feature_names_set]
            missing_theoretical = [f for f in theoretical_features if f not in all_feature_names_set]
            
            comparison_results[group_name] = {
                'theoretical_count': len(theoretical_features),
                'actual_count': 0,
                'missing_from_actual': sorted(theoretical_features),
                'missing_count': len(theoretical_features),
                'existing_theoretical': sorted(existing_theoretical),
                'missing_theoretical': sorted(missing_theoretical),
                'status': 'skipped',
                'skip_reason': skip_info.get('reason', 'unknown'),
                'existing_count': skip_info.get('existing_count', 0)
            }
        else:
            # Group exists in theoretical but not in validated or skipped (shouldn't happen, but handle it)
            existing_theoretical = [f for f in theoretical_features if f in all_feature_names_set]
            missing_theoretical = [f for f in theoretical_features if f not in all_feature_names_set]
            
            comparison_results[group_name] = {
                'theoretical_count': len(theoretical_features),
                'actual_count': 0,
                'missing_from_actual': sorted(theoretical_features),
                'missing_count': len(theoretical_features),
                'existing_theoretical': sorted(existing_theoretical),
                'missing_theoretical': sorted(missing_theoretical),
                'status': 'missing'
            }
    
    # Write comparison report
    report_path = os.path.join(fsea_base_folder, 'FSEA_THEORETICAL_VS_ACTUAL_COMPARISON.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THEORETICAL vs ACTUAL GROUP COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write("This report compares:\n")
        f.write("  - THEORETICAL: Groups created by applying rules to all features (no filtering)\n")
        f.write("  - ACTUAL: Groups that passed validation (existence check + min_size filter)\n\n")
        
        # Summary statistics
        validated_count = sum(1 for r in comparison_results.values() if r['status'] == 'validated')
        skipped_count = sum(1 for r in comparison_results.values() if r['status'] == 'skipped')
        missing_count = sum(1 for r in comparison_results.values() if r['status'] == 'missing')
        total_missing_features = sum(r['missing_count'] for r in comparison_results.values())
        groups_with_missing = sum(1 for r in comparison_results.values() if r['missing_count'] > 0)
        
        f.write("SUMMARY:\n")
        f.write(f"  Total theoretical groups: {len(comparison_results)}\n")
        f.write(f"  Validated groups: {validated_count}\n")
        f.write(f"  Skipped groups: {skipped_count}\n")
        f.write(f"  Missing groups (not in validated or skipped): {missing_count}\n")
        f.write(f"  Groups with missing features: {groups_with_missing}\n")
        f.write(f"  Total features missing from actual groups: {total_missing_features}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED COMPARISON BY GROUP\n")
        f.write("="*80 + "\n\n")
        
        for group_name in sorted(comparison_results.keys()):
            info = comparison_results[group_name]
            f.write(f"{group_name}\n")
            f.write(f"  Status: {info['status']}\n")
            f.write(f"  Theoretical features: {info['theoretical_count']}\n")
            f.write(f"  Actual features: {info['actual_count']}\n")
            
            if info['status'] == 'skipped':
                f.write(f"  Skip reason: {info.get('skip_reason', 'unknown')}\n")
                if 'existing_count' in info:
                    f.write(f"  Existing features in dataset: {info['existing_count']}\n")
            
            if info['missing_count'] > 0:
                f.write(f"  Missing from actual: {info['missing_count']} features\n")
                # Show first 20 missing features
                missing_to_show = info['missing_from_actual'][:20]
                for feat in missing_to_show:
                    exists = "EXISTS" if feat in all_feature_names_set else "MISSING"
                    f.write(f"    - {feat} [{exists}]\n")
                if len(info['missing_from_actual']) > 20:
                    f.write(f"    ... and {len(info['missing_from_actual']) - 20} more\n")
            
            if info.get('extra_count', 0) > 0:
                f.write(f"  Extra in actual (shouldn't be there): {info['extra_count']} features\n")
                for feat in info['extra_in_actual'][:10]:
                    f.write(f"    - {feat}\n")
                if len(info['extra_in_actual']) > 10:
                    f.write(f"    ... and {len(info['extra_in_actual']) - 10} more\n")
            
            if info['status'] == 'skipped' and info.get('existing_theoretical'):
                f.write(f"  Note: {len(info['existing_theoretical'])} theoretical features DO exist in dataset\n")
            
            f.write("\n")
    
    print(f"Comparison report saved to: {report_path}")
    
    # Also create a CSV with missing features
    csv_path = os.path.join(fsea_base_folder, 'FSEA_MISSING_FEATURES_BY_GROUP.csv')
    rows = []
    for group_name in sorted(comparison_results.keys()):
        info = comparison_results[group_name]
        if info['missing_count'] > 0:
            for feat in info['missing_from_actual']:
                rows.append({
                    'group_name': group_name,
                    'missing_feature': feat,
                    'feature_exists_in_dataset': feat in all_feature_names_set,
                    'group_status': info['status'],
                    'skip_reason': info.get('skip_reason', '') if info['status'] == 'skipped' else ''
                })
    
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Missing features CSV saved to: {csv_path}")
        print(f"  Total missing features: {len(rows)}")
    else:
        print("No missing features found - all theoretical features are in actual groups.")

def _clean_fsea_results_df(df):
    """
    Clean FSEA results DataFrame by removing 'Name' column and renaming 'Term' to 'Feature_Set'.
    
    Args:
        df: FSEA results DataFrame from gsea.prerank().res2d
    
    Returns:
        Cleaned DataFrame with 'Name' removed and 'Term' renamed to 'Feature_Set'
    """
    df_cleaned = df.copy()
    
    # Remove 'Name' column if it exists
    if 'Name' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['Name'])
    
    # Rename 'Term' to 'Feature_Set' if it exists
    if 'Term' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'Term': 'Feature_Set'})
    
    return df_cleaned
