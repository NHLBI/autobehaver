"""Declared category scopes, independent of observed animals and kinematics."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd

POSE_LABELS = ('Walking', 'Standing', 'Scramble', 'Groom-FrontLegs', 'Groom-Head',
               'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs', 'Jump')
ORIENT_LABELS = ('Floor', 'Ceiling', 'Wall-Side', 'Wall-Up', 'Wall-Down')
SUMMARY_ORIENT_LABELS = tuple(o for o in ORIENT_LABELS if o not in ('Wall-Up', 'Wall-Down'))
SUMMARY_COMBINED_LABELS = tuple(f'{p} + {o}' for p in POSE_LABELS
                              if p not in ('Jump', 'Scramble') for o in SUMMARY_ORIENT_LABELS)
SUMMARY_LABELS = {'Pose': POSE_LABELS, 'Orient': SUMMARY_ORIENT_LABELS,
                  'Pose+Orient': SUMMARY_COMBINED_LABELS}


def validate_labels(labels, allowed, *, category):
    """Reject nonmissing labels outside the declared category alphabet."""
    if labels is None:
        return
    unexpected = sorted({str(x) for x in labels if pd.notna(x) and str(x) != ''} - set(allowed))
    if unexpected:
        raise ValueError(f'Undeclared {category} labels: {unexpected}. Supply an explicit supported category contract.')


def adjacent_changes(labels):
    """Counts from neighboring frames only; missing intervals never bridge."""
    if labels is None:
        return Counter()
    result = Counter()
    for previous, current in zip(labels[:-1], labels[1:]):
        if pd.notna(previous) and pd.notna(current) and previous != '' and current != '' and previous != current:
            result[(str(previous), str(current))] += 1
    return result


def category_transition_tables(*, pose_valid, orient_valid, combined_valid,
                               source_file, individual, group):
    """Build adjacent-frame transition counts and combined-context switch contributions.

    Combined reports include orientation-independent Jump/Scramble; repertoire switches
    retain the declared 24-context budget scope.
    """
    from .transitions import _is_impossible_pose_orient_transition
    pose = np.asarray(pose_valid, dtype=object)
    orient = None if orient_valid is None else np.asarray(orient_valid, dtype=object).copy()
    combined = None if combined_valid is None else np.asarray(combined_valid, dtype=object).copy()
    if orient is not None:
        orient[np.isin(orient, ('Wall-Up', 'Wall-Down'))] = np.nan
    report_combined = None if combined is None else combined.copy()
    if report_combined is not None:
        specials = np.isin(pose, ('Jump', 'Scramble'))
        report_combined[specials] = pose[specials]
    native = {'Pose': pose, 'Orient': orient, 'Pose+Orient': report_combined}
    switch_views = {'Pose': pose, 'Orient': orient, 'Pose+Orient': combined}
    rows = []
    for category, labels in native.items():
        for (frm, to), count in adjacent_changes(labels).items():
            if category == 'Pose+Orient' and _is_impossible_pose_orient_transition(frm, to):
                continue
            rows.append(dict(source_file=source_file, individual=individual, group=group,
                             category=category, from_behavior=frm, to_behavior=to, count=count))
    switches = {cat: Counter() for cat in switch_views}
    for category, labels in switch_views.items():
        for (frm, _), count in adjacent_changes(labels).items():
            switches[category][frm] += count
    return pd.DataFrame(rows, columns=('source_file','individual','group','category',
                                     'from_behavior','to_behavior','count')), switches


def normalized_budget_entropy(master, prefix, allowed):
    """Fixed declared alphabet; zero absent states, unknown states rejected."""
    columns = [c for c in master if c.startswith(prefix)]
    if not columns:
        return None
    expected = {prefix + label: label for label in allowed}
    unexpected = [c for c in columns if c not in expected]
    # Unsupported displayed states can remain present only as exact observed
    # zero columns; they do not expand the actual producer's alphabet.
    invalid_extra = [c for c in unexpected if not np.all(master[c].to_numpy(dtype=float) == 0)]
    if invalid_extra:
        raise ValueError(f'Undeclared nonzero/unknown time-budget columns: {invalid_extra}')
    columns = [c for c in columns if c in expected]
    values = master.reindex(columns=list(expected), fill_value=0).to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(np.isfinite(values) & (values < 0)):
        raise ValueError('Time budgets must be finite nonnegative values or missing.')
    # A row with missing supplied values is unknown, not an observed zero.
    unknown = np.isnan(master[columns].to_numpy(dtype=float)).any(axis=1)
    mass = np.nansum(values, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = values / mass[:, None]
        terms = np.where(p > 0, p * np.log(p), 0.0)
        h = -np.sum(terms, axis=1) / np.log(len(allowed))
    h[(mass <= 0) | unknown] = np.nan
    return h
