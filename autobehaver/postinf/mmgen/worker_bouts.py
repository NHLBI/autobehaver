"""Build MMGen pose, orientation, and combined bouts and their valid per-frame label
arrays.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .bouts import extract_bouts_from_label_sequence, _build_valid_label_array_from_bouts
from .categories import POSE_LABELS, ORIENT_LABELS, validate_labels


def build_bouts_and_valid_arrays(
    *,
    masked_poses: Sequence[Any],
    masked_orients: Optional[Sequence[Any]],
    has_orient: bool,
    min_bout: int,
    num_frames: int,
):
    """Build bout lists and corresponding "valid" per-frame label arrays."""
    if len(masked_poses) != num_frames or (has_orient and (masked_orients is None or len(masked_orients) != num_frames)):
        raise ValueError('Behavior/orientation timelines must match num_frames.')
    validate_labels(masked_poses, POSE_LABELS, category='Pose')
    if not any(pd.notna(x) and str(x) != '' for x in masked_poses):
        raise ValueError('Recording contains no observed behavior labels; unknown data cannot be certified as zero events.')
    if has_orient:
        validate_labels(masked_orients, ORIENT_LABELS, category='Orient')
        observed_poses = {str(x) for x in masked_poses if pd.notna(x) and str(x) != ''}
        if not any(pd.notna(x) and str(x) != '' for x in masked_orients) and not observed_poses.issubset({'Jump','Scramble'}):
            raise ValueError('Present orientation column contains no observed labels; remove an unavailable column or supply valid data.')

    # Category 1: Pose-Only
    pose_bouts = extract_bouts_from_label_sequence(masked_poses, min_bout, exempt_labels={"Jump"})

    # Category 2: Pose+Orient
    pose_orient_bouts = []
    if has_orient and masked_orients is not None:
        pose_orient_seq = [
            (str(p) if pd.notna(p) and p in ('Jump', 'Scramble') else f"{p} + {o}"
             if pd.notna(p) and p != '' and pd.notna(o) and o != '' else np.nan)
            for p, o in zip(masked_poses, masked_orients)
        ]
        pose_orient_bouts_raw = extract_bouts_from_label_sequence(pose_orient_seq, min_bout)
        pose_orient_bouts = [b for b in pose_orient_bouts_raw if "+" in str(b["label"]) and "nan" not in str(b["label"])]

    # Category 3: Orient-Only
    orient_bouts = []
    if has_orient and masked_orients is not None:
        orient_bouts = extract_bouts_from_label_sequence(masked_orients, min_bout)

    # Filter out Wall-Up and Wall-Down orientations from pose+orient bouts for BoutSummary
    pose_orient_bouts_for_boutsummary = [
        b
        for b in pose_orient_bouts
        if "Wall-Up" not in str(b.get("label", ""))
        and "Wall-Down" not in str(b.get("label", ""))
    ]

    # Filter out Wall-Up and Wall-Down orientations from orient-only bouts for BoutSummary
    orient_bouts_for_boutsummary = [
        b for b in orient_bouts if b.get("label", "") not in ["Wall-Up", "Wall-Down"]
    ]

    masked_pose_valid = _build_valid_label_array_from_bouts(pose_bouts, num_frames)

    masked_orient_valid = None
    masked_pose_orient_valid = None
    if has_orient and masked_orients is not None:
        masked_orient_valid = _build_valid_label_array_from_bouts(orient_bouts, num_frames)
        masked_pose_orient_valid = _build_valid_label_array_from_bouts(
            pose_orient_bouts_for_boutsummary, num_frames
        )

    return (
        pose_bouts,
        pose_orient_bouts,
        orient_bouts,
        pose_orient_bouts_for_boutsummary,
        orient_bouts_for_boutsummary,
        masked_pose_valid,
        masked_orient_valid,
        masked_pose_orient_valid,
    )
