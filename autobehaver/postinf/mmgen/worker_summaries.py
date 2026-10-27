"""Build recording-level MMGen bout and sleep summaries.

Define sleep from standing duration and assign qualifying bouts to their dominant
orientation when available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bouts import calculate_bout_summary_features, calculate_sleep_features
from .categories import SUMMARY_LABELS


def _complete_category_summary(bouts, category, num_frames, fps, observed):
    """Absent events contribute zero rates; zero-event durations stay unknown."""
    table = calculate_bout_summary_features(bouts, num_frames, fps=fps)
    present = set(table['bout_type']) if not table.empty else set()
    columns = ('bout_type','bout_frequency','bout_duration_mean_s',
               'bout_duration_coeff_of_var','inter_bout_interval_mean_s',
               'inter_bout_interval_std_s','initiation_prob','n_bouts','total_length_frames','observed_frames',
               'observed_seconds','duration_sum_s','duration_sumsq_s',
               'interval_count','interval_sum_s','interval_sumsq_s')
    rows = []
    for label in SUMMARY_LABELS[category]:
        if label in present:
            continue
        row = {c: np.nan for c in columns}
        row['bout_type'] = label
        if observed and num_frames > 0:
            row.update(bout_frequency=0.0, initiation_prob=0.0, n_bouts=0,
                       total_length_frames=0, observed_frames=float(num_frames),
                       observed_seconds=float(num_frames / fps), duration_sum_s=0.0,
                       duration_sumsq_s=0.0, interval_count=0, interval_sum_s=0.0, interval_sumsq_s=0.0)
        rows.append(row)
    if rows:
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    table['category'] = category
    return table


def compute_bout_and_sleep_summaries(
    *,
    pose_bouts: List[Dict[str, Any]],
    pose_orient_bouts_for_boutsummary: List[Dict[str, Any]],
    orient_bouts_for_boutsummary: List[Dict[str, Any]],
    masked_orients: Optional[np.ndarray],
    has_orient: bool,
    num_frames: int,
    fps: float,
    sleep_threshold_frames: int,
    group: str,
    individual_id: str,
    sex: str,
    source_file: str,
    masked_poses: Optional[np.ndarray] = None,
    switch_counts: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bout summary and sleep summary tables for a single recording.

    Parameters
    ----------
    pose_bouts:
        Pose-only bouts.
    pose_orient_bouts_for_boutsummary:
        Pose+Orient bouts (filtered for summary purposes).
    orient_bouts_for_boutsummary:
        Orient-only bouts (filtered for summary purposes).
    masked_orients:
        Framewise orientation labels (may be None).
    has_orient:
        Whether orientation labels exist.
    num_frames:
        Number of frames in the inference CSV.
    fps:
        Video frames per second.
    sleep_threshold_frames:
        Minimum Standing-bout length (in frames) to count as sleep.
    group, individual_id, sex:
        Metadata.
    source_file:
        Basename of the inference CSV.

    Returns
    -------
    bout_summary_df, sleep_summary_df
    """

    # Bout summaries (Pose / Pose+Orient / Orient)
    pose_frames = sum(pd.notna(v) and v != '' for v in masked_poses) if masked_poses is not None else num_frames
    orient_frames = sum(pd.notna(v) and v != '' for v in masked_orients) if masked_orients is not None else 0
    pair_frames = sum(pd.notna(p) and p != '' and pd.notna(o) and o != ''
                      for p, o in zip(masked_poses, masked_orients)) if masked_poses is not None and masked_orients is not None else orient_frames
    pose_observed = pose_frames > 0
    orient_observed = has_orient and orient_frames > 0
    pose_summary_df = _complete_category_summary(pose_bouts, "Pose", pose_frames, fps, pose_observed)

    pose_orient_summary_df = pd.DataFrame()
    if has_orient:
        pair_observed = pair_frames > 0
        pose_orient_summary_df = _complete_category_summary(
            pose_orient_bouts_for_boutsummary, "Pose+Orient", pair_frames, fps, pair_observed)

    orient_summary_df = pd.DataFrame()
    if has_orient:
        orient_summary_df = _complete_category_summary(
            orient_bouts_for_boutsummary, "Orient", orient_frames, fps, orient_observed)

    bout_summary_df = pd.concat(
        [pose_summary_df, pose_orient_summary_df, orient_summary_df], ignore_index=True
    )
    if not bout_summary_df.empty:
        if switch_counts is not None:
            observed_by_category = {'Pose': pose_observed, 'Orient': orient_observed,
                                    'Pose+Orient': pair_observed if has_orient else False}
            bout_summary_df['n_switches'] = [
                float(switch_counts.get(cat, {}).get(label, 0)) if observed_by_category[cat] else np.nan
                for cat, label in zip(bout_summary_df['category'], bout_summary_df['bout_type'])]
        bout_summary_df["group"] = group
        bout_summary_df["individual"] = individual_id
        bout_summary_df["sex"] = sex
        bout_summary_df["source_file"] = source_file

    # Sleep summaries
    # Define sleep bouts using Pose-only Standing (robust to orientation blips)
    standing_pose_bouts = [b for b in pose_bouts if "Standing" in str(b.get("label", "")) and b.get("complete", True)]

    # 1) General (orientation-nonspecific) sleep features
    sleep_general_df = calculate_sleep_features(
        standing_pose_bouts,
        num_frames,
        fps=fps,
        sleep_threshold_frames=sleep_threshold_frames,
    )
    if sleep_general_df.empty and pose_observed and num_frames > 0:
        sleep_general_df = pd.DataFrame([dict(bout_type='Standing', n_sleep_bouts=0,
            sleep_bout_duration_mean_s=np.nan, sleep_bout_duration_std_s=np.nan,
            total_sleep_time_s=0.0, total_length_frames=0, sleep_time_budget=0.0)])

    # 2) Orientation-specific sleep features (derived from sleep bouts)
    sleep_orient_df = pd.DataFrame()
    if pose_observed and has_orient and masked_orients is not None:
        # Use the declared orientation categories for sleep summaries.
        orient_order_sleep = ["Floor", "Ceiling", "Wall-Side", "Wall-Up", "Wall-Down"]
        # Include Unknown so we don't silently lose bouts if orientation is missing
        orient_order_sleep_with_unknown = orient_order_sleep + ["Unknown"]

        # Identify Pose-only sleep bouts (Standing bouts that meet the sleep threshold)
        sleep_pose_bouts: List[Dict[str, int]] = []
        for b in standing_pose_bouts:
            length_frames = b.get("length", (b["end"] - b["start"] + 1))
            if length_frames >= sleep_threshold_frames:
                sleep_pose_bouts.append(
                    {
                        "start": int(b["start"]),
                        "end": int(b["end"]),
                        "length": int(length_frames),
                    }
                )

        # Assign each sleep bout to a dominant orientation
        orient_to_lengths: Dict[str, List[int]] = {o: [] for o in orient_order_sleep_with_unknown}

        for b in sleep_pose_bouts:
            start = b["start"]
            end = b["end"]
            length_frames = b["length"]

            # Slice orientation labels for this sleep bout
            orient_slice = masked_orients[start : end + 1]
            orient_slice = [o for o in orient_slice if pd.notna(o)]

            if not orient_slice:
                dominant_orient = "Unknown"
            else:
                # Mode orientation (first mode if tie)
                mode_vals = pd.Series(orient_slice).mode()
                dominant_orient = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"

            if dominant_orient not in orient_to_lengths:
                dominant_orient = "Unknown"

            orient_to_lengths[dominant_orient].append(length_frames)

        # Build a summary table that matches calculate_sleep_features output schema
        rows = []
        for orient in orient_order_sleep_with_unknown:
            lengths = orient_to_lengths.get(orient, [])
            n = len(lengths)

            total_length_frames = int(np.sum(lengths)) if n > 0 else 0
            mean_length_frames = float(np.mean(lengths)) if n > 0 else np.nan

            if n == 0:
                std_length_frames = np.nan
            elif n == 1:
                std_length_frames = 0.0
            else:
                std_length_frames = float(np.std(lengths, ddof=0))

            rows.append(
                {
                    "bout_type": f"Standing + {orient}",
                    "n_sleep_bouts": n,
                    "sleep_bout_duration_mean_s": (mean_length_frames / fps) if n > 0 else np.nan,
                    "sleep_bout_duration_std_s": (std_length_frames / fps) if n > 0 else np.nan,
                    "total_sleep_time_s": total_length_frames / fps,
                    "total_length_frames": total_length_frames,
                    "sleep_time_budget": 0.0,  # recalculated after aggregation
                }
            )

        sleep_orient_df = pd.DataFrame(rows)

        # If we have a 'Standing' row from the general sleep df, verify sums match.
        try:
            if sleep_general_df is not None and not sleep_general_df.empty:
                standing_row = sleep_general_df[
                    sleep_general_df["bout_type"].astype(str).str.strip().str.lower() == "standing"
                ]
                if not standing_row.empty:
                    standing_n = float(standing_row["n_sleep_bouts"].iloc[0])
                    standing_len = float(standing_row["total_length_frames"].iloc[0])

                    orient_n_sum = float(sleep_orient_df["n_sleep_bouts"].sum())
                    orient_len_sum = float(sleep_orient_df["total_length_frames"].sum())

                    if (standing_n != orient_n_sum) or (standing_len != orient_len_sum):
                        print(
                            f"WARNING: Sleep orient breakdown does not sum to Standing sleep. "
                            f"Standing(n={standing_n}, frames={standing_len}) vs "
                            f"OrientSum(n={orient_n_sum}, frames={orient_len_sum}). "
                            f"(This can happen if orientation is missing; check 'Standing + Unknown'.)"
                        )
        except Exception as e:
            print(f"WARNING: Sleep orient sum validation failed: {e}")

    # Combine general + orient-specific sleep summaries
    if sleep_orient_df is not None and not sleep_orient_df.empty:
        sleep_summary_df = pd.concat([sleep_general_df, sleep_orient_df], ignore_index=True)
    else:
        sleep_summary_df = sleep_general_df

    # Prepare sleep_summary_df as separate category (not merged into bout_summary)
    if not sleep_summary_df.empty:
        # Rename total_length_frames to sleep_total_length_frames
        if "total_length_frames" in sleep_summary_df.columns:
            sleep_summary_df = sleep_summary_df.rename(
                columns={"total_length_frames": "sleep_total_length_frames"}
            )
        sleep_summary_df["observed_frames"] = float(pose_frames)
        # Add metadata columns for later aggregation
        sleep_summary_df["group"] = group
        sleep_summary_df["individual"] = individual_id
        sleep_summary_df["sex"] = sex
        sleep_summary_df["source_file"] = source_file

    return bout_summary_df, sleep_summary_df
