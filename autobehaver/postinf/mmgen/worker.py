"""Per-recording worker for mmgen.

This module owns the multiprocessing-safe per-file processing logic for mmgen.
It delegates to the smaller ``worker_*`` helper modules and receives any
runtime-dependent values explicitly through the worker ``config`` payload.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .kinematics import KIN_FRAMES_SIGNAL_COLS as DEFAULT_KIN_FRAMES_SIGNAL_COLS
from .worker_bouts import build_bouts_and_valid_arrays
from .worker_identity import resolve_group_individual_sex
from .worker_io import extract_smoothed_labels, load_inference_and_poscalc
from .worker_kinematics import compute_kinematics_outputs
from .worker_measurements import compute_measurement_result
from .worker_positional import build_positional_df
from .worker_postural import compute_postural_features_df
from .worker_summaries import compute_bout_and_sleep_summaries

DEFAULT_MM_PER_PX: float = 0.0169896744624207
DEFAULT_VIDEO_FPS: float = 40.0
DEFAULT_SLEEP_THRESHOLD_SECONDS: float = 180.0  # 3 minutes in the standard mmgen workflow


def _resolve_runtime_float(runtime: Any, attr_name: str, default: float) -> float:
    """Return a runtime calibration value, falling back when unset/None."""

    if runtime is None:
        return float(default)
    value = getattr(runtime, attr_name, default)
    if value is None:
        return float(default)
    return float(value)


def _sleep_threshold_frames_from_fps(
    fps: float, *, seconds: float = DEFAULT_SLEEP_THRESHOLD_SECONDS
) -> int:
    """Convert the canonical sleep duration threshold to frames."""

    fps = float(fps)
    seconds = float(seconds)
    return max(1, int(round(fps * seconds)))


def _build_failure_info(
    *,
    inf_csv_path: str,
    stage: str,
    error_type: str,
    error_message: str,
    traceback_text: str = "",
) -> dict[str, str]:
    return {
        "source_file": os.path.basename(str(inf_csv_path)),
        "error_stage": str(stage),
        "error_type": str(error_type),
        "error_message": str(error_message),
        "traceback": str(traceback_text or ""),
    }


def process_video_data(args):
    """Process one inference CSV + matching keypoint-features file.

    Parameters are passed as a tuple so the function remains directly usable via
    ``multiprocessing.Pool.imap``.
    """

    inf_csv_path, poscalc_dir, min_bout, edge_skip, inf_tag, config = args
    stage = "load_inputs"
    try:
        df, poscalc_df, aligned_poscalc_df = load_inference_and_poscalc(
            inf_csv_path,
            poscalc_dir,
            edge_skip=edge_skip,
            inf_tag=inf_tag,
        )
        if df is None or poscalc_df is None or aligned_poscalc_df is None:
            return (
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                _build_failure_info(
                    inf_csv_path=inf_csv_path,
                    stage=stage,
                    error_type="InputLoadError",
                    error_message="Failed to load or align inference/poscalc inputs.",
                ),
            )

        stage = "extract_labels"
        masked_poses, masked_orients, has_orient = extract_smoothed_labels(df)
        group, individual_id, sex = resolve_group_individual_sex(inf_csv_path, config)

        stage = "build_bouts"
        (
            pose_bouts,
            pose_orient_bouts,
            orient_bouts,
            pose_orient_bouts_for_boutsummary,
            orient_bouts_for_boutsummary,
            masked_pose_valid,
            masked_orient_valid,
            masked_pose_orient_valid,
        ) = build_bouts_and_valid_arrays(
            masked_poses=masked_poses,
            masked_orients=masked_orients,
            has_orient=has_orient,
            min_bout=min_bout,
            num_frames=len(df),
        )

        runtime = config.get("runtime_settings")
        video_fps = _resolve_runtime_float(runtime, "video_fps", DEFAULT_VIDEO_FPS)
        mm_per_px = _resolve_runtime_float(runtime, "mm_per_px", DEFAULT_MM_PER_PX)
        kin_frames_signal_cols = tuple(
            DEFAULT_KIN_FRAMES_SIGNAL_COLS
            if runtime is None or getattr(runtime, "kin_frames_signal_cols", None) is None
            else getattr(runtime, "kin_frames_signal_cols", DEFAULT_KIN_FRAMES_SIGNAL_COLS)
        )
        num_frames = len(df)
        sleep_threshold_frames = _sleep_threshold_frames_from_fps(video_fps)

        stage = "summaries"
        bout_summary_df, sleep_summary_df = compute_bout_and_sleep_summaries(
            pose_bouts=pose_bouts,
            pose_orient_bouts_for_boutsummary=pose_orient_bouts_for_boutsummary,
            orient_bouts_for_boutsummary=orient_bouts_for_boutsummary,
            masked_orients=masked_orients,
            has_orient=has_orient,
            num_frames=num_frames,
            fps=video_fps,
            sleep_threshold_frames=sleep_threshold_frames,
            group=group,
            individual_id=individual_id,
            sex=sex,
            source_file=os.path.basename(inf_csv_path),
        )

        stage = "kinematics"
        bout_features_dict, kin_frames_df, aligned_poscalc_df_kin = compute_kinematics_outputs(
            aligned_poscalc_df=aligned_poscalc_df,
            pose_bouts=pose_bouts,
            pose_orient_bouts=pose_orient_bouts,
            masked_pose_valid=masked_pose_valid,
            masked_pose_orient_valid=masked_pose_orient_valid,
            num_frames=num_frames,
            fps=video_fps,
            mm_per_px=mm_per_px,
            kin_frames_signal_cols=kin_frames_signal_cols,
            group=group,
            individual_id=individual_id,
            sex=sex,
            source_file=os.path.basename(inf_csv_path),
        )

        stage = "measurements"
        measurement_result = compute_measurement_result(
            config=config,
            masked_poses=masked_poses,
            masked_orients=masked_orients,
            aligned_poscalc_df=aligned_poscalc_df,
            individual_id=individual_id,
            group=group,
            sex=sex,
        )

        stage = "positional"
        positional_df = build_positional_df(
            config=config,
            masked_poses=masked_poses,
            masked_orients=masked_orients,
            has_orient=has_orient,
            masked_pose_valid=masked_pose_valid,
            masked_orient_valid=masked_orient_valid,
            masked_pose_orient_valid=masked_pose_orient_valid,
            aligned_poscalc_df=aligned_poscalc_df,
            aligned_poscalc_df_kin=aligned_poscalc_df_kin,
            group=group,
            individual_id=individual_id,
            source_file=os.path.basename(inf_csv_path),
        )

        stage = "postural"
        postural_features_df = compute_postural_features_df(
            config=config,
            poscalc_df=aligned_poscalc_df,
            df_inference=df,
            masked_poses=masked_poses,
            masked_orients=masked_orients,
            edge_skip=edge_skip,
            group=group,
            individual_id=individual_id,
            video_path=inf_csv_path,
            mm_per_px=mm_per_px,
        )

        return (
            bout_features_dict,
            measurement_result,
            len(df),
            positional_df,
            postural_features_df,
            bout_summary_df,
            sleep_summary_df,
            kin_frames_df,
            None,
        )
    except Exception as e:
        print(f"Error processing {inf_csv_path}: {e}")
        return (
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            _build_failure_info(
                inf_csv_path=inf_csv_path,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_text=traceback.format_exc(),
            ),
        )
