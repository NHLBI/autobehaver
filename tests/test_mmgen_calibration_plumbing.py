from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autobehaver.postinf.mmgen import postural
from autobehaver.postinf.mmgen.worker import (
    DEFAULT_MM_PER_PX,
    DEFAULT_SLEEP_THRESHOLD_SECONDS,
    DEFAULT_VIDEO_FPS,
    _resolve_runtime_float,
    _sleep_threshold_frames_from_fps,
)
from autobehaver.postinf.mmgen.worker_postural import compute_postural_features_df


LEG_TIP_KPS = [
    "FrontTarsusLeft",
    "FrontTarsusRight",
    "MidTarsusLeft",
    "MidTarsusRight",
    "BackTarsusLeft",
    "BackTarsusRight",
]
BODY_AXIS_KPS = ["Antennae", "AbdomenTip"]
LR_PARTS = ["Eyes", "Thorax", "Abdomen", "Wings", "FrontLegs", "MidLegs", "BackLegs"]
TB_PARTS = ["Head", "Thorax", "Abdomen"]


def _minimal_postural_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    row: dict[str, float] = {
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "stance_area": 4.0,
    }

    coords = {
        "FrontTarsusLeft": (3.0, 4.0),
        "FrontTarsusRight": (0.0, 4.0),
        "MidTarsusLeft": (0.0, 3.0),
        "MidTarsusRight": (0.0, 3.0),
        "BackTarsusLeft": (8.0, 0.0),
        "BackTarsusRight": (8.0, 0.0),
        "Antennae": (1.0, 0.0),
        "AbdomenTip": (-1.0, 0.0),
    }
    for kp in LEG_TIP_KPS + BODY_AXIS_KPS:
        x, y = coords[kp]
        row[f"{kp}_x"] = x
        row[f"{kp}_y"] = y
        row[f"{kp}_likelihood"] = 0.99

    for part in LR_PARTS:
        row[f"asym_lr_x_{part}"] = 0.0
        row[f"asym_lr_y_{part}"] = 0.0
    for part in TB_PARTS:
        row[f"asym_tb_x_{part}"] = 0.0
        row[f"asym_tb_y_{part}"] = 0.0

    poscalc_df = pd.DataFrame([row])
    df_inference = pd.DataFrame({"Smoothed Pose Label": ["Standing"], "Smoothed Orient Label": ["Floor"]})
    masked_poses = np.array(["Standing"], dtype=object)
    masked_orients = np.array(["Floor"], dtype=object)
    return poscalc_df, df_inference, masked_poses, masked_orients


def test_sleep_threshold_frames_scale_with_runtime_fps() -> None:
    assert _sleep_threshold_frames_from_fps(40.0) == 7200
    assert _sleep_threshold_frames_from_fps(20.0) == 3600
    assert _sleep_threshold_frames_from_fps(50.0) == 9000
    assert _sleep_threshold_frames_from_fps(0.01) == 2
    assert DEFAULT_SLEEP_THRESHOLD_SECONDS == 180.0


def test_runtime_float_resolution_falls_back_when_unset() -> None:
    runtime = SimpleNamespace(video_fps=None, mm_per_px=None)
    assert _resolve_runtime_float(runtime, "video_fps", DEFAULT_VIDEO_FPS) == DEFAULT_VIDEO_FPS
    assert _resolve_runtime_float(runtime, "mm_per_px", DEFAULT_MM_PER_PX) == DEFAULT_MM_PER_PX
    assert _resolve_runtime_float(None, "video_fps", DEFAULT_VIDEO_FPS) == DEFAULT_VIDEO_FPS


def test_worker_postural_receives_and_applies_mm_per_px() -> None:
    poscalc_df, df_inference, masked_poses, masked_orients = _minimal_postural_inputs()

    result = compute_postural_features_df(
        config={"calculate_postural_features": True},
        poscalc_df=poscalc_df,
        df_inference=df_inference,
        masked_poses=masked_poses,
        masked_orients=masked_orients,
        edge_skip=0,
        group="g1",
        individual_id="fly1_movie0",
        video_path="/tmp/fly1.csv",
        mm_per_px=0.5,
    )

    assert result is not None
    assert not result.empty
    row = result.iloc[0]
    assert row["stance_area"] == 1.0
    assert row["extension_raw_FrontLegs_avg"] == 2.25
    assert row["group"] == "g1"
    assert row["individual"] == "fly1_movie0"


def test_postural_summary_no_longer_accepts_stale_mm_per_px_kwarg() -> None:
    params = inspect.signature(postural.calculate_and_save_postural_summary).parameters
    assert "mm_per_px" not in params
    assert "save_outputs" in params
