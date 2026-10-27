"""Initialize and run the Autocropper stage."""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any

from .config import AutocropperConfig
from .thread_limits import configure_native_threading


def _sync_runtime_args(*, cfg: AutocropperConfig, args: Any) -> None:
    """Apply runtime configuration values to the argparse namespace."""

    args.require_exclusions = bool(cfg.require_exclusions)

    file_filter = cfg.file_filter
    if file_filter in (None, ""):
        file_filter = getattr(args, "file_filter", None)
    args.file_filter = str(file_filter) if file_filter not in (None, "") else None

    trial_number = getattr(cfg, "trial_number", None)
    if trial_number in (None, ""):
        trial_number = getattr(args, "trial_number", None)
    args.trial_number = int(trial_number) if trial_number not in (None, "") else None

    args.heartbeat_interval = int(cfg.heartbeat_interval)
    args.timeout = int(cfg.timeout)
    args.crop_crf = int(getattr(cfg, "crop_crf", getattr(args, "crop_crf", 12)))
    args.native_threads = max(1, int(getattr(cfg, "native_threads", getattr(args, "native_threads", 1))))
    args.ffmpeg_threads = max(1, int(getattr(cfg, "ffmpeg_threads", getattr(args, "ffmpeg_threads", 1))))
    args.partial_cleanup_min_age_sec = max(0, int(getattr(cfg, "partial_cleanup_min_age_sec", getattr(args, "partial_cleanup_min_age_sec", 3600))))

    args.crop_dim_2k = int(cfg.crop_dim_2k)
    args.chamber_calibration_json = cfg.chamber_calibration_json


def run_autocropper(*, cfg: AutocropperConfig, args: Any) -> None:
    """Initialize runtime options and run chamber cropping."""

    _sync_runtime_args(cfg=cfg, args=args)

    # For reproducible random frame sampling.
    random.seed(42)

    # Windows safety for multiprocessing.
    mp.freeze_support()

    # Cap native-library threads before importing OpenCV/ffmpeg-facing modules.
    configure_native_threading(getattr(args, "native_threads", 1), overwrite=False)

    # Import heavy dependencies only when the stage actually runs.
    from .pipeline import run_pipeline, set_runtime

    set_runtime(cfg=cfg, args=args)
    run_pipeline()
