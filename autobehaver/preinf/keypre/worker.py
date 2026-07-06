"""Per-file worker implementation for keypre.

All computations are performed by the keypre worker implementation.
"""

from __future__ import annotations

import logging
import os

from . import pose
from .angular_kinematics import add_angular_kinematics_and_curvature
from .centroid_kinematics import add_centroid_kinematics_features
from .dlc_io import load_dlc_csv_flat
from .egocentric import add_asymmetry_features, add_centered_and_rotated_keypoints
from .export_keypoints import add_raw_keypoints_and_likelihoods, add_rotated_keypoints_and_likelihoods
from .geometry import add_bodyshape_features
from .logging_utils import configure_logging
from .part_kinematics import add_keypoint_kinematics_features
from .pairwise_angular import add_pairwise_angular_kinematics_features
from .spec import parts
from .writer import finalize_save_and_summarize

logger = logging.getLogger(__name__)


# Runtime config (set via init_worker / set_run_config)
output_dir: str | None = None
clobber: bool = True
raw_only: bool = False


def set_run_config(config: dict) -> None:
    """Set run-time globals used by the worker function(s)."""
    global output_dir, clobber, raw_only

    output_dir = str(config["output_dir"])
    clobber = bool(config["clobber"])
    pose.movie_dim = int(config["movie_dim"])
    raw_only = bool(config.get("raw_only", False))


def init_worker(config: dict, log_level: str) -> None:
    """Initializer for multiprocessing workers."""
    configure_logging(log_level)
    set_run_config(config)


def process_csv_file(csv_file: str) -> dict:
    """Worker function for processing a single DLC CSV file.

    Returns a status dict with keys:
      - status: 'success' | 'failed' | 'skipped'
      - file: input CSV path
      - error: error string if failed else None
      - out: output CSV path (if known)
      - n_rows, n_cols: output shape (if success)
    """
    try:
        if output_dir is None:
            raise RuntimeError(
                "output_dir is not set. "
                "This usually means init_worker()/set_run_config() was not called."
            )

        # Output path
        poscalc_file_name = os.path.basename(csv_file).replace(".csv", "_positional_calcs.csv")
        poscalc_file_path = os.path.join(output_dir, poscalc_file_name)

        # Skip if requested
        if (not clobber) and os.path.exists(poscalc_file_path):
            logger.info("Skipping existing output: %s", poscalc_file_path)
            return {
                "status": "skipped",
                "file": csv_file,
                "error": None,
                "out": poscalc_file_path,
                "n_rows": None,
                "n_cols": None,
            }

        # 1) Load DLC CSV
        input_df = load_dlc_csv_flat(csv_file)

        # 2) Collect outputs in insertion-order
        poscorr_dict: dict = {}

        # --- Raw-only export mode ---
        if raw_only:
            # Export only the DLC x/y (+ likelihoods) for each keypoint in `parts`.
            # (Front/Back are intentionally excluded; they are not in `spec.parts`.)
            add_raw_keypoints_and_likelihoods(input_df, poscorr_dict, parts)
            # Raw-only exports do not apply the "edge skip" NaN padding.
            return finalize_save_and_summarize(
                poscorr_dict,
                csv_file,
                poscalc_file_path,
                apply_edge_nans=False,
            )

        # 3) Pose features (centroid + angle + polar)
        input_df = pose.add_pose_features(input_df, poscorr_dict)

        # 4) Centroid kinematics (speed/forward/strafe + derivatives)
        input_df = add_centroid_kinematics_features(input_df, poscorr_dict)

        # 5) Angular kinematics + curvature
        input_df = add_angular_kinematics_and_curvature(input_df, poscorr_dict)

        # 6) Body-shape features (l-ratio, lengths, angles, head angles, areas)
        input_df = add_bodyshape_features(input_df, poscorr_dict)

        # 7) Egocentric keypoints (center + rotate)
        input_df = add_centered_and_rotated_keypoints(input_df, parts)

        # 8) Asymmetry features (uses rotated coords)
        input_df = add_asymmetry_features(input_df, poscorr_dict)

        # 9) Write final keypoints (rotated) + likelihoods into output dict
        add_rotated_keypoints_and_likelihoods(input_df, poscorr_dict, parts)

        # 10) Per-keypoint kinematics (velocity/accel/jerk from rotated coords)
        input_df = add_keypoint_kinematics_features(input_df, poscorr_dict)

        # 11) Pairwise angular kinematics
        input_df = add_pairwise_angular_kinematics_features(input_df, poscorr_dict)

        # 12) Finalize + save
        return finalize_save_and_summarize(poscorr_dict, csv_file, poscalc_file_path)

    except Exception as e:
        logger.exception("Error processing %s", csv_file)
        return {
            "status": "failed",
            "file": csv_file,
            "error": str(e),
            "out": None,
            "n_rows": None,
            "n_cols": None,
        }
