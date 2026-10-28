"""Derive and publish KeyPre features for individual DeepLabCut CSV files."""

from __future__ import annotations

import logging
import os

from . import pose
from .angular_kinematics import add_angular_kinematics_and_curvature
from .centroid_kinematics import add_centroid_kinematics_features
from .dlc_io import dlc_chunks
from .egocentric import add_asymmetry_features, add_centered_and_rotated_keypoints
from .export_keypoints import add_raw_keypoints_and_likelihoods, add_rotated_keypoints_and_likelihoods
from .geometry import add_bodyshape_features
from .logging_utils import configure_logging
from .part_kinematics import add_keypoint_kinematics_features
from .pairwise_angular import add_pairwise_angular_kinematics_features
from .spec import parts
from .writer import save_feature_chunks
from .contract import output_path, inspect_keypre_output, keypre_request
from autobehaver.completion import verified_run

logger = logging.getLogger(__name__)


# Runtime config (set via init_worker / set_run_config)
output_dir: str | None = None
clobber: bool = True
raw_only: bool = False
chunk_rows: int = 16384


def set_run_config(config: dict) -> None:
    """Set process-local runtime options used by the file worker."""
    global output_dir, clobber, raw_only, chunk_rows

    output_dir = str(config["output_dir"])
    clobber = bool(config["clobber"])
    pose.movie_dim = int(config["movie_dim"])
    raw_only = bool(config.get("raw_only", False))
    chunk_rows = int(config.get("chunk_rows", 16384))


def init_worker(config: dict, log_level: str) -> None:
    """Initialize process-local runtime options and logging."""
    configure_logging(log_level)
    set_run_config(config)


@verified_run
def process_csv_file(csv_file: str) -> dict:
    """Derive and validate features for one DeepLabCut CSV.

    Return a status dictionary with input/output paths, any error, and output
    shape when successful; skip only compatible, validated existing outputs.
    """
    try:
        if output_dir is None:
            raise RuntimeError(
                "output_dir is not set. "
                "This usually means init_worker()/set_run_config() was not called."
            )

        poscalc_file_path = str(output_path(csv_file, output_dir))
        identity = keypre_request(csv_file, pose.movie_dim, raw_only)

        if (not clobber) and inspect_keypre_output(csv_file, poscalc_file_path, movie_dim=pose.movie_dim, raw_only=raw_only, require_receipt=True)[0]:
            logger.info("Skipping existing output: %s", poscalc_file_path)
            return {
                "status": "skipped",
                "file": csv_file,
                "error": None,
                "out": poscalc_file_path,
                "n_rows": None,
                "n_cols": None,
            }

        from . import settings
        # Three centered derivatives plus the largest centered rolling radius.
        halo = 0 if raw_only else 3 + max(settings.rolling_windows, default=1) // 2
        timeline = {'origin': None, 'rows': None}
        def frames():
            for frame, start, stop, first, last in dlc_chunks(csv_file, chunk_rows=chunk_rows, halo=halo):
                if first and len(frame):
                    timeline['origin'] = int(frame.index[0])
                if last:
                    timeline['rows'] = int(frame.index[-1]) - timeline['origin'] + 1 if len(frame) else 0
                features = derive_features(frame, raw_only=raw_only)
                yield features.iloc[start:stop].copy(), first, last
        return save_feature_chunks(frames(), csv_file, poscalc_file_path,
                                   apply_edge_nans=not raw_only, identity=identity,
                                   movie_dim=pose.movie_dim, source_rows=lambda: timeline['rows'])

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


def derive_features(input_df, *, raw_only=False):
    """Apply the unchanged feature formulas to one frame with temporal context."""
    import pandas as pd
    poscorr_dict = {}
    if raw_only:
        add_raw_keypoints_and_likelihoods(input_df, poscorr_dict, parts)
        return pd.DataFrame(poscorr_dict)
    # Pose features (centroid + angle + polar)
    input_df = pose.add_pose_features(input_df, poscorr_dict)

    # Centroid kinematics (speed/forward/strafe + derivatives)
    input_df = add_centroid_kinematics_features(input_df, poscorr_dict)

    # Angular kinematics + curvature
    input_df = add_angular_kinematics_and_curvature(input_df, poscorr_dict)

    # Body-shape features (l-ratio, lengths, angles, head angles, areas)
    input_df = add_bodyshape_features(input_df, poscorr_dict)

    # Egocentric keypoints (center + rotate)
    input_df = add_centered_and_rotated_keypoints(input_df, parts)

    # Asymmetry features (uses rotated coords)
    input_df = add_asymmetry_features(input_df, poscorr_dict)

    # Write final keypoints (rotated) + likelihoods into output dict
    add_rotated_keypoints_and_likelihoods(input_df, poscorr_dict, parts)

    # Per-keypoint kinematics (velocity/accel/jerk from rotated coords)
    input_df = add_keypoint_kinematics_features(input_df, poscorr_dict)

    # Pairwise angular kinematics
    input_df = add_pairwise_angular_kinematics_features(input_df, poscorr_dict)

    return pd.DataFrame(poscorr_dict)
