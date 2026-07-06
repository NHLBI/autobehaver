"""Final keypoint export (rotated coords + likelihoods) for keypre."""

from __future__ import annotations

import logging

import numpy as np

from . import settings

logger = logging.getLogger(__name__)

include_likelihoods = settings.include_likelihoods


def add_rotated_keypoints_and_likelihoods(input_df, poscorr_dict, parts):
    """Write final keypoint columns (rotated) and likelihoods to poscorr_dict."""
    for part in parts:
        rx = f"{part}_rotated_x"
        ry = f"{part}_rotated_y"

        if rx in input_df.columns and ry in input_df.columns:
            poscorr_dict[f"{part}_x"] = input_df[rx]
            poscorr_dict[f"{part}_y"] = input_df[ry]
        else:
            logger.warning(f"Warning: Rotated columns for {part} not found in DataFrame.")

        if include_likelihoods:
            lk = f"{part}_likelihood"
            if lk in input_df.columns:
                poscorr_dict[lk] = input_df[lk]
            else:
                logger.warning(f"Warning: Likelihood column for {part} not found in DataFrame.")


def add_raw_keypoints_and_likelihoods(input_df, poscorr_dict, parts):
    """Write raw DLC keypoint columns (x/y) and likelihoods to poscorr_dict.

    This is intended for ablation experiments where you want to test models on
    only the direct DLC outputs (no centering/rotation/kinematics/body-shape).

    Notes:
      - Output column names are suffixed with ``_raw`` (e.g. ``Ocelli_x_raw``)
        to avoid confusion with the standard keypre export (which outputs
        rotated/corrected coordinates as ``*_x``/``*_y``).
      - ``parts`` should exclude derived helpers like Front/Back; keypre's
        canonical list in :mod:`autobehaver.preinf.keypre.spec` already does.
    """

    n = int(len(input_df))

    for part in parts:
        x_in = f"{part}_x"
        y_in = f"{part}_y"

        x_out = f"{part}_x_raw"
        y_out = f"{part}_y_raw"

        if x_in in input_df.columns:
            poscorr_dict[x_out] = input_df[x_in]
        else:
            logger.warning("Raw x column missing for %s; filling NaNs.", part)
            poscorr_dict[x_out] = np.full(n, np.nan, dtype=np.float32)

        if y_in in input_df.columns:
            poscorr_dict[y_out] = input_df[y_in]
        else:
            logger.warning("Raw y column missing for %s; filling NaNs.", part)
            poscorr_dict[y_out] = np.full(n, np.nan, dtype=np.float32)

        if include_likelihoods:
            lk_in = f"{part}_likelihood"
            lk_out = f"{part}_likelihood_raw"
            if lk_in in input_df.columns:
                poscorr_dict[lk_out] = input_df[lk_in]
            else:
                logger.warning("Raw likelihood column missing for %s; filling NaNs.", part)
                poscorr_dict[lk_out] = np.full(n, np.nan, dtype=np.float32)
