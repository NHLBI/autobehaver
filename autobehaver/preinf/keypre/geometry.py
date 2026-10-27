"""Derive likelihood ratios, distances, angles, and polygon areas for KeyPre."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .spec import (
    body_area_side_vertices,
    body_area_top_vertices,
    bodypart_triplets,
    length_pairs,
    stance_area_vertices,
)
from .utils import add_existing_columns, shoelace_area

logger = logging.getLogger(__name__)


def calculate_log_l_ratios(dataframe):
    # Add a small constant to avoid division by zero
    """Calculate natural-log likelihood ratios with an epsilon in numerator and denominator."""
    epsilon = 1e-6

    l_ratio_combinations = {
        "log_l_ratio_FB": ("Antennae_likelihood", "AbdomenTip_likelihood"),
        "log_l_ratio_head_LR": ("EyeLeft_likelihood", "EyeRight_likelihood"),
        "log_l_ratio_head_RL": ("EyeRight_likelihood", "EyeLeft_likelihood"),
        "log_l_ratio_head_TB": ("Ocelli_likelihood", "Mouthparts_likelihood"),
        "log_l_ratio_head_BT": ("Mouthparts_likelihood", "Ocelli_likelihood"),
        "log_l_ratio_thorax_LR": ("ThoraxLeft_likelihood", "ThoraxRight_likelihood"),
        "log_l_ratio_thorax_RL": ("ThoraxRight_likelihood", "ThoraxLeft_likelihood"),
        "log_l_ratio_thorax_TB": ("ThoraxTop_likelihood", "ThoraxBottom_likelihood"),
        "log_l_ratio_thorax_BT": ("ThoraxBottom_likelihood", "ThoraxTop_likelihood"),
        "log_l_ratio_abdomen_LR": ("AbdomenLeft_likelihood", "AbdomenRight_likelihood"),
        "log_l_ratio_abdomen_RL": ("AbdomenRight_likelihood", "AbdomenLeft_likelihood"),
        "log_l_ratio_abdomen_TB": ("AbdomenTop_likelihood", "AbdomenBottom_likelihood"),
        "log_l_ratio_abdomen_BT": ("AbdomenBottom_likelihood", "AbdomenTop_likelihood"),
    }

    for new_col, (col1, col2) in l_ratio_combinations.items():
        if col1 in dataframe.columns and col2 in dataframe.columns:
            dataframe[new_col] = np.log((dataframe[col1] + epsilon) / (dataframe[col2] + epsilon))
        else:
            logger.warning(
                f"Warning: Columns {col1} and/or {col2} not found in DataFrame. Skipping {new_col}."
            )

    return dataframe


def calculate_length_features(input_df, length_pairs):
    """Calculate Euclidean distances between the specified body parts in pixels."""

    new_columns = {}

    for partA, partB, designation in length_pairs:
        if partA.lower() == "centroid":
            xA, yA = "centroid_x", "centroid_y"
        else:
            xA, yA = f"{partA}_x", f"{partA}_y"

        if partB.lower() == "centroid":
            xB, yB = "centroid_x", "centroid_y"
        else:
            xB, yB = f"{partB}_x", f"{partB}_y"

        if not all(col in input_df.columns for col in [xA, yA, xB, yB]):
            logger.warning(f"Warning: Required columns for {partA} and/or {partB} not found in DataFrame.")
            continue

        distances = np.sqrt((input_df[xA] - input_df[xB]) ** 2 + (input_df[yA] - input_df[yB]) ** 2)

        new_col = f"length_{designation}"
        new_columns[new_col] = distances

    input_df = pd.concat([input_df, pd.DataFrame(new_columns, index=input_df.index)], axis=1)

    return input_df


def calculate_angle_features(dataframe, bodypart_triplets):
    """Calculate unsigned vertex angles in degrees, preserving NaNs on invalid rows."""

    new_columns = {}

    for partA, partB, partC, designation in bodypart_triplets:
        xA, yA = f"{partA}_x", f"{partA}_y"
        xB, yB = f"{partB}_x", f"{partB}_y"
        xC, yC = f"{partC}_x", f"{partC}_y"

        required_cols = [xA, yA, xB, yB, xC, yC]
        if not all(col in dataframe.columns for col in required_cols):
            logger.warning(
                f"Warning: Required columns for {partA}, {partB}, and/or {partC} not found in DataFrame."
            )
            continue

        new_col = f"anglefeature_{designation}"
        angle_series = pd.Series(np.nan, index=dataframe.index)

        # Identify valid rows where none of the required coordinates is NaN
        valid_rows = (
            (~dataframe[xA].isna())
            & (~dataframe[yA].isna())
            & (~dataframe[xB].isna())
            & (~dataframe[yB].isna())
            & (~dataframe[xC].isna())
            & (~dataframe[yC].isna())
        )

        if valid_rows.sum() == 0:
            new_columns[new_col] = angle_series
            continue

        # Calculate vectors: from B to A and from B to C
        vecAB = dataframe.loc[valid_rows, [xA, yA]].values - dataframe.loc[valid_rows, [xB, yB]].values
        vecBC = dataframe.loc[valid_rows, [xC, yC]].values - dataframe.loc[valid_rows, [xB, yB]].values

        dot_product = np.einsum("ij,ij->i", vecAB, vecBC)
        magnitudeAB = np.linalg.norm(vecAB, axis=1)
        magnitudeBC = np.linalg.norm(vecBC, axis=1)

        # Compute the cosine of the angle and clip to avoid floating point issues
        cos_theta = dot_product / (magnitudeAB * magnitudeBC)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angles = np.degrees(np.arccos(cos_theta))

        angle_series.loc[valid_rows] = angles
        new_columns[new_col] = angle_series

    dataframe = pd.concat([dataframe, pd.DataFrame(new_columns, index=dataframe.index)], axis=1)

    return dataframe


def calculate_head_orientation(dataframe):
    """Calculate signed angles between body and head reference lines in degrees.

    Recenter the Body_Eyes angle by 90 degrees and wrap it to [-180, 180).
    """

    new_data_cols = {}

    def calculate_angle(partA, partB, partC, partD, angle_name):
        """Calculate the signed angle between two reference vectors on valid rows."""
        xA, yA = f"{partA}_x", f"{partA}_y"
        xB, yB = f"{partB}_x", f"{partB}_y"
        xC, yC = f"{partC}_x", f"{partC}_y"
        xD, yD = f"{partD}_x", f"{partD}_y"

        required_cols = [xA, yA, xB, yB, xC, yC, xD, yD]
        if not all(col in dataframe.columns for col in required_cols):
            logger.warning(
                f"Warning: Required columns for {partA}, {partB}, {partC}, and/or {partD} not found in DataFrame."
            )
            return

        angle_series = pd.Series(np.nan, index=dataframe.index)

        # Identify valid rows: those where none of the required coordinates is missing (NaN)
        valid_rows = (
            (~dataframe[xA].isna())
            & (~dataframe[yA].isna())
            & (~dataframe[xB].isna())
            & (~dataframe[yB].isna())
            & (~dataframe[xC].isna())
            & (~dataframe[yC].isna())
            & (~dataframe[xD].isna())
            & (~dataframe[yD].isna())
        )

        if valid_rows.sum() > 0:
            vecAB = dataframe.loc[valid_rows, [xB, yB]].values - dataframe.loc[valid_rows, [xA, yA]].values
            vecCD = dataframe.loc[valid_rows, [xD, yD]].values - dataframe.loc[valid_rows, [xC, yC]].values

            dot_product = np.einsum("ij,ij->i", vecAB, vecCD)
            magnitudeAB = np.linalg.norm(vecAB, axis=1)
            magnitudeCD = np.linalg.norm(vecCD, axis=1)

            # Calculate cosine of the angle and clip to avoid issues near the limits [-1, 1]
            cos_theta = dot_product / (magnitudeAB * magnitudeCD)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angles = np.degrees(np.arccos(cos_theta))

            # Determine the sign of the angle using the cross product
            cross_product = vecAB[:, 0] * vecCD[:, 1] - vecAB[:, 1] * vecCD[:, 0]
            angles = np.where(cross_product < 0, -angles, angles)

            # Store the computed angles; invalid (NaN) rows will remain NaN
            angle_series.loc[valid_rows] = angles

        new_data_cols[angle_name] = angle_series

    calculate_angle("Back", "Front", "EyeLeft", "EyeRight", "headanglefeature_Body_Eyes")
    calculate_angle(
        "Front", "Back", "Ocelli", "Mouthparts", "headanglefeature_Body_Ocelli_Mouth"
    )

    # Re-center the Body_Eyes angle so that 0 represents a straight head
    if "headanglefeature_Body_Eyes" in new_data_cols:
        corrected_angle = new_data_cols["headanglefeature_Body_Eyes"] - 90
        new_data_cols["headanglefeature_Body_Eyes"] = (corrected_angle + 180) % 360 - 180

    dataframe = pd.concat([dataframe, pd.DataFrame(new_data_cols, index=dataframe.index)], axis=1)

    return dataframe


def calculate_area_features(df):
    """Calculate polygon areas in pixels squared using the canonical vertex order."""
    new_data_cols = {}

    area_definitions = {
        "stance_area": stance_area_vertices,
        "body_area_top": body_area_top_vertices,
        "body_area_side": body_area_side_vertices,
    }

    for area_name, vertices in area_definitions.items():
        required_cols = [f"{v}_x" for v in vertices] + [f"{v}_y" for v in vertices]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Warning: Missing required columns for '{area_name}'. Skipping.")
            continue

        new_data_cols[area_name] = shoelace_area(df, vertices)

    df = pd.concat([df, pd.DataFrame(new_data_cols, index=df.index)], axis=1)

    return df


def add_bodyshape_features(input_df, poscorr_dict):
    """Add likelihood ratios, lengths, angles, head orientation, and areas to the feature mapping."""
    input_df = calculate_log_l_ratios(input_df)
    add_existing_columns(
        poscorr_dict,
        input_df,
        [
            "log_l_ratio_FB",
            "log_l_ratio_head_LR",
            "log_l_ratio_head_RL",
            "log_l_ratio_head_TB",
            "log_l_ratio_head_BT",
            "log_l_ratio_thorax_LR",
            "log_l_ratio_thorax_RL",
            "log_l_ratio_thorax_TB",
            "log_l_ratio_thorax_BT",
            "log_l_ratio_abdomen_LR",
            "log_l_ratio_abdomen_RL",
            "log_l_ratio_abdomen_TB",
            "log_l_ratio_abdomen_BT",
        ],
    )

    input_df = calculate_length_features(input_df, length_pairs)
    length_columns = [c for c in input_df.columns if c.startswith("length_")]
    for c in length_columns:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_angle_features(input_df, bodypart_triplets)
    angle_columns = [c for c in input_df.columns if c.startswith("anglefeature_")]
    for c in angle_columns:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_head_orientation(input_df)
    headangle_columns = [c for c in input_df.columns if c.startswith("headanglefeature_")]
    for c in headangle_columns:
        poscorr_dict[c] = input_df[c]

    input_df = calculate_area_features(input_df)
    for col in ["stance_area", "body_area_top", "body_area_side"]:
        if col in input_df.columns:
            poscorr_dict[col] = input_df[col]

    return input_df
