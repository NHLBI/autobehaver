"""Keypre feature specification.

This module contains the bodypart lists and feature-definition specs used by keypre.
Ordering controls downstream output column order.
"""

from __future__ import annotations

# List of all bodyparts
parts = [
    "Ocelli",
    "Antennae",
    "Mouthparts",
    "EyeLeft",
    "EyeRight",
    "FrontTarsusLeft",
    "FrontJointLeft",
    "FrontTarsusRight",
    "FrontJointRight",
    "MidTarsusLeft",
    "MidJointLeft",
    "MidTarsusRight",
    "MidJointRight",
    "BackTarsusLeft",
    "BackJointLeft",
    "BackTarsusRight",
    "BackJointRight",
    "ThoraxTop",
    "ThoraxLeft",
    "ThoraxRight",
    "ThoraxBottom",
    "AbdomenTop",
    "AbdomenLeft",
    "AbdomenRight",
    "AbdomenBottom",
    "AbdomenTip",
    "WingLeft",
    "WingRight",
]


# Define pairs of body parts for length calculations
length_pairs = [
    ("Front", "Back", "FB"),  # Overall body length
    ("FrontTarsusLeft", "FrontJointLeft", "LegTipFL"),
    ("FrontTarsusRight", "FrontJointRight", "LegTipFR"),
    ("MidTarsusLeft", "MidJointLeft", "LegTipML"),
    ("MidTarsusRight", "MidJointRight", "LegTipMR"),
    ("BackTarsusLeft", "BackJointLeft", "LegTipBL"),
    ("BackTarsusRight", "BackJointRight", "LegTipBR"),
    ("FrontTarsusLeft", "MidTarsusLeft", "LegDist_FL_ML"),
    ("FrontTarsusLeft", "BackTarsusLeft", "LegDist_FL_BL"),
    ("MidTarsusLeft", "BackTarsusLeft", "LegDist_ML_BL"),
    ("FrontTarsusRight", "MidTarsusRight", "LegDist_FR_MR"),
    ("FrontTarsusRight", "BackTarsusRight", "LegDist_FR_BR"),
    ("MidTarsusRight", "BackTarsusRight", "LegDist_MR_BR"),
    ("FrontTarsusLeft", "FrontTarsusRight", "LegDist_FL_FR"),
    ("MidTarsusLeft", "MidTarsusRight", "LegDist_ML_MR"),
    ("BackTarsusLeft", "BackTarsusRight", "LegDist_BL_BR"),
    ("centroid", "FrontTarsusLeft", "LegFull_FL"),
    ("centroid", "FrontTarsusRight", "LegFull_FR"),
    ("centroid", "MidTarsusLeft", "LegFull_ML"),
    ("centroid", "MidTarsusRight", "LegFull_MR"),
    ("centroid", "BackTarsusLeft", "LegFull_BL"),
    ("centroid", "BackTarsusRight", "LegFull_BR"),
    ("centroid", "WingLeft", "WingL"),
    ("centroid", "WingRight", "WingR"),
    ("WingLeft", "WingRight", "interWing"),
]


# Define body part triplets for angle calculations
bodypart_triplets = [
    ("FrontTarsusLeft", "centroid", "MidTarsusLeft", "Legs_FL_ML"),
    ("FrontTarsusLeft", "centroid", "BackTarsusLeft", "Legs_FL_BL"),
    ("MidTarsusLeft", "centroid", "BackTarsusLeft", "Legs_ML_BL"),
    ("FrontTarsusRight", "centroid", "MidTarsusRight", "Legs_FR_MR"),
    ("FrontTarsusRight", "centroid", "BackTarsusRight", "Legs_FR_BR"),
    ("MidTarsusRight", "centroid", "BackTarsusRight", "Legs_MR_BR"),
    ("FrontTarsusLeft", "centroid", "FrontTarsusRight", "Legs_FL_FR"),
    ("MidTarsusLeft", "centroid", "MidTarsusRight", "Legs_ML_MR"),
    ("BackTarsusLeft", "centroid", "BackTarsusRight", "Legs_BL_BR"),
    ("WingLeft", "centroid", "WingRight", "Wings_L_R"),
]


# Define body parts for velocity calculations
keypoints_velocity_parts = [
    "Ocelli",
    "Antennae",
    "Mouthparts",
    "EyeLeft",
    "EyeRight",
    "FrontTarsusLeft",
    "FrontJointLeft",
    "FrontTarsusRight",
    "FrontJointRight",
    "MidTarsusLeft",
    "MidJointLeft",
    "MidTarsusRight",
    "MidJointRight",
    "BackTarsusLeft",
    "BackJointLeft",
    "BackTarsusRight",
    "BackJointRight",
    "ThoraxTop",
    "ThoraxLeft",
    "ThoraxRight",
    "ThoraxBottom",
    "AbdomenTop",
    "AbdomenLeft",
    "AbdomenRight",
    "AbdomenBottom",
    "AbdomenTip",
    "WingLeft",
    "WingRight",
]


# Define body part pairs for pairwise angular changes
angular_velocity_pairs = [
    # All joints and their corresponding tarsus
    ("FrontJointLeft", "FrontTarsusLeft"),
    ("FrontJointRight", "FrontTarsusRight"),
    ("MidJointLeft", "MidTarsusLeft"),
    ("MidJointRight", "MidTarsusRight"),
    ("BackJointLeft", "BackTarsusLeft"),
    ("BackJointRight", "BackTarsusRight"),
    # Centroid and all tarsus
    ("centroid", "FrontTarsusLeft"),
    ("centroid", "FrontTarsusRight"),
    ("centroid", "MidTarsusLeft"),
    ("centroid", "MidTarsusRight"),
    ("centroid", "BackTarsusLeft"),
    ("centroid", "BackTarsusRight"),
    # Centroid and all joints
    ("centroid", "FrontJointLeft"),
    ("centroid", "FrontJointRight"),
    ("centroid", "MidJointLeft"),
    ("centroid", "MidJointRight"),
    ("centroid", "BackJointLeft"),
    ("centroid", "BackJointRight"),
    # Centroid and both wings
    ("centroid", "WingLeft"),
    ("centroid", "WingRight"),
    # ThoraxTop with Ocelli and AbdomenTop
    ("ThoraxTop", "Ocelli"),
    ("ThoraxTop", "AbdomenTop"),
    # ThoraxLeft/Right with EyeLeft/Right and AbdomenLeft/Right
    ("ThoraxLeft", "EyeLeft"),
    ("ThoraxRight", "EyeRight"),
    ("ThoraxLeft", "AbdomenLeft"),
    ("ThoraxRight", "AbdomenRight"),
    # ThoraxBottom with Mouthparts and AbdomenBottom
    ("ThoraxBottom", "Mouthparts"),
    ("ThoraxBottom", "AbdomenBottom"),
]


# Define vertices for area calculations using the Shoelace formula
stance_area_vertices = [
    "FrontTarsusLeft",
    "MidTarsusLeft",
    "BackTarsusLeft",
    "BackTarsusRight",
    "MidTarsusRight",
    "FrontTarsusRight",
]

body_area_top_vertices = [
    "Antennae",
    "EyeRight",
    "ThoraxRight",
    "AbdomenRight",
    "AbdomenTip",
    "AbdomenLeft",
    "ThoraxLeft",
    "EyeLeft",
]

body_area_side_vertices = [
    "Antennae",
    "Ocelli",
    "ThoraxTop",
    "AbdomenTop",
    "AbdomenTip",
    "AbdomenBottom",
    "ThoraxBottom",
    "Mouthparts",
]


# Define keypoint pairs for asymmetry calculations
asymmetry_lr_pairs = [
    ("EyeLeft", "EyeRight", "Eyes"),
    ("ThoraxLeft", "ThoraxRight", "Thorax"),
    ("AbdomenLeft", "AbdomenRight", "Abdomen"),
    ("WingLeft", "WingRight", "Wings"),
    ("FrontTarsusLeft", "FrontTarsusRight", "FrontLegs"),
    ("MidTarsusLeft", "MidTarsusRight", "MidLegs"),
    ("BackTarsusLeft", "BackTarsusRight", "BackLegs"),
]

asymmetry_tb_pairs = [
    ("Ocelli", "Mouthparts", "Head"),
    ("ThoraxTop", "ThoraxBottom", "Thorax"),
    ("AbdomenTop", "AbdomenBottom", "Abdomen"),
]
