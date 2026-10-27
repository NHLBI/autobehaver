"""Validate the KeyPre positional-feature contract consumed by MMGen.

Keypoints are centered and rotated, image centroids are absolute, and angle_sin/cos
encode navigation bearings. Signed speed derivatives are distinct from acceleration and
jerk magnitudes.
"""
from __future__ import annotations

from .kinematics import HEAD_KEYPOINTS, KINEMATIC_SIGNAL_COLS

DERIVED_HEAD_SIGNALS = frozenset(
    ("head_velocity_inst", "head_acceleration_inst", "head_jerk_inst")
)
LEG_TIPS = tuple(f"{leg}Tarsus{side}" for leg in ("Front", "Mid", "Back") for side in ("Left", "Right"))
MEASUREMENT_PARTS = (
    "Antennae", "AbdomenTip", "EyeRight", "EyeLeft", "Ocelli", "Mouthparts",
    "ThoraxLeft", "ThoraxRight", "ThoraxTop", "ThoraxBottom",
    "AbdomenLeft", "AbdomenRight", "AbdomenTop", "AbdomenBottom", "WingLeft", "WingRight",
)

def navigation_to_image_angle(angle):
    """Convert navigation bearing to atan2 angle in image coordinates.

    Navigation: 0=N, 90=E. Image atan2: 0=E, 90=S, -90=N.
    Works with arrays, Series and scalars and preserves missing values.
    """
    return (angle - 90.0 + 180.0) % 360.0 - 180.0

def required_keypre_columns(*, include_postural=True, include_spatial=True, include_measurements=True):
    """Return positional and kinematic columns required by the enabled MMGen analyses."""
    required = set(KINEMATIC_SIGNAL_COLS) - DERIVED_HEAD_SIGNALS
    for part in HEAD_KEYPOINTS:
        required.add(f"{part}_likelihood")
        for prefix in ("partvelocity", "partaccel", "partjerk"):
            required.add(f"{prefix}_{part}_inst")
    if include_spatial:
        required.update(("centroid_x", "centroid_y", "angle_sin", "angle_cos"))
        required.update(("anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR", "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR"))
    if include_postural:
        for part in LEG_TIPS + ("Antennae", "AbdomenTip"):
            required.update(f"{part}_{suffix}" for suffix in ("x", "y", "likelihood"))
        required.add("stance_area")
        for part in ("Eyes", "Thorax", "Abdomen", "Wings", "FrontLegs", "MidLegs", "BackLegs"):
            required.update(f"asym_lr_{axis}_{part}" for axis in ("x", "y"))
        for part in ("Head", "Thorax", "Abdomen"):
            required.update(f"asym_tb_{axis}_{part}" for axis in ("x", "y"))
        for part in ("Ocelli", "Mouthparts", "ThoraxTop", "ThoraxBottom", "AbdomenTop", "AbdomenBottom"):
            required.add(f"{part}_likelihood")
    if include_measurements:
        for part in MEASUREMENT_PARTS:
            required.update(f"{part}_{suffix}" for suffix in ("x", "y", "likelihood"))
        required.update(("body_area_top", "body_area_side"))
    return tuple(sorted(required))

def validate_keypre_inputs(frame, **options):
    """Fail before recording processing when required scientific inputs are absent.

    Finite-value/bout eligibility is separate: a present but partly missing column
    is allowed. Raw-only/feature-disabled ablations are not full Mmgen inputs.
    """
    missing = sorted(set(required_keypre_columns(**options)) - set(frame.columns))
    if missing:
        raise ValueError(
            "Mmgen input is incompatible with the current Keypre contract; missing columns: "
            + ", ".join(missing)
            + ". Supply full Keypre positional_calcs with signed acceleration_speed_inst/jerk_speed_inst, "
            "centered/rotated keypoints, likelihoods, and the enabled stage inputs. "
            "Legacy magnitude columns are not interchangeable with signed speed derivatives."
        )
