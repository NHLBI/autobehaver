"""Resolve numbered pipeline folders and the project registry for stage CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

_STAGE_LAYOUT: dict[str, dict[str, str]] = {
    "moviechecker": {
        "input_dir": "1_full_movies",
        "exclusions_csv": "1_full_movies/exclusions.csv",
    },
    "autocropper": {
        "input_dir": "1_full_movies",
        "output_dir": "2_cropped_movies",
    },
    "dlc": {
        "input_dir": "2_cropped_movies",
        "output_dir": "3_dlc",
    },
    "keypre": {
        "input_dir": "3_dlc",
        "output_dir": "4_keypoint_features",
    },
    "keyprenorm": {
        "input_dir": "4_keypoint_features",
        "output_dir": "5_normalized_features",
    },
    "posepred_inference": {
        "input_dir": "5_normalized_features",
        "output_dir": "6_posepred_inference",
    },
    "infmovie": {
        "movie_dir": "2_cropped_movies",
        "inference_csv_dir": "6_posepred_inference",
        "output_dir": "7_infmovies",
    },
    "mmgen": {
        "poscalc_dir": "4_keypoint_features",
        "inference_csv_dir": "6_posepred_inference",
        "output_dir": "8_master_matrix",
    },
    "featureimp": {
        "matrix_dir": "8_master_matrix",
        "results_dir": "9_feature_importance",
    },
}


def stage_paths_for_main_dir(stage: str, main_dir: str) -> Dict[str, str]:
    """Return stage paths under the numbered-folder project root.

    ``stage`` is a canonical key such as ``keyprenorm``; unknown keys raise
    KeyError. Explicit CLI overrides are handled separately.
    """

    stage_key = str(stage or "").strip().lower()
    if stage_key not in _STAGE_LAYOUT:
        raise KeyError(f"Unknown pipeline stage for main_dir resolution: {stage!r}")

    root = Path(main_dir).expanduser().resolve()
    return {k: str((root / rel).resolve()) for k, rel in _STAGE_LAYOUT[stage_key].items()}


def registry_csv_for_main_dir(main_dir: str) -> str:
    """Return the default project registry CSV path for a project root."""

    root = Path(main_dir).expanduser().resolve()
    return str((root / "registry.csv").resolve())


def _is_missing_pathlike(value: Any) -> bool:
    if value is None:
        return True
    try:
        return str(value).strip() == ""
    except Exception:
        return False


def apply_main_dir_overrides(namespace: Any, *, stage: str) -> Any:
    """Fill missing CLI namespace paths from ``main_dir``.

    Explicit CLI values win because only empty / unset destinations are
    populated. If registry integration is enabled and ``registry_csv`` is not
    provided, it defaults to ``<main_dir>/registry.csv``.
    """

    main_dir = getattr(namespace, "main_dir", None)
    if _is_missing_pathlike(main_dir):
        return namespace

    derived = stage_paths_for_main_dir(stage, str(main_dir))
    for dest, path in derived.items():
        if hasattr(namespace, dest) and _is_missing_pathlike(getattr(namespace, dest, None)):
            setattr(namespace, dest, path)

    if getattr(namespace, "registry_enabled", False) and hasattr(namespace, "registry_csv"):
        if _is_missing_pathlike(getattr(namespace, "registry_csv", None)):
            setattr(namespace, "registry_csv", registry_csv_for_main_dir(str(main_dir)))

    return namespace


__all__ = [
    "stage_paths_for_main_dir",
    "registry_csv_for_main_dir",
    "apply_main_dir_overrides",
]
