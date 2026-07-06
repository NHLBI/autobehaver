"""Typed runtime settings helpers for FeatureImp submodules.

These dataclasses let submodules consume the exact settings they need without
relying on cross-module mutation of module-level globals. They are intentionally
small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import DEFAULT_SETTINGS


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True)
class SummaryRuntimeSettings:
    validation_error_on_fail: bool = bool(DEFAULT_SETTINGS.get("VALIDATION_ERROR_ON_FAIL", False))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SummaryRuntimeSettings":
        if mapping is None:
            mapping = {}
        return cls(
            validation_error_on_fail=bool(
                mapping.get("VALIDATION_ERROR_ON_FAIL", DEFAULT_SETTINGS.get("VALIDATION_ERROR_ON_FAIL", False))
            ),
        )


@dataclass(frozen=True)
class ValidationRuntimeSettings:
    global_random_seed: Optional[int] = _coerce_optional_int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ValidationRuntimeSettings":
        if mapping is None:
            mapping = {}
        return cls(
            global_random_seed=_coerce_optional_int(
                mapping.get("GLOBAL_RANDOM_SEED", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
            ),
        )


@dataclass(frozen=True)
class FSEARuntimeSettings:
    global_random_seed: Optional[int] = _coerce_optional_int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
    num_workers: int = int(DEFAULT_SETTINGS.get("num_workers", 1) or 1)
    fsea_debug_mode: bool = bool(DEFAULT_SETTINGS.get("fsea_debug_mode", False))
    fsea_fdr_threshold: float = float(DEFAULT_SETTINGS.get("fsea_fdr_threshold", 0.25) or 0.25)
    fsea_min_size: int = int(DEFAULT_SETTINGS.get("fsea_min_size", 15) or 15)
    fsea_max_size: int = int(DEFAULT_SETTINGS.get("fsea_max_size", 500) or 500)
    fsea_permutations: int = int(DEFAULT_SETTINGS.get("fsea_permutations", 100000) or 100000)
    fsea_plot_save_dpi: int = int(DEFAULT_SETTINGS.get("fsea_plot_save_dpi", 300) or 300)
    fsea_plot_sig_only: bool = bool(DEFAULT_SETTINGS.get("fsea_plot_sig_only", True))
    fsea_use_multiprocessing: bool = bool(DEFAULT_SETTINGS.get("fsea_use_multiprocessing", True))
    fsea_viz_plot_all_groups: bool = bool(DEFAULT_SETTINGS.get("fsea_viz_plot_all_groups", False))
    fsea_viz_top_n: int = int(DEFAULT_SETTINGS.get("fsea_viz_top_n", 30) or 30)
    fsea_viz_color_percentile: Optional[float] = (
        None if DEFAULT_SETTINGS.get("fsea_viz_color_percentile", 95) is None
        else float(DEFAULT_SETTINGS.get("fsea_viz_color_percentile", 95))
    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FSEARuntimeSettings":
        if mapping is None:
            mapping = {}
        raw_color_percentile = mapping.get(
            "fsea_viz_color_percentile",
            DEFAULT_SETTINGS.get("fsea_viz_color_percentile", 95),
        )
        return cls(
            global_random_seed=_coerce_optional_int(
                mapping.get("GLOBAL_RANDOM_SEED", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
            ),
            num_workers=int(mapping.get("num_workers", DEFAULT_SETTINGS.get("num_workers", 1)) or 1),
            fsea_debug_mode=bool(mapping.get("fsea_debug_mode", DEFAULT_SETTINGS.get("fsea_debug_mode", False))),
            fsea_fdr_threshold=float(
                mapping.get("fsea_fdr_threshold", DEFAULT_SETTINGS.get("fsea_fdr_threshold", 0.25)) or 0.25
            ),
            fsea_min_size=int(mapping.get("fsea_min_size", DEFAULT_SETTINGS.get("fsea_min_size", 15)) or 15),
            fsea_max_size=int(mapping.get("fsea_max_size", DEFAULT_SETTINGS.get("fsea_max_size", 500)) or 500),
            fsea_permutations=int(
                mapping.get("fsea_permutations", DEFAULT_SETTINGS.get("fsea_permutations", 100000)) or 100000
            ),
            fsea_plot_save_dpi=int(
                mapping.get("fsea_plot_save_dpi", DEFAULT_SETTINGS.get("fsea_plot_save_dpi", 300)) or 300
            ),
            fsea_plot_sig_only=bool(
                mapping.get("fsea_plot_sig_only", DEFAULT_SETTINGS.get("fsea_plot_sig_only", True))
            ),
            fsea_use_multiprocessing=bool(
                mapping.get(
                    "fsea_use_multiprocessing",
                    DEFAULT_SETTINGS.get("fsea_use_multiprocessing", True),
                )
            ),
            fsea_viz_plot_all_groups=bool(
                mapping.get(
                    "fsea_viz_plot_all_groups",
                    DEFAULT_SETTINGS.get("fsea_viz_plot_all_groups", False),
                )
            ),
            fsea_viz_top_n=int(
                mapping.get("fsea_viz_top_n", DEFAULT_SETTINGS.get("fsea_viz_top_n", 30)) or 30
            ),
            fsea_viz_color_percentile=(
                None if raw_color_percentile is None else float(raw_color_percentile)
            ),
        )
