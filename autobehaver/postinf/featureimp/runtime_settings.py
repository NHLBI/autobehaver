"""Provide typed subsets of FeatureImp runtime settings for summary, validation, and
FSEA helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import DEFAULT_SETTINGS
from .integrity import strict_bool
import math


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True)
class SummaryRuntimeSettings:
    """Hold the failure policy used by SHAP-summary validation."""
    validation_error_on_fail: bool = bool(DEFAULT_SETTINGS.get("VALIDATION_ERROR_ON_FAIL", False))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SummaryRuntimeSettings":
        """Resolve summary-validation settings from explicit values and defaults."""
        if mapping is None:
            mapping = {}
        return cls(
            validation_error_on_fail=strict_bool(
                mapping.get("VALIDATION_ERROR_ON_FAIL", DEFAULT_SETTINGS.get("VALIDATION_ERROR_ON_FAIL", False))
            ),
        )


@dataclass(frozen=True)
class ValidationRuntimeSettings:
    """Hold the deterministic seed used by validation helpers."""
    global_random_seed: Optional[int] = _coerce_optional_int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ValidationRuntimeSettings":
        """Resolve the validation seed from explicit values and defaults."""
        if mapping is None:
            mapping = {}
        return cls(
            global_random_seed=_coerce_optional_int(
                mapping.get("GLOBAL_RANDOM_SEED", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
            ),
        )


@dataclass(frozen=True)
class FSEARuntimeSettings:
    """Hold enrichment calculation and visualization settings."""
    global_random_seed: Optional[int] = _coerce_optional_int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
    num_workers: int = int(DEFAULT_SETTINGS.get("num_workers", 1))
    fsea_debug_mode: bool = bool(DEFAULT_SETTINGS.get("fsea_debug_mode", False))
    fsea_fdr_threshold: float = float(DEFAULT_SETTINGS.get("fsea_fdr_threshold", 0.15))
    fsea_min_size: int = int(DEFAULT_SETTINGS.get("fsea_min_size", 15))
    fsea_max_size: int = int(DEFAULT_SETTINGS.get("fsea_max_size", 500))
    fsea_permutations: int = int(DEFAULT_SETTINGS.get("fsea_permutations", 100000))
    fsea_plot_save_dpi: int = int(DEFAULT_SETTINGS.get("fsea_plot_save_dpi", 300))
    fsea_plot_sig_only: bool = bool(DEFAULT_SETTINGS.get("fsea_plot_sig_only", True))
    fsea_use_multiprocessing: bool = bool(DEFAULT_SETTINGS.get("fsea_use_multiprocessing", True))
    fsea_viz_plot_all_groups: bool = bool(DEFAULT_SETTINGS.get("fsea_viz_plot_all_groups", False))
    fsea_viz_top_n: int = int(DEFAULT_SETTINGS.get("fsea_viz_top_n", 30))
    fsea_viz_color_percentile: Optional[float] = (
        None if DEFAULT_SETTINGS.get("fsea_viz_color_percentile", 95) is None
        else float(DEFAULT_SETTINGS.get("fsea_viz_color_percentile", 95))
    )

    def __post_init__(self):
        """Require finite thresholds and valid enrichment sizes and runtime limits."""
        for key in ('fsea_fdr_threshold', 'fsea_viz_color_percentile'):
            value = getattr(self, key)
            if value is not None and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if not 0 <= self.fsea_fdr_threshold <= 1 or self.fsea_min_size < 1 or self.fsea_max_size < self.fsea_min_size:
            raise ValueError('Invalid FSEA probability/group sizes')
        if self.num_workers < 1 or self.fsea_permutations < 1 or self.fsea_plot_save_dpi < 1 or self.fsea_viz_top_n < 0:
            raise ValueError('Invalid FSEA workers/permutations/plot settings')
        if self.fsea_viz_color_percentile is not None and not 0 <= self.fsea_viz_color_percentile <= 100:
            raise ValueError('FSEA color percentile must be in [0,100]')

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FSEARuntimeSettings":
        """Resolve and validate enrichment settings from explicit values and defaults."""
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
            num_workers=int(mapping.get("num_workers", DEFAULT_SETTINGS.get("num_workers", 1))),
            fsea_debug_mode=strict_bool(mapping.get("fsea_debug_mode", DEFAULT_SETTINGS.get("fsea_debug_mode", False))),
            fsea_fdr_threshold=float(
                mapping.get("fsea_fdr_threshold", DEFAULT_SETTINGS.get("fsea_fdr_threshold", 0.15))
            ),
            fsea_min_size=int(mapping.get("fsea_min_size", DEFAULT_SETTINGS.get("fsea_min_size", 15))),
            fsea_max_size=int(mapping.get("fsea_max_size", DEFAULT_SETTINGS.get("fsea_max_size", 500))),
            fsea_permutations=int(
                mapping.get("fsea_permutations", DEFAULT_SETTINGS.get("fsea_permutations", 100000))
            ),
            fsea_plot_save_dpi=int(
                mapping.get("fsea_plot_save_dpi", DEFAULT_SETTINGS.get("fsea_plot_save_dpi", 300))
            ),
            fsea_plot_sig_only=strict_bool(
                mapping.get("fsea_plot_sig_only", DEFAULT_SETTINGS.get("fsea_plot_sig_only", True))
            ),
            fsea_use_multiprocessing=strict_bool(
                mapping.get(
                    "fsea_use_multiprocessing",
                    DEFAULT_SETTINGS.get("fsea_use_multiprocessing", True),
                )
            ),
            fsea_viz_plot_all_groups=strict_bool(
                mapping.get(
                    "fsea_viz_plot_all_groups",
                    DEFAULT_SETTINGS.get("fsea_viz_plot_all_groups", False),
                )
            ),
            fsea_viz_top_n=int(
                mapping.get("fsea_viz_top_n", DEFAULT_SETTINGS.get("fsea_viz_top_n", 30))
            ),
            fsea_viz_color_percentile=(
                None if raw_color_percentile is None else float(raw_color_percentile)
            ),
        )
