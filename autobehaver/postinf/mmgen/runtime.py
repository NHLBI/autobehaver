"""Explicit runtime settings for mmgen.

This module centralizes the per-run settings that were previously propagated by
mutating module-level variables across the mmgen submodules. The goal is to
keep a single explicit source of truth for one mmgen invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .kinematics import KIN_FRAMES_SIGNAL_COLS as DEFAULT_KIN_FRAMES_SIGNAL_COLS


@dataclass(frozen=True)
class MMGenRuntimeSettings:
    """Resolved runtime settings for one mmgen run.

    These settings are derived from :class:`MMGenConfig` and passed explicitly
    into the orchestrator, workers, and downstream analysis helpers.
    """

    num_workers: int = 16
    input_batch_size: int = 20
    keep_temp_files: bool = True
    emit_master_matrix: bool = True
    emit_full_outputs: bool = True

    make_measurements: bool = True
    calculate_occupancy: bool = True
    calculate_postural_features: bool = True
    make_detailed_transition_outputs: bool = True

    generate_quiver_plots: bool = True
    run_transition_statistics: bool = True

    filter_features_by_bout_count: bool = True
    min_bout_count_for_features: int = 6
    min_bout_count_for_transitions: int = 6

    random_seed: int = 42
    chamber_diameter_px: int | None = 530
    mm_per_px: float | None = 0.0169896744624207
    video_fps: float | None = 40.0
    kin_frames_signal_cols: tuple[str, ...] = tuple(DEFAULT_KIN_FRAMES_SIGNAL_COLS)

    @classmethod
    def from_config(cls, cfg) -> "MMGenRuntimeSettings":
        """Build runtime settings from an ``MMGenConfig``-like object."""

        return cls(
            num_workers=int(cfg.num_workers),
            input_batch_size=int(cfg.input_batch_size),
            keep_temp_files=bool(cfg.keep_temp_files),
            emit_master_matrix=bool(cfg.emit_master_matrix),
            emit_full_outputs=bool(cfg.emit_full_outputs),
            make_measurements=bool(cfg.make_measurements),
            calculate_occupancy=bool(cfg.calculate_occupancy),
            calculate_postural_features=bool(cfg.calculate_postural_features),
            make_detailed_transition_outputs=bool(cfg.make_detailed_transition_outputs),
            generate_quiver_plots=bool(cfg.generate_quiver_plots),
            run_transition_statistics=bool(cfg.run_transition_statistics),
            filter_features_by_bout_count=bool(cfg.filter_features_by_bout_count),
            min_bout_count_for_features=int(cfg.min_bout_count_for_features),
            min_bout_count_for_transitions=int(cfg.min_bout_count_for_transitions),
            random_seed=int(cfg.random_seed),
            chamber_diameter_px=(None if cfg.chamber_diameter_px is None else int(cfg.chamber_diameter_px)),
            mm_per_px=(None if cfg.mm_per_px is None else float(cfg.mm_per_px)),
            video_fps=(None if cfg.video_fps is None else float(cfg.video_fps)),
            kin_frames_signal_cols=tuple(DEFAULT_KIN_FRAMES_SIGNAL_COLS),
        )
