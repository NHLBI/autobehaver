"""Configuration dataclass for keyprenorm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KeypreNormConfig:
    """Runtime configuration for the keyprenorm stage."""

    input_dir: str

    # Optional stage output root directory.
    # If provided, normalized H5 files and the default norm_params.csv are
    # written directly into this folder. Otherwise, outputs are written under
    # <input_dir>/normalization_h5 and <input_dir>/norm_params/.
    output_dir: Optional[str] = None

    # Input discovery
    pattern: str = "*_positional_calcs.csv"

    # Optional: only process files whose filename contains this substring (case-insensitive)
    file_filter: Optional[str] = None

    # Optional labels
    labels_dir: Optional[str] = None

    # Norm params (load/save)
    norm_params: Optional[str] = None
    calculate_norm: bool = False
    norm_only: bool = False

    # Processing behavior
    clobber: bool = False

    num_workers: int = 14
    edge_skip: int = 5

    # Fixed-length mode (max_rows>0) vs variable-length (max_rows=0)
    max_rows: int = 64750
    require_full_length: bool = False

    # Optional utilities
    size_check: bool = False
    batch_size: int = 10

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Project registry (optional)
    # ------------------------------------------------------------------
    # When enabled, keyprenorm can:
    #   * discover inputs from the project registry (keypre_csv_relpath)
    #   * upsert outputs back into the registry (keyprenorm_h5_relpath,
    #     keyprenorm_params_relpath)
    registry_enabled: bool = False
    registry_csv: Optional[str] = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60

    def resolved_registry_csv(self) -> Optional[str]:
        """Return the registry CSV path if enabled."""
        return self.registry_csv if self.registry_enabled else None
