"""Define runtime options and output paths for KeyPreNorm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KeypreNormConfig:
    """Store runtime options for KeyPreNorm."""

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

    # 0 = automatic; positive values are an upper limit, still constrained by RAM/CPUs.
    num_workers: int = 0
    memory_gb: float | None = None
    edge_skip: int = 5

    batch_size: int = 10

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Project registry (optional)
    # When enabled, keyprenorm can:
    #   * discover inputs from the project registry (keypre_csv_relpath)
    #   * upsert outputs back into the registry (keyprenorm_h5_relpath,
    #     keyprenorm_params_relpath)
    registry_enabled: bool = False
    registry_csv: Optional[str] = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60
    file_list: str | None = None

    def resolved_registry_csv(self) -> Optional[str]:
        """Return the registry CSV path if enabled."""
        return self.registry_csv if self.registry_enabled else None
