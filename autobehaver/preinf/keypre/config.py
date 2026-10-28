"""Define runtime options and output paths for KeyPre."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypreConfig:
    """Store runtime options for KeyPre."""

    # Paths
    input_dir: str
    output_dir: str | None = None

    # Optional: only process input CSVs whose *filename* contains this substring
    # (case-insensitive). Useful for bundling jobs without subfolders.
    file_filter: str | None = None

    # Runtime
    # 0 = automatic; positive values are an upper limit, still constrained by RAM/CPUs.
    num_workers: int = 0
    memory_gb: float | None = None
    clobber: bool = False
    movie_dim: int = 600
    dry_run: bool = False
    log_level: str = "INFO"

    # Modes
    # If True, bypass all feature engineering/corrections and export only the
    # raw DLC keypoints (x/y/likelihood per part, excluding Front/Back).
    raw_only: bool = False

    # Internal engine defaults
    chunk_size: int = 100
    pool_maxtasksperchild: int = 100

    # Project registry (optional)
    # When enabled, keypre discovers inputs from the project registry
    # (dlc_csv_relpath) instead of scanning the filesystem, and then upserts
    # keypre outputs (keypre_csv_relpath + keypre_version) back into the
    # registry.
    registry_enabled: bool = False
    registry_csv: str | None = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60
    file_list: str | None = None

    def resolved_registry_csv(self) -> str | None:
        """Return the registry CSV path if enabled."""
        return self.registry_csv if self.registry_enabled else None
