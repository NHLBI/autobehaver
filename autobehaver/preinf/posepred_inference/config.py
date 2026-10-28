"""Define model, averaging, smoothing, and output options for PosePred inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

def default_models():
    """Return the bundled final nh16/32/64 epoch-301 checkpoint paths."""
    root = Path(__file__).resolve().parents[2] / "resources" / "posepred"
    return [str(root / f"nh{n}_e301.pth") for n in (16,32,64)]



@dataclass
class PosePredInferenceConfig:
    """Store runtime options for the PosePred inference stage."""

    # I/O
    input_dir: str
    output_dir: Optional[str] = None

    # Models (one or more; multiple implies ensembling)
    inference_model: List[str] = field(default_factory=default_models)

    # Inference performance
    num_workers: int = -1
    batch: int = 0
    memory_gb: Optional[float] = None
    noise_workers: int = 0
    vram_fraction: float = .8

    # Test-time averaging / augmentation
    inf_average: int = 10
    use_tta: bool = True
    tta_noise_level: float = 0.01
    tta_seed: int = 42

    # Smoothing (bout-length in frames)
    min_pose_bout: int = 5
    min_orient_bout: int = 5

    # Sequence length fallback when a checkpoint does not provide one.
    sequence_length: int = 41

    # Pipeline conveniences
    file_filter: Optional[str] = None
    clobber: bool = False
    archive_existing: bool = False

    # Project registry integration (optional)
    # When enabled, inputs are discovered from the registry (keyprenorm_h5_relpath)
    # and outputs are upserted back (posepred_inference_csv_relpath, posepred_models).
    registry_enabled: bool = False
    registry_csv: Optional[str] = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60

    # Output layout:
    #   flat_output=True  -> write <output_dir>/<stem>_inference.csv and <output_dir>/inference_details/
    #   flat_output=False -> write <output_dir>/inferences/<stem>_inference.csv
    flat_output: bool = True

    # Optional: write a small run summary/manifest
    write_run_log: bool = True
    file_list: str | None = None
