"""Define video selection, model, and rendering options for DeepLabCut inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from pathlib import Path


def default_dlc_config_path() -> str:
    """Return the bundled portable DeepLabCut project config path."""
    return str(
        Path(__file__).resolve().parents[2]
        / "resources"
        / "dlc"
        / "m1"
        / "config.yaml"
    )


DEFAULT_DLC_CONFIG_PATH: str | None = default_dlc_config_path()


@dataclass(frozen=True)
class DLCInferencerConfig:
    """Store runtime options for DeepLabCut inference."""

    # Required
    input_dir: str

    # DLC project config
    dlc_config: str

    # Video selection
    videos: list[str] | None = None
    video_glob: str | None = None
    recursive: bool = False
    videotype: str = ".mp4"  # DLC uses leading dot convention

    # Subset selection
    file_filter: str | None = None  # case-insensitive substring match on filename

    # Inference options
    shuffle: int = 1
    max_retries: int = 10
    dlc_python: str | None = None
    output_dir: str | None = None
    scratch_dir: str | None = None

    # Pipeline behavior
    clobber: bool = False

    # Optional exact scorer-tag substring to remove from CSV filenames.
    dlc_tag_to_remove: str | None = None

    # Keep scorer tags when explicitly requested despite dlc_tag_to_remove.
    keep_dlc_tag: bool = False

    # Labeled movie options
    make_movie: bool = False
    skip_analysis: bool = False
    dotsize: int = 24
    pcutoff: float = 0.6
    colormap: str = "rainbow"
    exclude_bodyparts: list[str] = field(default_factory=lambda: ["Front", "Back"])
    color_by: str = "bodypart"  # bodypart | individual
    trailpoints: int = 0
    fastmode: bool = True
    codec: str = "mp4v"
    overwrite: bool = True

    # Project registry (single CSV at project root)
    registry_enabled: bool = False
    registry_csv: str | None = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60
    file_list: str | None = None

    def resolved_registry_csv(self) -> str | None:
        """Return the explicit or inferred registry path, or None when disabled."""

        if not bool(self.registry_enabled):
            return None

        if self.registry_csv:
            return str(Path(self.registry_csv).expanduser().resolve())

        in_dir = Path(self.input_dir).expanduser().resolve()
        out_dir = Path(self.output_dir).expanduser().resolve() if self.output_dir else None

        if out_dir is not None:
            try:
                root = Path(os.path.commonpath([str(in_dir), str(out_dir)])).resolve()
            except Exception:
                root = in_dir.parent
        else:
            root = in_dir.parent if re.match(r"^\d+_", in_dir.name) else in_dir

        return str(root / "registry.csv")
