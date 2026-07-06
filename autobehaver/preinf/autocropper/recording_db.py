"""Recording database helpers (label lookup).

The lab's recording script produces timestamped backups like:
  recording_database_YYYYMMDD_HHMMSS.csv

Each row corresponds to a (experiment, subexperiment, trial_number) session and
contains a giant `camera_mappings` string mapping:
  rpi-<n>_chamber<m> : <LABEL_PREFIX>

Autocropper uses this mapping to prefix each cropped chamber movie so the label
propagates through the rest of the pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DB_GLOB = "recording_database_*.csv"


@dataclass(frozen=True)
class RecordingDBIndex:
    """In-memory index for recording database label lookups."""

    csv_path: Path
    # (experiment, subexperiment, trial_number) -> { "rpi-1_chamber1": "LABEL_", ... }
    label_map: dict[tuple[str, str, int], dict[str, str]]

    def get_label_prefix(
        self,
        *,
        experiment: str,
        subexperiment: str,
        trial_number: int,
        rpi_number: int,
        chamber_number: int,
    ) -> str:
        key = (str(experiment), str(subexperiment), int(trial_number))
        session = self.label_map.get(key)
        if session is None:
            raise KeyError(
                f"No recording DB entry found for (experiment={experiment!r}, subexperiment={subexperiment!r}, trial={trial_number})."
            )
        rc_key = f"rpi-{int(rpi_number)}_chamber{int(chamber_number)}"
        label = session.get(rc_key)
        if label is None:
            raise KeyError(
                f"No label found in recording DB for {rc_key} within (experiment={experiment!r}, subexperiment={subexperiment!r}, trial={trial_number})."
            )
        return str(label)


def _parse_db_timestamp_from_name(path: Path) -> datetime | None:
    """Parse YYYYMMDD_HHMMSS from recording_database_YYYYMMDD_HHMMSS.csv."""

    m = re.match(r"^recording_database_(\d{8})_(\d{6})$", path.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def find_latest_recording_database_csv(db_dir: Path) -> Path:
    """Find the newest recording_database_*.csv in a directory."""

    db_dir = Path(db_dir).expanduser().resolve()
    candidates = sorted(db_dir.glob(DB_GLOB))
    if not candidates:
        raise FileNotFoundError(f"No '{DB_GLOB}' files found in: {db_dir}")

    scored: list[tuple[float, Path]] = []
    for p in candidates:
        ts = _parse_db_timestamp_from_name(p)
        if ts is not None:
            scored.append((ts.timestamp(), p))
        else:
            # Fallback: file modified time
            scored.append((p.stat().st_mtime, p))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def parse_camera_mappings(camera_mappings: str) -> dict[str, str]:
    """Parse the camera_mappings blob into a dict.

    Expected format:
      rpi-1_chamber1:LABEL_,rpi-1_chamber2:LABEL_,...
    """

    out: dict[str, str] = {}
    if camera_mappings is None:
        return out

    s = str(camera_mappings).strip()
    if not s or s.lower() == "nan":
        return out

    for raw in s.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" not in entry:
            continue
        key, label = entry.split(":", 1)
        key = key.strip()
        label = label.strip()
        if key:
            out[key] = label
    return out


def load_recording_database_index(csv_path: Path) -> RecordingDBIndex:
    """Load the recording database and build a fast lookup map."""

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Recording database CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"experiment", "subexperiment", "trial_number", "camera_mappings"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(
            f"Recording DB is missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}"
        )

    label_map: dict[tuple[str, str, int], dict[str, str]] = {}

    for _i, row in df.iterrows():
        exp = str(row["experiment"])
        subexp = str(row["subexperiment"])
        try:
            trial = int(row["trial_number"])
        except Exception:
            # Keep message actionable; this should never happen if DB is well-formed.
            raise ValueError(f"Invalid trial_number in recording DB row: {row.get('trial_number')!r}")

        cam_map = parse_camera_mappings(row["camera_mappings"])
        key = (exp, subexp, trial)
        label_map[key] = cam_map

    logging.info(
        f"Loaded recording database: {csv_path} ({len(label_map)} sessions indexed)"
    )
    return RecordingDBIndex(csv_path=csv_path, label_map=label_map)
