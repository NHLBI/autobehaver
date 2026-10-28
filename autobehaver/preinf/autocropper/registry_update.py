"""Update the project registry from Autocropper per-movie logs.

Write once in the parent process to avoid worker contention.
"""

from __future__ import annotations

import json
import logging
from autobehaver.portable_paths import original_name
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import (
    extract_label_prefix,
    parse_full_movie_stem,
    parse_genotype_sex_from_prefix,
    relpath_str,
    upsert_registry_rows,
)


def _is_failure_string(s: Any) -> bool:
    if not isinstance(s, str):
        return True
    return s.strip().lower().startswith("failed")


def _parse_chamber_number(chamber_id: str) -> int | None:
    """Return the chamber number from a chamber identifier, if present."""
    m = re.search(r"chamber(\d+)", str(chamber_id), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def update_registry_from_autocropper_logs(
    *,
    registry_csv: Path,
    project_root: Path,
    input_movies_dir: Path,
    cropped_movies_dir: Path,
    run_dir: Path,
    recording_database_csv: Path | None,
    posepred_models: list[str] | None = None,
    movie_names: Iterable[str] | None = None,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert cropped outputs and mark explicitly excluded chambers inactive."""

    movie_logs_dir = Path(run_dir) / "movie_logs"
    if not movie_logs_dir.exists():
        logging.info(f"Registry update skipped (no movie_logs dir): {movie_logs_dir}")
        return

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    selected_names = {str(name).strip() for name in movie_names or [] if str(name).strip()}

    for log_path in sorted(movie_logs_dir.glob("*_log.json")):
        try:
            with log_path.open("r") as f:
                data = json.load(f) or {}
        except Exception:
            logging.warning(f"Skipping unreadable autocropper log: {log_path}")
            continue

        movie_name = str(data.get("movie_name") or "").strip()
        if not movie_name:
            continue
        if selected_names and movie_name not in selected_names:
            continue

        full_movie_path = Path(data['source_path']).resolve() if data.get('source_path') else (Path(input_movies_dir) / movie_name).resolve()
        full_movie_stem = Path(original_name(full_movie_path)).stem

        identity: dict[str, Any] = {}
        try:
            identity = parse_full_movie_stem(full_movie_stem)
        except Exception:
            identity = {}

        for k_log, k_out in (
            ("experiment", "experiment"),
            ("subexperiment", "subexperiment"),
            ("trial_number", "trial_number"),
            ("rpi_number", "rpi_number"),
        ):
            if k_log in data and data.get(k_log) not in (None, ""):
                identity[k_out] = data.get(k_log)

        cropped_files = data.get("cropped_files") or {}
        if not isinstance(cropped_files, dict):
            cropped_files = {}

        for chamber_id in data.get("excluded_chambers") or []:
            rows.append({
                "recording_id": f"{full_movie_stem}_{chamber_id}",
                "active": False,
                "updated_at": now,
            })

        for chamber_id, file_info in sorted(cropped_files.items()):
            if chamber_id in (data.get('missing_required_chambers') or []) or chamber_id in (data.get('crop_failures') or {}):
                continue
            if _is_failure_string(file_info):
                continue

            cropped_path = Path(cropped_movies_dir) / str(file_info)
            if not cropped_path.exists():
                p2 = Path(str(file_info))
                if p2.exists():
                    cropped_path = p2
                else:
                    continue

            chamber_number = _parse_chamber_number(str(chamber_id))
            label_prefix = (data.get("chamber_labels") or {}).get(str(chamber_id)) or extract_label_prefix(
                cropped_stem=Path(original_name(cropped_path)).stem,
                full_movie_stem=full_movie_stem,
                chamber_id=str(chamber_id),
            )
            genotype, sex = parse_genotype_sex_from_prefix(label_prefix)

            recording_id = f"{full_movie_stem}_{chamber_id}"
            animal_id = None
            if all(k in identity for k in ("experiment", "subexperiment", "trial_number", "rpi_number")):
                animal_id = (
                    f"{identity['experiment']}_{identity['subexperiment']}_"
                    f"trial{identity['trial_number']}_rpi-{identity['rpi_number']}_{chamber_id}"
                )

            row: dict[str, Any] = {
                "recording_id": recording_id,
                "animal_id": animal_id,
                "experiment": identity.get("experiment"),
                "subexperiment": identity.get("subexperiment"),
                "trial_number": identity.get("trial_number"),
                "rpi_number": identity.get("rpi_number"),
                "movie_number": identity.get("movie_number"),
                "timestamp": identity.get("timestamp"),
                "chamber_number": chamber_number,
                "genotype": genotype,
                "sex": sex,
                "label_prefix": label_prefix if label_prefix else None,
                "full_movie_relpath": relpath_str(full_movie_path, project_root),
                "cropped_movie_relpath": relpath_str(cropped_path, project_root),
                "recording_database_csv_relpath": (
                    relpath_str(recording_database_csv, project_root)
                    if recording_database_csv is not None
                    else None
                ),
                "active": True,
                "updated_at": now,
            }

            if posepred_models:
                row["posepred_models"] = ";".join([str(p) for p in posepred_models])

            rows.append(row)

    if not rows:
        logging.info("Registry update: no cropped outputs found in autocropper logs")
        return

    upsert_registry_rows(
        csv_path=Path(registry_csv),
        rows=rows,
        lock_timeout_sec=lock_timeout_sec,
        lock_stale_sec=lock_stale_sec,
    )
    logging.info(f"Registry updated: {registry_csv} (+{len(rows)} rows upserted)")
