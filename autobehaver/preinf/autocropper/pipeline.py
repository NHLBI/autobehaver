"""Autocropper orchestration.

This module coordinates:
  1) movie discovery and validation
  2) circle detection and chamber matching
  3) optional CNN chamber validation
  4) cropping and logging
  5) retry logic for robustness

Autocropper reads ``exclusions.csv`` and can process subsets via
``--file_filter`` or ``--movie_list``.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import AutocropperConfig
from .cropping import crop_chambers_in_parallel
from .detection import detect_and_match_chambers, draw_circles
from .thread_limits import configure_native_threading
from .io_utils import (
    delete_incomplete_files,
    ensure_user_permissions,
    find_processing_log,
    get_movie_files,
    is_video_file_valid,
    log_movie_results,
    save_chamber_coordinates_image,
    setup_logging,
    summarize_results,
)
from .validator import ChamberValidator

from .recording_db import (
    RecordingDBIndex,
    find_latest_recording_database_csv,
    load_recording_database_index,
)

from ..moviechecker.core import derive_pair_key
from ..moviechecker.exclusions import ExclusionsStore


# -----------------------------------------------------------------------------
# Runtime state (set by core.run_autocropper)
# -----------------------------------------------------------------------------

cfg: AutocropperConfig | None = None
args: Any | None = None

# Full movies folder
input_dir: Path | None = None

# Stage run directory (logs, validation_results, chamber_images, movie_logs)
run_dir: Path | None = None

# Where cropped chamber movies are written
crop_output_dir: Path | None = None

# moviechecker exclusions
exclusions_csv_path: Path | None = None
exclusions_store: ExclusionsStore | None = None

# recording database labeling
recording_db_path: Path | None = None
recording_db_index: RecordingDBIndex | None = None

chamber_validator: ChamberValidator | None = None


def set_runtime(*, cfg: AutocropperConfig, args: Any) -> None:
    """Set module-level runtime configuration for multiprocessing workers."""

    globals_dict = globals()
    globals_dict["cfg"] = cfg
    globals_dict["args"] = args

    globals_dict["input_dir"] = Path(cfg.input_dir)
    globals_dict["run_dir"] = Path(cfg.resolved_run_dir())
    globals_dict["crop_output_dir"] = Path(cfg.resolved_crop_output_dir())

    globals_dict["exclusions_csv_path"] = Path(cfg.resolved_exclusions_csv())
    globals_dict["exclusions_store"] = None

    globals_dict["recording_db_path"] = None
    globals_dict["recording_db_index"] = None


def _exclusions_optional() -> bool:
    """Return whether exclusions.csv is optional for this run."""
    assert args is not None
    return not bool(getattr(args, "require_exclusions", True))


def _file_filter_value() -> str | None:
    """Return the configured file-filter value."""
    assert args is not None

    value = getattr(args, "file_filter", None)
    if value in (None, ""):
        return None
    value = str(value).strip()
    return value or None


def _trial_number_value() -> int | None:
    """Return the exact trial-number selector, if configured."""
    assert args is not None
    value = getattr(args, "trial_number", None)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _trial_token_pattern(trial_number: int) -> re.Pattern[str]:
    return re.compile(rf"(?:^|_)trial{int(trial_number)}(?:_|$)", flags=re.IGNORECASE)


def _trial_number_from_name(name: str) -> int | None:
    m = re.search(r"(?:^|_)trial(\d+)(?:_|$)", str(name), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _warn_if_trial_filter_is_ambiguous(selected: list[Path], filter_value: str) -> None:
    """Warn when a substring trial filter selects other trials, e.g. trial1 -> trial14."""
    m = re.fullmatch(r"trial(\d+)", str(filter_value).strip(), flags=re.IGNORECASE)
    if not m:
        return
    requested = int(m.group(1))
    selected_trials = sorted({n for n in (_trial_number_from_name(p.name) for p in selected) if n is not None})
    if any(n != requested for n in selected_trials):
        logging.warning(
            "file_filter=%r is a substring match and selected trial numbers %s. "
            "Use --trial_number %s or --file_filter _trial%s_ for exact trial selection.",
            filter_value,
            selected_trials,
            requested,
            requested,
        )


def _validation_enabled() -> bool:
    """Return whether chamber validation should run."""
    assert args is not None
    return (not bool(getattr(args, "skip_validation", False))) and bool(
        getattr(args, "validate_chambers", True)
    )


def _heartbeat_interval_seconds() -> int:
    assert args is not None
    return max(1, int(getattr(args, "heartbeat_interval", 10)))


def _is_failure_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith("failed")


def _assess_required_crops(
    movie: Path,
    output_dir: Path,
    *,
    required_chambers: list[str] | set[str],
    raw_cropped_files: dict[str, str],
) -> dict[str, Any]:
    """Resolve final crop status for the chambers required in this attempt."""
    final_files: dict[str, str] = {}
    missing_required: list[str] = []
    crop_failures: dict[str, str] = {}

    for chamber_id in sorted({str(cid) for cid in required_chambers}):
        raw_value = raw_cropped_files.get(chamber_id)
        if _is_failure_marker(raw_value):
            crop_failures[chamber_id] = str(raw_value)

        res = _resolve_existing_crop(movie, output_dir, chamber_id)
        if res.get("status") == "missing":
            missing_required.append(chamber_id)
            continue

        p = res.get("path")
        if p is not None:
            final_files[chamber_id] = Path(p).name
            continue

        matches = list(res.get("matches", []))
        if matches:
            final_files[chamber_id] = str(matches[0])

    for chamber_id, file_info in raw_cropped_files.items():
        if chamber_id in final_files:
            continue
        if _is_failure_marker(file_info):
            continue
        final_files[chamber_id] = str(file_info)

    return {
        "cropped_files": final_files,
        "missing_required_chambers": missing_required,
        "crop_failures": crop_failures,
    }


# -----------------------------------------------------------------------------
# Exclusions helpers
# -----------------------------------------------------------------------------


def _load_exclusions_if_available() -> None:
    """Load exclusions.csv into an in-memory store.

    If the file is missing and exclusions are required, the run exits.
    """
    global exclusions_store

    assert args is not None
    if input_dir is None:
        raise RuntimeError("input_dir is not set (did you call run_autocropper()?)")
    if exclusions_csv_path is None:
        raise RuntimeError("exclusions_csv_path is not set")

    # Resolve default exclusions path relative to input_dir, if user passed a relative path
    ex_path = Path(exclusions_csv_path)
    if not ex_path.is_absolute():
        ex_path = (Path(input_dir) / ex_path).resolve()

    if ex_path.exists():
        store = ExclusionsStore(csv_path=str(ex_path))
        store.load()
        exclusions_store = store
        logging.info(f"Loaded exclusions file: {ex_path}")
        return

    # Missing
    if _exclusions_optional():
        exclusions_store = None
        logging.warning(
            f"No exclusions.csv found at {ex_path}. Proceeding because --no_require_exclusions is enabled."
        )
        return

    raise SystemExit(
        f"No exclusions.csv found at {ex_path}.\n"
        f"Please run moviechecker first (it writes exclusions.csv), or rerun autocropper with --no_require_exclusions."
    )


def _excluded_set_for_movie(movie: Path) -> set[str]:
    """Return a set of chamber IDs (e.g., {'chamber2','chamber6'}) to exclude."""
    if exclusions_store is None:
        return set()

    key = derive_pair_key(str(movie))
    rec = exclusions_store.get(key)
    if rec is None:
        # Missing record: treat as no exclusions for safety.
        return set()

    excluded: set[str] = set()
    for i, flag in enumerate(rec.exclude_ch):
        if int(flag) == 1:
            excluded.add(f"chamber{i+1}")
    return excluded


def _require_exclusions_coverage(movie_files: list[Path]) -> None:
    """Ensure exclusions.csv contains entries for the movies we are about to process."""
    assert args is not None

    if exclusions_store is None:
        return

    missing_keys: list[str] = []
    missing_movies: list[str] = []

    for m in movie_files:
        key = derive_pair_key(str(m))
        if not exclusions_store.has(key):
            missing_keys.append(key)
            missing_movies.append(m.name)

    if missing_movies:
        # Make the failure message actionable.
        preview = "\n".join(f"  - {n}" for n in missing_movies[:30])
        more = "" if len(missing_movies) <= 30 else f"\n  ... and {len(missing_movies)-30} more"
        raise SystemExit(
            "exclusions.csv is present but does not contain entries for all selected movies.\n"
            "Run moviechecker again (it will skip already-reviewed movies by default) and review the missing items.\n\n"
            "Missing movies:\n"
            f"{preview}{more}"
        )


# -----------------------------------------------------------------------------
# Recording database labeling helpers
# -----------------------------------------------------------------------------


def _load_recording_db_if_needed() -> None:
    """Load the recording database into memory (module global).

    Behavior:
      - If args.skip_labeling is True -> do nothing (labeling disabled).
      - Else:
          * Resolve CSV path:
              - if args.recording_database_csv provided -> use it
              - else: look in args.recording_database_dir (default: input_dir) and pick latest
          * Load into RecordingDBIndex for fast lookups.
      - Missing DB when labeling is enabled -> raise SystemExit with an actionable message.
    """

    global recording_db_path, recording_db_index

    assert args is not None
    if input_dir is None:
        raise RuntimeError("input_dir is not set")

    if bool(getattr(args, "skip_labeling", False)):
        recording_db_path = None
        recording_db_index = None
        logging.info("Recording DB labeling: disabled (--skip_labeling)")
        return

    # Explicit CSV overrides directory scanning.
    explicit_csv = getattr(args, "recording_database_csv", None)
    if explicit_csv:
        db_csv = Path(str(explicit_csv)).expanduser().resolve()
    else:
        db_dir_raw = getattr(args, "recording_database_dir", None)
        db_dir = Path(str(db_dir_raw)).expanduser().resolve() if db_dir_raw else Path(input_dir)
        try:
            db_csv = find_latest_recording_database_csv(db_dir)
        except FileNotFoundError:
            raise SystemExit(
                f"No recording_database_*.csv found in: {db_dir}\n"
                "Labeling is enabled by default.\n"
                "Either copy a recording_database_*.csv into the full-movies folder, set --recording_database_dir, "
                "or rerun with --skip_labeling."
            )

    try:
        recording_db_index = load_recording_database_index(db_csv)
        recording_db_path = db_csv
    except Exception as e:
        raise SystemExit(
            f"Failed to load recording database CSV: {db_csv}\n"
            f"Error: {e}\n"
            "If you want to run without DB labeling, rerun with --skip_labeling."
        )


def _parse_full_movie_identity(movie: Path) -> tuple[str, str, int, int]:
    """Parse (experiment, subexperiment, trial_number, rpi_number) from a full-movie filename.

    Expected filename convention:
        <experiment>_<subexperiment>_trial<trial>_rpi-<n>_movie<m>_...
    """

    stem = Path(movie).stem

    # Fast path: strict tokens
    parts = stem.split("_")
    if len(parts) >= 5 and parts[2].startswith("trial") and parts[3].startswith("rpi") and parts[4].startswith(
        "movie"
    ):
        experiment = parts[0]
        subexperiment = parts[1]
        try:
            trial = int(re.sub(r"^trial", "", parts[2], flags=re.IGNORECASE))
        except Exception:
            trial = None  # type: ignore
        m_rpi = re.match(r"^rpi-(\d+)$", parts[3], flags=re.IGNORECASE)
        rpi = int(m_rpi.group(1)) if m_rpi else None  # type: ignore
        if trial is not None and rpi is not None:
            return str(experiment), str(subexperiment), int(trial), int(rpi)

    raise ValueError(
        f"Could not parse movie identity from filename: {movie.name}\n"
        "Expected something like: <experiment>_<subexperiment>_trial1_rpi-6_movie0_<timestamp>.mp4"
    )


def _expected_crop_output_path(movie: Path, output_dir: Path, chamber_id: str) -> Path:
    """Compute the expected cropped output path for a chamber.

    If recording DB labeling is enabled, the filename is prefixed with the DB label.
    When labeling is disabled, the filename uses the movie stem and chamber ID.
    """

    base_name = Path(movie).stem

    if recording_db_index is None or bool(getattr(args, "skip_labeling", False)):
        return Path(output_dir) / f"{base_name}_{chamber_id}.mp4"

    exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
    chamber_num = int(str(chamber_id).replace("chamber", ""))
    label_prefix = recording_db_index.get_label_prefix(
        experiment=exp,
        subexperiment=subexp,
        trial_number=trial,
        rpi_number=rpi,
        chamber_number=chamber_num,
    )
    return Path(output_dir) / f"{label_prefix}{base_name}_{chamber_id}.mp4"


def _resolve_existing_crop(
    movie: Path,
    output_dir: Path,
    chamber_id: str,
) -> dict[str, Any]:
    """Resolve whether a crop already exists for a given movie/chamber.

    Returns a dict with:
      - status: one of {"expected", "exists_other", "missing"}
      - expected_name: expected output filename
      - path: Path to an existing crop (if present)
      - matches: list[str] of matching filenames (when ambiguous)

    Important
    ---------
    We treat *any* existing crop matching '*<base>_<chamber>.mp4' as "present" to avoid
    accidental re-cropping when label prefixes change.
    """

    expected_path = _expected_crop_output_path(movie, output_dir, chamber_id)
    base_name = Path(movie).stem

    if expected_path.exists():
        return {
            "status": "expected",
            "expected_name": expected_path.name,
            "expected_path": expected_path,
            "path": expected_path,
            "matches": [expected_path.name],
        }

    matches = sorted(Path(output_dir).glob(f"*{base_name}_{chamber_id}.mp4"))
    if not matches:
        return {
            "status": "missing",
            "expected_name": expected_path.name,
            "expected_path": expected_path,
            "path": None,
            "matches": [],
        }


    # Some other prefixed file exists; do NOT recrop.
    return {
        "status": "exists_other",
        "expected_name": expected_path.name,
        "expected_path": expected_path,
        "path": matches[0],
        "matches": [m.name for m in matches],
    }


def _require_recording_db_coverage(movie_files: list[Path]) -> None:
    """Ensure the recording DB can provide labels for selected movies.

    We require labels for *non-excluded* chambers only (exclusions already loaded).
    """

    assert args is not None

    if bool(getattr(args, "skip_labeling", False)):
        return
    if recording_db_index is None:
        return

    expected_total = int(args.chamber_columns) * int(args.chamber_rows)
    missing_rows: list[str] = []
    missing_keys: list[str] = []

    for movie in movie_files:
        exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
        session_key = (exp, subexp, trial)
        session = recording_db_index.label_map.get(session_key)
        if session is None:
            missing_rows.append(f"{movie.name} -> (experiment={exp}, subexperiment={subexp}, trial={trial})")
            continue

        excluded = _excluded_set_for_movie(movie)
        for i in range(1, expected_total + 1):
            chamber_id = f"chamber{i}"
            if chamber_id in excluded:
                continue
            rc_key = f"rpi-{rpi}_chamber{i}"
            if rc_key not in session:
                missing_keys.append(f"{movie.name} -> missing {rc_key} in session {session_key}")

    if missing_rows or missing_keys:
        preview = "\n".join(f"  - {s}" for s in (missing_rows + missing_keys)[:30])
        more_n = max(0, (len(missing_rows) + len(missing_keys)) - 30)
        more = "" if more_n == 0 else f"\n  ... and {more_n} more"
        raise SystemExit(
            "Recording DB labeling is enabled, but the recording database does not fully cover the selected movies.\n"
            "Fix the database entry (experiment/subexperiment/trial and camera_mappings), or rerun with --skip_labeling.\n\n"
            "Missing coverage:\n"
            f"{preview}{more}"
        )


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _init_chamber_validator() -> None:
    """Initialize chamber validation model if enabled."""
    global chamber_validator

    assert args is not None
    if run_dir is None:
        raise RuntimeError("run_dir is not set (did you call run_autocropper()?)")

    chamber_validator = None

    validation_requested = _validation_enabled()

    if validation_requested:
        model_path = Path(args.model_path)
        norm_params_path = Path(args.norm_params_path)

        if model_path.exists() and norm_params_path.exists():
            logging.info("\n=== Chamber validation enabled ===")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Normalization params path: {norm_params_path}")
            chamber_validator = ChamberValidator(
                model_path,
                norm_params_path,
                base_output_dir=run_dir,
            )
        else:
            # Model files missing -> force skip_validation so verification logic is consistent.
            logging.warning("\n" + "=" * 70)
            logging.warning("!!! CHAMBER VALIDATION SKIPPED !!!")
            logging.warning("Could not find the model and/or normalization files at the specified paths.")
            logging.warning(f"Searched for model: {model_path}")
            logging.warning(f"Searched for params: {norm_params_path}")
            logging.warning("The script will proceed by cropping ALL detected chambers (except excluded ones).")
            logging.warning("=" * 70)
            args.skip_validation = True
            chamber_validator = None
    else:
        logging.info("\n=== Chamber validation disabled by user flag ===")
        chamber_validator = None


def _prepare_validation_mode() -> None:
    """Resolve validation mode in the parent process without loading torch models."""
    assert args is not None

    if not _validation_enabled():
        logging.info("\n=== Chamber validation disabled by user flag ===")
        return

    model_path = Path(args.model_path)
    norm_params_path = Path(args.norm_params_path)
    if model_path.exists() and norm_params_path.exists():
        logging.info("\n=== Chamber validation enabled ===")
        logging.info("Model will be initialized lazily inside worker processes.")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Normalization params path: {norm_params_path}")
        return

    logging.warning("\n" + "=" * 70)
    logging.warning("!!! CHAMBER VALIDATION SKIPPED !!!")
    logging.warning("Could not find the model and/or normalization files at the specified paths.")
    logging.warning(f"Searched for model: {model_path}")
    logging.warning(f"Searched for params: {norm_params_path}")
    logging.warning("The script will proceed by cropping ALL detected chambers (except excluded ones).")
    logging.warning("=" * 70)
    args.skip_validation = True


def _worker_init(worker_cfg: AutocropperConfig, worker_args: Any) -> None:
    """Initialize per-process runtime state for spawn-safe multiprocessing."""
    random_seed = 42
    configure_native_threading(getattr(worker_args, "native_threads", 1), overwrite=False)
    set_runtime(cfg=worker_cfg, args=worker_args)
    if run_dir is None or crop_output_dir is None:
        raise RuntimeError("run_dir/crop_output_dir is not set in worker initializer")

    run_dir.mkdir(parents=True, exist_ok=True)
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, log_level=str(getattr(worker_args, "log_level", "INFO")))
    _load_exclusions_if_available()
    _load_recording_db_if_needed()
    _init_chamber_validator()

    # Keep frame sampling deterministic across workers/process launches.
    import random

    random.seed(random_seed)


def get_occupied_chambers(movie: Path) -> list[str] | None:
    """Read occupied chamber IDs from a movie's validation JSON file."""
    if run_dir is None:
        raise RuntimeError("run_dir is not set (did you call run_autocropper()?)")

    movie_name = Path(movie).stem
    validation_file = run_dir / "validation_results" / f"{movie_name}_validation.json"
    if not validation_file.exists():
        logging.info(f"No validation results found for {movie}, needs processing")
        return None

    try:
        with validation_file.open("r") as f:
            validation_results = json.load(f)
            occupied = [
                chamber_id
                for chamber_id, result in validation_results.items()
                if isinstance(result, dict) and result.get("is_occupied", False)
            ]
            logging.info(
                f"Found {len(occupied)} occupied chambers in validation results for {movie_name}"
            )
            return occupied
    except json.JSONDecodeError:
        logging.error(f"Error reading validation results for {movie}, file may be corrupted")
        return None
    except Exception as e:
        logging.error(f"Error processing validation results for {movie}: {str(e)}")
        return None


def all_cropped_movies_exist(
    movie: Path,
    output_dir: Path,
    *,
    excluded_chambers: set[str] | None = None,
) -> bool:
    """Check if all required chambers for a movie have already been cropped.

    - If validation is skipped: require all non-excluded chambers.
    - If validation is enabled: require all *occupied* non-excluded chambers.
    """
    assert args is not None

    excluded = excluded_chambers or set()

    expected_total = int(args.chamber_columns) * int(args.chamber_rows)
    all_expected = [f"chamber{i}" for i in range(1, expected_total + 1)]
    expected = [cid for cid in all_expected if cid not in excluded]

    # If everything is excluded, there's nothing to crop.
    if not expected:
        return True

    # If validation is skipped, check if all expected chambers have been cropped
    if getattr(args, "skip_validation", False):
        for chamber_id in expected:
            res = _resolve_existing_crop(movie, output_dir, chamber_id)
            if res["status"] == "missing":
                logging.debug(
                    f"Missing cropped video for {chamber_id} (validation skipped); expected {res['expected_name']}"
                )
                return False
            if res["status"] == "exists_other":
                logging.warning(
                    f"Crop exists for {movie.name} {chamber_id} but filename does not match expected label. "
                    f"Expected: {res['expected_name']} | Found: {res.get('matches', [])}"
                )
        return True

    occupied_chambers = get_occupied_chambers(movie)

    # If validation hasn't been done yet, return False to trigger processing
    if occupied_chambers is None:
        return False

    # Only require occupied chambers among the non-excluded set
    occupied_required = [cid for cid in occupied_chambers if cid in expected]

    # If validation is done and no required chambers are occupied, we're done
    if not occupied_required:
        logging.info(
            f"No occupied (non-excluded) chambers found for {movie}, considering it complete."
        )
        return True

    # Check if all required occupied chambers have been cropped
    for chamber_id in occupied_required:
        res = _resolve_existing_crop(movie, output_dir, chamber_id)
        if res["status"] == "missing":
            logging.debug(
                f"Missing cropped video for occupied {chamber_id}; expected {res['expected_name']}"
            )
            return False
        if res["status"] == "exists_other":
            logging.warning(
                f"Crop exists for {movie.name} {chamber_id} but filename does not match expected label. "
                f"Expected: {res['expected_name']} | Found: {res.get('matches', [])}"
            )
    return True


def verify_all_cropped_movies(movie_files: list[Path], output_dir: Path) -> list[Path]:
    """Verify that all required chambers have been cropped for all movies."""
    incomplete_movies: list[Path] = []
    for movie in movie_files:
        excluded = _excluded_set_for_movie(movie)
        if not all_cropped_movies_exist(movie, output_dir, excluded_chambers=excluded):
            incomplete_movies.append(movie)
    return incomplete_movies


def run_chamber_validation(
    movie_path: Path,
    all_coords: dict[str, tuple[int, int]],
    validator: ChamberValidator | None,
    crop_dim: int,
):
    if validator is None:
        logging.info(f"Skipping validation - will crop all {len(all_coords)} detected chambers")
        return all_coords, {}

    logging.info(f"Validating {len(all_coords)} chambers...")

    occupied_coords: dict[str, tuple[int, int]] = {}
    validation_results: dict[str, Any] = {}

    for chamber_id, (x, y) in all_coords.items():
        result = validator.validate_chamber(str(movie_path), chamber_id, x, y, crop_dim)
        validation_results[chamber_id] = result

        if result["is_occupied"]:
            occupied_coords[chamber_id] = (int(x), int(y))
            logging.info(f"✓ {chamber_id}: OCCUPIED (confidence: {result['confidence']:.3f})")
        else:
            logging.info(f"✗ {chamber_id}: EMPTY (confidence: {result['confidence']:.3f})")

    logging.info(f"Validation complete: {len(occupied_coords)}/{len(all_coords)} chambers occupied")
    return occupied_coords, validation_results


# -----------------------------------------------------------------------------
# Movie selection
# -----------------------------------------------------------------------------


def _load_movie_list(list_path: Path, *, base_dir: Path) -> list[Path]:
    """Load a user/provided list of movies (one per line)."""
    movies: list[Path] = []
    if not list_path.exists():
        logging.error(f"movie_list file not found: {list_path}")
        return movies

    for raw in list_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        movies.append(p)

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in movies:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _select_movie_files(all_movie_files: list[Path]) -> list[Path]:
    assert args is not None
    if input_dir is None:
        raise RuntimeError("input_dir is not set")

    # Explicit list takes precedence
    movie_list_path = getattr(args, "movie_list", None)
    if movie_list_path:
        selected = _load_movie_list(Path(movie_list_path), base_dir=input_dir)
        if selected:
            return selected
        # If list file is empty/missing, fall back to no selection (but warn)
        logging.warning("movie_list was provided but yielded no valid movie paths; falling back to scanning folder.")

    trial_number = _trial_number_value()
    if trial_number is not None:
        pattern = _trial_token_pattern(trial_number)
        return [m for m in all_movie_files if pattern.search(m.stem)]

    filt = _file_filter_value()
    if filt:
        needle = str(filt).lower()
        selected = [m for m in all_movie_files if needle in m.name.lower()]
        _warn_if_trial_filter_is_ambiguous(selected, str(filt))
        return selected

    return all_movie_files


# -----------------------------------------------------------------------------
# Multiprocessing entrypoint
# -----------------------------------------------------------------------------


def process_single_movie(movie: Path) -> None:
    assert args is not None
    if input_dir is None:
        raise RuntimeError("input_dir is not set (did you call run_autocropper()?)")
    if run_dir is None or crop_output_dir is None:
        raise RuntimeError("run_dir/crop_output_dir is not set")

    movie_name = Path(movie).name
    logging.info(f"Processing {movie_name}...")

    output_dir = crop_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_user_permissions(output_dir)

    movie_data: dict[str, Any] = {"movie_name": movie_name}

    excluded = _excluded_set_for_movie(movie)
    if excluded:
        movie_data["excluded_chambers"] = sorted(excluded)

    # Record recording DB provenance.
    if recording_db_path is not None:
        movie_data["recording_database_csv"] = str(recording_db_path)

    try:
        # Parse identity info (used for DB labeling). Only fatal when labeling is enabled.
        try:
            exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
            movie_data.update(
                {
                    "experiment": exp,
                    "subexperiment": subexp,
                    "trial_number": int(trial),
                    "rpi_number": int(rpi),
                }
            )
        except Exception as e:
            if not bool(getattr(args, "skip_labeling", False)):
                raise
            movie_data["identity_parse_warning"] = str(e)

        # Fast skip path: if all required crops already exist, do not redo detection/validation.
        if all_cropped_movies_exist(movie, output_dir, excluded_chambers=excluded):
            movie_data["skipped"] = True
            movie_data["status"] = "skipped"
            movie_data["skip_reason"] = "already processed"
            # Still record which cropped files exist so downstream tooling (registry, QC) can use the log.
            existing: dict[str, str] = {}
            try:
                expected_total = int(args.chamber_columns) * int(args.chamber_rows)
                for i in range(1, expected_total + 1):
                    chamber_id = f"chamber{i}"
                    if chamber_id in excluded:
                        continue
                    res = _resolve_existing_crop(movie, output_dir, chamber_id)
                    p = res.get("path")
                    if p is not None:
                        existing[chamber_id] = Path(p).name
                movie_data["cropped_files"] = existing
            except Exception:
                pass
            logging.info(f"✓ {movie_name}: Already processed, skipping")
            return

        # Step 1: Find all potential chambers
        all_coords, viz_frame, circles, crop_dim = detect_and_match_chambers(movie, args)
        movie_data["chamber_coordinates"] = {
            cid: {"x": int(x), "y": int(y)} for cid, (x, y) in all_coords.items()
        }

        # Apply exclusions *before* validation/cropping
        allowed_coords = {cid: xy for cid, xy in all_coords.items() if cid not in excluded}

        if not allowed_coords:
            logging.info(f"✓ {movie_name}: All chambers excluded, skipping cropping")
            movie_data["status"] = "skipped"
            movie_data["skip_reason"] = "all chambers excluded"
            movie_data["cropped_files"] = {}
            return

        # Step 2: Filter for occupied chambers using the CNN validator
        occupied_coords, validation_results = run_chamber_validation(
            movie, allowed_coords, chamber_validator, crop_dim
        )
        movie_data["validation_results"] = validation_results

        # Step 3: Create and save visualization of detected (and occupied) chambers
        draw_circles(
            viz_frame,
            circles,
            args.down_factor,
            occupied_coords,
            approx_radius=int(args.approx_radius),
        )
        save_chamber_coordinates_image(run_dir, movie, viz_frame)

        # Step 4: Crop (only what is missing) while respecting DB-based naming
        crop_status: dict[str, Any] = {}
        existing_or_final: dict[str, str] = {}
        coords_to_crop: dict[str, tuple[int, int]] = {}
        output_paths_to_crop: dict[str, Path] = {}

        for chamber_id, (x, y) in occupied_coords.items():
            res = _resolve_existing_crop(
                movie, output_dir, chamber_id
            )

            # Store only JSON-serializable fields
            crop_status[chamber_id] = {
                "status": str(res.get("status")),
                "expected": str(res.get("expected_name")),
                "found": list(res.get("matches", [])),
            }

            if res.get("status") == "missing":
                coords_to_crop[chamber_id] = (int(x), int(y))
                expected_path = Path(res["expected_path"])  # internal use only
                output_paths_to_crop[chamber_id] = expected_path
            else:
                p = res.get("path")
                if p is not None:
                    existing_or_final[chamber_id] = Path(p).name

        # Crop only missing chambers
        cropped_files: dict[str, str] = dict(existing_or_final)
        if coords_to_crop:
            source_size = None
            try:
                source_size = (int(viz_frame.shape[1]), int(viz_frame.shape[0]))
            except Exception:
                source_size = None
            newly_cropped = crop_chambers_in_parallel(
                movie,
                coords_to_crop,
                output_paths_to_crop,
                crop_dim,
                args,
                source_size=source_size,
            )
            cropped_files.update(newly_cropped)

        movie_data["crop_status"] = crop_status

        assessment = _assess_required_crops(
            movie,
            output_dir,
            required_chambers=list(occupied_coords.keys()),
            raw_cropped_files=cropped_files,
        )
        movie_data["cropped_files"] = assessment["cropped_files"]
        if assessment["crop_failures"]:
            movie_data["crop_failures"] = assessment["crop_failures"]
        if assessment["missing_required_chambers"]:
            movie_data["missing_required_chambers"] = assessment["missing_required_chambers"]

        missing_required = assessment["missing_required_chambers"]
        failed_attempts = assessment["crop_failures"]
        n_present = len(assessment["cropped_files"])
        n_required = len(occupied_coords)

        if missing_required or failed_attempts:
            movie_data["status"] = "incomplete"
            logging.warning(
                f"△ {movie_name}: Finished attempt with incomplete required crops "
                f"({n_present}/{n_required} present; {len(missing_required)} still missing, "
                f"{len(failed_attempts)} crop failures this attempt)"
            )
        else:
            movie_data["status"] = "completed"
            logging.info(
                f"✓ {movie_name}: All required crops present "
                f"({len(coords_to_crop)} new chambers cropped, {len(existing_or_final)} already present)"
            )

    except Exception as e:
        logging.error(f"✗ {movie_name}: Failed - {str(e)}")
        movie_data["status"] = "failed"
        movie_data["error"] = str(e)

    finally:
        # Step 5: Always save the log file for this movie
        log_movie_results(movie, run_dir, movie_data)

    return {"movie_name": movie_name, "error": movie_data.get("error")}


def process_movies_in_parallel(movie_files: list[Path]) -> None:
    """Process movies in parallel with spawn-safe worker initialization."""
    assert args is not None
    assert cfg is not None

    if not movie_files:
        return

    num_workers = max(1, int(getattr(args, "num_workers", 2)))
    heartbeat = _heartbeat_interval_seconds()
    ctx = mp.get_context("spawn")

    logging.info(
        f"Starting worker pool with {num_workers} process(es) using start method 'spawn'."
    )

    completed = 0
    total = len(movie_files)
    last_progress_log = 0.0

    with cf.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cfg, args),
    ) as executor:
        future_to_movie = {executor.submit(process_single_movie, movie): movie for movie in movie_files}

        while future_to_movie:
            done, _pending = cf.wait(
                tuple(future_to_movie.keys()),
                timeout=heartbeat,
                return_when=cf.FIRST_COMPLETED,
            )

            now = time.monotonic()
            if not done:
                if (now - last_progress_log) >= heartbeat:
                    logging.info(
                        f"Autocropper progress: {completed}/{total} movie tasks finished in this attempt, {len(future_to_movie)} still running."
                    )
                    last_progress_log = now
                continue

            for future in done:
                movie = future_to_movie.pop(future)
                try:
                    future.result()
                except Exception:
                    logging.exception(f"Worker failed while processing {movie.name}")
                    for pending in future_to_movie:
                        pending.cancel()
                    raise
                completed += 1

            if (now - last_progress_log) >= heartbeat or completed == total:
                logging.info(
                    f"Autocropper progress: {completed}/{total} movie tasks finished in this attempt, {len(future_to_movie)} still running."
                )
                last_progress_log = now


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------


def run_pipeline() -> None:
    """Main function to orchestrate autocropping for all movies."""
    assert args is not None
    if input_dir is None or run_dir is None or crop_output_dir is None:
        raise RuntimeError("runtime state is not set (did you call run_autocropper()?)")

    # Setup logging first (log to run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, log_level=str(getattr(args, "log_level", "INFO")))

    logging.info(f"Full movies dir: {input_dir}")
    logging.info(f"Crop output dir: {crop_output_dir}")
    logging.info(f"Run dir: {run_dir}")

    configured_movie_workers = max(1, int(getattr(args, "num_workers", 2)))
    configured_crop_workers = max(1, int(getattr(args, "max_workers", 6)))
    configured_ffmpeg_threads = max(1, int(getattr(args, "ffmpeg_threads", 1)))
    configured_native_threads = max(1, int(getattr(args, "native_threads", 1)))
    effective_crop_jobs = configured_movie_workers * configured_crop_workers
    estimated_crop_threads = effective_crop_jobs * configured_ffmpeg_threads
    logging.info(
        "Configured parallelism: "
        f"{configured_movie_workers} movie worker(s) x {configured_crop_workers} chamber worker(s) per movie "
        f"= up to {effective_crop_jobs} concurrent chamber crop jobs; "
        f"ffmpeg_threads={configured_ffmpeg_threads}, native_threads={configured_native_threads}."
    )

    slurm_cpus = 0
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    except Exception:
        slurm_cpus = 0
    if slurm_cpus:
        logging.info(f"Detected SLURM_CPUS_PER_TASK={slurm_cpus} for this autocropper command.")
        # Add movie workers as a small overhead on top of ffmpeg crop threads.
        estimated_threads = estimated_crop_threads + configured_movie_workers
        if estimated_threads > slurm_cpus:
            logging.warning(
                "Configured autocropper parallelism can exceed CPUs allocated to this swarm subjob: "
                f"estimated runnable crop/movie threads ~ {estimated_threads}, SLURM_CPUS_PER_TASK={slurm_cpus}. "
                "Reduce --num_workers/--max_workers/--ffmpeg_threads or request more -t CPUs."
            )
    else:
        host_cpu_count = os.cpu_count() or 0
        if host_cpu_count and estimated_crop_threads > (host_cpu_count * 2):
            logging.warning(
                "Configured autocropper parallelism substantially exceeds detected CPU count; "
                "if you see decoder/thread errors, reduce num_workers and/or max_workers."
            )

    # Load exclusions (required by default)
    _load_exclusions_if_available()

    # Load recording database (labeling) if enabled
    _load_recording_db_if_needed()

    # Resolve validation mode after logging is ready. The validator itself is
    # initialized lazily inside worker processes to avoid parent-process torch
    # initialization before multiprocessing starts.
    _prepare_validation_mode()

    # Find the processing log (in run_dir)
    log_file = find_processing_log(run_dir)
    if not log_file:
        log_file = run_dir / "autocropper_created_processing_log.txt"
        logging.warning(
            f"No pipeline processing log found. A new log file will be created at: {log_file}"
        )

    # Create directory for individual movie JSON logs
    movie_logs_dir = run_dir / "movie_logs"
    movie_logs_dir.mkdir(parents=True, exist_ok=True)
    ensure_user_permissions(movie_logs_dir)

    # Get all movie files (full-frame) and apply selection
    all_movie_files = get_movie_files(input_dir)
    selected_movies = _select_movie_files(all_movie_files)
    logging.info(f"Movie selection yielded {len(selected_movies)} candidate movie(s).")

    # Validate movie files before processing
    movie_files: list[Path] = []
    invalid_movie_files: list[Path] = []
    missing_listed_files: list[Path] = []

    for movie in selected_movies:
        if not movie.exists():
            missing_listed_files.append(movie)
            continue
        if is_video_file_valid(movie):
            movie_files.append(movie)
        else:
            invalid_movie_files.append(movie)

    if missing_listed_files:
        logging.warning(f"{len(missing_listed_files)} movies listed/selected but not found on disk.")
        for m in missing_listed_files[:20]:
            logging.warning(f"  missing: {m}")

    # If there are no valid movies, exit early
    if not movie_files:
        logging.info("No valid movie files found to process. Exiting.")
        return

    # Ensure exclusions.csv has coverage for this subset when required.
    if not _exclusions_optional():
        _require_exclusions_coverage(movie_files)

    # Ensure recording DB covers this subset (unless --skip_labeling)
    _require_recording_db_coverage(movie_files)

    # Add initial entry to pipeline log
    with log_file.open("a") as f:
        f.write("\n2. Autocropper\n")
        f.write(f"   Started at: {datetime.now().isoformat()}\n")
        f.write(f"   Full movies dir: {input_dir}\n")
        f.write(f"   Crop output dir: {crop_output_dir}\n")

        if exclusions_csv_path is not None:
            f.write(f"   Exclusions CSV: {exclusions_csv_path}\n")
        f.write(f"   require_exclusions: {not _exclusions_optional()}\n")

        f.write(f"   skip_labeling: {bool(getattr(args, 'skip_labeling', False))}\n")
        if not bool(getattr(args, 'skip_labeling', False)):
            f.write(
                f"   Recording DB CSV: {str(recording_db_path) if recording_db_path else 'None'}\n"
            )

        selection_desc = "all movies"
        if getattr(args, "movie_list", None):
            selection_desc = f"movie_list={args.movie_list}"
        elif _trial_number_value() is not None:
            selection_desc = f"trial_number={_trial_number_value()}"
        else:
            file_filter = _file_filter_value()
            if file_filter:
                selection_desc = f"file_filter={file_filter}"
        f.write(f"   Selection: {selection_desc}\n")
        f.write(f"   Num workers: {int(getattr(args, 'num_workers', 2))}\n")
        f.write(f"   Chamber crop workers/movie: {int(getattr(args, 'max_workers', 6))}\n")
        f.write(f"   ffmpeg threads/crop: {int(getattr(args, 'ffmpeg_threads', 1))}\n")
        f.write(f"   native threads/process: {int(getattr(args, 'native_threads', 1))}\n")
        f.write(
            f"   Effective max concurrent chamber crop jobs: "
            f"{int(getattr(args, 'num_workers', 2)) * int(getattr(args, 'max_workers', 6))}\n"
        )
        f.write(f"   Heartbeat interval (s): {_heartbeat_interval_seconds()}\n")
        f.write(f"   Crop timeout (s): {int(getattr(args, 'timeout', 600))}\n")
        f.write(f"   Crop CRF: {int(getattr(args, 'crop_crf', 12))}\n")

        validation_enabled = _validation_enabled()
        f.write(f"   Chamber validation: {'enabled' if validation_enabled else 'disabled'}\n")
        if validation_enabled:
            f.write(f"   CNN Model path: {args.model_path}\n")
            f.write(f"   Normalization params: {args.norm_params_path}\n")

        f.write("\n")
        f.write("   Files to process:\n")
        for movie in movie_files:
            f.write(f"   - {Path(movie).name}\n")

        if invalid_movie_files:
            logging.warning(f"Skipping {len(invalid_movie_files)} corrupted/incomplete files")
            f.write("\n   Skipped corrupted files:\n")
            for movie in invalid_movie_files:
                f.write(f"   - {Path(movie).name}\n")

        if missing_listed_files:
            f.write("\n   Missing selected files:\n")
            for movie in missing_listed_files:
                f.write(f"   - {movie}\n")

    # Output directories
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_user_permissions(crop_output_dir)

    # Create and set permissions for other directories (in run_dir)
    validation_dir = run_dir / "validation_results"
    chamber_images_dir = run_dir / "chamber_images"

    for directory in [validation_dir, chamber_images_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        ensure_user_permissions(directory)

    # Retry loop
    movies_to_process = movie_files[:]
    max_retries = int(getattr(args, "max_retries", 10))

    for attempt in range(max_retries + 1):
        if not movies_to_process:
            break

        logging.info("-" * 50)
        if attempt > 0:
            logging.warning(
                f"RETRY ATTEMPT {attempt}/{max_retries} for {len(movies_to_process)} incomplete movies."
            )
        else:
            logging.info(f"Starting initial processing for {len(movies_to_process)} movies.")

        process_movies_in_parallel(movies_to_process)
        delete_incomplete_files(
            crop_output_dir,
            movie_stems=[Path(m).stem for m in movie_files],
            min_partial_age_seconds=int(getattr(args, "partial_cleanup_min_age_sec", 3600)),
        )

        verified_incomplete = verify_all_cropped_movies(movie_files, crop_output_dir)
        verified_complete = len(movie_files) - len(verified_incomplete)
        logging.info(
            f"Verification after attempt {attempt + 1}/{max_retries + 1}: "
            f"{verified_complete}/{len(movie_files)} movies have all required crops; "
            f"{len(verified_incomplete)} still incomplete."
        )
        movies_to_process = verified_incomplete

    logging.info("-" * 50)

    if not movies_to_process:
        logging.info("✓ All movies processed successfully.")
    else:
        logging.error(f"Processing finished, but hit max retry limit ({max_retries}).")
        logging.error("The following movies failed permanently:")
        for movie in movies_to_process:
            logging.error(f"  - {movie.name}")

    summarize_results(run_dir, movie_names=[Path(m).name for m in movie_files])

    # ------------------------------------------------------------------
    # Project registry update (single CSV at project root)
    # ------------------------------------------------------------------
    try:
        if cfg is not None and bool(getattr(cfg, "registry_enabled", False)):
            reg_csv = cfg.resolved_registry_csv()
            if reg_csv:
                from .registry_update import update_registry_from_autocropper_logs

                registry_csv_path = Path(reg_csv).expanduser().resolve()
                project_root = registry_csv_path.parent

                update_registry_from_autocropper_logs(
                    registry_csv=registry_csv_path,
                    project_root=project_root,
                    input_movies_dir=Path(cfg.input_dir).expanduser().resolve(),
                    cropped_movies_dir=Path(cfg.resolved_crop_output_dir()).expanduser().resolve(),
                    run_dir=Path(cfg.resolved_run_dir()).expanduser().resolve(),
                    recording_database_csv=(Path(recording_db_path).resolve() if recording_db_path else None),
                    movie_names=[Path(m).name for m in movie_files],
                    lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                    lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                )
    except Exception:
        logging.exception("Registry update failed (autocropper)")
