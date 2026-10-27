"""Coordinate movie selection, metadata exclusions, chamber cropping, and retries."""

from __future__ import annotations

from autobehaver.cli_summary import reported_stage, validation_details

from autobehaver.progress import Progress, drain_events, set_worker_queue

import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from autobehaver.validation import StageValidationReport

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

from .recording_db import (
    RecordingDBIndex,
    find_latest_recording_database_csv,
    load_recording_database_index,
)

from ..moviechecker.core import derive_pair_key
from ..moviechecker.exclusions import ExclusionsStore


# Runtime state (set by core.run_autocropper)

cfg: AutocropperConfig | None = None
args: Any | None = None

# Full movies folder
input_dir: Path | None = None

# Stage run directory (logs, chamber_images, movie_logs)
run_dir: Path | None = None

# Where cropped chamber movies are written
crop_output_dir: Path | None = None

# moviechecker exclusions
exclusions_csv_path: Path | None = None
exclusions_store: ExclusionsStore | None = None

# recording database labeling
recording_db_path: Path | None = None
recording_db_index: RecordingDBIndex | None = None


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
    """Build an exact trial-token pattern that excludes longer trial numbers."""
    return re.compile(rf"(?:^|_)trial{int(trial_number)}(?:_|$)", flags=re.IGNORECASE)


def _trial_number_from_name(name: str) -> int | None:
    """Return the trial number from an exact filename token, if present."""
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
    selected_trials = sorted({n for n in (_trial_number_from_name(original_name(p)) for p in selected) if n is not None})
    if any(n != requested for n in selected_trials):
        logging.warning(
            "file_filter=%r is a substring match and selected trial numbers %s. "
            "Use --trial_number %s or --file_filter _trial%s_ for exact trial selection.",
            filter_value,
            selected_trials,
            requested,
            requested,
        )


def _heartbeat_interval_seconds() -> int:
    """Return the configured positive progress-report interval in seconds."""
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

    return {
        "cropped_files": final_files,
        "missing_required_chambers": missing_required,
        "crop_failures": crop_failures,
    }


# Exclusions helpers


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
    """Combine manual MovieChecker exclusions with recording-time chamber exclusions."""
    excluded: set[str] = set()
    if exclusions_store is not None:
        key = derive_pair_key(str(movie))
        rec = exclusions_store.get(key)
        if rec is not None:
            for i, flag in enumerate(rec.exclude_ch):
                if int(flag) == 1:
                    excluded.add(f"chamber{i+1}")

    # Combine recording-time occupancy with visual review of recording faults.
    if recording_db_index is not None:
        exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
        excluded.update(recording_db_index.get_excluded_chambers(
            experiment=exp,
            subexperiment=subexp,
            trial_number=trial,
            rpi_number=rpi,
        ))
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


# Recording database labeling helpers


def _load_recording_db_if_needed() -> None:
    """Load indexed recording metadata when labeling is enabled.

    An explicit CSV overrides discovery; missing or invalid metadata aborts
    the run before cropping.
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
                f"No recording_database*.csv or crop_occupancy_*.csv found in: {db_dir}\n"
                "Labeling is enabled by default.\n"
                "Either put recording metadata in the full-movies folder, set --recording_database_dir, "
                "or rerun with --skip_labeling."
            )

    try:
        recording_db_index = load_recording_database_index(db_csv)
        recording_db_path = db_csv
    except Exception as e:
        raise SystemExit(
            f"Failed to load recording metadata: {db_csv}\n"
            f"Error: {e}\n"
            "If you want to run without DB labeling, rerun with --skip_labeling."
        )


def _parse_full_movie_identity(movie: Path) -> tuple[str, str, int, int]:
    """Resolve experiment, subexperiment, trial, and camera from a movie name.

    Prefer indexed recording metadata; retain the legacy filename-only fallback
    for callers that have not loaded it.
    """

    if recording_db_index is not None:
        return recording_db_index.resolve_movie_identity(Path(original_name(movie)))

    stem = Path(original_name(movie)).stem

    match = re.match(
        r"^(?P<prefix>.+)_trial(?P<trial>\d+)_rpi-(?P<rpi>\d+)_movie\d+(?:_|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        prefix = match.group("prefix")
        trial = int(match.group("trial"))
        rpi = int(match.group("rpi"))
        # Compatibility fallback for callers that parse without loading metadata.
        # New recordings sanitize internal underscores to dashes, so one underscore
        # reliably denotes a subexperiment; old ambiguous names need the DB resolver.
        identity_parts = prefix.split("_", 1)
        experiment = identity_parts[0]
        subexperiment = identity_parts[1] if len(identity_parts) == 2 else ""
        return experiment, subexperiment, trial, rpi

    raise ValueError(
        f"Could not parse movie identity from filename: {movie.name}\n"
        "Expected <experiment>[_<subexperiment>]_trial1_rpi-6_movie0_<timestamp>.mp4"
    )


from autobehaver.portable_paths import generated_path, original_name
from .contract import crop_request, valid_crop, video_shape
from autobehaver.completion import verified_run, file_signature, file_digest, write_receipt, preserve_previous, read_receipt
from .selection import write_selection

_movie_requests = {}
_movie_shapes = {}


def _crop_identity(movie, chamber_id):
    """Share immutable input and code identity across chambers in this process."""
    key = file_signature(movie)
    if key not in _movie_requests:
        _movie_requests[key] = crop_request(movie, chamber_id, args)
    return crop_request(movie, chamber_id, args, base_request=_movie_requests[key])


def _checked_source_shape(movie):
    """Cache validated video dimensions and frame count by source file signature."""
    key = file_signature(movie)
    if key not in _movie_shapes:
        _movie_shapes[key] = video_shape(movie)
    return _movie_shapes[key]

def _expected_crop_output_path(movie: Path, output_dir: Path, chamber_id: str) -> Path:
    """Return the portable chamber output path with the current metadata label."""

    # Preserve legacy identity text while removing accidental empty filename
    # components such as the old blank-subexperiment ``experiment__trial1``.
    base_name = re.sub(r"_+", "_", Path(original_name(movie)).stem).strip("_")

    if recording_db_index is None or bool(getattr(args, "skip_labeling", False)):
        return generated_path(output_dir, f"{base_name}_{chamber_id}.mp4", reserve=65)

    exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
    chamber_num = int(str(chamber_id).replace("chamber", ""))
    label_prefix = recording_db_index.get_label_prefix(
        experiment=exp,
        subexperiment=subexp,
        trial_number=trial,
        rpi_number=rpi,
        chamber_number=chamber_num,
    )
    # Both legacy labels (which conventionally end in ``_``) and newer labels
    # are accepted. Publish exactly one separator before the recording name.
    normalized_prefix = re.sub(r"_+", "_", str(label_prefix).strip()).rstrip("_")
    if not normalized_prefix:
        raise ValueError(f"Recording metadata returned a blank label for {movie.name} {chamber_id}")
    return generated_path(output_dir, f"{normalized_prefix}_{base_name}_{chamber_id}.mp4", reserve=65)


def _resolve_existing_crop(
    movie: Path,
    output_dir: Path,
    chamber_id: str,
) -> dict[str, Any]:
    """Find a unique crop that matches the source request and passes integrity checks.

    Return status, expected paths, matched names, and a usable path when present.
    Republish verified pixels under a corrected label and preserve the old version.
    """

    expected_path = _expected_crop_output_path(movie, output_dir, chamber_id)
    base_name = Path(original_name(movie)).stem

    matches = sorted(set(([expected_path] if expected_path.exists() else []) + [
        path for path in Path(output_dir).glob("*.mp4")
        if original_name(path).endswith(f"{base_name}_{chamber_id}.mp4")
    ]))
    identity = _crop_identity(movie, chamber_id) if matches else None
    candidates = matches
    matches = [path for path in candidates if valid_crop(movie, path, identity,
                                                        source_shape=_checked_source_shape(movie))]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous valid crops for {movie.name} {chamber_id}: {[p.name for p in matches]}')
    usable = bool(matches)
    if usable:
        for invalid in set(candidates) - set(matches):
            preserve_previous(invalid)
    if usable and matches[0] != expected_path:
        # A deliberate label correction changes identity text, not crop pixels.
        # Publish those verified pixels under the current label, then retain the
        # old version in the stage history rather than leaving two active crops.
        previous_path = matches[0]
        producer_identity = read_receipt(previous_path)['request']
        temporary = Path(output_dir) / (".crop-label-" + uuid.uuid4().hex[:12] + ".tmp")
        before = file_signature(previous_path)
        try:
            shutil.copyfile(previous_path, temporary)
            if file_signature(previous_path) != before or file_digest(temporary) != file_digest(previous_path):
                raise ValueError(f"Crop changed during label correction: {previous_path}")
            preserve_previous(previous_path)
            os.replace(temporary, expected_path)
            write_receipt(expected_path, producer_identity, relabelled_from=previous_path.name)
            matches = [expected_path]
        finally:
            if temporary.exists():
                temporary.unlink()
    return {'status': ('expected' if matches[0] == expected_path else 'exists_other') if usable else 'missing',
            'expected_name':expected_path.name, 'expected_path':expected_path,
            'path':matches[0] if usable else None, 'matches':[p.name for p in matches]}


def _require_recording_db_coverage(movie_files: list[Path]) -> None:
    """Require metadata labels for every selected, non-excluded chamber."""

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


# Worker initialization and crop completion


def _worker_init(worker_cfg: AutocropperConfig, worker_args: Any, metadata_snapshot=None, progress_events=None) -> None:
    """Initialize per-process runtime state for spawn-safe multiprocessing."""
    set_worker_queue(progress_events)
    random_seed = 42
    _movie_requests.clear()
    _movie_shapes.clear()
    configure_native_threading(getattr(worker_args, "native_threads", 1), overwrite=False)
    set_runtime(cfg=worker_cfg, args=worker_args)
    if run_dir is None or crop_output_dir is None:
        raise RuntimeError("run_dir/crop_output_dir is not set in worker initializer")

    run_dir.mkdir(parents=True, exist_ok=True)
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, log_level=str(getattr(worker_args, "log_level", "INFO")))
    if metadata_snapshot is None:
        _load_exclusions_if_available()
        _load_recording_db_if_needed()
    else:
        global exclusions_store, recording_db_path, recording_db_index
        exclusions_store, recording_db_path, recording_db_index = metadata_snapshot

    # Keep frame sampling deterministic across workers/process launches.
    import random

    random.seed(random_seed)


def all_cropped_movies_exist(
    movie: Path,
    output_dir: Path,
    *,
    excluded_chambers: set[str] | None = None,
    resolutions: dict | None = None,
) -> bool:
    """Check that all non-excluded chambers have validated crops.

    Use recording metadata and MovieChecker exclusions, excluding historical
    automatic occupancy decisions.
    """
    assert args is not None

    excluded = excluded_chambers or set()

    expected_total = int(args.chamber_columns) * int(args.chamber_rows)
    all_expected = [f"chamber{i}" for i in range(1, expected_total + 1)]
    expected = [cid for cid in all_expected if cid not in excluded]

    # If everything is excluded, there's nothing to crop.
    if not expected:
        return True

    for chamber_id in expected:
        res = _resolve_existing_crop(movie, output_dir, chamber_id)
        if resolutions is not None:
            resolutions[chamber_id] = res
        if res["status"] == "missing":
            logging.debug(
                f"Missing cropped video for {chamber_id}; expected {res['expected_name']}"
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


# Movie selection


def _load_movie_list(list_path: Path, *, base_dir: Path) -> list[Path]:
    """Read an optional selection file with one movie path per line.

    Resolve relative movie paths under base_dir, ignore blank/comment lines,
    and preserve first occurrence order while removing duplicates.
    """
    movies: list[Path] = []
    if not list_path.is_file():
        raise FileNotFoundError(f"Supplied movie_list file not found: {list_path}")

    for raw in list_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Movie listed in {list_path} was not found: {p}")
        movies.append(p)

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in movies:
        k = os.path.normcase(str(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _select_movie_files(all_movie_files: list[Path]) -> list[Path]:
    """Select movies by explicit list, exact trial, substring, or the complete folder."""
    assert args is not None
    if input_dir is None:
        raise RuntimeError("input_dir is not set")

    # Explicit list takes precedence
    movie_list_path = getattr(args, "movie_list", None)
    if movie_list_path:
        return _load_movie_list(Path(movie_list_path), base_dir=input_dir)

    trial_number = _trial_number_value()
    if trial_number is not None:
        pattern = _trial_token_pattern(trial_number)
        return [m for m in all_movie_files if pattern.search(Path(original_name(m)).stem)]

    filt = _file_filter_value()
    if filt:
        needle = str(filt).lower()
        selected = [m for m in all_movie_files if needle in original_name(m).lower()]
        _warn_if_trial_filter_is_ambiguous(selected, str(filt))
        return selected

    return all_movie_files


# Multiprocessing entrypoint


@verified_run
def process_single_movie(movie: Path) -> None:
    """Crop missing included chambers and save the per-movie outcome log."""
    _movie_requests.clear()
    _movie_shapes.clear()
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

    movie_data: dict[str, Any] = {"movie_name": movie_name, "source_path": str(Path(movie).resolve())}

    excluded = _excluded_set_for_movie(movie)
    movie_data["expected_chamber_count"] = int(args.chamber_columns) * int(args.chamber_rows)
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
            if recording_db_index is not None and not bool(getattr(args, "skip_labeling", False)):
                movie_data["chamber_labels"] = {
                    f"chamber{i}": recording_db_index.get_label_prefix(
                        experiment=exp, subexperiment=subexp, trial_number=trial,
                        rpi_number=rpi, chamber_number=i,
                    )
                    for i in range(1, movie_data["expected_chamber_count"] + 1)
                    if f"chamber{i}" not in excluded
                }
        except Exception as e:
            if not bool(getattr(args, "skip_labeling", False)):
                raise
            movie_data["identity_parse_warning"] = str(e)

        # Fast skip path: if all required crops already exist, do not redo chamber detection.
        existing_resolutions = {}
        if all_cropped_movies_exist(movie, output_dir, excluded_chambers=excluded,
                                   resolutions=existing_resolutions):
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
                    res = existing_resolutions[chamber_id]
                    p = res.get("path")
                    if p is not None:
                        existing[chamber_id] = Path(p).name
                movie_data["cropped_files"] = existing
            except Exception:
                pass
            logging.info(f"[OK] {movie_name}: Already processed, skipping")
            return

        # Detect the complete grid before applying exclusions.
        args._detection_diagnostic_dir = str(run_dir / 'chamber_images')
        all_coords, viz_frame, circles, crop_dim = detect_and_match_chambers(movie, args)
        movie_data['grid_fit'] = getattr(args, '_last_grid_fit', {})
        movie_data["chamber_coordinates"] = {
            cid: {"x": int(x), "y": int(y)} for cid, (x, y) in all_coords.items()
        }

        # Apply exclusions *before* validation/cropping
        allowed_coords = {cid: xy for cid, xy in all_coords.items() if cid not in excluded}

        if not allowed_coords:
            logging.info(f"[OK] {movie_name}: All chambers excluded, skipping cropping")
            movie_data["status"] = "skipped"
            movie_data["skip_reason"] = "all chambers excluded"
            movie_data["cropped_files"] = {}
            return

        draw_circles(
            viz_frame,
            circles,
            args.down_factor,
            allowed_coords,
            approx_radius=int(args.approx_radius),
        )
        save_chamber_coordinates_image(run_dir, movie, viz_frame)

        # Crop (only what is missing) while respecting DB-based naming
        crop_status: dict[str, Any] = {}
        existing_or_final: dict[str, str] = {}
        coords_to_crop: dict[str, tuple[int, int]] = {}
        output_paths_to_crop: dict[str, Path] = {}

        for chamber_id, (x, y) in allowed_coords.items():
            res = existing_resolutions.get(chamber_id)
            if res is None:
                res = _resolve_existing_crop(movie, output_dir, chamber_id)

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
                identities={cid: _crop_identity(movie, cid) for cid in coords_to_crop},
            )
            cropped_files.update(newly_cropped)

        movie_data["crop_status"] = crop_status

        assessment = _assess_required_crops(
            movie,
            output_dir,
            required_chambers=list(allowed_coords.keys()),
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
        n_required = len(allowed_coords)

        if missing_required or failed_attempts:
            movie_data["status"] = "incomplete"
            logging.warning(
                f"Ã¢â€“Â³ {movie_name}: Finished attempt with incomplete required crops "
                f"({n_present}/{n_required} present; {len(missing_required)} still missing, "
                f"{len(failed_attempts)} crop failures this attempt)"
            )
        else:
            movie_data["status"] = "completed"
            logging.info(
                f"[OK] {movie_name}: All required crops present "
                f"({len(coords_to_crop)} new chambers cropped, {len(existing_or_final)} already present)"
            )

    except Exception as e:
        logging.error(f"[FAILED] {movie_name}: Failed - {str(e)}")
        movie_data["status"] = "failed"
        movie_data["error"] = str(e)

    finally:
        # Preserve per-movie outcomes even when cropping fails.
        log_movie_results(movie, run_dir, movie_data)

    return {"movie_name": movie_name, "error": movie_data.get("error")}


def process_movies_in_parallel(movie_files: list[Path]) -> None:
    """Process movies in parallel with spawn-safe worker initialization."""
    assert args is not None
    assert cfg is not None

    if not movie_files:
        return

    num_workers = min(len(movie_files), max(1, int(getattr(args, "num_workers", 2))))
    heartbeat = _heartbeat_interval_seconds()
    ctx = mp.get_context("spawn")

    logging.info(
        f"Starting worker pool with {num_workers} process(es) using start method 'spawn'."
    )

    completed = 0
    total = len(movie_files)
    events = ctx.Queue(maxsize=64)

    try:
        with Progress(total=total, desc="Autocropper movies") as display, cf.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(cfg, args, (exclusions_store, recording_db_path, recording_db_index), events),
        ) as executor:
            future_to_movie = {executor.submit(process_single_movie, movie): movie for movie in movie_files}

            while future_to_movie:
                done, _pending = cf.wait(
                    tuple(future_to_movie.keys()),
                    timeout=min(heartbeat, 1),
                    return_when=cf.FIRST_COMPLETED,
                )

                drain_events(events, display)
                if not done:
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

                if not future_to_movie:
                    display.detail = ""
                display.update(len(done))
    finally:
        events.close()
        events.join_thread()


# Main orchestrator


@reported_stage("Autocropper")
@verified_run
def run_pipeline() -> None:
    """Validate selected movies, crop required chambers with retries, and record outcomes."""
    _movie_requests.clear()
    _movie_shapes.clear()
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
    # Each movie shares decoder/filter pools; encoders retain separate caps.
    # FFmpeg can also create helper/I/O threads beyond this estimate.
    estimated_crop_threads = (2 * configured_movie_workers + effective_crop_jobs) * configured_ffmpeg_threads
    logging.info(
        "Configured parallelism: "
        f"{configured_movie_workers} movie worker(s) x {configured_crop_workers} chamber worker(s) per movie "
        f"= up to {effective_crop_jobs} chamber encoders sharing movie decoders; "
        f"ffmpeg_threads={configured_ffmpeg_threads} per decoder/filter/encoder pool, "
        f"native_threads={configured_native_threads}."
    )

    slurm_cpus = 0
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    except Exception:
        slurm_cpus = 0
    if slurm_cpus:
        logging.info(f"Detected SLURM_CPUS_PER_TASK={slurm_cpus} for this autocropper command.")
        # An indicative pool budget, not a measured peak or a hard CPU bound.
        estimated_threads = estimated_crop_threads + configured_movie_workers * configured_native_threads
        if estimated_threads > slurm_cpus:
            logging.warning(
                "Configured autocropper parallelism can exceed CPUs allocated to this swarm subjob: "
                f"configured decoder/filter/encoder/native pools ~ {estimated_threads} threads "
                f"plus helper threads, SLURM_CPUS_PER_TASK={slurm_cpus}. "
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
    for describe, key in (('native', lambda p: p.stem.casefold()),
                          ('logical', lambda p: Path(original_name(p)).stem.casefold())):
        seen = {}
        for movie in selected_movies:
            name = key(movie)
            if name in seen and seen[name].resolve() != movie.resolve():
                raise ValueError(f'Autocropper {describe} movie name collision: {seen[name]}; {movie}')
            seen[name] = movie
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

    # A selected-but-unreadable input is a failed run, not an empty successful run.
    if not movie_files:
        validation = StageValidationReport.for_items(
            "autocropper",
            [str(movie) for movie in selected_movies],
            metadata={"crop_output_directory": str(crop_output_dir)},
        )
        if not selected_movies:
            validation.add_issue(
                "no_inputs_selected",
                "error",
                "No full-movie inputs matched the requested selection.",
            )
        for movie in invalid_movie_files:
            validation.add_outcome(str(movie), "failed", reason="video could not be opened and decoded")
        for movie in missing_listed_files:
            validation.add_outcome(str(movie), "failed", reason="selected movie does not exist")
        validation.write_run_reports(run_dir / "AUTOCROPPER_VALIDATION.json")
        validation_details(validation, unit="Movies", output=str(crop_output_dir))
        validation.raise_for_errors()

    # Ensure exclusions.csv has coverage for this subset when required.
    if not _exclusions_optional():
        _require_exclusions_coverage(movie_files)

    # Ensure recording DB covers this subset (unless --skip_labeling)
    _require_recording_db_coverage(movie_files)

    # Publish current selections once in the parent, before workers run. Old
    # crops remain on disk but cannot override an explicit chamber exclusion.
    selection_records = {}
    for movie in movie_files:
        excluded = _excluded_set_for_movie(movie)
        if recording_db_index is not None and not bool(getattr(args, "skip_labeling", False)):
            exp, subexp, trial, rpi = _parse_full_movie_identity(movie)
        for i in range(1, int(args.chamber_columns) * int(args.chamber_rows) + 1):
            record = {
                "included": f"chamber{i}" not in excluded,
            }
            if record["included"] and recording_db_index is not None and not bool(getattr(args, "skip_labeling", False)):
                record["label_prefix"] = recording_db_index.get_label_prefix(
                    experiment=exp, subexperiment=subexp, trial_number=trial,
                    rpi_number=rpi, chamber_number=i,
                )
            selection_records[f"{Path(original_name(movie)).stem}_chamber{i}"] = record
    write_selection(crop_output_dir, selection_records)

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
    chamber_images_dir = run_dir / "chamber_images"

    for directory in [chamber_images_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        ensure_user_permissions(directory)

    # Retry loop
    movies_to_process = movie_files[:]
    max_retries = int(getattr(args, "max_retries", 10))
    worker_failure = None

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

        try:
            process_movies_in_parallel(movies_to_process)
        except Exception as exc:
            worker_failure = f'{type(exc).__name__}: {exc}'
            logging.exception('Autocropper worker pool failed; stopping automatic retries')
        delete_incomplete_files(
            crop_output_dir,
            movie_stems=[Path(original_name(m)).stem for m in movie_files],
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
        if worker_failure is not None:
            break

    logging.info("-" * 50)

    if not movies_to_process:
        logging.info("All required crops are present; completing final checks.")
    else:
        logging.error(f"Processing finished, but hit max retry limit ({max_retries}).")
        logging.error("The following movies failed permanently:")
        for movie in movies_to_process:
            logging.error(f"  - {movie.name}")

    summarize_results(run_dir, movie_names=[Path(m).name for m in movie_files])

    # Project registry update (single CSV at project root)
    registry_error: str | None = None
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
    except Exception as exc:
        registry_error = str(exc)
        logging.exception("Registry update failed (autocropper)")

    validation = StageValidationReport.for_items(
        "autocropper",
        [str(movie) for movie in selected_movies],
        metadata={
            "crop_output_directory": str(crop_output_dir),
            "recording_database": str(recording_db_path) if recording_db_path else None,
        },
    )
    incomplete = {str(movie) for movie in movies_to_process}
    invalid = {str(movie) for movie in invalid_movie_files}
    missing = {str(movie) for movie in missing_listed_files}
    expected_chambers = int(args.chamber_columns) * int(args.chamber_rows)
    for movie in selected_movies:
        item = str(movie)
        if item in missing:
            validation.add_outcome(item, "failed", reason="selected movie does not exist")
        elif item in invalid:
            validation.add_outcome(item, "failed", reason="video could not be opened and decoded")
        elif item in incomplete:
            validation.add_outcome(
                item,
                "failed",
                reason="one or more required chamber crops remain incomplete after retries",
            )
        else:
            excluded = _excluded_set_for_movie(movie)
            deliberately_empty = len(excluded) >= expected_chambers
            validation.add_outcome(
                item,
                "excluded" if deliberately_empty else "succeeded",
                reason="all chambers excluded by recording metadata or MovieChecker"
                if deliberately_empty
                else None,
                details={"excluded_chambers": sorted(excluded)},
            )
    if registry_error is not None:
        validation.add_issue(
            "registry_update_failed",
            "error",
            "Autocropper outputs were created, but the project registry could not be updated.",
            details={"error": registry_error},
        )
    if worker_failure is not None:
        validation.add_issue('worker_pool_failed', 'error',
            'A worker process failed; completed crops were checked and automatic retries stopped.',
            details={'error': worker_failure})
    validation.write_run_reports(run_dir / "AUTOCROPPER_VALIDATION.json")
    validation_details(validation, unit="Movies", output=str(crop_output_dir))
    validation.raise_for_errors()
