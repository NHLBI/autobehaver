"""Recording metadata discovery, label lookup, and occupancy filtering."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_GLOB = "recording_database*.csv"
CROP_OCCUPANCY_GLOB = "crop_occupancy_*.csv"
EMPTY_LABEL_PREFIX = "EMPTY::"
_TRUE_VALUES = {"1", "true", "yes", "y", "occupied"}
_FALSE_VALUES = {"0", "false", "no", "n", "empty"}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _identity_component(value: object) -> str:
    return re.sub(r"[-_]+", "-", _clean_text(value)).strip("-").casefold()


def _strict_int(value: object, *, field: str, context: str, minimum: int = 1,
                maximum: int | None = None) -> int:
    """Parse a bounded integer without rounding fractional CSV values."""
    if not _clean_text(value) or isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer, got {value!r}")
    try:
        numeric = Decimal(str(value))
        number = int(numeric)
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: {field} must be an integer, got {value!r}") from exc
    if numeric != number:
        raise ValueError(f"{context}: {field} must be an integer, got {value!r}")
    if number < minimum or (maximum is not None and number > maximum):
        expected = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{context}: {field} must be {expected}, got {number}")
    return number


def _session_key(experiment: object, subexperiment: object, trial: object, *,
                 context: str = "recording metadata") -> tuple[str, str, int]:
    """Validate and return the experiment/subexperiment/trial identity."""
    experiment_name = _clean_text(experiment)
    if not experiment_name:
        raise ValueError(f"{context}: experiment must not be blank")
    trial_number = _strict_int(trial, field="trial_number", context=context)
    return (experiment_name, _clean_text(subexperiment), trial_number)


def _strict_bool(value: object, *, field: str, context: str) -> bool:
    """Parse supported occupancy booleans and reject unrecognized values."""
    if isinstance(value, bool):
        return value
    if _clean_text(value):
        normalized = str(value).strip().casefold()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        # Some CSV writers serialize integer boolean columns as floats.
        if normalized == "1.0":
            return True
        if normalized == "0.0":
            return False
    raise ValueError(
        f"{context}: {field} must be one of true/false, yes/no, occupied/empty, or 1/0; "
        f"got {value!r}"
    )


def _optional_crop_bool(row: dict[str, str], *, context: str) -> tuple[bool, bool]:
    """Return ``(occupied, use_for_crop)`` without treating blank cells as false."""
    occupied_raw = row.get("occupied")
    occupied = True if not _clean_text(occupied_raw) else _strict_bool(
        occupied_raw, field="occupied", context=context
    )
    use_raw = row.get("use_for_crop")
    use_for_crop = occupied if not _clean_text(use_raw) else _strict_bool(
        use_raw, field="use_for_crop", context=context
    )
    if use_for_crop and not occupied:
        raise ValueError(f"{context}: use_for_crop is true but occupied is false")
    return occupied, use_for_crop


def _read_metadata_csv(path: Path, *, nrows: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Read metadata without requiring the scientific runtime for MovieChecker."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            rows = []
            if nrows != 0:
                for row in reader:
                    if not row:
                        continue
                    if header is not None and len(row) != len(header):
                        raise ValueError(f"Recording metadata CSV has an incorrect field count at line {reader.line_num}: {path}")
                    rows.append(dict(zip(header or [], row)))
                    if nrows is not None and len(rows) >= nrows:
                        break
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read recording metadata CSV {path}: {exc}") from exc
    if not header:
        raise ValueError(f"Recording metadata CSV is empty: {path}")
    if any(not name.strip() for name in header):
        raise ValueError(f"Recording metadata CSV has an unnamed column: {path}")
    normalized = [name.strip() for name in header]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Recording metadata CSV has duplicate columns {duplicates}: {path}")
    if normalized != header:
        raise ValueError(f"Recording metadata CSV has whitespace around a column name: {path}")
    return header, rows


def _canonical_rpi(value: object, *, context: str) -> str:
    """Normalize a valid camera identifier to the rpi-N convention."""
    raw = _clean_text(value)
    match = re.fullmatch(r"(?:rpi-)?(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"{context}: rpi_name must look like 'rpi-2' or '2', got {value!r}")
    number = _strict_int(match.group(1), field="rpi_name", context=context)
    return f"rpi-{number}"


def _new_label_map_entry(labels: dict, excluded: dict, origins: dict, *, key: tuple[str, str, int],
                         rc_key: str, label: str, use_for_crop: bool, context: str) -> None:
    """Add a unique chamber label and exclusion decision with its metadata origin."""
    slot = (key, rc_key)
    if slot in origins:
        raise ValueError(
            f"{context}: duplicate chamber mapping for {rc_key} in session {key}; "
            f"first defined at {origins[slot]}"
        )
    if not label:
        raise ValueError(f"{context}: label is blank for {rc_key} in session {key}")
    labels.setdefault(key, {})[rc_key] = label
    excluded.setdefault(key, set())
    if not use_for_crop:
        excluded[key].add(rc_key)
    origins[slot] = context


def _validate_normalized_session_identities(labels: dict) -> None:
    """Reject session identities that collide after filename separator normalization."""
    seen: dict[tuple[str, int], tuple[str, str, int]] = {}
    for key in labels:
        joined = key[0] if not key[1] else f"{key[0]}_{key[1]}"
        canonical = (_identity_component(joined), key[2])
        other = seen.get(canonical)
        if other is not None and other != key:
            raise ValueError(
                "Recording metadata has ambiguous experiment/subexperiment identities after "
                f"underscore/dash normalization: {other} and {key}"
            )
        seen[canonical] = key


@dataclass(frozen=True)
class RecordingDBIndex:
    """Unified index built from either supported recording metadata layout."""

    csv_path: Path
    label_map: dict[tuple[str, str, int], dict[str, str]]
    excluded_map: dict[tuple[str, str, int], set[str]]

    def _resolve_key(self, experiment: str, subexperiment: str, trial_number: int) -> tuple[str, str, int]:
        """Resolve an exact session or its unique separator-normalized equivalent."""
        exact = (_clean_text(experiment), _clean_text(subexperiment), int(trial_number))
        if exact in self.label_map:
            return exact
        wanted = (_identity_component(experiment), _identity_component(subexperiment), int(trial_number))
        matches = [key for key in self.label_map if
                   (_identity_component(key[0]), _identity_component(key[1]), key[2]) == wanted]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"No recording metadata entry found for (experiment={experiment!r}, "
            f"subexperiment={subexperiment!r}, trial={trial_number})."
        )

    def resolve_filename_prefix(self, prefix: str, trial_number: int) -> tuple[str, str, int]:
        """Resolve the text before ``_trialN`` against known sessions."""
        wanted = _identity_component(prefix)
        matches = []
        for key in self.label_map:
            if key[2] != int(trial_number):
                continue
            joined = key[0] if not key[1] else f"{key[0]}_{key[1]}"
            if _identity_component(joined) == wanted:
                matches.append(key)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous recording metadata for filename prefix {prefix!r}, trial {trial_number}.")
        raise KeyError(f"No recording metadata matches filename prefix {prefix!r}, trial {trial_number}.")

    def resolve_movie_identity(self, movie: Path) -> tuple[str, str, int, int]:
        """Resolve a full-movie filename using the indexed session identities."""
        match = re.match(
            r"^(?P<prefix>.+)_trial(?P<trial>\d+)_rpi-(?P<rpi>\d+)_movie\d+(?:_|$)",
            Path(movie).stem, flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Could not parse movie identity from filename: {Path(movie).name}")
        experiment, subexperiment, trial = self.resolve_filename_prefix(
            match.group("prefix"), int(match.group("trial"))
        )
        return experiment, subexperiment, trial, int(match.group("rpi"))

    def get_movie_exclusion_flags(self, movie: Path) -> list[int]:
        """Return six recording exclusions, requiring complete camera metadata."""
        experiment, subexperiment, trial, rpi = self.resolve_movie_identity(movie)
        for chamber in range(1, 7):
            self.get_label_prefix(experiment=experiment, subexperiment=subexperiment,
                                  trial_number=trial, rpi_number=rpi, chamber_number=chamber)
        excluded = self.get_excluded_chambers(
            experiment=experiment, subexperiment=subexperiment, trial_number=trial, rpi_number=rpi
        )
        return [int(f"chamber{chamber}" in excluded) for chamber in range(1, 7)]

    def get_label_prefix(self, *, experiment: str, subexperiment: str, trial_number: int,
                         rpi_number: int, chamber_number: int) -> str:
        """Return the indexed label for a session, camera, and chamber."""
        key = self._resolve_key(experiment, subexperiment, trial_number)
        rc_key = f"rpi-{int(rpi_number)}_chamber{int(chamber_number)}"
        label = self.label_map[key].get(rc_key)
        if label is None:
            raise KeyError(f"No label found in recording metadata for {rc_key} within {key}.")
        return str(label)

    def get_excluded_chambers(self, *, experiment: str, subexperiment: str,
                              trial_number: int, rpi_number: int) -> set[str]:
        """Return chamber IDs excluded for the selected session and camera."""
        key = self._resolve_key(experiment, subexperiment, trial_number)
        camera_prefix = f"rpi-{int(rpi_number)}_"
        return {rc_key.removeprefix(camera_prefix)
                for rc_key in self.excluded_map.get(key, set())
                if rc_key.startswith(camera_prefix)}


def _parse_db_timestamp_from_name(path: Path) -> datetime | None:
    """Return the recording-database filename timestamp, if valid."""
    match = re.match(r"^recording_database_(\d{8})_(\d{6})$", path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def find_latest_recording_database_csv(db_dir: Path) -> Path:
    """Find a session database, or a directory containing detailed occupancy CSVs."""
    db_dir = Path(db_dir).expanduser().resolve()
    candidates = sorted(db_dir.glob(DB_GLOB), key=lambda path: path.name.casefold())
    if candidates:
        def score(path: Path) -> tuple[float, str]:
            timestamp = _parse_db_timestamp_from_name(path)
            when = timestamp.timestamp() if timestamp is not None else path.stat().st_mtime
            return when, path.name.casefold()
        return max(candidates, key=score)
    if any(db_dir.rglob(CROP_OCCUPANCY_GLOB)):
        return db_dir
    raise FileNotFoundError(f"No '{DB_GLOB}' or '{CROP_OCCUPANCY_GLOB}' files found in: {db_dir}")


def parse_camera_mappings(camera_mappings: str, *, context: str = "camera_mappings") -> dict[str, str]:
    """Parse unique rpi-N_chamberM label entries from session metadata."""
    out: dict[str, str] = {}
    serialized = _clean_text(camera_mappings)
    if not serialized:
        raise ValueError(f"{context}: camera_mappings must not be blank")
    for raw in serialized.split(","):
        if ":" not in raw:
            raise ValueError(f"{context}: malformed camera mapping entry {raw!r}")
        key, label = raw.strip().split(":", 1)
        match = re.fullmatch(r"rpi-(\d+)_chamber(\d+)", key.strip(), flags=re.IGNORECASE)
        if not match:
            raise ValueError(
                f"{context}: mapping key must look like 'rpi-2_chamber4', got {key.strip()!r}"
            )
        rpi_number = _strict_int(match.group(1), field="rpi number", context=context)
        chamber_number = _strict_int(
            match.group(2), field="chamber number", context=context, maximum=6
        )
        canonical_key = f"rpi-{rpi_number}_chamber{chamber_number}"
        if canonical_key in out:
            raise ValueError(f"{context}: duplicate camera mapping for {canonical_key}")
        cleaned_label = label.strip()
        if not cleaned_label or cleaned_label == EMPTY_LABEL_PREFIX:
            raise ValueError(f"{context}: label is blank for {canonical_key}")
        out[canonical_key] = cleaned_label
    return out


def _load_session_database(csv_path: Path) -> tuple[dict, dict]:
    """Load session-level chamber labels and explicit empty-chamber exclusions."""
    columns, rows = _read_metadata_csv(csv_path)
    required = {"experiment", "subexperiment", "trial_number", "camera_mappings"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Recording DB is missing required columns: {sorted(missing)}")
    labels: dict[tuple[str, str, int], dict[str, str]] = {}
    excluded: dict[tuple[str, str, int], set[str]] = {}
    origins: dict[tuple[tuple[str, str, int], str], str] = {}
    session_origins: dict[tuple[str, str, int], str] = {}
    for row_index, row in enumerate(rows):
        context = f"{csv_path.name} row {row_index + 2}"
        key = _session_key(
            row["experiment"], row["subexperiment"], row["trial_number"], context=context
        )
        if key in session_origins:
            raise ValueError(
                f"{context}: duplicate session {key}; first defined at {session_origins[key]}"
            )
        session_origins[key] = context
        for rc_key, raw_label in parse_camera_mappings(
            row["camera_mappings"], context=f"{context} camera_mappings"
        ).items():
            label = raw_label
            use_for_crop = True
            if label.startswith(EMPTY_LABEL_PREFIX):
                use_for_crop = False
                label = label[len(EMPTY_LABEL_PREFIX):]
            _new_label_map_entry(
                labels, excluded, origins, key=key, rc_key=rc_key, label=label,
                use_for_crop=use_for_crop, context=context,
            )
    _validate_normalized_session_identities(labels)
    return labels, excluded


def _load_detailed_occupancy(csv_paths: list[Path]) -> tuple[dict, dict]:
    """Load per-chamber labels and occupancy decisions, rejecting conflicting records."""
    labels: dict[tuple[str, str, int], dict[str, str]] = {}
    excluded: dict[tuple[str, str, int], set[str]] = {}
    origins: dict[tuple[tuple[str, str, int], str], str] = {}
    required = {"experiment", "subexperiment", "trial_number", "rpi_name", "chamber_number", "label"}
    for csv_path in csv_paths:
        columns, rows = _read_metadata_csv(csv_path)
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(rows):
            context = f"{csv_path.name} row {row_index + 2}"
            key = _session_key(
                row["experiment"], row["subexperiment"], row["trial_number"], context=context
            )
            rpi = _canonical_rpi(row["rpi_name"], context=context)
            chamber = _strict_int(
                row["chamber_number"], field="chamber_number", context=context, maximum=6
            )
            rc_key = f"{rpi}_chamber{chamber}"
            occupied, use_for_crop = _optional_crop_bool(row, context=context)
            status = _clean_text(row.get("recording_status")).casefold()
            unusable_statuses = {"empty", "recording_failed", "not_connected", "not_recorded"}
            if status in unusable_statuses and use_for_crop:
                raise ValueError(
                    f"{context}: recording_status is {status!r} but use_for_crop is true"
                )
            label = _clean_text(row.get("legacy_label"))
            if not label:
                label = _clean_text(row["label"])
                sex = _clean_text(row.get("sex")).upper()
                if sex in {"MALE", "FEMALE"}:
                    label = f"{label}_{sex}_"
            if label.startswith(EMPTY_LABEL_PREFIX):
                if use_for_crop:
                    raise ValueError(
                        f"{context}: label marks {rc_key} empty but use_for_crop is true"
                    )
                label = label[len(EMPTY_LABEL_PREFIX):]
            _new_label_map_entry(
                labels, excluded, origins, key=key, rc_key=rc_key, label=label,
                use_for_crop=use_for_crop, context=context,
            )
    _validate_normalized_session_identities(labels)
    return labels, excluded


def load_recording_database_index(source: Path) -> RecordingDBIndex:
    """Detect either supported metadata schema and build the unified chamber index."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Recording metadata not found: {source}")
    if source.is_dir():
        csv_paths = sorted(source.rglob(CROP_OCCUPANCY_GLOB))
        if not csv_paths:
            raise FileNotFoundError(f"No '{CROP_OCCUPANCY_GLOB}' files found in: {source}")
        labels, excluded = _load_detailed_occupancy(csv_paths)
    else:
        columns = set(_read_metadata_csv(source, nrows=0)[0])
        is_session = "camera_mappings" in columns
        is_detailed = bool({"rpi_name", "chamber_number"}.intersection(columns))
        if is_session and is_detailed:
            raise ValueError(
                f"Recording metadata schema is ambiguous (contains both session and per-chamber fields): {source}"
            )
        if is_session:
            labels, excluded = _load_session_database(source)
        else:
            labels, excluded = _load_detailed_occupancy([source])
    logging.info("Loaded recording metadata: %s (%d sessions indexed, %d empty chambers)",
                 source, len(labels), sum(map(len, excluded.values())))
    return RecordingDBIndex(csv_path=source, label_map=labels, excluded_map=excluded)
