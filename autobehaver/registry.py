"""Link cropped chamber movies to downstream artifacts in a project CSV.

Store portable project-relative paths or an explicit external storage root.
Use a process-held lock for incremental updates; stage modules determine the
artifacts and biological metadata to register.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

import pandas as pd

from ._version import __version__
from .atomic_io import atomic_write_dataframe_csv, atomic_write_json, sha256_file
from .file_locks import FileLock


# Schema


DEFAULT_REGISTRY_COLUMNS: list[str] = [
    "recording_id",  # unique per cropped movie (includes movie# + timestamp + chamber)
    "animal_id",  # unique per animal within a trial/camera/chamber (no movie# / timestamp)
    "experiment",
    "subexperiment",
    "trial_number",
    "rpi_number",
    "movie_number",
    "timestamp",
    "chamber_number",
    "genotype",
    "sex",
    "label_prefix",
    # Artifact paths, with explicit roots for external storage.
    "full_movie_relpath",
    "cropped_movie_relpath",
    "dlc_csv_relpath",
    "keypre_csv_relpath",
    "keyprenorm_h5_relpath",
    "posepred_inference_csv_relpath",
    # Provenance
    "recording_database_csv_relpath",
    "keypre_version",
    "keyprenorm_params_relpath",
    "posepred_models",
    # Control / misc
    "active",
    "notes",
    "created_at",
    "updated_at",
]


CANONICAL_ID_KEY = "recording_id"


class RegistryValidationError(ValueError):
    """Report ambiguous identities or invalid paths/control values in a registry."""


_STORAGE_PREFIX = ".autobehaver_roots"


class RegistryArtifactPath(str):
    """A relative registry value carrying its explicitly identified storage root."""

    def __new__(cls, value: str, storage_root: str = "."):
        result = super().__new__(cls, value)
        result.storage_root = storage_root
        return result


def _storage_column(column: str) -> str:
    return column[:-len("_relpath")] + "_storage_root"


def _storage_key(root: str) -> str:
    value = str(root).replace("\\", "/")
    if PureWindowsPath(value).is_absolute():
        value = value.casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _valid_storage_root(value: str) -> bool:
    if value == ".":
        return True
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    return (windows.is_absolute() or posix.is_absolute()) and ".." not in posix.parts


_TRUE_ACTIVE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_ACTIVE_VALUES = frozenset({"0", "false", "no", "n", "off"})
_UNIQUE_ARTIFACT_COLUMNS = (
    "cropped_movie_relpath",
    "dlc_csv_relpath",
    "keypre_csv_relpath",
    "keyprenorm_h5_relpath",
    "posepred_inference_csv_relpath",
)


# Registry settings


@dataclass(frozen=True)
class RegistrySettings:
    """Runtime settings for reading/writing the project registry."""

    enabled: bool
    csv_path: Path
    lock_timeout_sec: int = 120
    lock_stale_sec: int = 60 * 60  # 1 hour

    @property
    def project_root(self) -> Path:
        # Convention: registry lives at project root.
        return self.csv_path.parent


# Locking


class RegistryLock:
    """Hold the shared file lock until exit; never infer ownership from its age.

    ``stale_sec`` remains accepted for existing callers. Legacy directory locks
    cannot prove whether an older writer is still running and need explicit
    migration after that writer has stopped.
    """

    def __init__(self, lock_dir: Path, *, timeout_sec: int = 120, stale_sec: int = 3600):
        self.lock_dir = Path(lock_dir)
        self.timeout_sec = int(timeout_sec)
        self.stale_sec = int(stale_sec)
        self._lock = FileLock(self.lock_dir, timeout_sec=timeout_sec)

    def acquire(self) -> None:
        if self.lock_dir.is_dir():
            raise RegistryValidationError(
                f"Legacy registry lock directory: {self.lock_dir}. After confirming all "
                "older registry writers have stopped, remove that directory before retrying. "
                "Current file locks release automatically when their process exits."
            )
        try:
            self._lock.acquire(metadata={"stage": "registry"})
        except RuntimeError as exc:
            raise TimeoutError(f"Timed out waiting for registry writer: {self.lock_dir}") from exc

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "RegistryLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# Helpers


def relpath_str(path: Path, root: Path) -> str:
    """Keep portable project paths, retaining the root for external storage."""
    absolute = Path(path).expanduser().resolve()
    project_root = Path(root).expanduser().resolve()
    try:
        rel = os.path.relpath(str(absolute), str(project_root))
    except ValueError:
        rel = None
    if rel is not None and _validate_relative_artifact_path(rel):
        return RegistryArtifactPath(rel.replace("\\", "/"))

    anchor = absolute.anchor
    if not anchor:
        raise RegistryValidationError(f"Cannot determine storage root for {absolute}")
    storage_root = Path(anchor).as_posix()
    within = absolute.relative_to(Path(anchor)).as_posix()
    value = f"{_STORAGE_PREFIX}/{_storage_key(storage_root)}/{within}"
    return RegistryArtifactPath(value, storage_root)


def resolve_registry_path(project_root: Path, row: Any, column: str) -> Path:
    """Resolve a historical project-relative or an explicitly rooted artifact."""
    value = row.get(column)
    if _is_missing_scalar(value) or not str(value).strip():
        raise RegistryValidationError(f"Registry has no {column} value")
    raw = str(value).replace("\\", "/")
    if not _validate_relative_artifact_path(raw):
        raise RegistryValidationError(f"Invalid registry artifact path: {raw}")
    root_value = row.get(_storage_column(column), ".")
    storage_root = "." if _is_missing_scalar(root_value) or not str(root_value).strip() else str(root_value)
    parts = PurePosixPath(raw).parts
    if parts and parts[0] == _STORAGE_PREFIX:
        if (len(parts) < 3 or storage_root == "." or not _valid_storage_root(storage_root)
                or parts[1] != _storage_key(storage_root)):
            raise RegistryValidationError(f"Missing or inconsistent storage location for {column}: {raw}")
        if os.name != "nt" and PureWindowsPath(storage_root).is_absolute():
            raise RegistryValidationError(
                f"Windows storage location {storage_root} is not mounted on this platform; "
                "resolve the recorded storage location before using this registry."
            )
        if os.name == "nt" and not PureWindowsPath(storage_root).is_absolute():
            raise RegistryValidationError(
                f"POSIX storage location {storage_root} needs an explicit mounted Windows "
                "location before using this registry; it cannot be resolved against the current drive."
            )
        return Path(storage_root).joinpath(*parts[2:]).resolve()
    if storage_root != ".":
        raise RegistryValidationError(f"Unexpected external storage root for project path {raw}")
    return Path(project_root).joinpath(*parts).resolve()


def parse_full_movie_stem(stem: str) -> dict[str, Any]:
    """Extract experiment, trial, camera, movie and timestamp fields from a stem.

    Expect experiment_subexperiment_trialN_rpi-N_movieN_timestamp. Accept
    underscores within experiment fields and a 14-digit timestamp with an
    optional AM/PM suffix. Raise ValueError for an invalid stem; a missing
    timestamp is returned as None.
    """

    s = str(stem)
    m = re.search(
        r"^(?P<experiment>.+?)_(?P<subexperiment>.+?)_trial(?P<trial>\d+)_rpi-(?P<rpi>\d+)_movie(?P<movie>\d+)(?P<rest>.*)$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError(
            f"Could not parse full movie identity from stem: {stem!r}. "
            "Expected something like: <experiment>_<subexperiment>_trial1_rpi-6_movie0_YYYYMMDDhhmmssPM"
        )

    rest = m.group("rest") or ""
    rest = rest.lstrip("_")
    tokens = [t for t in rest.split("_") if t]
    ts = None
    for t in tokens:
        if re.fullmatch(r"\d{14}(?:AM|PM)?", t, flags=re.IGNORECASE):
            ts = t
            break

    return {
        "experiment": str(m.group("experiment")),
        "subexperiment": str(m.group("subexperiment")),
        "trial_number": int(m.group("trial")),
        "rpi_number": int(m.group("rpi")),
        "movie_number": int(m.group("movie")),
        "timestamp": str(ts) if ts else None,
    }


def extract_label_prefix(*, cropped_stem: str, full_movie_stem: str, chamber_id: str) -> str:
    """Return the label before the supplied movie/chamber suffix, or an empty string."""

    suffix = f"{full_movie_stem}_{chamber_id}"
    if str(cropped_stem).endswith(suffix):
        return str(cropped_stem)[: -len(suffix)]
    return ""


def parse_genotype_sex_from_prefix(label_prefix: str) -> tuple[str | None, str | None]:
    """Return (genotype, sex) from a GENOTYPE_MALE_ or GENOTYPE_FEMALE_ prefix.

    Require whole sex tokens; either value may be None when unrecognized.
    """

    p = str(label_prefix or "")
    if not p:
        return None, None

    # Match whole sex tokens in the label prefix.
    match = re.fullmatch(r"(.+)_(FEMALE|MALE)_", p, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).upper()

    return None, None


# Registry read/write


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in DEFAULT_REGISTRY_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _canonicalize_column_order(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    ordered: list[str] = []
    for c in DEFAULT_REGISTRY_COLUMNS:
        if c in cols:
            ordered.append(c)
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    return df[ordered]


def _is_missing_scalar(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_relative_artifact_path(value: object) -> bool:
    if _is_missing_scalar(value):
        return True
    raw = str(value).strip()
    if not raw:
        return True
    posix = PurePosixPath(raw.replace("\\", "/"))
    windows = PureWindowsPath(raw)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
        and ".." not in windows.parts
    )


def validate_registry_dataframe(df: pd.DataFrame, *, source: Path | None = None) -> None:
    """Validate row identity, control values and artifact storage locations.

    Allow missing optional values. Relative paths must stay within their
    declared project or external storage root, and rows must be unambiguous.
    """

    if df.empty:
        return

    location = f" {Path(source)}" if source is not None else ""
    errors: list[str] = []

    if CANONICAL_ID_KEY not in df.columns:
        errors.append(f"missing required column {CANONICAL_ID_KEY!r}")
    else:
        identifiers = df[CANONICAL_ID_KEY]
        missing_ids = identifiers.map(
            lambda value: _is_missing_scalar(value) or not str(value).strip()
        )
        if bool(missing_ids.any()):
            examples = [str(index) for index in df.index[missing_ids][:10]]
            errors.append(f"blank recording_id at row index(es) {examples}")
        normalized_ids = identifiers.map(
            lambda value: "" if _is_missing_scalar(value) else str(value).strip()
        )
        duplicated = normalized_ids.ne("") & normalized_ids.duplicated(keep=False)
        if bool(duplicated.any()):
            examples = sorted(set(normalized_ids[duplicated].tolist()))[:10]
            errors.append(f"duplicate recording_id value(s): {examples}")

    if "active" in df.columns:
        invalid_active: list[str] = []
        for value in df["active"]:
            if _is_missing_scalar(value) or isinstance(value, bool):
                continue
            normalized = str(value).strip().lower()
            if normalized in {"1.0", "0.0"}:
                continue
            if normalized not in _TRUE_ACTIVE_VALUES | _FALSE_ACTIVE_VALUES:
                invalid_active.append(str(value))
        if invalid_active:
            errors.append(
                "unknown active value(s): "
                f"{sorted(set(invalid_active))[:10]}; use true/false, yes/no, or 1/0"
            )

    relative_columns = [column for column in df.columns if column.endswith("_relpath")]
    for column in relative_columns:
        invalid = [
            str(value)
            for value in df[column]
            if not _validate_relative_artifact_path(value)
        ]
        if invalid:
            errors.append(
                f"{column} contains absolute or parent-traversing path(s): "
                f"{sorted(set(invalid))[:5]}"
            )
        root_column = _storage_column(column)
        roots = df[root_column].tolist() if root_column in df.columns else ["."] * len(df)
        for value, root_value in zip(df[column].tolist(), roots):
            if _is_missing_scalar(value) or not str(value).strip():
                continue
            raw = str(value).replace("\\", "/")
            storage_root = "." if _is_missing_scalar(root_value) or not str(root_value).strip() else str(root_value)
            parts = PurePosixPath(raw).parts
            external = bool(parts and parts[0] == _STORAGE_PREFIX)
            if not _valid_storage_root(storage_root) or (
                external and (len(parts) < 3 or storage_root == "." or parts[1] != _storage_key(storage_root))
            ) or (not external and storage_root != "."):
                errors.append(f"{column} has missing or inconsistent storage location metadata: {raw}")
                break

    for column in _UNIQUE_ARTIFACT_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].map(
            lambda value: "" if _is_missing_scalar(value) else str(value).strip()
        )
        duplicated = values.ne("") & values.duplicated(keep=False)
        if bool(duplicated.any()):
            examples = sorted(set(values[duplicated].tolist()))[:5]
            errors.append(f"{column} is assigned to multiple recordings: {examples}")

    numeric_contracts = {
        "trial_number": (1, None),
        "rpi_number": (1, None),
        "movie_number": (0, None),
        "chamber_number": (1, 6),
    }
    for column, (minimum, maximum) in numeric_contracts.items():
        if column not in df.columns:
            continue
        invalid_values: list[str] = []
        for value in df[column]:
            if _is_missing_scalar(value) or not str(value).strip():
                continue
            try:
                numeric = float(value)
                valid = numeric.is_integer() and numeric >= minimum
                if maximum is not None:
                    valid = valid and numeric <= maximum
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                invalid_values.append(str(value))
        if invalid_values:
            bounds = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
            errors.append(
                f"{column} must contain integer values {bounds}: "
                f"{sorted(set(invalid_values))[:10]}"
            )

    if errors:
        raise RegistryValidationError(
            f"Invalid project registry{location}: " + "; ".join(errors)
        )


def read_registry(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        df = pd.DataFrame(columns=DEFAULT_REGISTRY_COLUMNS)
        return _canonicalize_column_order(df)
    try:
        numeric_or_control = {"trial_number", "rpi_number", "movie_number", "chamber_number", "active"}
        text_columns = {column: str for column in DEFAULT_REGISTRY_COLUMNS if column not in numeric_or_control}
        df = pd.read_csv(csv_path, dtype=text_columns, keep_default_na=False, na_values=[""])
    except Exception as exc:
        raise RegistryValidationError(
            f"Existing project registry could not be read: {csv_path}: {exc}"
        ) from exc
    df = _ensure_columns(df)
    df = _canonicalize_column_order(df)
    validate_registry_dataframe(df, source=csv_path)
    return df


def upsert_registry_rows(
    *,
    csv_path: Path,
    rows: Iterable[dict[str, Any]],
    key: str = CANONICAL_ID_KEY,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Merge rows by key while retaining existing values for empty updates.

    Preserve created_at for existing rows, add missing schema columns, and
    publish the validated CSV and provenance under a process-held file lock.
    """

    csv_path = Path(csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lock_dir = csv_path.parent / ".autobehaver_registry.lock"
    with RegistryLock(lock_dir, timeout_sec=lock_timeout_sec, stale_sec=lock_stale_sec):
        df_old = read_registry(csv_path)

        rows_list = []
        for supplied in rows:
            row = dict(supplied)
            for column, value in list(row.items()):
                if column.endswith("_relpath") and isinstance(value, RegistryArtifactPath):
                    row[_storage_column(column)] = value.storage_root
                    row[column] = str(value)
                elif (column.endswith("_relpath") and not _is_missing_scalar(value)
                      and str(value).strip() and _storage_column(column) not in row
                      and PurePosixPath(str(value).replace("\\", "/")).parts[0] != _STORAGE_PREFIX):
                    row[_storage_column(column)] = "."
            rows_list.append(row)
        if not rows_list:
            return

        df_new = pd.DataFrame(rows_list)
        df_new = _ensure_columns(df_new)
        df_new = _canonicalize_column_order(df_new)
        validate_registry_dataframe(df_new, source=csv_path)

        df_new = df_new.replace({"": pd.NA})

        if key not in df_new.columns:
            raise ValueError(f"New registry rows are missing key column {key!r}")

        if df_old.empty:
            df_out = df_new.copy()
        else:
            df_old = _ensure_columns(df_old)
            df_old = _canonicalize_column_order(df_old)

            old = df_old.set_index(key, drop=False)
            new = df_new.set_index(key, drop=False)

            # Preserve created_at for existing rows (write-once semantics).
            created_at_old = None
            if "created_at" in old.columns:
                try:
                    created_at_old = old["created_at"].copy()
                except Exception:
                    created_at_old = None

            # Empty CSV columns may be inferred as float by pandas. Registry
            # fields mix paths, identifiers and numeric metadata; explicit object
            # storage permits later string values without changing their content.
            old = old.astype(object)
            new = new.astype(object)
            old = old.reindex(columns=list(dict.fromkeys([*old.columns, *new.columns]))).astype(object)
            old.update(new)

            if created_at_old is not None and "created_at" in old.columns:
                try:
                    mask = created_at_old.notna()
                    old.loc[mask, "created_at"] = created_at_old.loc[mask]
                except Exception:
                    pass

            missing_keys = [k for k in new.index if k not in old.index]
            if missing_keys:
                old = pd.concat([old, new.loc[missing_keys]], axis=0)

            df_out = old.reset_index(drop=True)

        # Stable sort (helps manual browsing). Prefer full_movie_relpath then chamber.
        sort_cols = [c for c in ("full_movie_relpath", "chamber_number") if c in df_out.columns]
        if sort_cols:
            try:
                df_out = df_out.sort_values(sort_cols, kind="stable")
            except Exception:
                pass

        if "created_at" in df_out.columns:
            try:
                if "updated_at" in df_out.columns:
                    df_out["created_at"] = df_out["created_at"].fillna(df_out["updated_at"])
            except Exception:
                pass

        df_out = _canonicalize_column_order(df_out)
        validate_registry_dataframe(df_out, source=csv_path)
        atomic_write_dataframe_csv(df_out, csv_path, index=False)
        atomic_write_json(
            {
                "schema": "autobehaver.registry.provenance.v1",
                "status": "complete",
                "autobehaver_version": __version__,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "registry": {
                    "filename": csv_path.name,
                    "sha256": sha256_file(csv_path),
                    "row_count": int(len(df_out)),
                    "columns": [str(column) for column in df_out.columns],
                    "key": key,
                },
            },
            csv_path.with_suffix(csv_path.suffix + ".provenance.json"),
        )
