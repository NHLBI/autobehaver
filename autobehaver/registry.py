"""Project registry helpers.

The **project registry** is a single CSV (usually at the project root) that
links each *cropped chamber movie* (i.e., each animal) to downstream artifacts
(DLC outputs, keypoint features, normalized features, posepred inference, etc.).

Design goals
------------
* One file per project (no sidecar-per-movie clutter).
* Paths stored **relative to project root** so the registry can be moved between
  HPC/local mirrors.
* Safe to update incrementally from multiple pipeline stages via a simple
  directory lock.

This module intentionally focuses on **mechanics** (locking, upsert) and a few
small parsing helpers. Stage-specific logic (e.g., how autocropper discovers
cropped outputs) lives in the stage modules.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------


DEFAULT_REGISTRY_COLUMNS: list[str] = [
    # Identity
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
    # Artifacts (relative paths)
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Locking
# -----------------------------------------------------------------------------


class RegistryLock:
    """A simple directory lock (cross-platform, no fcntl required)."""

    def __init__(self, lock_dir: Path, *, timeout_sec: int = 120, stale_sec: int = 3600):
        self.lock_dir = Path(lock_dir)
        self.timeout_sec = int(timeout_sec)
        self.stale_sec = int(stale_sec)

    def acquire(self) -> None:
        start = time.time()
        while True:
            try:
                self.lock_dir.mkdir(parents=False, exist_ok=False)
                return
            except FileExistsError:
                # If lock is stale, remove it.
                try:
                    age = time.time() - self.lock_dir.stat().st_mtime
                    if age > self.stale_sec:
                        logging.warning(
                            f"Registry lock appears stale (age={age:.0f}s). Removing: {self.lock_dir}"
                        )
                        try:
                            for p in self.lock_dir.glob("*"):
                                if p.is_file():
                                    p.unlink()
                            self.lock_dir.rmdir()
                        except Exception:
                            # Another process may be racing us; just continue waiting.
                            pass
                        continue
                except Exception:
                    # If we can't stat it, keep waiting.
                    pass

                if (time.time() - start) > self.timeout_sec:
                    raise TimeoutError(
                        f"Timed out waiting for registry lock after {self.timeout_sec}s: {self.lock_dir}"
                    )
                time.sleep(0.25)

    def release(self) -> None:
        try:
            if self.lock_dir.exists():
                for p in self.lock_dir.glob("*"):
                    if p.is_file():
                        try:
                            p.unlink()
                        except Exception:
                            pass
                self.lock_dir.rmdir()
        except Exception:
            # Best-effort; do not crash pipeline.
            pass

    def __enter__(self) -> "RegistryLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def relpath_str(path: Path, root: Path) -> str:
    """Return a POSIX-style relative path string."""

    try:
        rel = os.path.relpath(str(Path(path).resolve()), str(Path(root).resolve()))
    except Exception:
        # If relpath fails (different drives on Windows), fall back to basename.
        rel = Path(path).name
    # Normalize separators for CSV readability / cross-platform use.
    return rel.replace("\\", "/")


def parse_full_movie_stem(stem: str) -> dict[str, Any]:
    """Parse identity fields from a full-movie filename stem.

    Expected pattern (recommended):
        <experiment>_<subexperiment>_trial<trial>_rpi-<n>_movie<m>_<timestamp>

    Notes
    -----
    * `experiment`/`subexperiment` are parsed using a non-greedy regex up to
      `_trial`, allowing underscores inside those fields.
    * `timestamp` is detected as the first token after `_movie<m>_` that matches
      ``\\d{14}(AM|PM)?`` (e.g., `20250819015233PM`).
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
    """Extract the label prefix from a cropped filename stem.

    Cropped filename convention:
        <LABEL_PREFIX><full_movie_stem>_<chamber_id>.mp4

    If no label prefix is present, returns an empty string.
    """

    suffix = f"{full_movie_stem}_{chamber_id}"
    if str(cropped_stem).endswith(suffix):
        return str(cropped_stem)[: -len(suffix)]
    return ""


def parse_genotype_sex_from_prefix(label_prefix: str) -> tuple[str | None, str | None]:
    """Parse genotype + sex from a label prefix like `GENOTYPE_MALE_`.

    Returns (genotype, sex) where either may be None.
    """

    p = str(label_prefix or "")
    if not p:
        return None, None

    # Canonical lab convention: <genotype>_<sex>_
    up = p.upper()
    if up.endswith("MALE_"):
        genotype = p[: -len("MALE_")].rstrip("_")
        return genotype if genotype else None, "MALE"
    if up.endswith("FEMALE_"):
        genotype = p[: -len("FEMALE_")].rstrip("_")
        return genotype if genotype else None, "FEMALE"

    return None, None


# -----------------------------------------------------------------------------
# Registry read/write
# -----------------------------------------------------------------------------


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
    # Preserve any extra columns at the end.
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    return df[ordered]


def read_registry(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        df = pd.DataFrame(columns=DEFAULT_REGISTRY_COLUMNS)
        return _canonicalize_column_order(df)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logging.exception(f"Failed to read registry CSV: {csv_path}")
        df = pd.DataFrame(columns=DEFAULT_REGISTRY_COLUMNS)
    df = _ensure_columns(df)
    return _canonicalize_column_order(df)


def upsert_registry_rows(
    *,
    csv_path: Path,
    rows: Iterable[dict[str, Any]],
    key: str = CANONICAL_ID_KEY,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert rows into the registry CSV.

    Rules:
      * `key` identifies a unique row (`recording_id`).
      * New non-null values overwrite existing values.
      * Null/empty values do NOT overwrite existing values.
    """

    csv_path = Path(csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Hidden lock dir at project root to avoid clutter.
    lock_dir = csv_path.parent / ".autobehaver_registry.lock"
    with RegistryLock(lock_dir, timeout_sec=lock_timeout_sec, stale_sec=lock_stale_sec):
        df_old = read_registry(csv_path)

        rows_list = list(rows)
        if not rows_list:
            return

        df_new = pd.DataFrame(rows_list)
        df_new = _ensure_columns(df_new)
        df_new = _canonicalize_column_order(df_new)

        # Normalize empties -> NA so they don't overwrite via DataFrame.update
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

            # Update existing rows with non-NA values.
            old.update(new)

            if created_at_old is not None and "created_at" in old.columns:
                try:
                    mask = created_at_old.notna()
                    old.loc[mask, "created_at"] = created_at_old.loc[mask]
                except Exception:
                    pass

            # Append brand-new keys
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

        # Fill created_at for any brand-new rows that don't have it yet.
        if "created_at" in df_out.columns:
            try:
                if "updated_at" in df_out.columns:
                    df_out["created_at"] = df_out["created_at"].fillna(df_out["updated_at"])
            except Exception:
                pass

        df_out = _canonicalize_column_order(df_out)
        df_out.to_csv(csv_path, index=False)
