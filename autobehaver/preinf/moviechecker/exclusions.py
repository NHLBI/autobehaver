"""Read and write six-chamber exclusions for paired full-movie recordings."""

from __future__ import annotations

import csv
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


EXCLUSIONS_HEADER = [
    "pair_key",
    "reviewed_movie",
    "exclude_ch1",
    "exclude_ch2",
    "exclude_ch3",
    "exclude_ch4",
    "exclude_ch5",
    "exclude_ch6",
    "updated_at",
]


@dataclass
class ExclusionRecord:
    """Store one movie-pair identity and six binary chamber exclusion flags."""
    pair_key: str
    reviewed_movie: str
    exclude_ch: List[int]  # length 6, values 0/1
    updated_at: float

    def to_row(self) -> Dict[str, str]:
        """Serialize the record into the exclusions CSV column layout."""
        row: Dict[str, str] = {
            "pair_key": self.pair_key,
            "reviewed_movie": self.reviewed_movie,
            "updated_at": f"{self.updated_at:.3f}",
        }
        for i in range(6):
            row[f"exclude_ch{i+1}"] = str(int(self.exclude_ch[i]))
        return row

    @staticmethod
    def from_row(row: Dict[str, str]) -> "ExclusionRecord":
        """Parse and validate one exclusions row."""
        pair_key = str(row.get("pair_key", "")).strip()
        if not pair_key:
            raise ValueError("pair_key is blank")
        reviewed_movie = str(row.get("reviewed_movie", "")).strip()
        exclude_ch: List[int] = []
        for i in range(6):
            column = f"exclude_ch{i+1}"
            v = str(row.get(column, "")).strip().lower()
            if v not in {"0", "1", "0.0", "1.0", "false", "true", "no", "yes", "n", "y", "off", "on"}:
                raise ValueError(f"{pair_key}: {column} must be 0/1, false/true or no/yes; got {row.get(column)!r}")
            exclude_ch.append(int(v in {"1", "1.0", "true", "yes", "y", "on"}))
        try:
            updated_at = float(row.get("updated_at", "0") or 0.0)
        except Exception:
            updated_at = 0.0
        return ExclusionRecord(
            pair_key=pair_key,
            reviewed_movie=reviewed_movie,
            exclude_ch=exclude_ch,
            updated_at=updated_at,
        )


class ExclusionsStore:
    """Cache exclusions and write the complete review state to CSV.

    Retry atomic replacement for temporary storage locks, then try direct
    overwrite for shares that reject rename-over-existing. Preserve a complete
    temporary recovery copy when both approaches fail.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records: Dict[str, ExclusionRecord] = {}

    def load(self) -> None:
        """Load validated exclusions, rejecting conflicting duplicate pair keys."""
        loaded: Dict[str, ExclusionRecord] = {}
        if not os.path.exists(self.csv_path):
            self.records = loaded
            return

        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            required = ["pair_key", *(f"exclude_ch{i}" for i in range(1, 7))]
            missing = [column for column in required if column not in fields]
            if missing or len(fields) != len(set(fields)):
                raise ValueError(f"{self.csv_path}: invalid exclusions header; missing columns {missing}; duplicate columns are not permitted")
            for row_number, row in enumerate(reader, start=2):
                try:
                    rec = ExclusionRecord.from_row(row)
                except ValueError as exc:
                    raise ValueError(f"{self.csv_path}, row {row_number}: {exc}") from exc
                previous = loaded.get(rec.pair_key)
                if previous is not None and previous.exclude_ch != rec.exclude_ch:
                    raise ValueError(f"{self.csv_path}, row {row_number}: conflicting exclusions for duplicate pair_key {rec.pair_key!r}")
                if previous is None or rec.updated_at >= previous.updated_at:
                    loaded[rec.pair_key] = rec
        self.records = loaded

    def has(self, pair_key: str) -> bool:
        return pair_key in self.records

    def get(self, pair_key: str) -> Optional[ExclusionRecord]:
        return self.records.get(pair_key)

    def upsert(self, pair_key: str, reviewed_movie: str, exclude_ch: List[int]) -> None:
        """Replace the in-memory review record for one movie pair."""
        if len(exclude_ch) != 6:
            raise ValueError("exclude_ch must be length 6")
        self.records[pair_key] = ExclusionRecord(
            pair_key=pair_key,
            reviewed_movie=reviewed_movie,
            exclude_ch=[int(bool(x)) for x in exclude_ch],
            updated_at=time.time(),
        )

    def _write_csv_file(self, path: str) -> None:
        """Write the complete sorted exclusions table and flush buffered data."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXCLUSIONS_HEADER)
            writer.writeheader()
            for key in sorted(self.records.keys()):
                writer.writerow(self.records[key].to_row())
            # Flush so a successful replace points at a complete file.  Some
            # network filesystems do not support fsync cleanly, so fsync failure
            # should not turn a successful CSV write into a hard failure.
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    @staticmethod
    def _replace_error_is_retryable(exc: OSError) -> bool:
        # WinError 5 = access denied; 32/33 = sharing/lock-style violations.
        # On non-Windows platforms these attributes are usually absent, but
        # PermissionError still covers the common retryable case.
        return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}

    def write_atomic(self, *, attempts: int = 8, delay_s: float = 0.25) -> None:
        """Save exclusions with atomic-replace retries and a direct-write fallback.

        Retain a complete recovery copy if publication fails.
        """
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        tmp_dir = os.path.dirname(self.csv_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="exclusions_", suffix=".csv", dir=tmp_dir)
        os.close(fd)

        remove_tmp = True
        try:
            self._write_csv_file(tmp_path)

            n_attempts = max(1, int(attempts))
            for attempt in range(n_attempts):
                try:
                    os.replace(tmp_path, self.csv_path)
                    remove_tmp = False  # os.replace moved it; there is no temp file left.
                    return
                except OSError as exc:
                    if attempt >= n_attempts - 1 or not self._replace_error_is_retryable(exc):
                        break
                    time.sleep(max(0.0, float(delay_s)) * (attempt + 1))

            # Some Windows/SMB shares reject rename-over-existing but allow
            # direct writes. Retain a recovery copy if this fallback also fails.
            try:
                self._write_csv_file(self.csv_path)
                return
            except OSError as direct_exc:
                remove_tmp = False
                msg = (
                    "Failed to update exclusions.csv after retrying atomic replace "
                    "and direct overwrite. The destination may be locked, read-only, "
                    "or temporarily unavailable on the network share. "
                    f"Destination: {self.csv_path!r}. Recovery copy with the latest "
                    f"review state: {tmp_path!r}. Close any program using the CSV, "
                    "then rename/copy the recovery file to exclusions.csv if needed."
                )
                # Report the final publication error with the recovery path.
                raise RuntimeError(msg) from direct_exc
        finally:
            if remove_tmp:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
