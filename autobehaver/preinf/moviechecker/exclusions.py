"""Read/write exclusions CSV for moviechecker.

The exclusions file is intentionally a simple CSV so users can manually adjust
it if file paths/names change.

Each row corresponds to a *paired* full-movie unit (movie0/movie1).
"""

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
    pair_key: str
    reviewed_movie: str
    exclude_ch: List[int]  # length 6, values 0/1
    updated_at: float

    def to_row(self) -> Dict[str, str]:
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
        pair_key = str(row.get("pair_key", "")).strip()
        reviewed_movie = str(row.get("reviewed_movie", "")).strip()
        exclude_ch: List[int] = []
        for i in range(6):
            v = str(row.get(f"exclude_ch{i+1}", "0")).strip()
            exclude_ch.append(1 if v in {"1", "true", "True", "YES", "yes"} else 0)
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
    """A small in-memory cache + CSV writer.

    The normal save path writes a complete temporary CSV and atomically replaces
    ``exclusions.csv``.  Windows network drives can occasionally refuse the
    final replace if the destination file is briefly locked by another process
    or if the share allows writing but not rename-over-existing.  To make manual
    review less fragile, saving now retries the replace and then falls back to a
    direct overwrite before giving up.  If every save strategy fails, the fully
    written temporary CSV is deliberately left behind as a recovery file.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records: Dict[str, ExclusionRecord] = {}

    def load(self) -> None:
        self.records = {}
        if not os.path.exists(self.csv_path):
            return

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = ExclusionRecord.from_row(row)
                if rec.pair_key:
                    self.records[rec.pair_key] = rec

    def has(self, pair_key: str) -> bool:
        return pair_key in self.records

    def get(self, pair_key: str) -> Optional[ExclusionRecord]:
        return self.records.get(pair_key)

    def upsert(self, pair_key: str, reviewed_movie: str, exclude_ch: List[int]) -> None:
        if len(exclude_ch) != 6:
            raise ValueError("exclude_ch must be length 6")
        self.records[pair_key] = ExclusionRecord(
            pair_key=pair_key,
            reviewed_movie=reviewed_movie,
            exclude_ch=[int(bool(x)) for x in exclude_ch],
            updated_at=time.time(),
        )

    def _write_csv_file(self, path: str) -> None:
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

            # Some Windows/SMB shares allow ordinary writes but reject atomic
            # rename-over-existing.  This is not atomic, but it is better than
            # crashing during manual review when the source of truth is a small
            # user-editable CSV.
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
                # Chain the direct write error, because it is the last operation
                # attempted.  The message tells the user the replace was also
                # retried first.
                raise RuntimeError(msg) from direct_exc
        finally:
            if remove_tmp:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
