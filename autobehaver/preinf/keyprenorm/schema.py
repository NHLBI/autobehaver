"""Read feature headers and identify KeyPreNorm schema mismatches.

Canonical order comes from the first input or normalization parameters.
The coordinator rejects mismatched column sets before normalization; matching
sets in a different order are reindexed.
"""

from __future__ import annotations

import logging
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("keyprenorm")



def find_processing_log(input_dir: str) -> Optional[str]:
    """Locate an existing processing log file in a directory.

    Returns the first file ending with "_processing_log.txt", or None.
    """
    try:
        for file in os.listdir(input_dir):
            if file.endswith("_processing_log.txt"):
                return os.path.join(input_dir, file)
    except FileNotFoundError:
        return None
    return None


def read_csv_header_columns(csv_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Read a CSV header and return (columns, error), with columns None on failure."""
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as handle:
            cols = next(csv.reader(handle))
        if not cols or any(not c.strip() for c in cols) or len(cols) != len(set(cols)):
            raise ValueError('Blank or duplicate feature headers are not a valid schema.')
        return cols, None
    except Exception as e:
        return None, str(e)


def filter_files_by_schema(
    csv_files: Sequence[str],
    canonical_columns: Sequence[str],
    max_examples: int = 10,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Select files with the canonical column set and report every rejected input.

    Accept different column ordering because normalization reindexes it. Return
    (good_files, bad_details) for the coordinator to enforce the stage contract.
    """
    canonical_set = set(canonical_columns)
    good: List[str] = []
    bad: List[Dict[str, object]] = []

    for csv_path in csv_files:
        cols, err = read_csv_header_columns(csv_path)
        if err is not None or cols is None:
            bad.append({
                "file": csv_path,
                "reason": "header_read_error",
                "missing": [],
                "extra": [],
                "error": err,
            })
            continue

        col_set = set(cols)
        missing = sorted(canonical_set - col_set)
        extra = sorted(col_set - canonical_set)

        if missing or extra:
            bad.append({
                "file": csv_path,
                "reason": "schema_mismatch",
                "missing": missing,
                "extra": extra,
                "error": None,
            })
        else:
            good.append(csv_path)

    if bad:
        banner = "=" * 90
        logger.warning(banner)
        logger.warning("Some input CSV files do NOT match the canonical column schema; they will be SKIPPED.")
        logger.warning(
            "Canonical columns: %d | Bad files: %d / %d | Good files: %d / %d",
            len(canonical_columns),
            len(bad),
            len(csv_files),
            len(good),
            len(csv_files),
        )

        by_reason: Dict[str, int] = {}
        for item in bad:
            by_reason[str(item["reason"])] = by_reason.get(str(item["reason"]), 0) + 1
        for reason, count in by_reason.items():
            logger.warning(" - %s: %d", reason, count)

        logger.warning("Examples (showing up to %d):", max_examples)
        shown = 0
        for item in bad:
            if shown >= max_examples:
                break
            base = os.path.basename(str(item["file"]))
            if item["reason"] == "header_read_error":
                logger.warning(" - %s: header read error: %s", base, item.get("error"))
            else:
                missing = item.get("missing", [])
                extra = item.get("extra", [])
                logger.warning(" - %s: missing %d cols, extra %d cols", base, len(missing), len(extra))
                if missing:
                    logger.warning("   missing (first 10): %s", missing[:10])
                if extra:
                    logger.warning("   extra   (first 10): %s", extra[:10])
            shown += 1
        if len(bad) > max_examples:
            logger.warning("... (showing first %d of %d)", max_examples, len(bad))
        logger.warning(banner)

    return good, bad


def resolve_glob_pattern(input_folder: str, pattern: str) -> str:
    """Resolve a glob pattern relative to input_folder unless pattern is absolute."""
    if os.path.isabs(pattern):
        return pattern
    return os.path.join(input_folder, pattern)


def summarize_bad_schema(bad_schema: List[Dict[str, object]], max_items: int = 50) -> List[Dict[str, object]]:
    """Limit schema-rejection details for run records."""
    out: List[Dict[str, object]] = []
    for item in bad_schema[:max_items]:
        missing = item.get("missing", []) or []
        extra = item.get("extra", []) or []
        out.append({
            "file": os.path.basename(str(item.get("file"))),
            "reason": item.get("reason"),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_first10": missing[:10],
            "extra_first10": extra[:10],
            "error": item.get("error"),
        })
    return out
