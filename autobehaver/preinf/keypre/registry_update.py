"""Record KeyPre positional-calculations outputs in the project registry.

Update once in the parent process to avoid worker contention.
"""

from __future__ import annotations

import logging
from .contract import output_path
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import read_registry, relpath_str, resolve_registry_path, upsert_registry_rows


def _expected_keypre_name(dlc_csv: Path) -> str:
    """Return the expected KeyPre output filename for a DeepLabCut CSV."""
    name = dlc_csv.name
    if name.lower().endswith(".csv"):
        return name[:-4] + "_positional_calcs.csv"
    return name + "_positional_calcs.csv"


def update_registry_from_keypre_outputs(
    *,
    registry_csv: Path,
    project_root: Path,
    dlc_csv_files: Iterable[Path],
    output_dir: Path,
    keypre_version: str | None,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert KeyPre output paths for the supplied DeepLabCut CSVs."""

    registry_csv = Path(registry_csv).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not registry_csv.exists():
        logging.warning(f"Registry update (keypre) skipped: registry CSV not found: {registry_csv}")
        return

    df = read_registry(registry_csv)
    if df.empty:
        logging.info("Registry update (keypre): registry is empty; nothing to update")
        return

    # Bind to the exact registry-resolved artifact, including external storage roots.
    rel_to_id: dict[str, str] = {}
    for _, r in df.iterrows():
        rel = r.get("dlc_csv_relpath")
        rid = r.get("recording_id")
        if isinstance(rel, str) and rel.strip() and isinstance(rid, str) and rid.strip():
            path = str(resolve_registry_path(project_root, r, 'dlc_csv_relpath'))
            if path in rel_to_id and rel_to_id[path] != rid:
                raise ValueError(f'Multiple recording IDs refer to DLC input: {path}')
            rel_to_id[path] = rid

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for dlc_csv in dlc_csv_files:
        try:
            inp = Path(dlc_csv).expanduser().resolve()
        except Exception:
            inp = Path(dlc_csv)

        rid = rel_to_id.get(str(inp))

        if not rid:
            continue

        out_path = output_path(inp, output_dir)
        if not out_path.exists():
            continue

        row: dict[str, Any] = {
            "recording_id": rid,
            "keypre_csv_relpath": relpath_str(out_path, project_root),
            "updated_at": now,
        }
        if keypre_version:
            row["keypre_version"] = str(keypre_version)
        rows.append(row)

    if not rows:
        logging.info("Registry update (keypre): no keypre outputs found to upsert")
        return

    upsert_registry_rows(
        csv_path=registry_csv,
        rows=rows,
        lock_timeout_sec=lock_timeout_sec,
        lock_stale_sec=lock_stale_sec,
    )
    logging.info(f"Registry updated (keypre): {registry_csv} (+{len(rows)} rows upserted)")
