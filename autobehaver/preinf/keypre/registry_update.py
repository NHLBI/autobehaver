"""Project-registry update for the keypre stage.

Keypre consumes DLC output CSVs and produces per-video positional-calcs CSVs.
In the Autobehaver pipeline we record the primary output path for each animal
in the **project registry**.

This helper is called once per keypre run (in the parent process) to avoid
multiprocessing contention.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows


def _expected_keypre_name(dlc_csv: Path) -> str:
    """Return the expected keypre output filename for a DLC CSV."""
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
    """Upsert keypre outputs into the project registry."""

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

    # Map dlc_csv_relpath -> recording_id
    rel_to_id: dict[str, str] = {}
    for _, r in df.iterrows():
        rel = r.get("dlc_csv_relpath")
        rid = r.get("recording_id")
        if isinstance(rel, str) and rel.strip() and isinstance(rid, str) and rid.strip():
            rel_to_id[rel.replace("\\", "/")] = rid

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for dlc_csv in dlc_csv_files:
        try:
            inp = Path(dlc_csv).expanduser().resolve()
        except Exception:
            inp = Path(dlc_csv)

        rel = relpath_str(inp, project_root)
        rid = rel_to_id.get(rel)

        if not rid:
            # Fallback: match by basename if registry was created on a different mirror.
            bn = inp.name
            matches = [k for k in rel_to_id.keys() if Path(k).name == bn]
            if len(matches) == 1:
                rid = rel_to_id[matches[0]]

        if not rid:
            continue

        out_path = output_dir / _expected_keypre_name(inp)
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
