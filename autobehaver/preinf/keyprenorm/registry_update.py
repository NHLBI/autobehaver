"""Record normalized feature and parameter paths in the project registry.

Update once in the parent process to avoid worker contention.
"""

from __future__ import annotations

import logging
from autobehaver.portable_paths import normalized_path
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows


def _expected_norm_h5_name(keypre_csv: Path) -> str:
    """Return the expected normalized HDF5 filename for a KeyPre CSV."""
    name = keypre_csv.name
    if name.lower().endswith(".csv"):
        return name[:-4] + "_normalized.h5"
    return name + "_normalized.h5"


def update_registry_from_keyprenorm_outputs(
    *,
    registry_csv: Path,
    project_root: Path,
    keypre_csv_files: Iterable[Path],
    normalized_h5_dir: Path,
    norm_params_path: Path,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert KeyPreNorm output and parameter paths for the supplied feature CSVs."""

    registry_csv = Path(registry_csv).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    normalized_h5_dir = Path(normalized_h5_dir).expanduser().resolve()
    norm_params_path = Path(norm_params_path).expanduser().resolve()

    if not registry_csv.exists():
        logging.warning(f"Registry update (keyprenorm) skipped: registry CSV not found: {registry_csv}")
        return

    df = read_registry(registry_csv)
    if df.empty:
        logging.info("Registry update (keyprenorm): registry is empty; nothing to update")
        return

    # Map keypre_csv_relpath -> recording_id
    rel_to_id: dict[str, str] = {}
    for _, r in df.iterrows():
        rel = r.get("keypre_csv_relpath")
        rid = r.get("recording_id")
        if isinstance(rel, str) and rel.strip() and isinstance(rid, str) and rid.strip():
            rel_to_id[rel.replace("\\", "/")] = rid

    params_rel = None
    if norm_params_path.exists():
        params_rel = relpath_str(norm_params_path, project_root)

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for keypre_csv in keypre_csv_files:
        try:
            inp = Path(keypre_csv).expanduser().resolve()
        except Exception:
            inp = Path(keypre_csv)

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

        out_path = normalized_path(inp, normalized_h5_dir)
        if not out_path.exists():
            continue

        row: dict[str, Any] = {
            "recording_id": rid,
            "keyprenorm_h5_relpath": relpath_str(out_path, project_root),
            "updated_at": now,
        }
        if params_rel:
            row["keyprenorm_params_relpath"] = params_rel
        rows.append(row)

    if not rows:
        logging.info("Registry update (keyprenorm): no normalized outputs found to upsert")
        return

    upsert_registry_rows(
        csv_path=registry_csv,
        rows=rows,
        lock_timeout_sec=lock_timeout_sec,
        lock_stale_sec=lock_stale_sec,
    )
    logging.info(f"Registry updated (keyprenorm): {registry_csv} (+{len(rows)} rows upserted)")
