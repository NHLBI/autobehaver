"""Record PosePred prediction paths and verified model identities in the registry.

Update once in the parent process to avoid worker contention.
"""

from __future__ import annotations

import logging
import json
from autobehaver.portable_paths import prediction_path
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Set, List

from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows


def _expected_inference_name(norm_h5: Path) -> str:
    """Return the expected inference CSV filename for a normalized H5."""
    return f"{norm_h5.stem}_inference.csv"


def update_registry_from_posepred_inference_outputs(
    *,
    registry_csv: Path,
    project_root: Path,
    normalized_h5_files: Iterable[Path],
    output_root: Path,
    use_inferences_subdir: bool,
    model_paths: Optional[List[str]] = None,
    only_set_models_if_new: bool = True,
    existing_outputs_before: Optional[Set[str]] = None,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert prediction paths, preferring checkpoint identities from verified provenance."""

    registry_csv = Path(registry_csv).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if existing_outputs_before is None:
        existing_outputs_before = set()

    if not registry_csv.exists():
        logging.warning(
            f"Registry update (posepred_inference) skipped: registry CSV not found: {registry_csv}"
        )
        return

    df = read_registry(registry_csv)
    if df.empty:
        logging.info("Registry update (posepred_inference): registry is empty; nothing to update")
        return

    # Map keyprenorm_h5_relpath -> recording_id
    rel_to_id: dict[str, str] = {}
    for _, r in df.iterrows():
        rel = r.get("keyprenorm_h5_relpath")
        rid = r.get("recording_id")
        if isinstance(rel, str) and rel.strip() and isinstance(rid, str) and rid.strip():
            rel_to_id[rel.replace("\\", "/")] = rid

    inference_dir = (output_root / "inferences") if use_inferences_subdir else output_root

    # Store checkpoint basenames in the registry; full identities remain in provenance.
    models_str: Optional[str] = None
    if model_paths:
        try:
            models_str = ";".join([Path(p).name for p in model_paths])
        except Exception:
            models_str = ";".join([str(p) for p in model_paths])

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for norm_h5 in normalized_h5_files:
        try:
            inp = Path(norm_h5).expanduser().resolve()
        except Exception:
            inp = Path(norm_h5)

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

        out_path = prediction_path(inp, inference_dir)
        if not out_path.exists():
            continue

        # Determine whether this file was newly generated in this run.
        try:
            out_res = str(out_path.resolve())
        except Exception:
            out_res = str(out_path)
        is_new = out_res not in existing_outputs_before

        # Regenerated outputs can reuse a path; verified provenance identifies
        # the checkpoint content that actually produced them.
        actual_models = None
        manifest = Path(str(out_path) + '.provenance.json')
        if manifest.is_file():
            from autobehaver.inference import _completed_inference_matches
            from autobehaver.atomic_io import sha256_file
            try:
                metadata = json.loads(manifest.read_text(encoding='utf-8'))
                checkpoint_records = metadata.get('inputs', {}).get('checkpoints', [])
                if (metadata.get('fingerprint_sha256') and checkpoint_records
                    and metadata.get('inputs', {}).get('source_hdf5', {}).get('sha256') == sha256_file(inp)
                    and _completed_inference_matches(str(manifest), metadata)):
                    actual_models = ';'.join(Path(item['path']).name for item in checkpoint_records)
            except (OSError, ValueError, TypeError, KeyError):
                pass

        row: dict[str, Any] = {
            "recording_id": rid,
            "posepred_inference_csv_relpath": relpath_str(out_path, project_root),
            "updated_at": now,
        }

        if actual_models:
            row['posepred_models'] = actual_models
        elif models_str and (not only_set_models_if_new or is_new):
            row["posepred_models"] = models_str

        rows.append(row)

    if not rows:
        logging.info("Registry update (posepred_inference): no inference outputs found to upsert")
        return

    upsert_registry_rows(
        csv_path=registry_csv,
        rows=rows,
        lock_timeout_sec=lock_timeout_sec,
        lock_stale_sec=lock_stale_sec,
    )
    logging.info(
        f"Registry updated (posepred_inference): {registry_csv} (+{len(rows)} rows upserted)"
    )
