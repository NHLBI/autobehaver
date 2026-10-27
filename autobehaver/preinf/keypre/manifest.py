"""Record KeyPre feature specifications, runtime options, paths, and outcomes."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from autobehaver import __version__ as AUTOBEHAVER_VERSION

from . import settings
from . import spec as spec_mod


def _jsonable(obj):
    """Convert nested containers to JSON-serializable values."""

    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def _keypre_spec_snapshot() -> dict:
    """Return a serializable snapshot of KeyPre settings and ordered feature specifications."""

    snapshot = {
        "autobehaver_version": AUTOBEHAVER_VERSION,
        "settings": {
            "include_position": settings.include_position,
            "rolling_windows": list(settings.rolling_windows),
            "include_likelihoods": settings.include_likelihoods,
            "nan_edge_effects": settings.nan_edge_effects,
            "edge_skip": settings.edge_skip,
            "warn_on_internal_nans": settings.warn_on_internal_nans,
        },
        "spec": {
            "parts": list(spec_mod.parts),
            "length_pairs": list(spec_mod.length_pairs),
            "bodypart_triplets": list(spec_mod.bodypart_triplets),
            "keypoints_velocity_parts": list(spec_mod.keypoints_velocity_parts),
            "angular_velocity_pairs": list(spec_mod.angular_velocity_pairs),
            "stance_area_vertices": list(spec_mod.stance_area_vertices),
            "body_area_top_vertices": list(spec_mod.body_area_top_vertices),
            "body_area_side_vertices": list(spec_mod.body_area_side_vertices),
            "asymmetry_lr_pairs": list(spec_mod.asymmetry_lr_pairs),
            "asymmetry_tb_pairs": list(spec_mod.asymmetry_tb_pairs),
        },
    }
    return _jsonable(snapshot)


def _sha256_json(payload: dict) -> str:
    """Hash a dictionary using stable JSON serialization."""

    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str | None:
    """Return the file SHA256 digest, or None when unavailable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON to a temporary file and atomically replace the destination."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _build_run_manifest(
    *,
    args: argparse.Namespace,
    input_folder: str,
    csv_source_folder: str,
    out_dir: str,
    csv_files: list[str],
    start_time_iso: str,
    entrypoint: str | None = None,
) -> tuple[str, dict]:
    """Write the initial run manifest and return its path and dictionary."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    manifest_path = os.path.join(out_dir, f"keypre_run_manifest_{run_id}.json")

    # Prefer the actual entrypoint, if provided.
    script_path = None
    if entrypoint:
        try:
            abs_entry = os.path.abspath(entrypoint)
            if os.path.exists(abs_entry):
                script_path = abs_entry
        except Exception:
            script_path = None
    if script_path is None:
        script_path = os.path.abspath(__file__) if "__file__" in globals() else None

    spec_snapshot = _keypre_spec_snapshot()
    spec_sha256 = _sha256_json(spec_snapshot)
    manifest: dict = {
        "run_id": run_id,
        "script": {
            "name": os.path.basename(script_path) if script_path else None,
            "version": f"autobehaver_keypre@{AUTOBEHAVER_VERSION}",
            "path": script_path,
            "sha256": _sha256_file(script_path) if script_path else None,
        },
        "keypre_spec": {
            "sha256": spec_sha256,
            "snapshot": spec_snapshot,
        },
        "started_at": start_time_iso,
        "status": "running",
        "args": vars(args),
        "paths": {
            "input_folder": os.path.abspath(input_folder),
            "csv_source_folder": os.path.abspath(csv_source_folder),
            "output_folder": os.path.abspath(out_dir),
        },
        "discovery": {
            "csv_count": len(csv_files),
            "csv_pattern": os.path.join(csv_source_folder, "*.csv"),
            "example_csv_files": [os.path.basename(p) for p in csv_files[:10]],
            "csv_files": [os.path.abspath(p) for p in csv_files],
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "results": None,
        "finished_at": None,
        "duration_seconds": None,
        "error": None,
    }

    _atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest
