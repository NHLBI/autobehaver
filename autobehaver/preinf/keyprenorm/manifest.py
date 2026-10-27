"""Record KeyPreNorm runtime options, discovered inputs, paths, and outcomes."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import uuid
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger("keyprenorm")


def sha256_file(path: str) -> Optional[str]:
    """Return the file SHA256 digest, or None when unavailable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    """Write JSON to a temporary file and atomically replace the destination."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def build_run_manifest(
    *,
    out_dir: str,
    args: argparse.Namespace,
    input_folder: str,
    csv_glob: str,
    norm_params_path: str,
    start_time: datetime,
    all_csv_files: Sequence[str],
    stage_version: str,
    script_path: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    """Write the initial run manifest and return its path and dictionary."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    manifest_path = os.path.join(out_dir, f"keyprenorm_run_manifest_{run_id}.json")

    resolved_script_path: Optional[str] = None
    if script_path:
        resolved_script_path = os.path.abspath(script_path)
    else:
        # Best-effort: sys.argv[0] usually points to the top-level runner.
        candidate = None
        try:
            candidate = os.path.abspath(sys.argv[0])
        except Exception:
            candidate = None
        if candidate and os.path.exists(candidate):
            resolved_script_path = candidate
        else:
            # Fallback: this module's file.
            try:
                resolved_script_path = os.path.abspath(__file__)
            except Exception:
                resolved_script_path = None

    manifest: Dict[str, object] = {
        "run_id": run_id,
        "script": {
            "name": os.path.basename(resolved_script_path) if resolved_script_path else None,
            "version": stage_version,
            "path": resolved_script_path,
            "sha256": sha256_file(resolved_script_path) if resolved_script_path else None,
        },
        "started_at": start_time.isoformat(),
        "status": "running",
        "args": vars(args),
        "paths": {
            "input_folder": os.path.abspath(input_folder),
            "output_folder": os.path.abspath(out_dir),
            "norm_params_path": os.path.abspath(norm_params_path),
        },
        "discovery": {
            "csv_pattern": csv_glob,
            "csv_count": len(all_csv_files),
            "example_csv_files": [os.path.basename(p) for p in list(all_csv_files)[:10]],
            "csv_files": [os.path.abspath(p) for p in all_csv_files],
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
            "h5py_version": h5py.__version__,
        },
        "results": None,
        "finished_at": None,
        "duration_seconds": None,
        "error": None,
    }

    atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest
