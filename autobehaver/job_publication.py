"""Publish prepared jobs and a lightweight source snapshot through mounted storage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .source_inventory import source_files as release_source_files
from .preparation_io import io_results


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_jobs(prepared, destination, *, source_root):
    """Create a new bundle, check space, and verify every copied file.

    Models and normalization resources retain their existing paths. Existing
    bundles are never overwritten, including incomplete publications.
    """
    prepared, destination, source_root = map(Path, (prepared, destination, source_root))
    if destination.exists():
        raise FileExistsError(f"Publication destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"Mounted job-storage parent is unavailable: {destination.parent}")
    if destination.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Publish outside the source package")
    source_files = []
    for relative in release_source_files(source_root):
        source_files.append((source_root / relative, Path("package") / relative))
    job_files = [(path, path.relative_to(prepared)) for path in sorted(prepared.rglob("*")) if path.is_file()]
    files = job_files + source_files
    required = sum(path.stat().st_size for path, _ in files)
    available = shutil.disk_usage(destination.parent).free
    if available < required + 16 * 1024 * 1024:
        raise OSError(f"Insufficient mounted storage: need {required} bytes plus a small reserve; free {available}")
    expected = dict(io_results(lambda item: (item[1].as_posix(), _digest(item[0])), files,
                               label='Checking files before transfer', workers=4))
    destination.mkdir()
    pending = destination / "PUBLICATION_PENDING.txt"
    pending.write_text("Publication is incomplete until PUBLICATION_RECEIPT.json is present.\n")
    for name in ("logs", "resources"):
        (destination / name).mkdir()
    def transfer(item):
        path, relative = item
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        # Mounted Windows shares may not support chmod/copystat. Compute jobs
        # invoke scripts through Bash/Python; source bytes are what we verify.
        shutil.copyfile(path, output)
        key = relative.as_posix()
        if _digest(path) != expected[key] or _digest(output) != expected[key]:
            raise OSError(f"Publication verification failed for {key}; incomplete bundle retained at {destination}")
    list(io_results(transfer, files, label='Sending and verifying files', workers=4))
    receipt = {"verified": True, "files": expected, "file_count": len(files),
               "bytes": required, "free_bytes_before": available,
               "models_copied": 0, "package": "package"}
    (destination / "package" / "SOURCE_MANIFEST.json").write_text(json.dumps(
        {"files": {key.removeprefix("package/"): value for key, value in expected.items()
                   if key.startswith("package/")}}, indent=2)+"\n", encoding="utf-8")
    (destination / "PUBLICATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    pending.unlink()
    return receipt
