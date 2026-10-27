"""Allocate owned scratch workspaces separately from durable results.

Prefer job-local scratch when available. Remove successful workspaces and
retain failures by default; source data and published outputs are untouched.
"""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def scratch_root(scratch_dir=None):
    """Select explicit, environment, job-local or system temporary storage."""
    if scratch_dir is not None:
        return Path(scratch_dir).expanduser().resolve()
    configured = os.environ.get('AUTOBEHAVER_SCRATCH_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    for key in ('SLURM_TMPDIR',):
        value = os.environ.get(key)
        if value:
            candidate = Path(value).expanduser().resolve()
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate
    job_id = os.environ.get('SLURM_JOB_ID', '')
    if job_id.isdecimal():
        candidate = Path('/lscratch') / job_id
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate.resolve()
    configured = os.environ.get('TMPDIR')
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return Path(tempfile.gettempdir()).resolve()


@contextmanager
def scratch_workspace(stage, scratch_dir=None, keep_on_error=True, required_bytes=0):
    """Yield a unique owned workspace after checking requested free space.

    Remove it after success, or after failure when ``keep_on_error`` is false.
    Verify the ownership marker and exact parent before recursive deletion.
    """
    root = scratch_root(scratch_dir)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < int(required_bytes):
        raise OSError(f'Insufficient scratch space for {stage}: {root}')
    safe_stage = ''.join(c if c.isalnum() else '_' for c in str(stage))[:32] or 'stage'
    path = Path(tempfile.mkdtemp(prefix=f'ab_{safe_stage}_', dir=str(root))).resolve()
    marker = path / '.autobehaver-scratch.json'
    token = uuid.uuid4().hex
    marker.write_text(json.dumps({'stage': str(stage), 'token': token}), encoding='utf-8')
    failed = True
    try:
        yield path
        failed = False
    finally:
        if not failed or not keep_on_error:
            # Verify this exact owned child immediately before recursive deletion.
            if path.parent != root or not marker.is_file():
                raise RuntimeError('Scratch ownership check failed; preserving workspace')
            if json.loads(marker.read_text(encoding='utf-8')).get('token') != token:
                raise RuntimeError('Scratch marker changed; preserving workspace')
            try:
                shutil.rmtree(path)
            except OSError as exc:
                # Published results do not become invalid because an owned
                # temporary file is still open or briefly cannot be removed.
                logging.warning("Scratch cleanup incomplete; retained %s: %s", path, exc)
