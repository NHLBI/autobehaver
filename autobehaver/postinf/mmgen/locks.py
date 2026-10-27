"""Coordinate MMGen writers with crash-releasing ownership and legacy migration."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
from pathlib import Path
import subprocess

from autobehaver.file_locks import FileLock, file_lock_active


def _legacy_owner_active(path):
    """Verify an old directory-lock owner; unknown ownership remains protected."""
    try:
        info = json.loads((path / 'lock_info.json').read_text())
    except (OSError, ValueError):
        return True
    job = str(info.get('slurm_job_id') or '')
    if job.isdigit():
        try:
            result = subprocess.run(['sacct', '-j', job, '--noheader', '--parsable2',
                                     '--allocations', '--format=State'],
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return True
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                    'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
        states = [line.split('|')[0].split()[0].rstrip('+') for line in result.stdout.splitlines() if line.strip()]
        return result.returncode != 0 or not states or any(state not in terminal for state in states)
    if info.get('host') != socket.gethostname() or not isinstance(info.get('pid'), int):
        return True
    pid = info['pid']
    if pid <= 0:
        return True
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ShardLock(FileLock):
    """Crash-releasing shard lock with verified migration of old lock directories."""
    def __init__(self, lock_dir, *, timeout_sec=0.0):
        super().__init__(lock_dir, timeout_sec=timeout_sec)
        self.lock_dir = Path(lock_dir)

    def acquire(self, *, metadata=None):
        # Serialize conversion separately; never unlink an OS lock file, since
        # existing waiters must continue locking the same file identity.
        with FileLock(self.path.with_name(self.path.name + '.gate'), timeout_sec=1):
            if self.path.is_dir():
                if self.path.is_symlink() or _legacy_owner_active(self.path):
                    raise RuntimeError(f'Older MMGen lock has a live or unverified owner: {self.path}')
                entries = list(self.path.iterdir())
                if any(entry.name != 'lock_info.json' or not entry.is_file() or entry.is_symlink() for entry in entries):
                    raise RuntimeError(f'Older lock contains unexpected files: {self.path}')
                resolved = self.path.resolve()
                if resolved.parent != self.path.parent.resolve():
                    raise RuntimeError(f'Unexpected legacy lock path: {self.path}')
                for entry in entries:
                    entry.unlink()
                self.path.rmdir()
            super().acquire(metadata=metadata)


def shard_lock_active(path):
    path = Path(path)
    return _legacy_owner_active(path) if path.is_dir() else file_lock_active(path)


class DirLock(ShardLock):
    """Compatibility name for native locks that previously used directories.

    Native ownership releases on process death. A legacy directory is migrated
    only after its owner is verified dead; age alone never authorizes removal.
    """


CANONICAL_LOCK = '_mmgen_work/_mmgen_canonical.lock'
_canonical_owners = ContextVar('mmgen_canonical_owners', default=frozenset())


def _owner_key(root):
    # Include process and thread identity: inherited contexts cannot grant a
    # forked process or another thread permission to bypass the operating lock.
    return (os.getpid(), get_ident(), os.path.normcase(str(Path(root).resolve())))


def canonical_lock_held(root):
    return _owner_key(root) in _canonical_owners.get()


@contextmanager
def canonical_writer(root):
    """Own all canonical reads/publications; nested calls share this ownership."""
    key = _owner_key(root)
    if key in _canonical_owners.get():
        yield
        return
    path = Path(root) / CANONICAL_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = ShardLock(path)
    lock.acquire(metadata={'kind': 'mmgen_canonical', 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    token = _canonical_owners.set(_canonical_owners.get() | {key})
    try:
        yield
    finally:
        _canonical_owners.reset(token)
        lock.release()
