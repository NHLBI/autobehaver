"""Process-held file locks that release automatically when the owner exits."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import socket
import time


class FileLock:
    """Lock a persistent file; file existence alone never signifies ownership."""
    def __init__(self, path, *, timeout_sec=0.0, poll_interval_sec=0.05):
        self.path = Path(path)
        self.timeout_sec = float(timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.descriptor = None

    def _change(self, acquire):
        if os.name == 'nt':
            import msvcrt
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            msvcrt.locking(self.descriptor, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.descriptor, (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)

    def acquire(self, *, metadata=None):
        if self.descriptor is not None:
            raise RuntimeError(f'Lock instance already acquired: {self.path}')
        self.descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        deadline = time.monotonic() + self.timeout_sec
        acquired = False
        try:
            while True:
                try:
                    self._change(True)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'Another writer currently holds: {self.path}') from exc
                    time.sleep(self.poll_interval_sec)
            if metadata is not None:
                info = dict(metadata)
                info.update(lock_format='os-file-lock-v1', pid=os.getpid(), host=socket.gethostname(),
                            created_at_unix=time.time())
                data = json.dumps(info, indent=2, sort_keys=True).encode('utf-8')
                os.lseek(self.descriptor, 0, os.SEEK_SET)
                os.ftruncate(self.descriptor, 0)
                remaining = memoryview(data)
                while remaining:
                    remaining = remaining[os.write(self.descriptor, remaining):]
                os.fsync(self.descriptor)
        except BaseException:
            if acquired:
                self.release()
            else:
                os.close(self.descriptor)
                self.descriptor = None
            raise

    def release(self):
        if self.descriptor is not None:
            try:
                self._change(False)
            finally:
                os.close(self.descriptor)
                self.descriptor = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exception):
        self.release()


def file_lock_active(path):
    """Probe actual OS ownership, leaving persistent coordination files intact."""
    path = Path(path)
    if not path.exists():
        return False
    lock = FileLock(path)
    try:
        lock.acquire()
    except RuntimeError:
        return True
    else:
        lock.release()
        return False
