"""Lightweight directory-lock utilities for mmgen.

Why this exists
--------------
mmgen is often run on HPC where users may launch many jobs at once.

For deterministic sharding, *each shard writes to its own output directory*,
so shards do not race each other. The two main remaining race conditions are:

1) Accidentally launching two jobs with the same ``(num_shards, shard_index)``.
   Those jobs would write to the *same* shard folder and corrupt the shard
   ``temp_store.h5`` / double-count commits.

2) Running the merge step while shard jobs are still running.

To mitigate this, we use simple **directory locks** (no fcntl required).
Directory creation is atomic across platforms / network filesystems in the
common case.

Locking philosophy
------------------
* **Shard lock**: acquired at shard start, released only on successful shard
  completion. If a shard crashes, the lock directory remains as a strong signal
  that outputs may be incomplete.

* **Merge lock**: acquired for the full duration of the merge+finalize step.
  It is released on completion (even if the merge fails) so the user can rerun.

Users can always delete stale lock directories manually if needed.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DirLock:
    """A minimal directory lock.

    Parameters
    ----------
    lock_dir:
        Directory path used as the lock.
    timeout_sec:
        How long to wait for the lock. In mmgen we generally use 0 to fail fast.
    poll_interval_sec:
        Poll interval while waiting.
    stale_sec:
        If set, a lock older than this will be treated as stale and removed.
        (Default: None => never auto-remove).
    """

    lock_dir: Path
    timeout_sec: float = 0.0
    poll_interval_sec: float = 0.25
    stale_sec: Optional[float] = None

    def acquire(self, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        lock_dir = Path(self.lock_dir)
        start = time.time()

        while True:
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)

                # Best-effort metadata for debugging (PID, host, time, etc.).
                try:
                    info = dict(metadata or {})
                    info.setdefault("pid", os.getpid())
                    info.setdefault("host", socket.gethostname())
                    info.setdefault("created_at_unix", time.time())
                    (lock_dir / "lock_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
                except Exception:
                    pass

                return

            except FileExistsError:
                # Optional stale cleanup.
                if self.stale_sec is not None:
                    try:
                        age = time.time() - lock_dir.stat().st_mtime
                        if age > float(self.stale_sec):
                            _best_effort_remove_dir(lock_dir)
                            continue
                    except Exception:
                        pass

                if float(self.timeout_sec) <= 0:
                    raise RuntimeError(f"Lock already exists: {lock_dir}")

                if (time.time() - start) > float(self.timeout_sec):
                    raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")

                time.sleep(float(self.poll_interval_sec))

    def release(self) -> None:
        _best_effort_remove_dir(Path(self.lock_dir))


def _best_effort_remove_dir(path: Path) -> None:
    """Best-effort recursive directory removal (very small / shallow)."""

    try:
        if not path.exists():
            return
        for p in path.glob("*"):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
            except Exception:
                pass
        try:
            path.rmdir()
        except Exception:
            # If another process is racing, ignore.
            pass
    except Exception:
        pass
