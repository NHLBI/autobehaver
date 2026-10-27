"""Tiny process-kill and merge-probe checks; no recordings or cluster access."""
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from autobehaver.postinf.mmgen.locks import ShardLock, shard_lock_active


def test_shard_lock_releases_after_process_is_killed(tmp_path):
    package = Path(__file__).resolve().parents[1]
    worker = '''from pathlib import Path
import sys, time
from autobehaver.postinf.mmgen.locks import ShardLock, DirLock
root=Path(sys.argv[1])
lock=(DirLock if sys.argv[2]=='legacy' else ShardLock)(root/'shard.lock')
lock.acquire(metadata={'kind':'test_writer'})
(root/'ready').write_text('ready')
while True: time.sleep(1)
'''
    for mode in ('native', 'legacy'):
        root = tmp_path/mode
        root.mkdir()
        child = subprocess.Popen([sys.executable, '-B', '-c', worker, str(root), mode],
            env={**os.environ, 'PYTHONPATH':str(package)}, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+30
            while not (root/'ready').exists():
                if child.poll() is not None:
                    raise AssertionError(child.communicate())
                assert time.monotonic() < deadline
                time.sleep(0.01)
            path = root/'shard.lock'
            assert shard_lock_active(path)
            try:
                ShardLock(path).acquire()
            except RuntimeError:
                pass
            else:
                raise AssertionError('A live shard writer was not excluded')
            child.kill()
            child.communicate(timeout=10)
            assert not shard_lock_active(path)
            replacement = ShardLock(path)
            replacement.acquire(metadata={'kind':'replacement'})
            assert path.is_file()
            assert shard_lock_active(path)
            replacement.release()
            assert path.exists() and not shard_lock_active(path)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


def test_legacy_scheduler_state_protects_live_job_and_reclaims_finished_job(tmp_path):
    path = tmp_path/'shard.lock'
    path.mkdir()
    (path/'lock_info.json').write_text('{"slurm_job_id":"12345"}')
    with patch('autobehaver.postinf.mmgen.locks.subprocess.run', return_value=
               subprocess.CompletedProcess([], 0, stdout='RUNNING|\n', stderr='')):
        assert shard_lock_active(path)
        try:
            ShardLock(path).acquire()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Live older job lost its lock')
    with patch('autobehaver.postinf.mmgen.locks.subprocess.run', return_value=
               subprocess.CompletedProcess([], 0, stdout='FAILED|\n', stderr='')):
        assert not shard_lock_active(path)
        lock = ShardLock(path)
        lock.acquire(metadata={'kind':'replacement'})
        lock.release()
    assert path.is_file() and not shard_lock_active(path)


def test_missing_legacy_owner_is_preserved(tmp_path):
    path = tmp_path/'shard.lock'
    path.mkdir()
    assert shard_lock_active(path)
    try:
        ShardLock(path).acquire()
    except RuntimeError:
        pass
    else:
        raise AssertionError('An unverifiable older lock was removed')
    assert path.is_dir()


def test_merge_ignores_unheld_files_but_detects_live_locks(tmp_path):
    from autobehaver.postinf.mmgen.shard_merge import _active_shard_locks
    work = tmp_path/'shard_000_of_001/_mmgen_work'
    work.mkdir(parents=True)
    path = work/'_mmgen_shard.lock'
    lock = ShardLock(path)
    lock.acquire(metadata={'kind':'writer'})
    assert _active_shard_locks(shards_root=str(tmp_path), num_shards=1) == [str(path)]
    lock.release()
    assert path.exists()
    assert _active_shard_locks(shards_root=str(tmp_path), num_shards=1) == []
