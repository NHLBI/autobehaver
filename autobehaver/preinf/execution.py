"""Plan bounded preprocessing within the job's CPU and memory allocation.

Allowances are conservative estimates, not OS-enforced memory limits. Feature
tables have bounded rows; the scheduler/cgroup remains the hard allocation limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

GIB = 1024 ** 3
MIB = 1024 ** 2


def _slurm_mb(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", str(value).strip(), re.I)
    if not match:
        raise ValueError(f"Unrecognized Slurm memory allocation: {value!r}")
    amount = float(match[1]) * {'': MIB, 'K': 1024, 'M': MIB, 'G': GIB, 'T': 1024*GIB}[match[2].upper()]
    return int(amount)


def _cgroup_limits():
    """Read this process's v1/v2 hierarchy, including stricter ancestor limits.

    Mountinfo handles non-root mount roots and container cgroup namespaces.
    Unsupported/inaccessible files are ignored; never write cgroup controls.
    """
    limits = []
    try:
        memberships = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
        mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    except OSError:
        return limits
    for line in mounts:
        before, sep, after = line.partition(' - ')
        if not sep:
            continue
        fields, fs = before.split(), after.split()
        if len(fields) < 5 or len(fs) < 3 or fs[0] not in ('cgroup', 'cgroup2'):
            continue
        version2 = fs[0] == 'cgroup2'
        if not version2 and 'memory' not in fs[2].split(','):
            continue
        decode = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
        mount, mount_root = Path(decode(fields[4])), Path(decode(fields[3]))
        for _, controllers, membership in memberships:
            if (version2 and controllers) or (not version2 and 'memory' not in controllers.split(',')):
                continue
            member = Path(membership)
            try:
                relative = member.relative_to(mount_root)
            except ValueError:
                if membership != '/':
                    continue
                relative = Path('.')  # cgroup namespace exposes its root at the mount
            folder = mount / relative
            while True:
                try:
                    limit = int((folder / ('memory.max' if version2 else 'memory.limit_in_bytes')).read_text().strip())
                    if 0 < limit < (1 << 60):
                        limits.append((str(folder), limit))
                except (OSError, ValueError):
                    pass
                if folder == mount:
                    break
                folder = folder.parent
    return limits


def cpu_limit():
    candidates = [os.cpu_count() or 1]
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        candidates.append(int(os.environ['SLURM_CPUS_PER_TASK']))
    return max(1, min(candidates))


def memory_limit(memory_gb=None):
    """Return the smallest explicit, Slurm or cgroup limit; use local availability.

    Within Slurm, unknown allocation is an error rather than guessing node RAM.
    An explicit value is a cap, never permission to exceed a detected allocation.
    """
    candidates = []
    if memory_gb is not None:
        if not math.isfinite(memory_gb) or memory_gb <= 0:
            raise ValueError('--memory_gb must be finite and positive.')
        candidates.append(('argument', int(memory_gb * GIB)))
    if os.environ.get('SLURM_MEM_PER_NODE'):
        value = _slurm_mb(os.environ['SLURM_MEM_PER_NODE'])
        if value:
            candidates.append(('SLURM_MEM_PER_NODE', value))
    elif os.environ.get('SLURM_MEM_PER_CPU'):
        # This command is one task. Do not claim memory assigned to other tasks.
        cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if cpus:
            candidates.append(('SLURM_MEM_PER_CPU', _slurm_mb(os.environ['SLURM_MEM_PER_CPU']) * int(cpus)))
    candidates.extend(('cgroup:' + name, value) for name, value in _cgroup_limits())
    if not candidates and os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Cannot determine job RAM; pass --memory_gb with the allocated GiB.')
    # Available system memory can further constrain a job on an already busy host.
    try:
        import psutil
        candidates.append(('available system RAM', int(psutil.virtual_memory().available)))
    except ImportError:
        pass
    if not candidates:
        raise ValueError('Cannot determine available RAM; pass --memory_gb (GiB).')
    return min(candidates, key=lambda item: item[1]), candidates


@dataclass(frozen=True)
class ExecutionPlan:
    stage: str
    workers: int
    chunk_rows: int
    cpu_limit: int
    memory_bytes: int
    reserve_bytes: int
    worker_allowance_bytes: int
    memory_source: str
    detected_limits: list

    def describe(self):
        return (f'{self.stage}: {self.workers} workers, {self.chunk_rows} rows/chunk; '
                f'RAM budget {self.memory_bytes/GIB:.2f} GiB from {self.memory_source}, '
                f'reserve {self.reserve_bytes/GIB:.2f} GiB, '
                f'estimated allowance {self.worker_allowance_bytes/GIB:.2f} GiB/worker')


def plan_execution(stage, *, files, columns, num_workers=0, memory_gb=None):
    """Choose concurrency and bounded rows with 25% (at least 1 GiB) headroom.

    KeyPre allowance: 256 MiB baseline + 128 bytes per feature per context row.
    Normalization: 256 MiB + 48 bytes per feature per row, including statistics.
    These deliberately cover concurrent pandas arrays/parsing/serialization,
    not just the final float32 table. No claim of a measured universal bound.
    """
    if num_workers < 0 or columns < 1 or files < 0:
        raise ValueError('Worker ceiling must be nonnegative; columns positive; files nonnegative.')
    (source, total), limits = memory_limit(memory_gb)
    reserve = max(GIB, math.ceil(total * .25))
    usable = total - reserve
    cpu = cpu_limit()
    ceiling = min(cpu, num_workers or cpu, max(1, files))
    target, per_cell = (16384, 128) if stage == 'keypre' else (4096, 48)
    context = 32 if stage == 'keypre' else 0  # larger than the current 14-row halo
    baseline = 256 * MIB
    minimum_rows = 1024
    minimum = baseline + columns * per_cell * (minimum_rows + context)
    workers = min(ceiling, usable // minimum)
    if workers < 1:
        raise ValueError(f'RAM budget {total/GIB:.2f} GiB is too small for preprocessing plus safety reserve.')
    rows = min(target, (usable // workers - baseline) // (columns * per_cell) - context)
    rows = max(minimum_rows, int(rows) // 256 * 256)
    allowance = baseline + columns * per_cell * (rows + context)
    plan = ExecutionPlan(stage, int(workers), rows, cpu, total, reserve, allowance, source, limits)
    logging.getLogger(__name__).info(plan.describe())
    return plan


def worker_results(function, items, *, num_workers, initializer=None, initargs=(), tasks_per_worker=16):
    """Crash-aware, bounded submissions; recycle generations to release arenas.

    Match the established MMGen executor approach, with completion-order refill
    so one slow recording does not idle the other workers. Return input order for
    deterministic normalization-statistics reductions. An abruptly killed worker
    breaks the executor and terminates its siblings instead of losing a Pool task.
    """
    from concurrent.futures import FIRST_COMPLETED, wait
    items = iter(items)
    while True:
        generation = list(islice(items, max(1, num_workers) * tasks_per_worker))
        if not generation:
            return
        workers = min(num_workers, len(generation))
        if workers == 1:
            if initializer:
                initializer(*initargs)
            for item in generation:
                yield function(item)
            continue
        with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs,
                                 mp_context=mp.get_context('spawn')) as executor:
            remaining = iter(enumerate(generation))
            pending = {executor.submit(function, item): idx for idx, item in islice(remaining, workers)}
            results = {}
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    results[idx] = future.result()
                    next_item = next(remaining, None)
                    if next_item is not None:
                        index, item = next_item
                        pending[executor.submit(function, item)] = index
            for index in range(len(generation)):
                yield results[index]


def add_memory_arguments(parser):
    parser.add_argument('--memory_gb', type=float, default=None,
                        help='Total RAM budget in GiB; detected from Slurm/cgroup or local available RAM if omitted. Leaves safety headroom.')
    return parser
