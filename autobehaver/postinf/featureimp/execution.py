"""Allocation-aware operational limits; never change statistical settings."""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from autobehaver.preinf.execution import memory_limit, cpu_limit, GIB


def allowance():
    explicit = os.environ.get('AUTOBEHAVER_FI_MEMORY_GB')
    (source, total), _ = memory_limit(float(explicit) if explicit else None)
    import psutil
    resident = psutil.Process().memory_info().rss
    usable = total - max(GIB, int(total * .25)) - resident
    if usable <= 0:
        raise MemoryError('FeatureImp has no safe RAM allowance after current use and reserved headroom.')
    return source, total, usable


def worker_capacity(requested, estimated_bytes, label):
    source, total, usable = allowance()
    estimate = max(1, int(estimated_bytes))
    if estimate > usable:
        raise MemoryError(f'{label} needs an estimated {estimate/GIB:.2f} GiB beyond the parent and reserve; '
                          f'only {usable/GIB:.2f} GiB is available. Increase RAM before retrying.')
    workers = min(max(1, int(requested)), cpu_limit(), max(1, usable // estimate))
    print(f'[FeatureImp resources] {label}: {workers} workers; allocation/limit {total/GIB:.2f} GiB '
          f'({source}); estimated {estimate/GIB:.2f} GiB/worker; 25% or 1 GiB reserve.', flush=True)
    return workers


def fsea_preflight(features, permutations, requested_threads):
    # Conservative working estimate for the native permutation engine. Historical
    # 100k/6000-feature runs reached ~26 GiB; six float64 work arrays give ~27 GiB.
    # This is a preflight estimate, NOT a hard bound on an external library.
    estimate = GIB + 6 * 8 * int(features) * int(permutations)
    worker_capacity(1, estimate, 'FSEA permutation calculation')
    threads = min(max(1, int(requested_threads)), cpu_limit())
    print(f'[FeatureImp resources] FSEA: {features} features, {permutations} permutations, '
          f'{threads} threads; statistical settings unchanged.', flush=True)
    return threads


# Share terminal/log behavior with the other pipeline stages.
from autobehaver.progress import phase
