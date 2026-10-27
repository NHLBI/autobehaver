"""Memory-admitted recording workers and restartable positional report parts.

Estimates include pandas/object overhead and temporary arrays; they are not an
OS memory guarantee. The job/cgroup limit remains authoritative.
"""
from __future__ import annotations

from autobehaver.progress import progress
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from autobehaver.atomic_io import atomic_write_json, sha256_file
from autobehaver.preinf.execution import memory_limit, cpu_limit, GIB


def recording_ranges(store, key, chunk_rows=50000):
    """One sequential identity-column pass; support split/interleaved recordings."""
    ranges = {}
    offset = 0
    for chunk in progress(store.select(key, columns=['individual'], chunksize=chunk_rows), desc='MMGen indexing recording chunks'):
        values = chunk['individual'].astype(str).to_numpy()
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        stops = np.r_[starts[1:], len(values)]
        for start, stop in zip(starts, stops):
            spans = ranges.setdefault(str(values[start]), [])
            left, right = offset + int(start), offset + int(stop)
            if spans and spans[-1][1] == left:
                spans[-1][1] = right
            else:
                spans.append([left, right])
        offset += len(chunk)
    return ranges


def estimate_recording_bytes(rows, row_bytes):
    # On-disk width alone underestimates strings, copies, transform arrays and CSV
    # formatting. Include interpreter/import baseline separately for each worker.
    return GIB + int(rows) * max(4096, int(row_bytes) * 8)


def budgeted_results(function, tasks, estimates, *, num_workers, memory_gb=None):
    """Ordered results, bounded queued work, weighted memory admission.

    Workers return paths/compact summaries, never whole frame tables. Reserve
    covers the parent/reduction and leaves at least 25% of allocation untouched.
    """
    tasks, estimates = list(tasks), list(estimates)
    if len(tasks) != len(estimates):
        raise ValueError('One memory estimate is required per recording')
    if not tasks:
        return
    (source, total), _ = memory_limit(memory_gb)
    reserve = max(2 * GIB, int(total * .25))
    usable = total - reserve
    if max(estimates) > usable:
        required = (max(estimates) + max(2 * GIB, max(estimates) / 3)) / GIB
        raise MemoryError(f'A recording needs an estimated {max(estimates)/GIB:.2f} GiB; '
                          f'job budget is {total/GIB:.2f} GiB including reserve. '
                          f'Request at least {required:.1f} GiB; no recording was loaded.')
    workers = min(max(1, int(num_workers)), cpu_limit(), len(tasks), max(1, usable // max(estimates)))
    print(f'[MMGen finalize] ceiling {workers} workers; {total/GIB:.2f} GiB from {source}; '
          f'{reserve/GIB:.2f} GiB reserved; recording estimates '
          f'{min(estimates)/GIB:.2f}-{max(estimates)/GIB:.2f} GiB.', flush=True)
    # A persistent spawn pool avoids inheriting an open HDF handle and amortizes imports.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pending = deque()
        index, committed, adjustment = 0, 0, 1.0
        while index < len(tasks) or pending:
            while index < len(tasks) and len(pending) < workers:
                allowance = int(estimates[index] * adjustment)
                if committed + allowance > usable:
                    break
                pending.append((pool.submit(function, tasks[index]), allowance, estimates[index]))
                committed += allowance
                index += 1
            if not pending:
                raise MemoryError('Observed worker memory raised the estimate beyond this job budget; '
                                  'completed recording checkpoints are retained.')
            future, allowance, original = pending.popleft()
            result = future.result()
            committed -= allowance
            # Measured worker peak can only raise, never weaken, the conservative estimate.
            if isinstance(result, dict):
                adjustment = max(adjustment, float(result.get('peak_rss_bytes', 0)) * 1.25 / original)
            yield result


def peak_rss_bytes():
    try:
        import resource
        import sys
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    except ImportError:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, 'peak_wset', info.rss))


def report_part(task):
    """Transform one recording, serialize it and precompute plot reductions."""
    from .occupancy import _transform_positional_individual_df, _OccupancyAccumulator
    store_path, key, individual, spans, columns, radius, quiver, prefix = task
    prefix = Path(prefix)
    receipt = prefix.with_suffix('.json')
    csv_path, reduction_path = prefix.with_suffix('.csv'), prefix.with_suffix('.pkl')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if all(path.is_file() and sha256_file(path) == saved['hashes'].get(path.suffix)
               for path in (csv_path, reduction_path)):
            return saved
    started = time.monotonic()
    with pd.HDFStore(store_path, mode='r') as store:
        frames = [store.select(key, start=a, stop=b, columns=columns) for a, b in spans]
    frame = pd.concat(frames) if len(frames) > 1 else frames[0]
    del frames
    if not frame['individual'].astype(str).eq(individual).all():
        raise RuntimeError('Recording row lookup does not match the requested identity')
    frame, n, q1, q12 = _transform_positional_individual_df(frame, fixed_arena_radius=radius)
    acc = _OccupancyAccumulator()
    kde_values = {}
    if frame is not None and not frame.empty:
        for col in acc.behavior_cols:
            if col not in frame:
                continue
            for (behavior, group), gdf in frame.dropna(subset=[col, 'group']).groupby([col, 'group'], observed=False):
                behavior, group = str(behavior), str(group)
                coords = gdf[['centroid_x', 'centroid_y']].dropna()
                if not coords.empty:
                    acc.add_heatmap(col, behavior, group, coords['centroid_x'].to_numpy(dtype=float), coords['centroid_y'].to_numpy(dtype=float))
                if 'r_norm' in gdf:
                    # Preserve original RNG consumption exactly in the ordered parent merge.
                    kde_values[(col, behavior, group)] = gdf['r_norm'].to_numpy(dtype=float)
                if quiver and 'angle' in gdf:
                    x, y, angle = (gdf[c].to_numpy(dtype=float) for c in ('centroid_x', 'centroid_y', 'angle'))
                    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(angle)
                    if np.any(valid):
                        acc.add_quiver(col, behavior, group, x[valid], y[valid], angle[valid])
        frame.to_csv(str(csv_path) + '.tmp', index=False)
    else:
        Path(str(csv_path) + '.tmp').write_text('')
    with open(str(reduction_path) + '.tmp', 'wb') as stream:
        pickle.dump((acc, kde_values), stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(csv_path) + '.tmp', csv_path)
    os.replace(str(reduction_path) + '.tmp', reduction_path)
    result = dict(csv=str(csv_path), reduction=str(reduction_path), qc=[n, q1, q12],
                  rows=0 if frame is None else len(frame), peak_rss_bytes=peak_rss_bytes(),
                  seconds=time.monotonic()-started,
                  hashes={'.csv':sha256_file(csv_path), '.pkl':sha256_file(reduction_path)})
    atomic_write_json(result, receipt)
    return result


def merge_reduction(acc, result):
    # Only load locally generated, hash-verified parts bound to input and code identity.
    path = Path(result['reduction'])
    if sha256_file(path) != result['hashes']['.pkl']:
        raise RuntimeError('Recording reduction changed after checkpoint verification')
    with path.open('rb') as stream:
        part, values = pickle.load(stream)
    for name in ('_heat_counts', '_heat_n', '_q_sum_u', '_q_sum_v', '_q_count'):
        target = getattr(acc, name)
        for key, value in getattr(part, name).items():
            if key in target:
                target[key] += value
            else:
                target[key] = value
    for col in acc.behavior_cols:
        acc.behaviors[col].update(part.behaviors[col])
    for (col, behavior, group), value in values.items():
        acc.add_kde(col, behavior, group, value)


def prepare_report_tasks(store_path, key, directory, columns, radius, quiver):
    from . import occupancy
    path = Path(store_path).resolve()
    stat = path.stat()
    identity = dict(store=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    key=key, columns=columns, radius=radius, quiver=quiver,
                    implementation=sha256_file(__file__), transform=sha256_file(occupancy.__file__))
    # Upstream's content/provenance checks remain authoritative. Stat plus those
    # identities invalidate parts after any ordinary store replacement or mutation.
    for name in ('MMGEN_CALCULATION_IDENTITY.json', 'MMGEN_STORE_IDENTITY.json'):
        candidate = path.parent / name
        identity[name] = sha256_file(candidate) if candidate.exists() else None
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(directory) / '.occupancy_parts' / digest
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / 'ranges.json'
    with pd.HDFStore(path, mode='r') as store:
        row_bytes = store.get_storer(key).table.rowsize
        if index_path.exists():
            indexed = json.loads(index_path.read_text())
            ranges = indexed['ranges']
            encoded = json.dumps(ranges, sort_keys=True).encode()
            if hashlib.sha256(encoded).hexdigest() != indexed['sha256']:
                raise RuntimeError('Recording row lookup checksum failed; rebuild this checkpoint directory')
        else:
            print('[MMGen finalize] Building recording row lookup in one sequential pass.', flush=True)
            ranges = recording_ranges(store, key)
            atomic_write_json(dict(ranges=ranges, sha256=hashlib.sha256(json.dumps(ranges, sort_keys=True).encode()).hexdigest()), index_path)
    tasks, estimates = [], []
    for number, individual in enumerate(sorted(ranges)):
        spans = ranges[individual]
        tasks.append((str(path), key, individual, spans, columns, radius, quiver, str(folder / f'{number:06d}')))
        estimates.append(estimate_recording_bytes(sum(b-a for a,b in spans), row_bytes))
    atomic_write_json(dict(identity=identity, recordings=len(tasks)), folder / 'identity.json')
    return tasks, estimates
