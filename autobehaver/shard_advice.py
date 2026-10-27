"""Workload-based planning advice; does not submit or change existing jobs."""
from __future__ import annotations
import math


def recommend_shards(task_count, *, workers_per_shard=1, minimum_tasks_per_worker=4,
                     observed_seconds_per_task=None, target_seconds_per_shard=900):
    """Return a maximum and a recommendation, stating the estimation basis.

    With timing evidence, group enough tasks for the target allocation duration.
    Otherwise use a declared minimum amount of work per worker, not a fabricated
    timing estimate. The hard maximum prevents empty shards.
    """
    count, workers, minimum = int(task_count), int(workers_per_shard), int(minimum_tasks_per_worker)
    if count < 0 or workers < 1 or minimum < 1 or target_seconds_per_shard <= 0:
        raise ValueError('Task count must be nonnegative; workers, batch size and target time must be positive.')
    if not count:
        return {'task_count': 0, 'hard_maximum': 0, 'recommended_shards': 0, 'basis': 'no work'}
    batch = workers * minimum
    basis = 'minimum tasks per worker; no runtime prediction'
    if observed_seconds_per_task is not None:
        duration = float(observed_seconds_per_task)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Observed task duration must be positive and finite.')
        batch = max(batch, math.ceil(target_seconds_per_shard * workers / duration))
        basis = 'observed task duration and target allocation time'
    return {'task_count': count, 'hard_maximum': count,
            'recommended_shards': max(1, count // batch), 'minimum_tasks_per_shard': batch,
            'workers_per_shard': workers, 'basis': basis}


def main():
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--tasks', type=int, help='Actual FeatureImp tasks from its frozen plan, or a known recording count.')
    source.add_argument('--input-dir', help='Inspect this folder to count files automatically.')
    p.add_argument('--glob', default='*.mp4', help='Input filename pattern (default *.mp4).')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--observed-seconds-per-task', type=float)
    p.add_argument('--target-minutes', type=float, default=15)
    a = p.parse_args()
    if a.input_dir:
        from pathlib import Path
        directory=Path(a.input_dir)
        if not directory.is_dir(): p.error('Input folder does not exist')
        a.tasks=sum(p.is_file() for p in directory.glob(a.glob))
    print(json.dumps(recommend_shards(a.tasks, workers_per_shard=a.workers,
                     observed_seconds_per_task=a.observed_seconds_per_task,
                     target_seconds_per_shard=60*a.target_minutes), indent=2))


if __name__ == '__main__':
    main()
