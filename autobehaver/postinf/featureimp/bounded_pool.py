"""Persistent process workers with bounded pending results and explicit failures."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import sys
import time

from autobehaver.progress import progress as tqdm


def _measured_task(function, task):
    result = function(task)
    import psutil
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    if sys.platform != 'win32':
        import resource
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    return result, peak


class BoundedProcessRunner:
    """Reuse process workers across task groups and recycle them only after a group
    completes.

    Bound pending results and propagate task failures without retrying them.
    """

    def __init__(self, *, num_workers, initializer=None, initargs=(), preferred_start_method=None,
                 recycle_after_tasks=512, estimated_worker_bytes=None):
        from .mp_utils import choose_safe_start_method, _main_module_is_importable
        self.workers = max(1, int(num_workers or 1))
        self.initializer, self.initargs = initializer, tuple(initargs)
        self.start_method = choose_safe_start_method(preferred_start_method)
        if self.workers > 1 and not _main_module_is_importable():
            print(f'FeatureImp: using serial execution; {self.start_method} requires an importable __main__.')
            self.workers = 1
        self.recycle_limit = max(1, int(recycle_after_tasks)) * self.workers
        self.completed = 0
        self.executor = None
        self.serial_initialized = False
        self.estimated_worker_bytes = estimated_worker_bytes
        self.pilot_done = False
        self.observed_peak = 0

    def __enter__(self):
        return self

    def _start(self):
        if self.workers == 1:
            if not self.serial_initialized and self.initializer is not None:
                self.initializer(*self.initargs)
            self.serial_initialized = True
            return
        if self.executor is None:
            from .mp_utils import cap_worker_thread_env
            cap_worker_thread_env()
            self.executor = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=mp.get_context(self.start_method), initializer=self.initializer, initargs=self.initargs)
            self.completed = 0

    def _stop(self, *, failed=False):
        """Shut down this runner's workers, bounding cleanup and stopping failed native
        processes.
        """
        executor, self.executor = self.executor, None
        if executor is None:
            return
        processes = list((getattr(executor, '_processes', None) or {}).values())
        if failed:
            # Stop this runner's processes before shutdown; never leave a hung
            # native worker waiting during error propagation.
            for process in processes:
                if process.is_alive():
                    process.terminate()
        try:
            executor.shutdown(wait=False, cancel_futures=failed)
        except TypeError:
            executor.shutdown(wait=False)
        deadline = time.monotonic() + 5
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        lingering = [process for process in processes if process.is_alive()]
        if lingering and not failed:
            print('FeatureImp: completed workers did not exit during cleanup; stopping their native processes.')
        for process in lingering:
            process.terminate()
        deadline = time.monotonic() + 2
        for process in lingering:
            process.join(timeout=max(0, deadline - time.monotonic()))

    def run(self, worker_fn, tasks, *, desc, stage_name, show_progress=True):
        """Yield completed task results with a bounded submission queue and optional
        progress display.
        """
        total = len(tasks) if hasattr(tasks, '__len__') else None
        if self.estimated_worker_bytes and self.workers > 1:
            from .execution import worker_capacity
            if not self.pilot_done:
                iterator = iter(tasks)
                try:
                    first = next(iterator)
                except StopIteration:
                    return
                from .mp_utils import cap_worker_thread_env
                cap_worker_thread_env()
                print(f'{stage_name}: measuring one training task before admitting concurrent workers.', flush=True)
                try:
                    self.executor = ProcessPoolExecutor(max_workers=1,
                        mp_context=mp.get_context(self.start_method), initializer=self.initializer, initargs=self.initargs)
                    result, self.observed_peak = self.executor.submit(_measured_task, worker_fn, first).result()
                except BaseException:
                    self._stop(failed=True)
                    raise
                self._stop()
                self.pilot_done = True
                tasks = iterator
                if total is not None:
                    total -= 1
                yield result
                del result
            admitted = worker_capacity(self.workers,
                max(self.estimated_worker_bytes, int(self.observed_peak * 1.5)), stage_name + ' measured admission')
            if admitted != self.workers:
                self._stop()
                self.workers = admitted
        self._start()
        with tqdm(total=total, desc=desc, file=sys.stdout, mininterval=5,
                  disable=not show_progress or total == 0, leave=True) as progress:
            if self.workers == 1:
                for task in tasks:
                    yield worker_fn(task)
                    progress.update(1)
                return
            iterator = iter(enumerate(tasks))
            pending = {}
            def fill():
                while len(pending) < 2 * self.workers:
                    try:
                        idx, task = next(iterator)
                    except StopIteration:
                        break
                    future = self.executor.submit(_measured_task, worker_fn, task) if self.estimated_worker_bytes else self.executor.submit(worker_fn, task)
                    pending[future] = idx
            try:
                fill()
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        idx = pending.pop(future)
                        try:
                            result = future.result()
                            if self.estimated_worker_bytes:
                                result, peak = result
                                self.observed_peak = max(self.observed_peak, peak)
                        except BrokenProcessPool as exc:
                            raise RuntimeError(f'{stage_name} worker process crashed using {self.start_method}; '
                                               'check XGBoost/SHAP/native-library errors and available memory.') from exc
                        except BaseException as exc:
                            raise RuntimeError(f'{stage_name} task {idx + 1} raised {type(exc).__name__}: {exc}') from exc
                        self.completed += 1
                        progress.update(1)
                        yield result
                        del result
                    ready.clear()
                    fill()
            except BaseException:
                self._stop(failed=True)
                raise
        if self.completed >= self.recycle_limit:
            self._stop()

    def __exit__(self, exc_type, exc, tb):
        self._stop(failed=exc_type is not None)
