"""Bounded inference scheduling; no changes to models or scientific settings.

Host allowances are conservative estimates, not a guarantee against OS OOM.
Calibration measures the real ensemble/TTA path, including CPU noise preparation.
Only CUDA allocation failures are retried, before any rows are committed.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gc
import math
import time

import torch

from .augmentation import inference_noise, apply_inference_noise
from .preinf.execution import cpu_limit, memory_limit, GIB, MIB
from .resource_monitor import current_monitor, record_settings


@dataclass(frozen=True)
class InferencePlan:
    host_batch: int
    loader_workers: int
    noise_workers: int
    memory_bytes: int
    reserve_bytes: int
    fixed_bytes: int
    bytes_per_window: int


def plan_inference(*, rows, total_frames, columns, sequence_length, model_widths,
                   classes, batch=0, num_workers=-1, noise_workers=0,
                   memory_gb=None, use_noise=True, current_rss=None):
    """Joint allowance for source loading, full-timeline results and bounded queues.

    At most two auto loader processes; remaining CPUs prepare noise (up to eight).
    Positive worker arguments are ceilings. Zero loader workers is synchronous.
    The loader prefetch factor is one. Count both shared and pinned base buffers,
    active/collating batches, noise buffers, block temporaries and worker overhead.
    """
    if batch < 0 or num_workers < -1 or noise_workers < 0:
        raise ValueError('batch/noise_workers must be nonnegative; num_workers must be >= -1.')
    if min(rows, total_frames) < 0 or min(columns, sequence_length) < 1:
        raise ValueError('Invalid feature dimensions for inference memory planning.')
    (_, limit), _ = memory_limit(memory_gb)
    cpu = cpu_limit()
    automatic_loaders = 0 if rows <= 8192 else 2
    loaders = min(automatic_loaders if num_workers == -1 else num_workers, max(0, cpu - 2))
    workers = min(noise_workers or 8, max(1, cpu - loaders - 1)) if use_noise else 0
    if current_rss is None:
        import psutil
        current_rss = psutil.Process().memory_info().rss
    reserve = max(GIB, math.ceil(limit * .25))
    # Source loading/selection/float conversion/index plus output formatting/writes.
    # Float64 inputs and pandas/Python result rows require more than their raw bytes.
    persistent = (current_rss + rows * (columns * 32 + 128)
                  + total_frames * (4 * sum(model_widths) + 128 * classes * (len(model_widths) + 1) + 2048))
    window = sequence_length * columns * 4
    while True:
        fixed = persistent + loaders * 512 * MIB + workers * (64 * window * 3 + 16 * MIB)
        # Loader shared + pinned buffers (including cached smaller tails), active
        # batches, bounded pageable noise and one fixed pinned transfer buffer.
        # Pinned allocators can round allocations up; reserve twice that buffer.
        per_window = window * (4 * loaders + 4 + workers + (4 if use_noise else 0))
        cap = max(0, (limit - reserve - fixed) // per_window)
        if cap >= min(256, max(1, rows)) or not (loaders or workers > 1):
            break
        if loaders:
            loaders -= 1
        else:
            workers -= 1
    if cap < 1:
        raise ValueError('Insufficient RAM for this recording and inference buffers with safety headroom. '
                         'Increase allocated RAM or process shorter recordings; no work was started.')
    host_batch = min(batch or 8192, cap, max(1, rows))
    plan = InferencePlan(int(host_batch), loaders, workers, limit, reserve, fixed, per_window)
    print(f'Inference memory plan: base batches up to {plan.host_batch}; '
          f'{loaders} loader processes, {workers} noise preparation threads; '
          f'{limit/GIB:.2f} GiB RAM budget, {reserve/GIB:.2f} GiB reserve.')
    record_settings({'inference_' + k: v for k, v in asdict(plan).items()})
    return plan


class NoiseQueue:
    """Ordered, bounded futures. Native NumPy RNG runs concurrently outside the GIL."""
    def __init__(self, workers, pin_memory):
        self.workers = max(1, workers)
        self.pin_memory = pin_memory
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='posepred-noise')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def buffers(self, tasks):
        remaining = iter(tasks)
        pending = deque()
        try:
            for _ in range(self.workers):
                task = next(remaining, None)
                if task is None:
                    break
                pending.append(self.pool.submit(inference_noise, **task, pin_memory=self.pin_memory))
            while pending:
                future = pending.popleft()
                value = future.result()
                del future
                # Current buffer plus at most `workers` futures. The planner counts both.
                task = next(remaining, None)
                if task is not None:
                    pending.append(self.pool.submit(inference_noise, **task, pin_memory=self.pin_memory))
                yield value
                del value
        finally:
            for future in pending:
                future.cancel()
            # No previous-batch tasks/buffers survive a retry or an early exit.
            for future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass
            pending.clear()


class EnsembleRunner:
    def __init__(self, models, device, *, passes, pose_pred, orient_pred, use_tta,
                 noise_level, columns, seed, model_identities, noise_workers=1,
                 pose_permutations=None, orient_permutations=None, max_batch_size=None):
        self.models, self.device = models, device
        self.passes, self.pose_pred, self.orient_pred = passes, pose_pred, orient_pred
        self.use_tta, self.noise_level, self.columns = use_tta, noise_level, columns
        self.seed, self.model_identities = seed, model_identities
        self.pose_permutations, self.orient_permutations = pose_permutations, orient_permutations
        self.noise = NoiseQueue(noise_workers, False)
        self.max_batch_size = max_batch_size
        self.pinned_noise = None
        self.pinned_copy_done = None
        self.profile = current_monitor() is not None
        self.metrics = dict(noise_wait_seconds=0., host_noise_staging_seconds=0.)
        if self.profile and device.type == 'cuda':
            self.metrics.update(forward_cuda_seconds=0., transfer_and_noise_cuda_seconds=0.)

    def __enter__(self):
        self.noise.__enter__()
        return self

    def __exit__(self, *exc):
        self.noise.__exit__(*exc)
        try:
            if self.pinned_copy_done is not None:
                self.pinned_copy_done.synchronize()
        except RuntimeError:
            if not exc or exc[0] is None:
                raise
        finally:
            self.pinned_noise = self.pinned_copy_done = None

    def stage_noise(self, noise):
        """One reusable pinned buffer; wait for its previous copy before overwriting.

        Its capacity is the host admission ceiling, not each calibration size.
        This prevents pinned allocator caches accumulating a buffer per probe/pass.
        Pageable preparation arrays remain bounded by NoiseQueue's future count.
        """
        if self.device.type != 'cuda':
            return noise
        if self.pinned_copy_done is not None:
            self.pinned_copy_done.synchronize()
        capacity = max(len(noise), self.max_batch_size or len(noise))
        if self.pinned_noise is None or self.pinned_noise.shape[0] < capacity:
            self.pinned_noise = torch.empty((capacity, *noise.shape[1:]), dtype=noise.dtype, pin_memory=True)
        target = self.pinned_noise[:len(noise)]
        target.copy_(noise)
        return target

    @torch.no_grad()
    def forward(self, host, identities, frames):
        noise_enabled = self.use_tta and bool(self.columns) and self.noise_level != 0
        tasks = (dict(rows=len(host), sequence_length=int(host.shape[1]), columns_to_noise=self.columns,
                      seed=self.seed, recording_identities=identities, source_frames=frames,
                      model_identity=identity, pass_index=p)
                 for identity in self.model_identities for p in range(self.passes)) if noise_enabled else ()
        buffers = self.noise.buffers(tasks)
        events = []

        def start_event():
            if self.profile and self.device.type == 'cuda':
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event

        def finish_event(start, kind):
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                events.append((kind, start, end))

        try:
            event = start_event()
            sequences = host.to(self.device, non_blocking=True)
            finish_event(event, 'transfer_and_noise_cuda_seconds')
            pose, orient, contexts = [], [], []
            for model_idx, model in enumerate(self.models):
                acc_pose = acc_orient = acc_context = cached = None
                for pass_idx in range(self.passes):
                    if not self.use_tta and cached is not None:
                        logits_pose, logits_orient, context = cached
                    else:
                        values = sequences
                        if noise_enabled:
                            started = time.perf_counter()
                            noise = next(buffers)
                            self.metrics['noise_wait_seconds'] += time.perf_counter() - started
                            started = time.perf_counter()
                            noise = self.stage_noise(noise)
                            self.metrics['host_noise_staging_seconds'] += time.perf_counter() - started
                            event = start_event()
                            try:
                                values = apply_inference_noise(sequences, noise, self.columns, self.noise_level)
                            finally:
                                if self.device.type == 'cuda':
                                    # Also cover an OOM after the copy was enqueued.
                                    self.pinned_copy_done = torch.cuda.Event()
                                    self.pinned_copy_done.record()
                            # Transfer + noise application; no claim of transfer alone.
                            finish_event(event, 'transfer_and_noise_cuda_seconds')
                            del noise
                        event = start_event()
                        with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                            logits_pose, logits_orient, _, context = model(values)
                        finish_event(event, 'forward_cuda_seconds')
                        if not self.use_tta:
                            cached = (logits_pose, logits_orient, context)
                        del values
                    if self.pose_pred:
                        acc_pose = logits_pose.clone() if acc_pose is None else acc_pose + logits_pose
                    if self.orient_pred:
                        acc_orient = logits_orient.clone() if acc_orient is None else acc_orient + logits_orient
                    acc_context = context.clone() if acc_context is None else acc_context + context
                if self.pose_pred:
                    result = acc_pose / self.passes
                    if self.pose_permutations and self.pose_permutations[model_idx] is not None:
                        result = result[:, self.pose_permutations[model_idx]]
                    pose.append(result)
                if self.orient_pred:
                    result = acc_orient / self.passes
                    if self.orient_permutations and self.orient_permutations[model_idx] is not None:
                        result = result[:, self.orient_permutations[model_idx]]
                    orient.append(result)
                contexts.append(acc_context / self.passes)
            # Keep softmax/ensemble arithmetic on the original device. Only profiling
            # synchronizes here; the caller already copies results after this point.
            if events:
                events[-1][2].synchronize()
            for kind, start, end in events:
                self.metrics[kind] += start.elapsed_time(end) / 1000
            return pose, orient, contexts
        finally:
            buffers.close()


def cuda_attempt(call):
    """Drop the failed forward's traceback/tensors before releasing cached blocks."""
    try:
        return True, call()
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return False, None


def choose_calibration(samples):
    """Use the smallest batch within 5% of the best observed windows/second."""
    best = max(s['windows_per_second'] for s in samples)
    return min(s['batch'] for s in samples if s['windows_per_second'] >= best * .95)


def calibrate(runner, host, identities, frames, *, vram_fraction=.8):
    """Bounded geometric growth, real ensemble forwards, 20% device headroom.

    Two timed trials per size after a small warm-up. Stop after two sizes without
    a >5% gain, at the host limit, or before projected peaks exceed the VRAM ceiling.
    Memory includes all resident models. Foreign allocations reduce the ceiling.
    """
    if not 0 < vram_fraction <= .9:
        raise ValueError('vram_fraction must be > 0 and <= 0.9.')
    device = runner.device
    if device.type != 'cuda':
        return min(256, len(host)), []
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    baseline = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device)
    ceiling = min(total * vram_fraction, baseline + max(0, free + reusable - baseline) * vram_fraction)
    if baseline >= ceiling:
        raise RuntimeError('Resident models leave insufficient VRAM headroom on the selected GPU.')
    samples, best_rate, plateau = [], 0., 0
    size = min(256, len(host))
    # Warm-up kernels without touching output records or the run's random streams.
    warm = min(32, size)
    ok, output = cuda_attempt(lambda: runner.forward(host[:warm], identities[:warm], frames[:warm]))
    del output
    if not ok:
        size = 1
    while size >= 1:
        rates, peak, failed = [], 0, False
        for _ in range(2):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            ok, output = cuda_attempt(lambda: runner.forward(host[:size], identities[:size], frames[:size]))
            del output
            if not ok:
                failed = True
                break
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = max(peak, torch.cuda.max_memory_allocated(device))
            rates.append(size / max(elapsed, 1e-9))
        if failed or peak > ceiling:
            if samples:
                break
            if size == 1:
                raise RuntimeError('One inference window cannot fit with the configured VRAM headroom.')
            size = max(1, size // 2)
            continue
        rate = sum(rates) / len(rates)
        samples.append(dict(batch=size, windows_per_second=rate, peak_allocated_bytes=peak))
        print(f'Inference calibration: batch {size}, {rate:.1f} windows/s, peak allocated {peak/GIB:.2f} GiB.')
        plateau = plateau + 1 if rate < best_rate * 1.05 else 0
        best_rate = max(best_rate, rate)
        next_size = min(len(host), size * 2)
        projected = baseline + (peak - baseline) * (next_size / size) * 1.2
        if next_size == size or plateau >= 2 or projected >= ceiling:
            break
        size = next_size
    return choose_calibration(samples), samples


def computed_batches(dataloader, runner, identity_for, *, requested_batch, tuning_state,
                     vram_fraction=.8):
    """Yield only successful, uncommitted slices; OOM halves the remaining slices.

    A shared state reuses calibration across recordings and retains OOM backoff.
    No output is appended until a complete slice/ensemble has succeeded.
    """
    started = time.perf_counter()
    iterator = iter(dataloader)
    tuning_state['loader_startup_seconds'] = tuning_state.get('loader_startup_seconds', 0.) + time.perf_counter() - started
    while True:
        started = time.perf_counter()
        try:
            host, filenames, indices = next(iterator)
        except StopIteration:
            break
        tuning_state['loader_wait_seconds'] = tuning_state.get('loader_wait_seconds', 0.) + time.perf_counter() - started
        identities = [identity_for(name) for name in filenames] if runner.use_tta else []
        frames = indices.detach().cpu().numpy()
        input_limited = (not requested_batch and not tuning_state.get('oom_retries')
                         and tuning_state.get('calibration_host_windows', len(host)) < len(host)
                         and tuning_state.get('batch') == tuning_state.get('calibration_host_windows'))
        if 'batch' not in tuning_state or input_limited:
            if requested_batch:
                tuning_state['batch'] = requested_batch
            else:
                before = dict(runner.metrics)
                started = time.perf_counter()
                size, samples = calibrate(runner, host, identities, frames, vram_fraction=vram_fraction)
                tuning_state.update(batch=size, calibration=samples, calibration_host_windows=len(host),
                    calibration_seconds=tuning_state.get('calibration_seconds', 0.) + time.perf_counter()-started)
                runner.metrics = before  # Calibration is not newly completed scientific work.
                if runner.device.type == 'cuda':
                    # Report production peaks separately from the discarded probes.
                    torch.cuda.reset_peak_memory_stats(runner.device)
            print(f'Inference execution batch: {tuning_state["batch"]} windows.')
        offset = 0
        while offset < len(host):
            count = min(tuning_state['batch'], len(host) - offset)
            stop = offset + count
            ok, outputs = cuda_attempt(lambda: runner.forward(host[offset:stop], identities[offset:stop], frames[offset:stop]))
            if not ok:
                if count == 1:
                    raise RuntimeError('CUDA OOM for one window; increase GPU memory or reduce model memory.')
                tuning_state['batch'] = max(1, count // 2)
                tuning_state['oom_retries'] = tuning_state.get('oom_retries', 0) + 1
                print(f'CUDA memory backoff: retrying uncommitted windows with batch {tuning_state["batch"]}.')
                continue
            yield filenames[offset:stop], indices[offset:stop], outputs
            del outputs
            offset = stop
