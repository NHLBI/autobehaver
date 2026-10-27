"""Memory admission, exact parallel noise, ensemble arithmetic and safe retry."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
import torch

from autobehaver import inference_runtime as runtime
from autobehaver.augmentation import apply_inference_noise
from tests._frozen_tta_v1 import augment_inference_features as frozen_noise


def test_joint_budget_includes_all_queues_and_reduces_for_long_recordings(monkeypatch):
    monkeypatch.setattr(runtime, 'cpu_limit', lambda: 12)
    monkeypatch.setattr(runtime, 'memory_limit', lambda value: (('test', int(value * runtime.GIB)), []))
    kwargs = dict(rows=72000, total_frames=72010, columns=820, sequence_length=41,
                  model_widths=[1024]*3, classes=15, current_rss=2*runtime.GIB)
    plans = [runtime.plan_inference(**kwargs, memory_gb=ram) for ram in (8, 16, 32, 64)]
    assert plans[-1].noise_workers == 8 and plans[-1].loader_workers == 2
    assert plans[0].host_batch < plans[-1].host_batch
    for plan in plans:
        assert plan.fixed_bytes + plan.host_batch * plan.bytes_per_window + plan.reserve_bytes <= plan.memory_bytes
    long = runtime.plan_inference(**{**kwargs, 'rows': 288000, 'total_frames': 288010}, memory_gb=32)
    assert long.host_batch < plans[2].host_batch
    with pytest.raises(ValueError, match='Insufficient RAM'):
        runtime.plan_inference(**{**kwargs, 'rows': 288000, 'total_frames': 288010}, memory_gb=16)
    with pytest.raises(ValueError, match='Insufficient RAM'):
        runtime.plan_inference(**{**kwargs, 'total_frames': 10**9}, memory_gb=32)


@pytest.mark.parametrize('cpu', [1, 2, 4, 8, 12])
def test_worker_counts_respect_cpu_allocation(monkeypatch, cpu):
    monkeypatch.setattr(runtime, 'cpu_limit', lambda: cpu)
    monkeypatch.setattr(runtime, 'memory_limit', lambda _: (('test', 32*runtime.GIB), []))
    plan = runtime.plan_inference(rows=70000, total_frames=70010, columns=820, sequence_length=41,
                                 model_widths=[1024]*3, classes=15, current_rss=runtime.GIB)
    assert plan.loader_workers + plan.noise_workers <= cpu
    assert plan.loader_workers == 0 if cpu < 3 else plan.loader_workers <= 2


@pytest.mark.parametrize('workers', [1, 2, 8])
def test_parallel_noise_matches_frozen_across_batches_order_and_models(workers):
    data = torch.arange(137*5*7, dtype=torch.float32).reshape(137, 5, 7) / 10
    order = np.random.default_rng(11).permutation(137)
    frames = np.arange(61, 198)
    identities = ['A']*80 + ['B']*57
    for model in ['m1', 'm2', 'm3']:
        expected = frozen_noise(data, [1, 4, 5], .01, seed=42, model_identity=model,
                                pass_index=3, recording_identities=identities, source_frames=frames)
        tasks, splits = [], np.array_split(order, 9)
        for subset in splits:
            tasks.append(dict(rows=len(subset), sequence_length=5, columns_to_noise=[1,4,5],
                seed=42, recording_identities=[identities[i] for i in subset], source_frames=frames[subset],
                model_identity=model, pass_index=3))
        actual = torch.empty_like(data)
        with runtime.NoiseQueue(workers, False) as queue:
            for subset, buffer in zip(splits, queue.buffers(tasks)):
                actual[subset] = apply_inference_noise(data[subset], buffer, [1,4,5], .01)
        assert torch.equal(actual, expected)


def test_queue_is_bounded_and_drains_on_early_exit(monkeypatch):
    started, lock = [], threading.Lock()
    def make(**kwargs):
        with lock:
            started.append(kwargs['pass_index'])
        return torch.ones(2)
    monkeypatch.setattr(runtime, 'inference_noise', make)
    with runtime.NoiseQueue(3, False) as queue:
        stream = queue.buffers([dict(pass_index=i) for i in range(30)])
        next(stream)
        time.sleep(.02)
        assert len(started) <= 4
        stream.close()
    assert not any(t.name.startswith('posepred-noise') for t in threading.enumerate())


def test_queue_worker_exception_is_propagated_and_cleaned(monkeypatch):
    def fail(**kwargs):
        raise ValueError('invalid noise input')
    monkeypatch.setattr(runtime, 'inference_noise', fail)
    with pytest.raises(ValueError, match='invalid noise input'):
        with runtime.NoiseQueue(2, False) as queue:
            list(queue.buffers([dict(pass_index=i) for i in range(4)]))
    assert not any(t.name.startswith('posepred-noise') for t in threading.enumerate())


class TinyModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layer = torch.nn.Linear(7, 8)
        with torch.no_grad():
            self.layer.weight.copy_(torch.randn(8,7, generator=torch.Generator().manual_seed(seed)))
        self.d_model = 8

    def forward(self, x):
        context = self.layer(x[:, 2])
        return context[:, :3], context[:, 3:5], None, context


@pytest.mark.parametrize('tta', [True, False])
def test_three_model_ten_pass_logits_and_context_match_frozen_loop(tta):
    torch.set_num_threads(1)
    models = [TinyModel(i).eval() for i in range(3)]
    host = torch.linspace(-1, 1, 17*5*7).reshape(17,5,7)
    ids, frames = ['recording']*17, np.arange(60,77)
    expected = ([], [], [])
    with torch.no_grad():
        for idx, model in enumerate(models):
            sums = [None]*3
            for p in range(10):
                values = frozen_noise(host, [0,2,4], .01, seed=42, recording_identities=ids,
                    source_frames=frames, model_identity=str(idx), pass_index=p) if tta else host
                pose, orient, _, context = model(values)
                for i, value in enumerate((pose,orient,context)):
                    sums[i] = value.clone() if sums[i] is None else sums[i] + value
            for i, value in enumerate(sums):
                expected[i].append(value / 10)
    for workers in (1, 3, 8):
        with runtime.EnsembleRunner(models, torch.device('cpu'), passes=10, pose_pred=True,
            orient_pred=True, use_tta=tta, noise_level=.01, columns=[0,2,4], seed=42,
            model_identities=list(map(str, range(3))), noise_workers=workers) as runner:
            actual = runner.forward(host, ids, frames)
        for expected_group, actual_group in zip(expected, actual):
            for a, b in zip(expected_group, actual_group):
                assert torch.equal(a, b)
        assert torch.equal(torch.softmax(torch.stack(expected[0]).mean(0), 1),
                           torch.softmax(torch.stack(actual[0]).mean(0), 1))


def test_oom_retries_same_uncommitted_frames_and_retains_smaller_batch():
    attempted = []
    class Runner:
        use_tta = True
        def forward(self, host, ids, frames):
            attempted.append(list(frames))
            if len(host) > 3:
                raise torch.cuda.OutOfMemoryError('simulated')
            return host.clone()
    data = [(torch.arange(11).reshape(11,1,1), ['a']*11, torch.arange(11)),
            (torch.arange(11,16).reshape(5,1,1), ['a']*5, torch.arange(11,16))]
    state = {}
    results = list(runtime.computed_batches(data, Runner(), lambda name: name, requested_batch=8, tuning_state=state))
    assert torch.equal(torch.cat([row[1] for row in results]), torch.arange(16))
    assert torch.equal(torch.cat([row[2] for row in results]).flatten(), torch.arange(16))
    assert state['oom_retries'] == 2 and state['batch'] == 2
    assert attempted[:3] == [list(range(8)), list(range(4)), list(range(2))]


def test_non_cuda_errors_are_not_retried():
    with pytest.raises(ValueError, match='bad data'):
        runtime.cuda_attempt(lambda: (_ for _ in ()).throw(ValueError('bad data')))


def test_calibration_selects_smaller_near_plateau():
    assert runtime.choose_calibration([dict(batch=256, windows_per_second=100),
        dict(batch=512, windows_per_second=150), dict(batch=1024, windows_per_second=154)]) == 512


def test_calibration_headroom_and_oom_backoff_without_gpu(monkeypatch):
    state = dict(clock=0., peak=runtime.GIB, sizes=[])
    for name in ('empty_cache', 'synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *a: (7*runtime.GIB,8*runtime.GIB))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *a: runtime.GIB)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *a: runtime.GIB)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: state['peak'])
    monkeypatch.setattr(runtime.time, 'perf_counter', lambda: state['clock'])
    class Runner:
        device = torch.device('cuda')
        def forward(self, host, ids, frames):
            n = len(host)
            state['sizes'].append(n)
            if n > 128:
                raise torch.cuda.OutOfMemoryError('simulated')
            state['peak'] = runtime.GIB + n * 4 * runtime.MIB
            state['clock'] += .1 + n / 1000
            return host
    size, samples = runtime.calibrate(Runner(), torch.zeros(1024,1,1), ['a']*1024, np.arange(1024))
    assert size == 128
    assert all(s['peak_allocated_bytes'] < .8 * 8 * runtime.GIB for s in samples)
    assert 256 in state['sizes'] and 128 in state['sizes']


def test_inference_cli_auto_defaults_and_manual_overrides():
    from autobehaver.preinf.posepred_inference.cli import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.batch == 0 and args.num_workers == -1 and args.noise_workers == 0
    args = parser.parse_args(['--batch','512','--num_workers','0','--noise_workers','4','--memory_gb','16'])
    assert (args.batch, args.num_workers, args.noise_workers, args.memory_gb) == (512,0,4,16)


def test_short_recording_auto_avoids_loader_process_startup(monkeypatch):
    monkeypatch.setattr(runtime, 'cpu_limit', lambda: 12)
    monkeypatch.setattr(runtime, 'memory_limit', lambda _: (('test', 8*runtime.GIB), []))
    kwargs = dict(rows=500, total_frames=510, columns=820, sequence_length=41,
                  model_widths=[1024]*3, classes=15, current_rss=runtime.GIB)
    assert runtime.plan_inference(**kwargs).loader_workers == 0
    assert runtime.plan_inference(**kwargs, num_workers=2).loader_workers == 2
