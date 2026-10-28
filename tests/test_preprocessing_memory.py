"""Memory planning, temporal seam conservation, and killed-worker regressions."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from concurrent.futures.process import BrokenProcessPool

from autobehaver.preinf import execution
from autobehaver.preinf.keypre import worker
from autobehaver.preinf.keypre.dlc_io import dlc_chunks, load_dlc_csv_flat
from autobehaver.preinf.keypre.spec import parts


def make_dlc(path, length, origin=0):
    rng = np.random.default_rng(31)
    columns = pd.MultiIndex.from_product([['DLC'], parts + ['Front', 'Back'], ['x', 'y', 'likelihood']])
    values = rng.normal(300, 20, (length, len(columns)))
    values[:, 2::3] = rng.uniform(.1, 1, (length, len(parts)+2))
    # Motion, angle wraps, stationary periods, missing coordinates across seams.
    if length > 100:
        values[15:30] = values[15]
        values[61:67, 0] = np.nan
        values[127:131, 4] = np.nan
        values[255:260, -6:] = np.nan
    frame = pd.DataFrame(values, columns=columns, index=np.arange(origin, origin+length))
    frame.to_csv(path)
    return frame


@pytest.mark.parametrize('length', [0, 1, 5, 10, 11, 63, 64, 65, 71, 128, 129, 513])
@pytest.mark.parametrize('origin', [0, 1])
def test_streamed_features_match_whole_recording(tmp_path, length, origin):
    source = tmp_path / 'recording.csv'
    make_dlc(source, length, origin)
    reference = worker.derive_features(load_dlc_csv_flat(source)).astype(np.float32)
    if length > 10:
        reference.iloc[:5] = np.nan
        reference.iloc[-5:] = np.nan
    out = tmp_path / 'out'; out.mkdir()
    worker.set_run_config(dict(output_dir=str(out), clobber=True, movie_dim=600, chunk_rows=64))
    result = worker.process_csv_file(str(source))
    assert result['status'] == 'success', result
    actual = pd.read_csv(result['out'], dtype=np.float64)
    assert list(actual) == list(reference)
    assert len(actual) == length
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
    # Float32 serialization plus different rolling-state summation order.
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-5, equal_nan=True)
    assert result['inspection'][2] == (True, 'ok', length, 829)


def test_dlc_chunk_memory_and_frame_continuity(tmp_path):
    source = tmp_path / 'recording.csv'
    frame = make_dlc(source, 250)
    chunks = list(dlc_chunks(source, chunk_rows=32, halo=7))
    assert all(len(frame) <= 32 + 2*7 for frame, *_ in chunks)
    retained = [frame.iloc[start:stop] for frame, start, stop, _, _ in chunks]
    pd.testing.assert_frame_equal(pd.concat(retained), load_dlc_csv_flat(source))
    frame.index = list(range(32)) + list(range(33, 251))
    frame.to_csv(source)
    with pytest.raises(ValueError, match='chunk boundary'):
        list(dlc_chunks(source, chunk_rows=32, halo=7))


@pytest.fixture
def controlled_resources(monkeypatch):
    for name in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(execution, '_cgroup_limits', lambda: [])
    monkeypatch.setattr(execution, 'cpu_limit', lambda: 8)
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=128*execution.GIB))


def test_memory_budget_selects_workers_and_chunks(controlled_resources):
    large = execution.plan_execution('keypre', files=20, columns=829, memory_gb=32)
    small = execution.plan_execution('keypre', files=20, columns=829, memory_gb=4)
    assert large.workers == 8 and large.chunk_rows == 16384
    assert small.chunk_rows < large.chunk_rows
    for plan in (small, large):
        assert plan.workers * plan.worker_allowance_bytes + plan.reserve_bytes <= plan.memory_bytes
    tiny = execution.plan_execution('keypre', files=20, columns=829, memory_gb=2)
    assert tiny.workers < 8
    assert execution.plan_execution('keyprenorm', files=20, columns=829, memory_gb=32).workers == 8
    assert execution.plan_execution('keypre', files=20, columns=829, memory_gb=32, num_workers=2).workers == 2


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0, .2])
def test_invalid_or_inadequate_memory_rejected(controlled_resources, value):
    with pytest.raises(ValueError):
        execution.plan_execution('keypre', files=8, columns=829, memory_gb=value)


def test_slurm_and_cgroup_override_oversized_argument(controlled_resources, monkeypatch):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '8192')
    assert execution.memory_limit(64)[0][1] == 8*execution.GIB
    monkeypatch.setattr(execution, '_cgroup_limits', lambda: [('job', 4*execution.GIB)])
    assert execution.memory_limit(64)[0][1] == 4*execution.GIB
    monkeypatch.delenv('SLURM_MEM_PER_NODE')
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '2G')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    monkeypatch.setattr(execution, '_cgroup_limits', lambda: [])
    assert execution.memory_limit()[0][1] == 16*execution.GIB


def test_unknown_slurm_allocation_never_uses_node_ram(controlled_resources, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises(ValueError, match='allocated GiB'):
        execution.memory_limit()
    assert execution.memory_limit(8)[0][1] == 8*execution.GIB


@pytest.mark.parametrize('version', [1, 2])
def test_cgroup_reads_current_membership_and_stricter_parent(monkeypatch, version):
    mount = '/sys/fs/cgroup' if version == 2 else '/sys/fs/cgroup/memory'
    control = 'memory.max' if version == 2 else 'memory.limit_in_bytes'
    files = {
        '/proc/self/cgroup': '0::/slurm/job/step' if version == 2 else '5:memory:/slurm/job/step',
        '/proc/self/mountinfo': f'1 0 0:1 /slurm {mount} rw - '+('cgroup2 cgroup rw' if version == 2 else 'cgroup cgroup rw,memory'),
        f'{mount}/job/step/{control}': str(8*execution.GIB),
        f'{mount}/job/{control}': str(4*execution.GIB),
        f'{mount}/{control}': 'max' if version == 2 else str(1<<62),
    }
    def read(path, *args, **kwargs):
        value = files.get(path.as_posix())
        if value is None:
            raise FileNotFoundError(str(path))
        return value
    monkeypatch.setattr(Path, 'read_text', read)
    assert min(value for _, value in execution._cgroup_limits()) == 4*execution.GIB


def test_cli_accepts_explicit_budget_and_auto_workers():
    from autobehaver.preinf.keypre.cli import build_arg_parser as keypre_parser, _config_from_args as keypre_config
    from autobehaver.preinf.keyprenorm.cli import build_arg_parser as norm_parser, _config_from_args as norm_config
    for parser, config in ((keypre_parser, keypre_config), (norm_parser, norm_config)):
        value = config(parser().parse_args(['--input_dir', 'source', '--memory_gb', '12']))
        assert value.num_workers == 0 and value.memory_gb == 12


def test_killed_worker_marks_keypre_manifest_failed(tmp_path, monkeypatch):
    from autobehaver.preinf.keypre import core
    from autobehaver.preinf.keypre.config import KeypreConfig
    inputs = tmp_path/'in'; inputs.mkdir()
    make_dlc(inputs/'recording.csv', 11)
    def broken(*args, **kwargs):
        raise BrokenProcessPool('simulated killed worker')
    monkeypatch.setattr(core, 'execute_workers', broken)
    with pytest.raises(BrokenProcessPool):
        core.run_keypre(cfg=KeypreConfig(input_dir=str(inputs), output_dir=str(tmp_path/'out'), memory_gb=4))
    manifest = json.loads(next((tmp_path/'out').glob('keypre_run_manifest_*.json')).read_text())
    assert manifest['status'] == 'failed'
    assert 'killed worker' in manifest['error']


def _abrupt_exit(value):
    if value == 0:
        os._exit(7)
    return value


def _square(value):
    return value * value


def test_process_order_and_recycling():
    assert list(execution.worker_results(_square, range(7), num_workers=2, tasks_per_worker=2)) == [x*x for x in range(7)]


def test_worker_death_raises_instead_of_idle_pool():
    with pytest.raises(BrokenProcessPool):
        list(execution.worker_results(_abrupt_exit, range(4), num_workers=2))


def test_validation_reads_bounded_blocks_and_rejects_corruption(tmp_path, monkeypatch):
    from autobehaver.preinf.keypre.contract import inspect_keypre_output
    source = tmp_path / 'recording.csv'; make_dlc(source, 100)
    out = tmp_path / 'out'; out.mkdir()
    worker.set_run_config(dict(output_dir=str(out), clobber=True, movie_dim=600, chunk_rows=64))
    result = worker.process_csv_file(str(source)); assert result['status'] == 'success'
    read = pd.read_csv
    def bounded(*args, **kwargs):
        assert kwargs.get('chunksize') == 4096
        return read(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', bounded)
    assert inspect_keypre_output(source, result['out'], require_receipt=True)[0]
    with open(result['out'], 'a') as stream:
        stream.write(','.join(['inf'] * 829)+'\n')
    assert not inspect_keypre_output(source, result['out'], require_receipt=True)[0]


def test_source_change_does_not_commit(tmp_path, monkeypatch):
    source = tmp_path / 'recording.csv'; make_dlc(source, 100)
    out = tmp_path / 'out'; out.mkdir()
    worker.set_run_config(dict(output_dir=str(out), clobber=True, movie_dim=600, chunk_rows=64))
    original = worker.derive_features
    def mutate(frame, **kwargs):
        with open(source, 'a') as stream:
            stream.write('\n')
        return original(frame, **kwargs)
    monkeypatch.setattr(worker, 'derive_features', mutate)
    result = worker.process_csv_file(str(source))
    assert result['status'] == 'failed'
    assert not list(out.glob('*.csv'))


def test_dropped_feature_row_does_not_commit(tmp_path, monkeypatch):
    source = tmp_path / 'recording.csv'; make_dlc(source, 100)
    out = tmp_path / 'out'; out.mkdir()
    worker.set_run_config(dict(output_dir=str(out), clobber=True, movie_dim=600, chunk_rows=64))
    original = worker.derive_features
    monkeypatch.setattr(worker, 'derive_features', lambda frame, **kwargs: original(frame, **kwargs).iloc[:-1])
    result = worker.process_csv_file(str(source))
    assert result['status'] == 'failed'
    assert 'complete source timeline' in result['error']
    assert not list(out.glob('*.csv'))


def test_parallel_statistics_and_normalization_end_to_end(tmp_path):
    from autobehaver.preinf.keyprenorm.cli import main
    import h5py
    inputs = tmp_path/'input'; inputs.mkdir()
    out = tmp_path/'output'
    arrays = []
    for index in range(3):
        frame = pd.DataFrame({'a':np.arange(257, dtype=np.float32)+index*7,
                              'b':np.sin(np.arange(257, dtype=np.float32)),
                              'constant':np.ones(257, dtype=np.float32)})
        frame.to_csv(inputs/f'file{index}_positional_calcs.csv', index=False)
        arrays.append(pd.read_csv(inputs/f'file{index}_positional_calcs.csv', dtype=np.float32).to_numpy()[5:-5])
    main(['--input_dir', str(inputs), '--output_dir', str(out), '--calculate_norm',
          '--num_workers', '2', '--batch_size', '1', '--memory_gb', '4'])
    parameters = pd.read_csv(out/'norm_params.csv')
    values = np.concatenate(arrays).astype(np.float64)
    np.testing.assert_allclose(parameters.Mean, values.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parameters.StdDev, values.std(axis=0), rtol=1e-12, atol=1e-12)
    for index, original in enumerate(arrays):
        means = parameters.Mean.to_numpy(np.float32); stds = parameters.StdDev.to_numpy(np.float32)
        expected = original-means
        np.divide(expected, stds, out=expected, where=stds>0); expected[:,stds<=0] = 0
        with h5py.File(out/f'file{index}_positional_calcs_normalized.h5') as hf:
            np.testing.assert_array_equal(hf['data'][:], expected)
            np.testing.assert_array_equal(hf['source_frame_indices'][:], np.arange(5,252))
    manifest = json.loads(next(out.glob('keyprenorm_run_manifest_*.json')).read_text())
    assert manifest['status'] == 'complete'
    assert manifest['execution_plan']['workers'] == 2
