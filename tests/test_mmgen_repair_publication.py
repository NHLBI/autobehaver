"""Real process-death/ownership checks and retained-store publication contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from autobehaver.atomic_io import atomic_write_json, sha256_file
from autobehaver.postinf.mmgen import core, matrix_parts, pipeline, publication as pub
from autobehaver.postinf.mmgen.integrity import STORE_IDENTITY_FILENAME, bind_store_identity, read_store_identity
from autobehaver.postinf.mmgen.locks import canonical_writer, ShardLock, shard_lock_active
from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings
from tests.test_mmgen_repair_contracts import append_case, seal_generation, recording_name, write_pair


def hashes(root):
    return {name: sha256_file(root / name) for name in
            (*pub.CORE_ARTIFACTS, pub.IDENTITY_RELATIVE_PATH, pub.OUTPUT_RECEIPT_FILENAME)}


def run_child(code, *arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-B', '-c', code, *map(str, arguments)],
                          capture_output=True, text=True, env=env, timeout=45)


def test_canonical_modes_exclude_other_process_before_mutation(append_case):
    inf, pos, root, cfg, first = append_case
    before = hashes(root)
    code = '''
import sys
from autobehaver.postinf.mmgen.core import MMGenConfig, run_mmgen
from autobehaver.postinf.mmgen import pipeline, append_mode, matrix_parts
cfg=MMGenConfig(inference_csv_dir=sys.argv[1],poscalc_dir=sys.argv[2],output_dir=sys.argv[3],num_workers=1)
calls=[lambda:run_mmgen(cfg),lambda:run_mmgen(MMGenConfig(inference_csv_dir=sys.argv[1],poscalc_dir=sys.argv[2],output_dir=sys.argv[3],merge_shards=True,num_shards=2)),
lambda:matrix_parts.merge_matrix_part_files(output_dir=sys.argv[3],num_shards=2),
lambda:pipeline.run_post_inference_analysis(sys.argv[1],sys.argv[2],sys.argv[3],5,5,'_inference'),
lambda:append_mode._verified_generation(sys.argv[3])]
for call in calls:
    try:call()
    except RuntimeError as exc:
        assert 'Another writer currently holds' in str(exc),str(exc)
    else:raise AssertionError('Canonical writer was not excluded')
'''
    with canonical_writer(root):
        result = run_child(code, inf, pos, root)
    assert result.returncode == 0, result.stderr
    assert hashes(root) == before


@pytest.mark.parametrize('phase', ['backing_up', 'publishing', 'rolling_back'])
def test_killed_generation_recovers_all_originals_and_releases_lock(append_case, phase):
    inf, pos, root, cfg, first = append_case
    staged = root / 'stage'
    seal_generation(staged, cfg, [first], value=99)
    originals = hashes(root)
    store = root / 'temp_store.h5'
    store.write_bytes(b'retained working store is not a published artifact')
    code = '''
import os,sys
from pathlib import Path
from autobehaver.postinf.mmgen import publication as p
root,stage,phase=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
real=os.replace
def interrupted(source,target):
    source,target=Path(source),Path(target)
    moved=real(source,target)
    backup_target=any(x.startswith('published_backup_') for x in target.parts)
    backup_source=any(x.startswith('published_backup_') for x in source.parts)
    if ((phase=='backing_up' and backup_target)
        or (phase=='publishing' and source.name.endswith('.publish'))
        or (phase=='rolling_back' and backup_source)):
        os._exit(71)
    return moved
os.replace=interrupted
if phase=='rolling_back':
    with p.canonical_generation(root):
        (root/'MASTER_FEATURE_MATRIX.csv').write_bytes(b'partial replacement')
        raise RuntimeError('injected refresh failure')
else:
    p._publish_generation(root,stage,(*p.CORE_ARTIFACTS,p.IDENTITY_RELATIVE_PATH,p.OUTPUT_RECEIPT_FILENAME))
'''
    result = run_child(code, root, staged, phase)
    assert result.returncode == 71, result.stderr
    journal = json.loads((root / pub.PUBLICATION_JOURNAL).read_text())
    assert 'temp_store.h5' not in journal['originals']
    assert STORE_IDENTITY_FILENAME not in journal['originals']
    protected_bytes = sum((Path(journal['backup_dir']) / name).stat().st_size
                          if (Path(journal['backup_dir']) / name).is_file() else (root / name).stat().st_size
                          for name in journal['originals'])
    (root / 'publication_scope.json').write_text(json.dumps({'files': list(journal['originals']),
        'protected_bytes': protected_bytes, 'excluded': ['temp_store.h5', STORE_IDENTITY_FILENAME]}))
    pub._verified_generation(root)
    assert hashes(root) == originals
    assert store.read_bytes() == b'retained working store is not a published artifact'
    assert not (root / pub.PUBLICATION_JOURNAL).exists()
    with canonical_writer(root):
        pass


def test_ordinary_failed_refresh_restores_generation_and_retains_new_store(append_case, monkeypatch):
    inf, pos, root, cfg, first = append_case
    before = hashes(root)
    new_identity = {'calculation': {'new': True}, 'sources': {}}
    def fail(_cfg):
        bind_store_identity(root / '_mmgen_work', new_identity['calculation'], {}, reuse=False)
        (root / 'temp_store.h5').write_bytes(b'new retryable work')
        (root / 'MASTER_FEATURE_MATRIX.csv').write_bytes(b'partial new matrix')
        raise RuntimeError('injected calculation failure')
    monkeypatch.setattr(core, '_dispatch_mmgen', fail)
    with pytest.raises(RuntimeError, match='injected calculation failure'):
        core.run_mmgen(replace(cfg, append_new_data=False))
    assert hashes(root) == before
    assert (root / 'temp_store.h5').read_bytes() == b'new retryable work'
    assert read_store_identity(root / '_mmgen_work')['calculation'] == new_identity['calculation']
    bind_store_identity(root / '_mmgen_work', new_identity['calculation'], {}, reuse=True)
    pub._verified_generation(root)


def test_process_only_does_not_replace_published_identity(append_case, monkeypatch):
    inf, pos, root, cfg, first = append_case
    before = hashes(root)
    def process(*args, **kwargs):
        bind_store_identity(root / '_mmgen_work', {'pending': True}, {}, reuse=False)
        (root / 'temp_store.h5').write_bytes(b'pending finalization')
    monkeypatch.setattr(pipeline, '_run_post_inference_analysis_owned', process)
    # The real public config and dispatcher exercise the ownership decision;
    # the processing fixture intentionally produces no completed output receipt.
    monkeypatch.setattr(pipeline, 'run_post_inference_analysis', process)
    core.run_mmgen(replace(cfg, append_new_data=False, process_only=True))
    assert hashes(root) == before
    assert read_store_identity(root / '_mmgen_work')['calculation'] == {'pending': True}
    pub._verified_generation(root)


@pytest.mark.parametrize('held_name', ['_mmgen_shard.lock', '_mmgen_shard_matrix.lock'])
def test_empty_finalize_owns_validation_and_releases_partial_acquisition(tmp_path, held_name):
    shard = Path(matrix_parts.get_shard_output_dir(str(tmp_path), 2, 0))
    work = shard / '_mmgen_work'
    work.mkdir(parents=True)
    (shard / 'temp_store.h5').write_bytes(b'must not be read or deleted before ownership')
    with ShardLock(work / held_name):
        with pytest.raises(RuntimeError, match='Another writer currently holds'):
            matrix_parts.finalize_shard_temp_store(inference_csv_dir=str(tmp_path), poscalc_dir=str(tmp_path),
                output_dir=str(tmp_path), min_bout=5, edge_skip=5, inf_tag='_inference',
                runtime_settings=MMGenRuntimeSettings(), file_filter=None, num_shards=2,
                shard_index=0, shard_by='row_identity')
    assert not (work / matrix_parts.SHARD_MATRIX_DONE_FILENAME).exists()
    assert (shard / 'temp_store.h5').read_bytes() == b'must not be read or deleted before ownership'
    assert not shard_lock_active(work / '_mmgen_shard.lock')


def test_append_keeps_old_reports_under_their_old_source_identity(append_case):
    from autobehaver.postinf.mmgen import append_mode as app
    inf, pos, root, cfg, first = append_case
    report = root / 'old_full_report.csv'
    report.write_text('old report\n')
    receipt = json.loads((root / pub.OUTPUT_RECEIPT_FILENAME).read_text())
    old_source_hash = receipt['sources_sha256']
    receipt['artifacts'][report.name] = sha256_file(report)
    atomic_write_json(receipt, root / pub.OUTPUT_RECEIPT_FILENAME)
    second = recording_name(movie=1)
    write_pair(inf, pos, second)
    plan, selected, _, batch = app._create_append_plan(cfg)
    seal_generation(batch, cfg, [first, second], value=7)
    app._apply_delta_batch_to_canonical(cfg=cfg, plan=plan, delta_output_dir=batch)
    current = pub._verified_generation(root)['receipt']
    assert current['sources_sha256'] != old_source_hash
    assert report.name not in current['artifacts']
    assert current['historical_artifacts'][report.name]['sources_sha256'] == old_source_hash
    assert current['historical_artifacts'][report.name]['sha256'] == sha256_file(report)
    assert report.read_text() == 'old report\n'


def test_failed_publication_removes_only_declared_new_files(append_case, monkeypatch):
    inf, pos, root, cfg, first = append_case
    before = hashes(root)
    stage = root / 'stage'
    seal_generation(stage, cfg, [first], value=19)
    (stage / 'MASTER_FEATURE_MATRIX_SCHEMA.json').write_text('{}')
    unrelated = root / 'author_notes.txt'
    unrelated.write_text('preserve me')
    replace_file = os.replace
    def fail_receipt(source, target):
        if Path(source).name == pub.OUTPUT_RECEIPT_FILENAME + '.publish':
            raise OSError('injected receipt failure')
        return replace_file(source, target)
    monkeypatch.setattr(pub.os, 'replace', fail_receipt)
    with pytest.raises(OSError, match='receipt failure'):
        pub._publish_generation(root, stage, (*pub.CORE_ARTIFACTS, 'MASTER_FEATURE_MATRIX_SCHEMA.json',
            pub.IDENTITY_RELATIVE_PATH, pub.OUTPUT_RECEIPT_FILENAME))
    assert hashes(root) == before
    assert not (root / 'MASTER_FEATURE_MATRIX_SCHEMA.json').exists()
    assert unrelated.read_text() == 'preserve me'


def test_full_refresh_failure_restores_report_retained_as_history_after_append(append_case):
    from autobehaver.postinf.mmgen import append_mode as app
    inf, pos, root, cfg, first = append_case
    report = root / '1_Individual_Bout_Summary.csv'
    report.write_text('the original full report\n')
    receipt = json.loads((root / pub.OUTPUT_RECEIPT_FILENAME).read_text())
    receipt['artifacts'][report.name] = sha256_file(report)
    atomic_write_json(receipt, root / pub.OUTPUT_RECEIPT_FILENAME)
    second = recording_name(movie=1)
    write_pair(inf, pos, second)
    plan, selected, _, batch = app._create_append_plan(cfg)
    seal_generation(batch, cfg, [first, second], value=7)
    app._apply_delta_batch_to_canonical(cfg=cfg, plan=plan, delta_output_dir=batch)
    before = hashes(root)
    historical = pub._verified_generation(root)['receipt']['historical_artifacts']
    assert report.name in historical
    with pytest.raises(RuntimeError, match='refresh failed'):
        with pub.canonical_generation(root):
            report.write_text('part of a newer report\n')
            (root / 'MASTER_FEATURE_MATRIX.csv').write_text('partial new matrix\n')
            raise RuntimeError('refresh failed')
    assert hashes(root) == before
    assert report.read_text() == 'the original full report\n'
    assert pub._verified_generation(root)['receipt']['historical_artifacts'] == historical
    report.write_text('unverified external replacement\n')
    with pytest.raises(RuntimeError, match='Canonical artifact changed or missing'):
        pub._verified_generation(root)
