"""Small independent checks for shared identity, reporting and recovery rules."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autobehaver.batch_inputs import select_batch_inputs
from autobehaver.portable_paths import generated_path, original_name, recording_name
from autobehaver.recording_identity import derive_pair_key
from autobehaver.registry import RegistryLock, RegistryValidationError, read_registry, upsert_registry_rows
from autobehaver.scratch import scratch_workspace
from autobehaver.validation import StageValidationReport


def test_existing_mapped_name_survives_root_relocation(tmp_path):
    source = tmp_path / 'longer_root'
    source.mkdir()
    logical = 'MixedCase_' + 'a' * 160 + '_inference.csv'
    artifact = generated_path(source, logical)
    artifact.write_text('value\n1\n')
    moved = tmp_path / 'short'
    shutil.copytree(source, moved)
    resumed = generated_path(moved, logical)
    assert resumed.name == artifact.name
    assert resumed.read_text() == 'value\n1\n'
    assert original_name(resumed) == logical


@pytest.mark.parametrize('scorer', ['DLC_resnet50_ModelDLC_name', 'DLCresnet50_model'])
def test_biological_dlc_token_is_preserved(scorer):
    recording = 'DLC_experiment_MALE_trial1_rpi-1_movie0_chamber1'
    assert recording_name(recording + scorer) == recording
    assert recording_name(recording) == recording
    assert recording_name('experimentDLCx') == 'experimentDLCx'


@pytest.mark.parametrize('stage_suffix', ['_positional_calcs_normalized_inference', '_positional_calcs_inference'])
def test_batch_never_substitutes_a_prefixed_recording(tmp_path, stage_suffix):
    requested = 'G_MALE_exp_trial1_rpi-1_movie0_chamber1'
    listing = tmp_path / 'batch.txt'
    listing.write_text(requested + '\n')
    other = tmp_path / ('OLD_' + requested + '_positional_calcs.csv')
    with pytest.raises(ValueError, match='matched 0'):
        select_batch_inputs([other], listing)
    correct = tmp_path / (requested + stage_suffix + '.csv')
    assert select_batch_inputs([other, correct], listing) == [correct]


def test_pair_key_supports_24_hour_timestamps_without_matching_movie10():
    assert derive_pair_key('e_trial1_movie0_20261019120000.mp4') == derive_pair_key('e_trial1_movie1_20261019130000.mp4')
    assert derive_pair_key('e_trial1_movie10_20261019120000.mp4') != derive_pair_key('e_trial1_movie1_20261019130000.mp4')


def test_registry_round_trip_preserves_text_tokens(tmp_path):
    path = tmp_path / 'registry.csv'
    upsert_registry_rows(csv_path=path, rows=[{'recording_id':'0012', 'animal_id':'NA', 'genotype':'None', 'trial_number':1, 'active':True}])
    row = read_registry(path).iloc[0]
    assert row['recording_id'] == '0012'
    assert row['animal_id'] == 'NA'
    assert row['genotype'] == 'None'
    assert row['trial_number'] == 1
    upsert_registry_rows(csv_path=path, rows=[{'recording_id':'0012', 'notes':'updated'}])
    result = read_registry(path)
    assert len(result) == 1 and result.iloc[0]['notes'] == 'updated'


def test_legacy_registry_directory_is_preserved(tmp_path):
    path = tmp_path / '.autobehaver_registry.lock'
    path.mkdir()
    with pytest.raises(RegistryValidationError, match='Legacy registry lock directory'):
        RegistryLock(path, timeout_sec=0, stale_sec=0).acquire()
    assert path.is_dir()


def test_registry_lock_releases_after_owner_exits(tmp_path):
    path = tmp_path / '.autobehaver_registry.lock'
    command = 'from pathlib import Path; from autobehaver.registry import RegistryLock; import os,sys; lock=RegistryLock(Path(sys.argv[1]),timeout_sec=0); lock.acquire(); os._exit(0)'
    subprocess.run([sys.executable, '-c', command, str(path)], check=True, timeout=20)
    assert path.is_file()
    with RegistryLock(path, timeout_sec=0):
        assert path.is_file()


@pytest.mark.skipif(os.name != 'nt', reason='Foreign POSIX root on Windows')
def test_foreign_registry_root_is_not_resolved_on_current_windows_drive(tmp_path):
    from autobehaver.registry import resolve_registry_path, _storage_key
    row = {'dlc_csv_relpath':'.autobehaver_roots/'+_storage_key('/data')+'/example.csv', 'dlc_csv_storage_root':'/data'}
    with pytest.raises(RegistryValidationError, match='explicit mounted Windows'):
        resolve_registry_path(tmp_path, row, 'dlc_csv_relpath')


def test_source_publication_excludes_development_and_nested_environment_files(tmp_path):
    from autobehaver.job_publication import publish_jobs
    source, prepared = tmp_path/'source', tmp_path/'prepared'
    prepared.mkdir()
    (prepared/'submit.sh').write_text('# fixture\n')
    names = ['autobehaver/main.py','autobehaver/schema.json','scripts/native.sh',
             'scripts/validate.ps1','scripts/validate.cmd',
             '.env','.env.local','.venv/fake.py','test_data/movie.mp4',
             'autobehaver/.env.local','autobehaver/.venv/fake.py','notes_private.txt']
    for name in names:
        path = source/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('inert fixture only')
    target = tmp_path/'published'
    receipt = publish_jobs(prepared,target,source_root=source)
    assert set(receipt['files']) == {'submit.sh','package/autobehaver/main.py',
                                  'package/autobehaver/schema.json','package/scripts/native.sh',
                                  'package/scripts/validate.ps1','package/scripts/validate.cmd'}


def test_early_status_does_not_poison_completed_report():
    report = StageValidationReport.for_items('test', ['a'])
    assert report.status == 'failed'
    report.add_outcome('a', 'succeeded')
    assert report.status == 'passed'
    assert not report.issues


def test_batch_reports_survive_later_batches(tmp_path):
    latest = tmp_path / 'stage.json'
    saved = []
    for item in ['a', 'b']:
        report = StageValidationReport.for_items('test', [item])
        report.add_outcome(item, 'succeeded')
        saved.append(report.write_run_reports(latest))
    assert saved[0] != saved[1]
    assert set(json.loads(saved[0].read_text())['outcomes']) == {'a'}
    assert set(json.loads(saved[1].read_text())['outcomes']) == {'b'}
    assert latest.read_bytes() == saved[1].read_bytes()


def test_scratch_cleanup_error_preserves_success_and_warns(tmp_path, monkeypatch, caplog):
    def cannot_delete(path):
        raise PermissionError('sharing conflict')
    with monkeypatch.context() as context:
        context.setattr('autobehaver.scratch.shutil.rmtree', cannot_delete)
        with scratch_workspace('test', tmp_path) as work:
            (work / 'small.txt').write_text('done')
    assert work.is_dir()
    assert 'Scratch cleanup incomplete' in caplog.text
