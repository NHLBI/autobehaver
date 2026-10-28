"""Small CLI handoff and temporary-store lifecycle checks."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from autobehaver.argparse_utils import parse_cli_args


def test_mmgen_defaults_and_direct_feature_options(monkeypatch, tmp_path):
    from autobehaver.postinf.mmgen import cli, core
    from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings

    assert not cli.build_parser().parse_args([]).keep_temp_files
    assert not core.MMGenConfig(inference_csv_dir='inf', poscalc_dir='pos', output_dir='out').keep_temp_files
    assert not MMGenRuntimeSettings().keep_temp_files
    captured = []
    monkeypatch.setattr(core, 'run_mmgen', captured.append)
    cli.main(['--main-dir', str(tmp_path), '--no-measurements', '--no-postural',
              '--no-bout-count-filter', '--min-bout-count-for-features', '11',
              '--min-bout-count-for-transitions=17', '--keep-temp-files'])
    cfg = captured[0]
    assert not cfg.make_measurements and not cfg.calculate_postural_features
    assert not cfg.filter_features_by_bout_count
    assert cfg.min_bout_count_for_features == 11 and cfg.min_bout_count_for_transitions == 17
    assert cfg.keep_temp_files
    assert cfg.inference_csv_dir == str((tmp_path/'6_posepred_inference').resolve())


@pytest.mark.parametrize('module_name', ['mmgen', 'featureimp', 'infmovie'])
def test_global_yaml_options_are_rejected(module_name):
    import importlib
    module = importlib.import_module(f'autobehaver.postinf.{module_name}.cli')
    build = getattr(module, 'build_parser', getattr(module, 'build_arg_parser', None))
    assert '--config' not in build().format_help()
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(build(), ['--config', 'unused.yaml'])
    assert exc.value.code == 2


def test_infmovie_dataclass_defaults_and_registry_handoff(monkeypatch, tmp_path):
    from autobehaver.postinf.infmovie import cli, pipeline
    from autobehaver.postinf.infmovie.config import InfMoviePipelineConfig

    args = cli.build_parser().parse_args([])
    assert {key: getattr(args, key) for key in asdict(InfMoviePipelineConfig())} == asdict(InfMoviePipelineConfig())
    captured = []
    monkeypatch.setattr(pipeline, 'run_infmovie_pipeline', captured.append)
    cli.main(['--registry-enabled', '--registry-csv', 'records.csv', '--output-dir', str(tmp_path),
              '--mode', 'qc', '--no-qc-use-switching', '--default-movie-fps', '25'])
    cfg = captured[0]
    assert cfg.registry_enabled and cfg.registry_csv == 'records.csv'
    assert cfg.mode == 'qc' and not cfg.qc_use_switching and cfg.default_movie_fps == 25
    assert cfg.write_plan_csv and cfg.write_run_log


def test_featureimp_main_dir_and_explicit_keys_keep_saved_settings_semantics(monkeypatch, tmp_path):
    from autobehaver.postinf.featureimp import cli

    captured = {}
    class StopAfterSettings(Exception):
        pass
    def apply_settings(settings, **kwargs):
        captured.update(settings=settings, **kwargs)
        raise StopAfterSettings()
    monkeypatch.setattr(cli, '_import_core', lambda: SimpleNamespace(apply_settings=apply_settings))
    with pytest.raises(StopAfterSettings):
        cli.main(['--main-dir', str(tmp_path), '--matrix-dir', 'chosen-matrix', '--stage', 'process_shard',
                  '--num-shards=3', '--no-postanalysis-strict'])
    assert captured['settings']['matrix_dir'] == 'chosen-matrix'
    assert captured['settings']['results_dir'] == str((tmp_path/'9_feature_importance').resolve())
    assert captured['settings']['num_shards'] == 3
    assert captured['explicit_keys'] == {'main_dir', 'matrix_dir', 'stage', 'num_shards', 'postanalysis_strict'}
    assert not captured['propagate']


def test_verified_cleanup_keeps_manifests_and_rejects_changed_or_foreign_store(tmp_path):
    from autobehaver.atomic_io import sha256_file
    from autobehaver.postinf.mmgen.hdf_store import cleanup_verified_temp_stores

    owner = tmp_path/'run'; owner.mkdir()
    store = owner/'temp_store.h5'; store.write_bytes(b'temp')
    manifest = owner/'MMGEN_BATCH_MANIFEST.json'; manifest.write_bytes(b'manifest')
    original = sha256_file(store)
    store.write_bytes(b'changed')
    cleanup_verified_temp_stores(str(owner), [(str(store), original)])
    assert store.exists() and manifest.read_bytes() == b'manifest'
    other = tmp_path/'temp_store.h5'; other.write_bytes(b'foreign')
    cleanup_verified_temp_stores(str(owner), [(str(store), sha256_file(store)), (str(other), sha256_file(other))])
    assert store.exists() and other.exists()
    cleanup_verified_temp_stores(str(owner), [(str(store), sha256_file(store))])
    assert not store.exists() and other.exists() and manifest.exists()


def test_verified_cleanup_read_failure_is_optional_and_retains_stores(monkeypatch, tmp_path):
    from autobehaver import atomic_io
    from autobehaver.postinf.mmgen.hdf_store import cleanup_verified_temp_stores

    store = tmp_path/'temp_store.h5'; store.write_bytes(b'completed store')
    digest = atomic_io.sha256_file(store)
    def unreadable(path):
        raise PermissionError('injected temporary cleanup read failure')
    monkeypatch.setattr(atomic_io, 'sha256_file', unreadable)
    cleanup_verified_temp_stores(str(tmp_path), [(str(store), digest)])
    assert store.read_bytes() == b'completed store'


@pytest.mark.parametrize('keep,fail', [(False, False), (True, False), (False, True)])
def test_completed_merge_cleanup_and_failure_retention(monkeypatch, tmp_path, keep, fail):
    from autobehaver.atomic_io import sha256_file
    from autobehaver.postinf.mmgen import core

    inf = tmp_path/'inf'; pos = tmp_path/'pos'; out = tmp_path/'out'
    inf.mkdir(); pos.mkdir(); out.mkdir()
    shard = out/'_mmgen_shards'/'shard_000_of_002'; shard.mkdir(parents=True)
    paths = [out/'temp_store.h5', shard/'temp_store.h5']
    for p in paths:
        p.write_bytes(b'owned temporary data')
    marker = shard/'SHARD_DONE.json'; marker.write_bytes(b'provenance')
    def merge(**kwargs):
        kwargs['verified_store_identities'].extend((str(p), sha256_file(p)) for p in paths)
        return str(paths[0])
    def finalize(*args, **kwargs):
        assert kwargs['finalize_only']
        if fail:
            raise RuntimeError('finalization failed')
        (out/'MMGEN_OUTPUT_RECEIPT.json').write_bytes(b'checked publication')
    monkeypatch.setattr(core, 'merge_shard_temp_stores', merge)
    monkeypatch.setattr(core.pipeline, 'run_post_inference_analysis', finalize)
    cfg = core.MMGenConfig(inference_csv_dir=str(inf), poscalc_dir=str(pos), output_dir=str(out),
                          merge_shards=True, keep_temp_files=keep)
    if fail:
        with pytest.raises(RuntimeError, match='finalization failed'):
            core.run_mmgen(cfg)
    else:
        core.run_mmgen(cfg)
    assert all(p.exists() for p in paths) == bool(keep or fail)
    assert marker.read_bytes() == b'provenance'


def test_standalone_finalize_shard_retains_required_temp(monkeypatch, tmp_path):
    from autobehaver.postinf.mmgen import core

    inf = tmp_path/'inf'; pos = tmp_path/'pos'; inf.mkdir(); pos.mkdir()
    calls = []
    monkeypatch.setattr(core, 'finalize_shard_temp_store', lambda **kwargs: calls.append(kwargs))
    cfg = core.MMGenConfig(inference_csv_dir=str(inf), poscalc_dir=str(pos), output_dir=str(tmp_path/'out'),
                          finalize_shard_matrix=True, num_shards=2, shard_by='row_identity')
    core.run_mmgen(cfg)
    assert calls[0]['runtime_settings'].keep_temp_files


@pytest.mark.parametrize('fail', [False, True])
def test_single_append_retains_temp_until_canonical_publication(monkeypatch, tmp_path, fail):
    import pandas as pd
    from autobehaver.postinf.mmgen import core, append_mode

    batch = tmp_path/'batch'; batch.mkdir()
    temp = batch/'temp_store.h5'
    cfg = core.MMGenConfig(inference_csv_dir='inf', poscalc_dir='pos', output_dir='out')
    monkeypatch.setattr(append_mode, '_validate_plan_against_cfg', lambda *args: None)
    def delta(*args, **kwargs):
        assert kwargs['runtime_settings'].keep_temp_files
        temp.write_bytes(b'required until publication')
    def publish(**kwargs):
        assert temp.exists()
        if fail:
            raise RuntimeError('canonical publication failed')
    monkeypatch.setattr(append_mode.pipeline, 'run_post_inference_analysis', delta)
    monkeypatch.setattr(append_mode, '_apply_delta_batch_to_canonical', publish)
    def run():
        append_mode._run_single_job_append_from_plan(cfg=cfg, plan={},
            selected_files_df=pd.DataFrame({'source_file': ['one.csv']}), batch_dir=batch)
    if fail:
        with pytest.raises(RuntimeError, match='canonical publication failed'):
            run()
    else:
        run()
    assert temp.exists() == fail


def test_native_default_cleanup_and_staged_merge_preserve_matrix_and_provenance(tmp_path, monkeypatch):
    import pandas as pd
    import numpy as np
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    from autobehaver.postinf.mmgen.core import MMGenConfig, run_mmgen

    inf = tmp_path/'inf'; pos = tmp_path/'pos'; inf.mkdir(); pos.mkdir()
    for movie in (0, 1):
        name = f'G_FEMALE_20260101_rpi-1_trial1_movie{movie}_chamber1_positional_calcs.csv'
        values = {column: np.full(400, .99 if column.endswith('_likelihood') else 1.)
                  for column in required_keypre_columns(include_postural=False, include_spatial=False,
                                                        include_measurements=False)}
        pd.DataFrame(values).to_csv(pos/name, index=False)
        labels = ['Standing']*100+['Walking']*50+['Groom-Head']*100+['Walking']*150
        pd.DataFrame({'Frame Number': np.arange(1, 401), 'Smoothed Pose Label': labels,
                      'Smoothed Orient Label': ['Floor']*400}).to_csv(inf/name.replace('.csv', '_inference.csv'), index=False)
    common = dict(inference_csv_dir=str(inf), poscalc_dir=str(pos), inf_tag='_inference',
                  num_workers=1, input_batch_size=1, edge_skip=0, make_measurements=False,
                  calculate_occupancy=False, calculate_postural_features=False, emit_full_outputs=False,
                  min_bout_count_for_features=1, min_bout_count_for_transitions=1)
    plain = tmp_path/'plain'
    run_mmgen(MMGenConfig(**common, output_dir=str(plain)))
    expected = pd.read_csv(plain/'MASTER_FEATURE_MATRIX.csv')
    assert not (plain/'temp_store.h5').exists()
    assert (plain/'MMGEN_BATCH_MANIFEST.json').exists() and (plain/'MMGEN_OUTPUT_RECEIPT.json').exists()

    staged = tmp_path/'staged'
    run_mmgen(MMGenConfig(**common, output_dir=str(staged), process_only=True))
    assert (staged/'temp_store.h5').exists()
    run_mmgen(MMGenConfig(**common, output_dir=str(staged), finalize_only=True))
    assert (staged/'temp_store.h5').exists()
    pd.testing.assert_frame_equal(expected, pd.read_csv(staged/'MASTER_FEATURE_MATRIX.csv'))

    parts = tmp_path/'parts'
    for index in range(2):
        run_mmgen(MMGenConfig(**common, output_dir=str(parts), process_only=True,
                             num_shards=2, shard_index=index))
    stores = list(parts.glob('_mmgen_shards/*/temp_store.h5'))
    assert len(stores) == 2
    run_mmgen(MMGenConfig(**common, output_dir=str(parts), merge_shards=True, num_shards=2))
    pd.testing.assert_frame_equal(expected, pd.read_csv(parts/'MASTER_FEATURE_MATRIX.csv'))
    assert not (parts/'temp_store.h5').exists() and not any(p.exists() for p in stores)
    assert len(list(parts.glob('_mmgen_shards/*/_mmgen_work/SHARD_DONE.json'))) == 2
    assert (parts/'MMGEN_BATCH_MANIFEST.json').exists() and (parts/'MMGEN_OUTPUT_RECEIPT.json').exists()

    import json
    from autobehaver.postinf.mmgen import append_mode as app, integrity, publication as pub

    canonical_names = (*app.CORE_ARTIFACTS, app.IDENTITY_RELATIVE_PATH, app.OUTPUT_RECEIPT_FILENAME)
    before = {name: (plain/name).read_bytes() for name in canonical_names}
    for folder in (inf, pos):
        source = next(folder.glob('*movie0*'))
        source.with_name(source.name.replace('movie0', 'movie2')).write_bytes(source.read_bytes())
    original_copy = pub._publication_copy
    def fail_after_two_publications(source, destination):
        if Path(destination).name == 'MMGEN_ROW_IDENTITY_SUMMARY.csv.publish':
            raise OSError('injected canonical publication failure')
        return original_copy(source, destination)
    with monkeypatch.context() as patcher:
        patcher.setattr(pub, '_publication_copy', fail_after_two_publications)
        with pytest.raises(OSError, match='injected canonical publication failure'):
            run_mmgen(MMGenConfig(**common, output_dir=str(plain), append_new_data=True, append_batch_id='cleanup_retry'))
    assert {name: (plain/name).read_bytes() for name in canonical_names} == before
    assert list(plain.glob('_mmgen_append_batches/*/temp_store.h5'))

    with monkeypatch.context() as patcher:
        def must_reuse_completed_delta(*args, **kwargs):
            raise AssertionError('Retry must reuse the already verified completed delta.')
        patcher.setattr(app.pipeline, 'run_post_inference_analysis', must_reuse_completed_delta)
        run_mmgen(MMGenConfig(**common, output_dir=str(plain), append_new_data=True, append_batch_id='cleanup_retry'))
    generation = app._verified_generation(plain)
    assert len(generation['identity']['sources']) == 3
    assert all('original_name' in identity for identity in generation['identity']['sources'].values())
    completed_batch = Path(json.loads((plain/'MMGEN_APPEND_LATEST.json').read_text())['delta_output_dir'])
    assert not (completed_batch/'temp_store.h5').exists()
    full = tmp_path/'full_after_append'
    run_mmgen(MMGenConfig(**common, output_dir=str(full)))
    pd.testing.assert_frame_equal(pd.read_csv(plain/'MASTER_FEATURE_MATRIX.csv'), pd.read_csv(full/'MASTER_FEATURE_MATRIX.csv'))

    identity_path = plain/app.IDENTITY_RELATIVE_PATH
    receipt_path = plain/app.OUTPUT_RECEIPT_FILENAME
    identity_bytes = identity_path.read_bytes(); receipt_bytes = receipt_path.read_bytes()
    preserved_outputs = {name: (plain/name).read_bytes() for name in app.CORE_ARTIFACTS}
    for change in ('alter_hash', 'remove_identity'):
        identity = json.loads(identity_bytes)
        name = next(iter(identity['sources']))
        if change == 'alter_hash':
            identity['sources'][name]['inference_sha256'] = '0'*64
        else:
            del identity['sources'][name]
        identity['sources_sha256'] = integrity.content_hash(identity['sources'])
        receipt = json.loads(receipt_bytes); receipt['sources_sha256'] = identity['sources_sha256']
        identity_path.write_text(json.dumps(identity)); receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(RuntimeError, match='ledger and calculation generation disagree'):
            app._verified_generation(plain)
        assert {name: (plain/name).read_bytes() for name in app.CORE_ARTIFACTS} == preserved_outputs
        identity_path.write_bytes(identity_bytes); receipt_path.write_bytes(receipt_bytes)
