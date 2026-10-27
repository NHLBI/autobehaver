"""Content/calculation identities and verified recording completion records."""
from __future__ import annotations

from dataclasses import asdict
from autobehaver.completion import code_digest
from autobehaver.portable_paths import original_name
from .io import find_matching_poscalc_file
import hashlib
import json
import math
import csv
import time
from pathlib import Path
import pandas as pd

from autobehaver.atomic_io import atomic_write_json, sha256_file

CONTRACT_VERSION = 2
IDENTITY_FILENAME = 'MMGEN_CALCULATION_IDENTITY.json'
STORE_IDENTITY_FILENAME = 'MMGEN_STORE_IDENTITY.json'
NONSCIENTIFIC_SETTINGS = {'num_workers', 'input_batch_size', 'keep_temp_files',
                         'emit_master_matrix', 'emit_full_outputs',
                         'generate_quiver_plots',
                         'run_transition_statistics'}


def content_hash(value) -> str:
    """Hash a canonical JSON identity with nonfinite values rejected."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def calculation_identity(runtime_settings, *, min_bout: int, edge_skip: int, inf_tag: str) -> dict:
    """Identify scientific settings, producer code, and the declared feature schema."""
    from .feature_schema import schema_metadata
    settings = asdict(runtime_settings)
    settings = {k:v for k,v in settings.items() if k not in NONSCIENTIFIC_SETTINGS}
    package = Path(__file__).resolve().parents[2]
    modules = {'scientific_syntax':code_digest([*(package/'postinf/mmgen').rglob('*.py'), package/'label_gaps.py'])}
    return json.loads(json.dumps({'contract_version':CONTRACT_VERSION, 'runtime_settings':settings,
            'min_bout':int(min_bout), 'edge_skip':int(edge_skip), 'inf_tag':str(inf_tag),
            'source_modules':modules,
            'declared_feature_schema_sha256':schema_metadata()['feature_schema_sha256']}, sort_keys=True, allow_nan=False))


def source_identities(files, *, inference_csv_dir, poscalc_dir, inf_tag, source_frame_offset=0) -> dict:
    """Record input hashes, positional partners, and effective original-frame alignment."""
    from .worker_io import prediction_provenance, verify_prediction_positional_source, effective_source_frame_offset
    from autobehaver.preinf.execution import cpu_limit
    from autobehaver.preparation_io import io_results, preparation_progress
    def inspect(filename):
        source = Path(inference_csv_dir) / filename
        partner_path = find_matching_poscalc_file(str(source),poscalc_dir,inf_tag)
        if partner_path is None: raise FileNotFoundError(f'Missing positional-calcs partner for {source}')
        partner = Path(partner_path)
        if not source.is_file() or not partner.is_file():
            raise FileNotFoundError(f'Mmgen requires both inputs: {source}; {partner}')
        info = prediction_provenance(source)
        linkage = verify_prediction_positional_source(source, partner, info)
        with source.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
        effective_offset = effective_source_frame_offset(info, explicit_indices='Source Frame Index' in header, requested=source_frame_offset)
        sidecar = Path(str(source)+'.provenance.json')
        return filename, {'inference_sha256':sha256_file(source), 'poscalc_sha256':sha256_file(partner),
                            'original_name':original_name(source), 'positional_source_linkage':linkage,
                            'effective_source_frame_offset':effective_offset,
                            'provenance_sha256':sha256_file(sidecar) if sidecar.is_file() else None}
    with preparation_progress(lambda message: print('[MMGen verification] ' + message, flush=True)):
        return dict(io_results(inspect, sorted(map(str, files)),
            label='Input recordings', workers=min(4, cpu_limit())))


def bind_run_identity(path, identity, sources, *, reuse: bool) -> dict:
    """Write a run identity or require an existing generation to match before reuse."""
    path = Path(path)
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    payload = {'calculation':identity, 'calculation_sha256':content_hash(identity),
               'sources':sources, 'sources_sha256':content_hash(sources)}
    if reuse:
        if not path.is_file():
            raise RuntimeError('Cannot reuse a legacy Mmgen store without content/calculation identity; use fresh outputs.')
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != payload:
            raise RuntimeError('Mmgen inputs or scientific calculation changed; existing store cannot be reused.')
    else:
        atomic_write_json(payload, path)
    return payload


def read_store_identity(work_dir) -> dict:
    """Read retained-store identity, with read-only fallback for legacy stores."""
    work_dir = Path(work_dir)
    path = work_dir / STORE_IDENTITY_FILENAME
    if not path.is_file():
        path = work_dir / IDENTITY_FILENAME
    return json.loads(path.read_text(encoding='utf-8'))


def bind_store_identity(work_dir, identity, sources, *, reuse: bool) -> dict:
    """Bind processing data without changing the last published generation."""
    work_dir = Path(work_dir)
    path = work_dir / STORE_IDENTITY_FILENAME
    if reuse and not path.is_file():
        # Verify the old binding before migrating it. Only new-format paths are
        # written; the published/legacy path remains unchanged.
        payload = bind_run_identity(work_dir / IDENTITY_FILENAME, identity, sources, reuse=True)
        atomic_write_json(payload, path)
        return payload
    return bind_run_identity(path, identity, sources, reuse=reuse)


def table_inventory(store, source_file: str, *, expected: dict | None = None) -> dict:
    """Return exact persisted rows for one source, including legitimate empty tables."""
    from .hdf_store import HDF_KEYS
    result = {}
    keys = set(HDF_KEYS.values()) - {HDF_KEYS['video_lengths_tmp']}
    for key in sorted(keys):
        count = 0
        if key in store:
            schema = store.select(key, start=0, stop=0)
            if 'source_file' not in schema.columns:
                raise RuntimeError(f'Cannot certify recording-level provenance for {key}: source_file absent')
            for chunk in store.select(key, iterator=True, chunksize=50_000):
                count += int(chunk['source_file'].astype(str).eq(source_file).sum())
        result[key] = count
    if expected is not None and result != expected:
        raise RuntimeError(f'Persisted Mmgen output inventory changed for {source_file}')
    return result


def empty_inventory() -> dict:
    """Return a zero-row inventory for every declared temporary table."""
    from .hdf_store import HDF_KEYS
    return {key: 0 for key in sorted(set(HDF_KEYS.values()) - {HDF_KEYS['video_lengths_tmp']})}



def verification_chunks(store, key):
    """Single HDF reader, adaptive bounded blocks; never load the entire table.

    The working allowance is 1/16 of detected RAM, capped at 256 MiB.
    One eighth targets the dataframe; the rest covers grouping/hash copies.
    Sample conservatively, grow at most 4x, and reassess after every block.
    This is an estimate, not an OS-enforced memory guarantee.
    """
    from autobehaver.preinf.execution import memory_limit
    (_, total), _ = memory_limit()
    target = max(1, min(256 * 1024**2, total // 16) // 8)
    rows, offset = 1024, 0
    bytes_per_row = 1
    count = int(store.get_storer(key).nrows)
    started = last = time.monotonic()
    print(f'[MMGen verification] {key}: 0/{count} rows; '
          f'dataframe target {target / 1024**2:.1f} MiB.', flush=True)
    while offset < count:
        chunk = store.select(key, start=offset, stop=min(count, offset + rows))
        if chunk.empty:
            raise RuntimeError(f'Unexpected empty verification block: {key} at {offset}')
        used = int(chunk.memory_usage(index=True, deep=True).sum())
        # Peak observed bytes per row prevents later growth based on a narrow block.
        bytes_per_row = max(bytes_per_row, used / len(chunk))
        next_rows = max(1, min(65536, rows * 4, int(target / bytes_per_row)))
        offset += len(chunk)
        yield chunk
        del chunk
        rows = next_rows
        now = time.monotonic()
        if now - last >= 10 or offset == count:
            print(f'[MMGen verification] {key}: {offset}/{count} rows '
                  f'({now-started:.1f}s).', flush=True)
            last = now


def all_table_inventories(store, *, fingerprints=None) -> dict:
    """Count rows and optionally certify scientific cells in one bounded pass."""
    from .hdf_store import HDF_KEYS
    result = {}
    for key in empty_inventory():
        expected = {name: tables[key] for name, tables in (fingerprints or {}).items() if key in tables}
        if key not in store:
            if expected:
                raise RuntimeError(f'Missing committed scientific table {key}')
            continue
        schema = store.select(key, start=0, stop=0)
        if 'source_file' not in schema:
            raise RuntimeError(f'Cannot certify recording provenance for {key}')
        digests = {name: hashlib.sha256() for name in expected}
        for chunk in verification_chunks(store, key):
            for filename, count in chunk['source_file'].astype(str).value_counts().items():
                result.setdefault(filename, empty_inventory())[key] += int(count)
            if expected:
                for name, frame in chunk.groupby('source_file', sort=False, observed=True):
                    name = str(name)
                    if name not in expected:
                        continue
                    columns = expected[name]['columns']
                    if not set(columns).issubset(frame.columns):
                        raise RuntimeError(f'Committed scientific columns missing: {name}, {key}')
                    extras = set(frame.columns) - set(columns)
                    if extras and frame[list(extras)].notna().any().any():
                        raise RuntimeError(f'Committed cells changed during schema expansion: {name}, {key}')
                    rows = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype='uint64')
                    digests[name].update(rows.tobytes())
        for name, digest in digests.items():
            if digest.hexdigest() != expected[name]['sha256']:
                raise RuntimeError(f'Committed scientific cells changed: {name}, {key}')
    return result


def frame_fingerprint(frame) -> dict:
    """Hash persisted scientific cells in row order, independent of HDF indexes."""
    columns = list(frame.columns)
    rows = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype='uint64')
    return {'columns':columns, 'sha256':hashlib.sha256(rows.tobytes()).hexdigest()}


def verify_table_fingerprints(store, fingerprints) -> None:
    """One streaming pass per table; schema expansion cannot hide changed cells."""
    for key in empty_inventory():
        expected = {filename:tables[key] for filename,tables in fingerprints.items() if key in tables}
        if not expected:
            continue
        if key not in store:
            raise RuntimeError(f'Missing committed scientific table {key}')
        digests = {name:hashlib.sha256() for name in expected}
        for chunk in store.select(key, iterator=True, chunksize=50_000):
            for name, frame in chunk.groupby('source_file', sort=False, observed=True):
                name = str(name)
                if name not in expected:
                    continue  # uncommitted rows are purged before retry
                columns = expected[name]['columns']
                if not set(columns).issubset(frame.columns):
                    raise RuntimeError(f'Committed scientific columns missing: {name}, {key}')
                extras = set(frame.columns)-set(columns)
                if extras and frame[list(extras)].notna().any().any():
                    raise RuntimeError(f'Committed cells changed during schema expansion: {name}, {key}')
                values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype='uint64')
                digests[name].update(values.tobytes())
        for name, digest in digests.items():
            if digest.hexdigest() != expected[name]['sha256']:
                raise RuntimeError(f'Committed scientific cells changed: {name}, {key}')


def commit_fingerprint(row) -> str:
    """Bind denominator and animal metadata as well as persisted numerical rows."""
    n_frames = float(row['n_frames'])
    if not math.isfinite(n_frames) or n_frames <= 0 or not n_frames.is_integer():
        raise RuntimeError('Invalid committed recording frame count')
    return content_hash({
        **{k: str(row[k]) for k in ['source_file','group','individual','sex',
                                    'calculation_sha256','input_sha256']},
        'n_frames': int(n_frames),
        'output_inventory': json.loads(row['output_inventory']),
        'output_fingerprints': json.loads(row['output_fingerprints']),
    })


def validate_commits(temp_store_path, identity_payload, *, selected_files=None,
                     allow_uncommitted: bool = False) -> set[str]:
    """Verify committed sources, denominators, table counts, and scientific cell
    fingerprints.
    """
    from .hdf_store import HDF_KEYS, open_store
    from .recording_commits import read_recording_commits
    if identity_payload['calculation_sha256'] != content_hash(identity_payload['calculation']) or identity_payload['sources_sha256'] != content_hash(identity_payload['sources']):
        raise RuntimeError('Mmgen identity contents changed')
    with open_store(str(temp_store_path), 'r') as store:
        permitted = set(map(str, selected_files)) if selected_files is not None else set(identity_payload['sources'])
        commits = read_recording_commits(store)
        if commits.empty:
            persisted = all_table_inventories(store)
            if set(persisted) - permitted:
                raise RuntimeError('Mmgen scientific rows contain unselected sources')
            if persisted and not allow_uncommitted:
                raise RuntimeError('Mmgen scientific rows have no recording commits')
            return set()
        required = {'source_file','group','individual','sex','n_frames','commit_sha256',
                    'calculation_sha256','input_sha256','output_inventory','output_fingerprints'}
        if not required.issubset(commits):
            raise RuntimeError('Legacy/unverified Mmgen commit cannot certify reuse or finalization')
        if commits['source_file'].astype(str).duplicated().any():
            raise RuntimeError('Duplicate Mmgen recording commits')
        fingerprints = {str(row['source_file']):json.loads(row['output_fingerprints']) for row in commits.to_dict('records')}
        persisted = all_table_inventories(store, fingerprints=fingerprints)
        if set(persisted) - permitted:
            raise RuntimeError('Mmgen scientific rows contain unselected sources')
        done = set()
        for row in commits.to_dict('records'):
            filename = str(row['source_file'])
            if filename not in identity_payload['sources']:
                raise RuntimeError(f'Unselected recording commit: {filename}')
            if row['calculation_sha256'] != identity_payload['calculation_sha256']:
                raise RuntimeError(f'Changed calculation for {filename}')
            if row['input_sha256'] != content_hash(identity_payload['sources'][filename]):
                raise RuntimeError(f'Changed inputs for {filename}')
            inventory = json.loads(row['output_inventory'])
            if set(inventory) != set(empty_inventory()) or any(type(v) is not int or v < 0 for v in inventory.values()):
                raise RuntimeError(f'Invalid committed table inventory for {filename}')
            if set(fingerprints[filename]) != {k for k, v in inventory.items() if v > 0}:
                raise RuntimeError(f'Incomplete committed scientific fingerprints for {filename}')
            if row['commit_sha256'] != commit_fingerprint(row):
                raise RuntimeError(f'Committed recording metadata changed for {filename}')
            if inventory != persisted.get(filename, empty_inventory()):
                raise RuntimeError(f'Persisted Mmgen output inventory changed for {filename}')
            done.add(filename)
        if not allow_uncommitted and set(persisted) - done:
            raise RuntimeError('Uncommitted scientific rows cannot be finalized')
        if selected_files is not None and not done.issubset(set(map(str,selected_files))):
            raise RuntimeError('Mmgen commit contains sources outside selected cohort')
        return done
