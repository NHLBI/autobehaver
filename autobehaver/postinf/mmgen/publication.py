"""Own and publish one MMGen generation without backing up working HDF stores.

The store identity binds retained processing data. The published calculation
identity changes only with a committed output generation. A durable journal
protects the prior receipt's artifacts by same-filesystem rename; interrupted
writers recover that generation before another reader or writer may proceed.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import shutil
import uuid

from autobehaver.atomic_io import atomic_write_json, sha256_file
from .integrity import content_hash, STORE_IDENTITY_FILENAME
from .io import read_metadata_csv
from .ledger import BATCH_MANIFEST_FILENAME, ROW_IDENTITY_SUMMARY_FILENAME, SOURCE_LEDGER_FILENAME
from .locks import canonical_writer, _owner_key, shard_lock_active

OUTPUT_RECEIPT_FILENAME = 'MMGEN_OUTPUT_RECEIPT.json'
IDENTITY_RELATIVE_PATH = '_mmgen_work/MMGEN_CALCULATION_IDENTITY.json'
PUBLICATION_JOURNAL = '_mmgen_work/MMGEN_APPEND_PUBLICATION.json'
CORE_ARTIFACTS = ('MASTER_FEATURE_MATRIX.csv', SOURCE_LEDGER_FILENAME,
                  ROW_IDENTITY_SUMMARY_FILENAME, BATCH_MANIFEST_FILENAME)
_transactions = ContextVar('mmgen_publication_transactions', default=frozenset())


def _publication_copy(source, target):
    """Copy staged artifacts without Windows' legacy MAX_PATH limit."""
    def native(value):
        path = os.path.abspath(os.fspath(value))
        if os.name != 'nt' or path.startswith('\\\\?\\'):
            return path
        return '\\\\?\\UNC\\' + path[2:] if path.startswith('\\\\') else '\\\\?\\' + path
    shutil.copy2(native(source), native(target))


def _artifact_path(root, relative):
    relative = Path(relative)
    target = root / relative
    if relative.is_absolute() or root.resolve() not in target.resolve().parents or target.is_symlink():
        raise RuntimeError(f'Publication target must be a regular file inside the output directory: {relative}')
    return target


def _verify_published_receipt(root):
    receipt_path, identity_path = root / OUTPUT_RECEIPT_FILENAME, root / IDENTITY_RELATIVE_PATH
    if not receipt_path.is_file() or not identity_path.is_file():
        raise RuntimeError('Legacy Mmgen outputs lack a verified content/calculation generation; use fresh outputs.')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    if (identity.get('calculation_sha256') != content_hash(identity.get('calculation'))
            or identity.get('sources_sha256') != content_hash(identity.get('sources'))):
        raise RuntimeError('Malformed Mmgen calculation/source identity.')
    if any(receipt.get(key) != identity[key] for key in ('calculation_sha256', 'sources_sha256')):
        raise RuntimeError('Canonical Mmgen receipt and scientific identity disagree.')
    artifacts = receipt.get('artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError('Canonical receipt lacks a certified artifact inventory.')
    retained_artifacts = dict(artifacts)
    historical = receipt.get('historical_artifacts', {})
    if not isinstance(historical, dict):
        raise RuntimeError('Canonical receipt has a malformed historical artifact inventory.')
    for relative, record in historical.items():
        if (relative in artifacts or not isinstance(record, dict)
                or not all(isinstance(record.get(key), str) for key in
                           ('sha256', 'calculation_sha256', 'sources_sha256'))):
            raise RuntimeError(f'Malformed historical artifact identity: {relative}')
        retained_artifacts[relative] = record['sha256']
    for relative, digest in retained_artifacts.items():
        path = _artifact_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f'Canonical artifact changed or missing: {relative}')
    return {'receipt': receipt, 'identity': identity, 'retained_artifacts': retained_artifacts,
            'receipt_sha256': sha256_file(receipt_path),
            'identity_sha256': sha256_file(identity_path)}


def _clean_backup(backup, originals):
    """Remove only files named by this owned journal; never recurse into stores."""
    parents = set()
    for relative in originals:
        path = _artifact_path(backup, relative)
        if path.is_file():
            path.unlink()
        parents.update(parent for parent in path.parents if parent == backup or backup in parent.parents)
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass  # Leave unrelated entries intact.


def _recover_publication(root, *, owned_failure=False):
    """Recover before canonical reads; legacy copy journals remain readable."""
    root = Path(root)
    with canonical_writer(root):
        journal_path = root / PUBLICATION_JOURNAL
        if not journal_path.exists():
            return
        legacy_lock = root / '_mmgen_work/_mmgen_append.lock'
        if legacy_lock.is_dir() and shard_lock_active(legacy_lock) and not owned_failure:
            raise RuntimeError('An older append writer has live or unverified ownership; recovery must wait.')
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        backup = Path(journal['backup_dir'])
        if root.resolve() not in backup.resolve().parents or backup.is_symlink():
            raise RuntimeError('Publication backup must remain inside the canonical output folder.')
        originals = journal['originals']
        for relative in originals:
            _artifact_path(root, relative)
            _artifact_path(backup, relative)
        if journal.get('status') == 'complete':
            if journal.get('backup_mode') == 'rename':
                _clean_backup(backup, originals)
            journal_path.unlink()
            return
        rename_backup = journal.get('backup_mode') == 'rename'
        # In an interrupted backup/rollback, each original is either already at
        # its canonical path or still in the backup. Verify all before restoring.
        locations = {}
        for relative, digest in originals.items():
            if digest is None:
                continue
            saved, target = backup / relative, root / relative
            if saved.is_file() and sha256_file(saved) == digest:
                locations[relative] = saved
            elif (rename_backup and not saved.exists()
                  and journal.get('status') in {'backing_up', 'rolling_back'}
                  and target.is_file() and sha256_file(target) == digest):
                locations[relative] = target
            else:
                raise RuntimeError(f'Interrupted publication cannot recover original {relative}')
        journal['status'] = 'rolling_back'
        atomic_write_json(journal, journal_path)
        for relative, digest in originals.items():
            target = root / relative
            if digest is None:
                if target.is_file():
                    target.unlink()
                continue
            source = locations[relative]
            if source == target:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if rename_backup:
                os.replace(source, target)
            else:
                restore = target.with_name(target.name + '.rollback')
                _publication_copy(source, restore)
                os.replace(restore, target)
        if rename_backup:
            _clean_backup(backup, originals)
        journal_path.unlink()


def _verified_generation(root, *, recover=True):
    """Verify the append-capable matrix, provenance, and exact source ledger."""
    root = Path(root)
    with canonical_writer(root):
        if recover:
            _recover_publication(root)
        generation = _verify_published_receipt(root)
        if not set(CORE_ARTIFACTS).issubset(generation['receipt']['artifacts']):
            raise RuntimeError('Canonical receipt must certify matrix, ledger, row identity and batch manifest together.')
        ledger = read_metadata_csv(root / SOURCE_LEDGER_FILENAME)
        required = {'source_file', 'row_identity', 'source_file_sha256', 'poscalc_file_sha256'}
        if not required.issubset(ledger) or ledger['source_file'].astype(str).duplicated().any():
            raise RuntimeError('Canonical source ledger lacks unique verified input hashes.')
        ledger_sources = {str(row['source_file']): {'inference_sha256': str(row['source_file_sha256']),
                         'poscalc_sha256': str(row['poscalc_file_sha256'])} for row in ledger.to_dict('records')}
        identity_sources = {name: {key: source.get(key) for key in ('inference_sha256', 'poscalc_sha256')}
                            for name, source in generation['identity']['sources'].items()}
        if ledger_sources != identity_sources:
            raise RuntimeError('Canonical source ledger and calculation generation disagree.')
        return generation


@contextmanager
def canonical_generation(root, *, producing=True, replacing=None):
    """Protect only published artifacts, retaining the working store for retry."""
    root = Path(root)
    with canonical_writer(root):
        key = _owner_key(root)
        if key in _transactions.get():
            yield
            return
        _recover_publication(root)
        if not producing or not (root / OUTPUT_RECEIPT_FILENAME).is_file():
            yield
            return
        generation = _verify_published_receipt(root)
        # Legacy stores used the published identity path. Preserve that binding
        # once before moving the published metadata; new processing writes its
        # own store identity and never changes this published file prematurely.
        store_identity = root / '_mmgen_work' / STORE_IDENTITY_FILENAME
        if (root / 'temp_store.h5').is_file() and not store_identity.exists():
            atomic_write_json(generation['identity'], store_identity)
        originals = dict(generation['retained_artifacts'])
        if replacing is not None:
            # Append updates its matrix/provenance only. Historical full reports
            # remain in place and retain their separately declared old identity.
            originals = {name: digest for name, digest in originals.items() if name in replacing}
            for name in replacing:
                target = _artifact_path(root, name)
                originals.setdefault(name, sha256_file(target) if target.is_file() else None)
        originals[IDENTITY_RELATIVE_PATH] = generation['identity_sha256']
        originals[OUTPUT_RECEIPT_FILENAME] = generation['receipt_sha256']
        backup = root / '_mmgen_work' / ('published_backup_' + uuid.uuid4().hex[:12])
        backup.mkdir()
        journal = {'status': 'backing_up', 'backup_mode': 'rename',
                   'backup_dir': str(backup.resolve()), 'originals': originals}
        atomic_write_json(journal, root / PUBLICATION_JOURNAL)
        token = _transactions.set(_transactions.get() | {key})
        try:
            for relative, digest in originals.items():
                if digest is None:
                    continue
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(root / relative, saved)
            journal['status'] = 'publishing'
            atomic_write_json(journal, root / PUBLICATION_JOURNAL)
            yield
            _verify_published_receipt(root)
            journal['status'] = 'complete'
            atomic_write_json(journal, root / PUBLICATION_JOURNAL)
            _clean_backup(backup, originals)
            (root / PUBLICATION_JOURNAL).unlink()
        except BaseException:
            _recover_publication(root, owned_failure=True)
            raise
        finally:
            _transactions.reset(token)


def _publish_generation(root, staged, relative_files):
    """Publish the staged append generation with the receipt last."""
    root, staged = Path(root), Path(staged)
    relative_files = [name for name in relative_files if name != OUTPUT_RECEIPT_FILENAME] + [OUTPUT_RECEIPT_FILENAME]
    for relative in relative_files:
        if not _artifact_path(staged, relative).is_file():
            raise RuntimeError(f'Staged publication missing {relative}')
    with canonical_generation(root, replacing=relative_files):
        for relative in relative_files:
            target = _artifact_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.publish')
            _publication_copy(staged / relative, temporary)
            os.replace(temporary, target)
        _verified_generation(root, recover=False)
