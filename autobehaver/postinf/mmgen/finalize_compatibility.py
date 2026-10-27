"""Explicit, digest-pinned reuse of an audited pre-update MMGen store.

This exception is only for finalization. Original input/commit validation stays
in force and the new execution identity is published alongside the old producer.
"""
import json
import os
from pathlib import Path
from .integrity import content_hash, read_store_identity
from autobehaver.atomic_io import atomic_write_json, sha256_file


def resolve_finalization_identity(identity, work_dir, *, finalize_only):
    receipt_path = os.environ.get('AUTOBEHAVER_MMGEN_FINALIZE_COMPATIBILITY')
    if not receipt_path:
        return identity, None
    if not finalize_only:
        raise ValueError('Finalization compatibility requires --finalize_only')
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    previous = read_store_identity(work_dir)
    old = previous['calculation']
    if receipt.get('contract') != 1 or not receipt.get('audit'):
        raise ValueError('Missing audited finalization compatibility contract')
    if Path(work_dir).parent.resolve() != Path(receipt['output_dir']).resolve():
        raise ValueError('Finalization compatibility belongs to a different project')
    if content_hash(old) != receipt['old_calculation_sha256'] or previous['calculation_sha256'] != content_hash(old):
        raise ValueError('Old calculation identity differs from audited store')
    if identity['source_modules'] != receipt['new_source_modules']:
        raise ValueError('Finalizer code differs from audited compatibility record')
    left, right = dict(old), dict(identity)
    left.pop('source_modules'); right.pop('source_modules')
    if left != right:
        raise ValueError('Scientific settings changed; finalization compatibility refused')
    return old, dict(contract=1, compatibility_receipt_sha256=sha256_file(receipt_path),
                     audit=receipt['audit'], producer_calculation=old,
                     finalizer_calculation=identity)


def publish_execution_record(record, output_dir):
    if record is not None:
        atomic_write_json(record, Path(output_dir)/'MMGEN_FINALIZATION_EXECUTION.json')
