#!/usr/bin/env python3
"""Write or verify the source release inventory; trained assets stay external."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autobehaver.source_inventory import source_files

GENERATED = {'RELEASE_MANIFEST.json', 'SHA256SUMS'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    version_tree = ast.parse((root / 'autobehaver/_version.py').read_text(encoding='utf-8-sig'))
    version = next(ast.literal_eval(node.value) for node in version_tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__version__'
                           for target in node.targets))
    records = []
    parsed = 0
    for relative in source_files(root):
        if relative.as_posix() in GENERATED:
            continue
        path = root / relative
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=relative.as_posix())
            parsed += 1
        records.append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                        'sha256': digest(path)})
    return {'version': version, 'scope': 'Distributable source; trained models and normalization assets are external.',
            'inventory_policy': 'autobehaver/source_inventory.py',
            'generated_files_excluded': sorted(GENERATED),
            'file_count': len(records), 'bytes': sum(record['bytes'] for record in records),
            'python_modules_parsed': parsed, 'files': records}


def update_or_verify(root, *, write=False):
    root = Path(root).resolve()
    expected = inventory(root)
    manifest_path = root / 'RELEASE_MANIFEST.json'
    encoded = (json.dumps(expected, indent=2) + '\n').encode('utf-8')
    checksum_lines = [f"{record['sha256']}  {record['path']}\n" for record in expected['files']]
    checksum_lines.append(f"{hashlib.sha256(encoded).hexdigest()}  RELEASE_MANIFEST.json\n")
    checksums = ''.join(checksum_lines).encode('utf-8')
    if write:
        manifest_path.write_bytes(encoded)
        (root / 'SHA256SUMS').write_bytes(checksums)
    else:
        if manifest_path.read_bytes() != encoded:
            raise ValueError('Release manifest does not match the current source/version. Regenerate after final edits.')
        if (root / 'SHA256SUMS').read_bytes() != checksums:
            raise ValueError('SHA256SUMS does not match the current source and manifest.')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Regenerate inventories after final source edits.')
    parser.add_argument('--root', type=Path, default=ROOT, help='Source checkout to inventory.')
    args = parser.parse_args(argv)
    result = update_or_verify(args.root, write=args.write)
    print(f"{'Wrote' if args.write else 'Verified'} {result['version']}: {result['file_count']} source files, "
          f"{result['python_modules_parsed']} Python files parsed; external assets excluded.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
