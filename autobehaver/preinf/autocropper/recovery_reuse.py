"""Resolve pinned earlier crop producers using the compute job's Python runtime."""
import hashlib
import json
from pathlib import Path
import sys
from autobehaver.completion import code_digest


def producer_codes(manifest):
    codes = []
    for source in manifest['sources']:
        root = Path(source['root'])
        paths = []
        for name, expected in source['files'].items():
            if name not in ('detection.py', 'grid_fit.py', 'cropping.py', 'contract.py'):
                raise ValueError('Unexpected crop producer source file.')
            path = root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Earlier crop source changed; recovery reuse stopped.')
            paths.append(path)
        if not {'detection.py', 'cropping.py', 'contract.py'} <= set(source['files']):
            raise ValueError('Incomplete pinned producer source.')
        codes.append(code_digest(paths))
    return sorted(set(codes))


if __name__ == '__main__':
    print(','.join(producer_codes(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))))
