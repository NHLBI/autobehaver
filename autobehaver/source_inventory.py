"""Select distributable source without collecting local environments or data."""
from pathlib import Path
import os


SOURCE_DIRECTORIES = ('autobehaver', 'scripts', 'environment', 'docs', 'tests', 'examples', 'assets', '.github')
SOURCE_ROOT_FILES = frozenset({
    'pyproject.toml', 'README.md', 'LICENSE.txt', 'ASSET_NOTICE.md',
    'THIRD_PARTY_NOTICES.md', 'ASSET_SHA256SUMS', '.gitignore',
    'RELEASE_MANIFEST.json', 'SHA256SUMS', 'AGENTS.md', '.gitattributes',
})
SOURCE_SUFFIXES = frozenset({
    '.py', '.pyi', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.sh', '.sbatch',
    '.csv', '.tsv', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.html', '.css', '.js',
    '.ps1', '.cmd',
})
_EXCLUDED_DIRECTORIES = frozenset({'__pycache__', 'build', 'dist', 'venv', 'env', 'test_data', 'test_runs', 'scratch', 'tmp'})


def source_files(root):
    """Yield checked relative paths for the maintained source distribution.

    Trained models and normalization resources remain external to job bundles.
    Development directories and unsupported file types are never copied merely
    because they happen to be beside the package.
    """
    root = Path(root).resolve()
    selected = []
    for name in SOURCE_ROOT_FILES:
        if (root / name).is_file():
            selected.append(Path(name))
    for name in SOURCE_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            continue
        if directory.is_symlink() or (hasattr(directory, 'is_junction') and directory.is_junction()):
            raise ValueError(f'Source directory must be local, not linked: {directory}')
        for current, directories, files in os.walk(directory):
            current = Path(current)
            directories[:] = sorted(child for child in directories
                if not child.startswith('.') and child not in _EXCLUDED_DIRECTORIES
                and not child.startswith('pytest-cache-files-')
                and (current / child).relative_to(root).parts != ('autobehaver', 'resources'))
            for child in directories:
                candidate = current / child
                if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
                    raise ValueError(f'Source directory must be local, not linked: {candidate}')
            for name in files:
                path = current / name
                if not name.startswith('.') and path.suffix.lower() in SOURCE_SUFFIXES:
                    selected.append(path.relative_to(root))
    for relative in sorted(selected, key=lambda path: path.as_posix()):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Source file must remain within the package: {path}')
        yield relative
