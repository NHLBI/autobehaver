"""Copy results into a shallow transfer tree with a reversible name map.

Preserve source files and verify copied content. Native processing stages
use their own integrated name maps.
"""
from pathlib import Path
import argparse
import json
import shutil
import os
import hashlib
from .completion import native_path,file_digest
from .portable_paths import PATH_BUDGET,TEMP_RESERVE,compact_component


def export_results(source,destination):
    """Copy and verify results into a separate empty destination.

    Check free space, path length and name collisions before copying. Return
    the file map also written to NAME_MAP.json, retaining original relative paths.
    """
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Export destination must be outside the source tree')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose an empty export destination; existing files are preserved')
    files=[source / Path(directory).relative_to(Path(native_path(source))) / name
           for directory,_,names in os.walk(native_path(source)) for name in names]
    if shutil.disk_usage(destination.parent).free < sum(os.stat(native_path(p)).st_size for p in files):
        raise OSError('Insufficient destination space')
    plans=[]
    used=set()
    for path in files:
        relative=path.relative_to(source)
        # Flatten excessive nesting, retaining the exact original relative path
        # in the manifest. Stable hashes separate equal basenames in different dirs.
        parent=relative.parent.as_posix()
        folder=('root' if parent=='.' else 'd_'+hashlib.sha256(parent.encode()).hexdigest()[:12])
        limit=min(140,PATH_BUDGET-TEMP_RESERVE-len(str(destination/folder))-1)
        if limit < 40: raise ValueError('Choose a shorter export destination')
        name=compact_component(path.name,limit)
        key=(folder+'/'+name).casefold()
        if key in used: raise ValueError('Portable export collision')
        used.add(key)
        plans.append((path,folder+'/'+name,relative.as_posix()))
    destination.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for path,short,original in plans:
        target=destination/short
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(native_path(path),native_path(target))
        digest=file_digest(target)
        if digest!=file_digest(path): raise OSError(f'Export checksum mismatch: {original}')
        manifest.append({'portable_path':short,'original_path':original,'sha256':digest})
    (destination/'NAME_MAP.json').write_text(json.dumps({'version':1,'files':manifest},indent=2),encoding='utf-8')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args(argv)
    files=export_results(args.source,args.destination)
    print(f'Copied and verified {len(files)} files; exact original paths are in NAME_MAP.json')

if __name__ == '__main__': main()
