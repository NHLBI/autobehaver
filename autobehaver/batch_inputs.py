"""Select explicit recording batches through the existing lossless filename maps."""
from __future__ import annotations

import re
from pathlib import Path

from .portable_paths import logical_recording_stem, original_name


def select_batch_inputs(paths, file_list=None):
    """Keep one input per listed logical recording stem; leave unbatched calls alone.

    Lists contain newline-separated chamber movie stems, without stage suffixes.
    Delimiter checks distinguish chamber1 from chamber10 and preserve renamed files.
    Missing or duplicate versions are reported before processing the batch.
    """
    paths = list(paths)
    if not file_list:
        return paths
    selectors = [re.sub(r"_+", "_", line.strip()) for line in
                 Path(file_list).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not selectors or len(set(s.casefold() for s in selectors)) != len(selectors):
        raise ValueError(f"Batch list is empty or contains duplicate recording identities: {file_list}")
    names = {str(path): re.sub(r"_+", "_", logical_recording_stem(original_name(path))).casefold()
             for path in paths}
    selected = set()
    for selector in selectors:
        matches = [path for path in paths if names[str(path)] == selector.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Batch recording {selector!r} matched {len(matches)} inputs; "
                             f"expected one. Check its upstream output and filename map. List: {file_list}")
        selected.add(str(matches[0]))
    return [path for path in paths if str(path) in selected]
