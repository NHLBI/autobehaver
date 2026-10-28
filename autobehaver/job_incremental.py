"""Read-only desktop selection of recordings absent from a committed MMGen ledger."""
import csv
import hashlib
import json
from pathlib import Path

from .portable_paths import logical_recording_stem, original_name


def new_recordings(root, movies, chambers, by_movie):
    root = Path(root)
    output = root / '8_master_matrix'
    ledger = output / 'MMGEN_SOURCE_LEDGER.csv'
    receipt = output / 'MMGEN_OUTPUT_RECEIPT.json'
    required = [ledger, receipt, output / 'MASTER_FEATURE_MATRIX.csv',
                output / '_mmgen_work/MMGEN_CALCULATION_IDENTITY.json']
    if any(not path.is_file() for path in required):
        raise ValueError('New-recordings mode needs an existing master matrix and its MMGen source ledger/receipts. Use a normal run first.')
    # Verify the ledger used for selection; the compute-side append engine verifies
    # the entire generation, scientific settings and previous input contents.
    data = ledger.read_bytes()
    receipt_data = receipt.read_bytes()
    metadata = json.loads(receipt_data.decode('utf-8'))
    entry = metadata.get('artifacts', {}).get(ledger.name)
    expected = entry.get('sha256') if isinstance(entry, dict) else entry
    if expected != hashlib.sha256(data).hexdigest():
        raise ValueError('MMGen source ledger does not match its receipt. Finish or repair the previous MMGen run first.')
    reader = csv.DictReader(data.decode('utf-8-sig').splitlines())
    if 'source_file' not in (reader.fieldnames or []):
        raise ValueError('MMGen source ledger lacks source_file.')
    previous = set()
    for row in reader:
        name = row['source_file']
        if not name or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid source filename in MMGen ledger.')
        previous.add(logical_recording_stem(original_name(root / '6_posepred_inference' / name)).casefold())
    if not previous:
        raise ValueError('MMGen source ledger is empty. Use a normal run first.')
    if receipt.read_bytes() != receipt_data:
        raise ValueError('MMGen generation changed during inventory. Wait for it to finish before building.')
    selected = [name for name in chambers if name.casefold() not in previous]
    if not selected:
        raise ValueError('No new recordings found relative to the committed master matrix.')
    chosen = set(selected)
    selected_movies = []
    for movie in movies:
        new = [name for name in by_movie[movie] if name in chosen]
        if new and len(new) != len(by_movie[movie]):
            raise ValueError('An existing full movie has both committed and new chambers. Start from existing crops or DLC to preserve its completed crops.')
        if new:
            selected_movies.append(movie)
    return selected_movies, selected, {m: by_movie[m] for m in selected_movies}, {
        'mode': 'new_recordings', 'previous_recordings': len(previous),
        'new_recordings': len(selected), 'ledger_sha256': hashlib.sha256(data).hexdigest(),
        'receipt_sha256': hashlib.sha256(receipt_data).hexdigest(),
        'matrix_update': 'append_new_data',
        'full_reports': 'Prior full reports are retained; new/affected-animal reports are in the append batch folder.'}
