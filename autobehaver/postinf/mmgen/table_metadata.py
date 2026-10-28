"""Read recording identities without materializing per-frame tables."""
from __future__ import annotations

import json

import pandas as pd

from .recording_commits import read_recording_commits


def read_individual_metadata(store, key, *, columns=('group', 'individual', 'sex'),
                             chunk_rows=4096):
    """Use recording inventories, or bounded reads for older standalone stores.

    Membership depends on rows in the requested table, not on complete bouts.
    This retains animals whose frames are usable but whose bouts are censored.
    """
    columns = list(columns)
    key = '/' + str(key).lstrip('/')
    if int(chunk_rows) < 1:
        raise ValueError('Metadata chunk size must be positive')
    if key not in store:
        return pd.DataFrame(columns=columns)
    commits = read_recording_commits(store)
    if not commits.empty and set(columns + ['output_inventory']).issubset(commits.columns):
        inventories = [json.loads(value) if isinstance(value, str) else value
                       for value in commits['output_inventory']]
        if all(isinstance(value, dict) and key in value for value in inventories):
            present = [int(value[key]) > 0 for value in inventories]
            return commits.loc[present, columns].drop_duplicates().reset_index(drop=True)

    # Older fixtures/standalone stores may not contain recording inventories.
    # Discard frame duplicates in each chunk before retaining the tiny result.
    metadata = pd.DataFrame(columns=columns)
    for chunk in store.select(key, columns=columns, iterator=True,
                              chunksize=int(chunk_rows)):
        metadata = pd.concat([metadata, chunk.drop_duplicates()], ignore_index=True)
        metadata = metadata.drop_duplicates().reset_index(drop=True)
    return metadata
