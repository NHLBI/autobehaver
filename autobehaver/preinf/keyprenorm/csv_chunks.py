"""Iterate over bounded CSV chunks while trimming both preprocessing edges."""
from __future__ import annotations

import numpy as np
import pandas as pd

CSV_CHUNK_ROWS = 4096
HDF_COMPRESSION_LEVEL = 1


class RetainedCsvChunks:
    """Retain original interior rows while counting the complete source timeline."""

    def __init__(self, path, columns, edge_skip, *, chunk_rows=CSV_CHUNK_ROWS):
        self.path = path
        self.columns = list(columns)
        self.edge_skip = int(edge_skip)
        self.chunk_rows = int(chunk_rows)
        if self.edge_skip < 0 or self.chunk_rows < 1:
            raise ValueError("Edge skip must be nonnegative and chunk size positive.")
        self.source_rows = 0

    def __iter__(self):
        """Yield numeric chunks in canonical column order after trimming both edges.

        Hold a bounded trailing buffer until the final source rows are known.
        """
        self.source_rows = 0
        tail = None
        with pd.read_csv(self.path, header=0, dtype=np.float32,
                         chunksize=self.chunk_rows) as reader:
            for block in reader:
                start = self.source_rows
                self.source_rows += len(block)
                block = block.reindex(columns=self.columns)
                skip = max(0, self.edge_skip - start)
                block = block.iloc[min(skip, len(block)):]
                if tail is not None and len(tail):
                    block = pd.concat([tail, block], ignore_index=True)
                retained = max(0, len(block) - self.edge_skip)
                if retained:
                    yield block.iloc[:retained]
                tail = block.iloc[retained:].copy()
        # The bounded tail is exactly the omitted trailing preprocessing edge.
