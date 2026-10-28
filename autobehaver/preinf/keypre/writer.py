"""Validate and atomically publish KeyPre feature CSVs."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from autobehaver.completion import preserve_previous, write_receipt, file_signature

import numpy as np
import pandas as pd

from . import settings

logger = logging.getLogger(__name__)

# Mirror settings globals (used by finalize_save_and_summarize)
nan_edge_effects = settings.nan_edge_effects
edge_skip = settings.edge_skip


def save_feature_chunks(chunks, csv_file, poscalc_file_path, *, apply_edge_nans=True, identity=None, movie_dim=600, source_rows=None):
    """Write bounded feature blocks; validate inside the producing worker.

    Only the true recording edges receive NaNs. The source and final output
    signatures let the coordinator reuse this numeric validation, while still
    checking the published content receipt.
    """
    from .contract import inspect_keypre_output, keypre_request
    temporary = str(Path(poscalc_file_path).with_name('.keypre-' + uuid.uuid4().hex[:12] + '.csv'))
    rows, columns = 0, None
    try:
        with open(temporary, 'w', newline='', encoding='utf-8') as stream:
            for frame, first, last in chunks:
                floats = frame.select_dtypes(include='float').columns
                if len(floats):
                    frame = frame.astype({col: np.float32 for col in floats})
                if columns is None:
                    columns = list(frame.columns)
                elif list(frame.columns) != columns:
                    raise ValueError('Feature column order changed between chunks.')
                if apply_edge_nans and nan_edge_effects and (not (first and last) or len(frame) > edge_skip * 2):
                    if first:
                        frame.iloc[:edge_skip, :] = np.nan
                    if last and edge_skip:
                        frame.iloc[-edge_skip:, :] = np.nan
                frame.to_csv(stream, index=False, header=first)
                rows += len(frame)
        # No generated feature arrays are retained during the readback scan.
        del frame
        if source_rows is not None and rows != source_rows():
            raise ValueError('Written features do not cover the complete source timeline.')
        check = inspect_keypre_output(csv_file, temporary, raw_only=not apply_edge_nans, expected_rows=rows)
        if not check[0]:
            raise ValueError(check[1])
        if identity is not None and keypre_request(csv_file, movie_dim, not apply_edge_nans) != identity:
            raise ValueError('DLC source changed during processing; output was not committed.')
        preserve_previous(poscalc_file_path)
        os.replace(temporary, poscalc_file_path)
        if identity is not None:
            write_receipt(poscalc_file_path, identity, columns=columns, rows=rows)
        logger.info('Processed and saved raw positional calculations: %s', poscalc_file_path)
        return dict(status='success', file=csv_file, error=None, out=poscalc_file_path,
                    n_rows=rows, n_cols=len(columns),
                    inspection=(file_signature(csv_file), file_signature(poscalc_file_path), check))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finalize_save_and_summarize(
    poscorr_dict: dict,
    csv_file: str,
    poscalc_file_path: str,
    *,
    apply_edge_nans: bool = True,
    identity: dict | None = None,
) -> dict:
    """Publish float32 feature columns and return the output status and shape.

    When enabled and the movie exceeds twice edge_skip, mark both edge regions
    NaN without removing rows. Raw-only export leaves those rows populated.
    """
    poscorr_df = pd.DataFrame(poscorr_dict)

    # Downcast float columns to float32 to save disk space.
    float_cols = poscorr_df.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        poscorr_df = poscorr_df.astype({col: np.float32 for col in float_cols})

    # In raw-only mode we intentionally do not apply edge NaNs.
    if apply_edge_nans and nan_edge_effects and len(poscorr_df) > edge_skip * 2:
        poscorr_df.iloc[:edge_skip, :] = np.nan
        poscorr_df.iloc[-edge_skip:, :] = np.nan

    tmp_path = str(Path(poscalc_file_path).with_name('.keypre-' + uuid.uuid4().hex[:12] + '.csv'))
    try:
        poscorr_df.to_csv(tmp_path, index=False)
        from .contract import inspect_keypre_output
        ok, reason, _, _ = inspect_keypre_output(csv_file, tmp_path, raw_only=not apply_edge_nans, expected_rows=len(poscorr_df))
        if not ok:
            raise ValueError(reason)
        preserve_previous(poscalc_file_path)
        os.replace(tmp_path, poscalc_file_path)
        if identity is not None:
            write_receipt(poscalc_file_path, identity, columns=list(poscorr_df.columns), rows=len(poscorr_df))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    logger.info(f"Processed and saved raw positional calculations: {poscalc_file_path}")

    return {
        "status": "success",
        "file": csv_file,
        "error": None,
        "out": poscalc_file_path,
        "n_rows": int(poscorr_df.shape[0]),
        "n_cols": int(poscorr_df.shape[1]),
    }
