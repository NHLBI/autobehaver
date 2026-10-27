"""Summarize positional CSVs with bounded reads and one animal per worker."""
from __future__ import annotations

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .finalize_execution import budgeted_results, GIB
from .io import read_metadata_csv

SPATIAL_COLUMNS = (
    'individual', 'source_file', 'frame_idx', 'r_norm', 'relative_angle', 'speed_mm_s',
    'masked_pose', 'masked_orient', 'masked_pose_orient',
    'masked_pose_valid', 'masked_orient_valid', 'masked_pose_orient_valid',
    'anglefeature_Wings_L_R', 'anglefeature_Legs_FL_FR',
    'anglefeature_Legs_ML_MR', 'anglefeature_Legs_BL_BR',
)


def _summarize_animal(task):
    from .feature_matrix import _compute_spatial_feature_summary_core
    path, bouts, identities, fps, minimum, gap = task
    fragments = []
    # These binary fragments are written only by this invocation in its private
    # temporary directory. No external pickle inputs are accepted.
    with open(path, 'rb') as stream:
        while stream.peek(1):
            fragments.append(pickle.load(stream))
    frame = pd.concat(fragments, ignore_index=True)
    del fragments
    return _compute_spatial_feature_summary_core(frame, bouts, identities,
        video_fps=fps, min_context_frames=minimum, groom_episode_max_gap_frames=gap)


def compute_spatial_feature_summary_csv(path, bout_summary_df, master_matrix, *,
                                        video_fps, num_workers, work_dir, chunk_rows=50_000):
    """Group even interleaved recordings on disk, preserving their exact values.

    Only a CSV chunk, one animal's frame data per worker, and the small feature
    summaries are held in memory. Temporary fragments are removed on exit.
    """
    from .feature_matrix import MIN_CONTEXT_DURATION_S, GROOM_EPISODE_MAX_NON_GROOM_GAP_S
    if int(chunk_rows) < 1:
        raise ValueError('Spatial CSV chunk size must be positive')
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    identities = master_matrix[['individual']].drop_duplicates().copy()
    with TemporaryDirectory(prefix='spatial_summary_', dir=work_dir) as temporary:
        files = {}
        with read_metadata_csv(path, usecols=lambda column: column in SPATIAL_COLUMNS,
                               chunksize=int(chunk_rows)) as reader:
            for chunk in reader:
                for individual, rows in chunk.groupby('individual', sort=False):
                    individual = str(individual)
                    if individual not in files:
                        files[individual] = str(Path(temporary) / f'{len(files):06d}.bin')
                    with open(files[individual], 'ab') as stream:
                        pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if not files:
            return None
        tasks = []
        for individual in sorted(files):
            bouts = (bout_summary_df.loc[bout_summary_df['individual'].astype(str).eq(individual)].copy()
                     if bout_summary_df is not None and not bout_summary_df.empty else None)
            master = identities.loc[identities['individual'].astype(str).eq(individual)].copy()
            tasks.append((files[individual], bouts, master, float(video_fps),
                          int(round(MIN_CONTEXT_DURATION_S * float(video_fps))),
                          int(round(GROOM_EPISODE_MAX_NON_GROOM_GAP_S * float(video_fps)))))
        print(f"[mmgen] Spatial summaries: {len(files)} animals, bounded CSV reads.", flush=True)
        estimates = [GIB + Path(task[0]).stat().st_size * 12 for task in tasks]
        summaries = list(budgeted_results(_summarize_animal, tasks, estimates,
                                          num_workers=max(1, int(num_workers))))
    result = pd.concat(summaries, ignore_index=True, sort=False)
    order = pd.Index(identities['individual'].astype(str))
    return result.set_index('individual').reindex(order).reset_index()
