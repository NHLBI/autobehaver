"""Read HDF5 feature headers and serialize portable artifact paths."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


__all__ = ['get_headers_from_first_hdf5', 'get_num_features_from_first_hdf5', 'portable_relpath']


def portable_relpath(path: str | os.PathLike[str], start: str | os.PathLike[str]) -> str:
    """Serialize a relative path with POSIX separators.

    Use native paths for filesystem access. Preserve an absolute path when
    Windows drives differ and no relative path can represent the location.
    """

    try:
        relative = os.path.relpath(os.fspath(path), start=os.fspath(start))
    except (OSError, ValueError):
        # Different Windows drives cannot be expressed as a relative path.
        # Preserve the absolute location, but still serialize separators
        # consistently.
        return Path(path).expanduser().resolve().as_posix()
    return Path(relative).as_posix()


def get_headers_from_first_hdf5(input_folder: str) -> list[str]:
    """Read headers from a usable HDF5 feature array in the folder.

    Feature data is expected as ``data`` with shape (T, F). Read ``columns``
    from file attributes, data attributes or a root dataset. Missing or
    length-mismatched headers fall back to col_0 through col_{F-1}; these names
    cannot preserve semantic feature selection.
    """
    files = os.listdir(input_folder)
    hdf5_files = sorted(
        f for f in files
        if (f.endswith('.h5') or f.endswith('.hdf5')) and ('metadata' not in f)
    )
    if not hdf5_files:
        raise FileNotFoundError("No suitable HDF5 files found in the input folder.")

    def _to_str_list(x) -> list[str]:
        arr = np.array(x)
        out: list[str] = []
        for item in arr.flatten().tolist():
            if isinstance(item, bytes):
                out.append(item.decode('utf-8', errors='ignore'))
            else:
                out.append(str(item))
        return out

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                dset = h5f['data']
                if len(dset.shape) != 2:
                    continue
                num_features = int(dset.shape[1])

                cols = None
                if 'columns' in h5f.attrs:
                    cols = h5f.attrs['columns']
                elif 'columns' in dset.attrs:
                    cols = dset.attrs['columns']
                elif 'columns' in h5f:
                    cols = h5f['columns'][()]

                if cols is not None:
                    headers = _to_str_list(cols)
                    if len(headers) != num_features:
                        print(
                            f"WARNING: Found column metadata in {fname} but length {len(headers)} "
                            f"does not match data.shape[1]={num_features}. Falling back to generic names."
                        )
                        return [f"col_{i}" for i in range(num_features)]
                    return headers
        except Exception:
            continue

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                num_features = int(h5f['data'].shape[1])
                print(
                    "WARNING: No feature-name metadata found in HDF5 files. "
                    "Falling back to generic column names col_0..col_{F-1}. "
                    "Scaling selection by name will likely be empty."
                )
                return [f"col_{i}" for i in range(num_features)]
        except Exception:
            continue

    raise FileNotFoundError("Could not find any suitable HDF5 file containing a 2D 'data' dataset.")


def get_num_features_from_first_hdf5(input_folder: str) -> int:
    """Return F from the first usable HDF5 ``data`` array of shape (T, F).

    Raise FileNotFoundError when no suitable array is found.
    """
    files = os.listdir(input_folder)
    hdf5_files = sorted(
        f for f in files
        if (f.endswith('.h5') or f.endswith('.hdf5')) and ('metadata' not in f)
    )
    if not hdf5_files:
        raise FileNotFoundError("No suitable HDF5 files found in the input folder.")

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                dset = h5f['data']
                if len(dset.shape) != 2:
                    continue
                return int(dset.shape[1])
        except Exception:
            continue

    raise FileNotFoundError("Could not find any suitable HDF5 file containing a 2D 'data' dataset.")
