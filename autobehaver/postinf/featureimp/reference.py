"""Write the frozen features, normalization schema, model inventory, and provenance
bundle for a completed FeatureImp run.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from autobehaver import __version__ as AUTOBEHAVER_VERSION
from autobehaver.io import portable_relpath

from .matrix import compute_feature_schema_sha256, load_normalization_params

REFERENCE_DIRNAME = "FEATUREIMP_REFERENCE"
REFERENCE_MANIFEST_NAME = "FEATUREIMP_REFERENCE_MANIFEST.json"



def ensure_reference_folder(analysis_folder: str) -> str:
    """Create and return the analysis provenance directory."""
    folder = os.path.join(str(analysis_folder), REFERENCE_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    return folder



def _safe_relpath(path: Optional[str], root: str) -> Optional[str]:
    if not path:
        return None
    try:
        return portable_relpath(str(path), str(root))
    except Exception:
        return str(path)



def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)



def _write_json_list(path: str, values: Sequence[Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(values), f, indent=2, default=str)


def _resolve_normalization_params_path(
    *,
    analysis_folder: str,
    normalized_matrix_path: str,
    normalization_params_path: Optional[str] = None,
) -> str:
    """Resolve the normalization-params sidecar used by a completed run.

    Normalized matrices can live in a child analysis subfolder while the analysis-wide
    NORMALIZATION_PARAMS.json remains in the parent analysis folder. Search a
    small, deterministic set of candidate locations so reference-bundle writing
    remains robust.
    """

    candidates: List[str] = []
    if normalization_params_path:
        candidates.append(str(normalization_params_path))

    analysis_folder = str(analysis_folder)
    normalized_matrix_path = str(normalized_matrix_path)
    candidates.append(os.path.join(analysis_folder, 'NORMALIZATION_PARAMS.json'))

    matrix_dir = os.path.dirname(normalized_matrix_path)
    if matrix_dir:
        candidates.append(os.path.join(matrix_dir, 'NORMALIZATION_PARAMS.json'))

    parent_dir = os.path.dirname(analysis_folder)
    if parent_dir:
        candidates.append(os.path.join(parent_dir, 'NORMALIZATION_PARAMS.json'))

    seen = set()
    ordered: List[str] = []
    for c in candidates:
        c = os.path.abspath(str(c))
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    for c in ordered:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        'Could not locate NORMALIZATION_PARAMS.json for reference bundle. '
        f'Searched: {ordered}'
    )




def _build_model_inventory_rows(
    performance_records: Iterable[Mapping[str, Any]],
    *,
    analysis_folder: str,
) -> List[Dict[str, Any]]:
    """Record model paths, task identities, feature lists, and file availability in
    stable order.
    """
    rows: List[Dict[str, Any]] = []
    for rec in performance_records:
        if not isinstance(rec, Mapping):
            continue
        model_path = rec.get('model_path')
        model_exists = bool(model_path) and os.path.exists(str(model_path))
        row = {
            'h_hit': rec.get('h_hit'),
            'subset_idx': rec.get('subset_idx'),
            'fold_idx': rec.get('fold_idx'),
            'is_permuted': bool(rec.get('is_permuted', False)),
            'permutation_idx': rec.get('permutation_idx'),
            'n_features': rec.get('n_features'),
            'accuracy_test': rec.get('accuracy_test', rec.get('accuracy')),
            'accuracy_train': rec.get('accuracy_train'),
            'model_path': str(model_path) if model_path else None,
            'model_relpath': _safe_relpath(str(model_path), analysis_folder) if model_path else None,
            'model_exists': bool(model_exists),
        }
        rows.append(row)
    rows.sort(key=lambda r: (
        int(r['permutation_idx']) if r['permutation_idx'] not in (None, '') else -1,
        int(r['h_hit']) if r['h_hit'] not in (None, '') else -1,
        int(r['subset_idx']) if r['subset_idx'] not in (None, '') else -1,
        int(r['fold_idx']) if r['fold_idx'] not in (None, '') else -1,
    ))
    return rows



def write_featureimp_reference_bundle(
    *,
    analysis_folder: str,
    normalized_matrix_path: str,
    selected_feature_columns: Sequence[str],
    id_columns: Sequence[str],
    control_group_tag: Optional[str],
    training_group_a: Optional[str],
    training_group_b: Optional[str],
    settings: Mapping[str, Any],
    performance_records: Optional[Iterable[Mapping[str, Any]]] = None,
    subset_weights_path: Optional[str] = None,
    weighting_manifest_path: Optional[str] = None,
    group_label_mapping_path: Optional[str] = None,
    normalization_params_path: Optional[str] = None,
) -> str:
    """Write the explicit reference/provenance bundle for a completed FeatureImp run."""
    analysis_folder = str(analysis_folder)
    ref_dir = ensure_reference_folder(analysis_folder)

    selected_feature_columns = [str(c) for c in selected_feature_columns]
    id_columns = [str(c) for c in id_columns]

    selected_features_path = os.path.join(ref_dir, 'FROZEN_FEATURE_COLUMNS.json')
    _write_json_list(selected_features_path, selected_feature_columns)

    params_path = _resolve_normalization_params_path(
        analysis_folder=analysis_folder,
        normalized_matrix_path=normalized_matrix_path,
        normalization_params_path=normalization_params_path,
    )
    norm_params = load_normalization_params(params_path)
    normalization_feature_columns = [
        str(c) for c in norm_params.get('numeric_feature_columns') or list(norm_params['means'].keys())
    ]
    normalization_features_path = os.path.join(ref_dir, 'NORMALIZATION_FEATURE_COLUMNS.json')
    _write_json_list(normalization_features_path, normalization_feature_columns)

    compositional_sets_path = os.path.join(ref_dir, 'COMPOSITIONAL_SETS.json')
    _write_json(compositional_sets_path, norm_params.get('clr_input_compositional_sets', norm_params.get('compositional_sets', {})))

    inventory_rows = _build_model_inventory_rows(
        performance_records or [],
        analysis_folder=analysis_folder,
    )
    inventory_path = os.path.join(ref_dir, 'MODEL_INVENTORY.csv')
    if inventory_rows:
        pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    else:
        with open(inventory_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'h_hit', 'subset_idx', 'fold_idx', 'is_permuted',
                'permutation_idx', 'n_features', 'accuracy_test', 'accuracy_train',
                'model_path', 'model_relpath', 'model_exists',
            ])

    models_folder = os.path.join(analysis_folder, 'SAVED_PIPELINE', 'models')
    real_model_count = sum(1 for row in inventory_rows if not bool(row.get('is_permuted', False)))
    perm_model_count = sum(1 for row in inventory_rows if bool(row.get('is_permuted', False)))

    manifest = {
        'reference_schema_version': 1,
        'autobehaver_version': AUTOBEHAVER_VERSION,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'analysis_folder_name': os.path.basename(analysis_folder),
        'analysis_folder': analysis_folder,
        'reference_dirname': REFERENCE_DIRNAME,
        'artifacts': {
            'normalized_matrix_csv': _safe_relpath(normalized_matrix_path, analysis_folder),
            'normalization_params_json': _safe_relpath(params_path, analysis_folder),
            'featureimp_run_settings_json': 'FEATUREIMP_RUN_SETTINGS.json' if os.path.exists(os.path.join(analysis_folder, 'FEATUREIMP_RUN_SETTINGS.json')) else None,
            'group_label_mapping_json': _safe_relpath(group_label_mapping_path, analysis_folder),
            'selected_feature_columns_json': f'{REFERENCE_DIRNAME}/{os.path.basename(selected_features_path)}',
            'normalization_feature_columns_json': f'{REFERENCE_DIRNAME}/{os.path.basename(normalization_features_path)}',
            'compositional_sets_json': f'{REFERENCE_DIRNAME}/{os.path.basename(compositional_sets_path)}',
            'model_inventory_csv': f'{REFERENCE_DIRNAME}/{os.path.basename(inventory_path)}',
            'subset_weights_csv': _safe_relpath(subset_weights_path, analysis_folder),
            'weighting_manifest_json': _safe_relpath(weighting_manifest_path, analysis_folder),
            'models_folder': _safe_relpath(models_folder, analysis_folder) if os.path.isdir(models_folder) else None,
        },
        'reference_contract': {
            'id_columns': id_columns,
            'control_group_tag': control_group_tag,
            'training_group_a': training_group_a,
            'training_group_b': training_group_b,
            'normalization_method': str(norm_params.get('normalization_method', 'clr_then_zscore_v1')),
            'selected_feature_count': int(len(selected_feature_columns)),
            'selected_feature_schema_sha256': compute_feature_schema_sha256(selected_feature_columns),
            'normalized_feature_count': int(len(normalization_feature_columns)),
            'normalized_feature_schema_sha256': compute_feature_schema_sha256(normalization_feature_columns),
        },
        'training_settings': dict(settings),
        'models': {
            'inventory_row_count': int(len(inventory_rows)),
            'real_model_count': int(real_model_count),
            'permuted_model_count': int(perm_model_count),
        },
    }

    manifest_path = os.path.join(analysis_folder, REFERENCE_MANIFEST_NAME)
    _write_json(manifest_path, manifest)
    return manifest_path


__all__ = [
    'REFERENCE_DIRNAME',
    'REFERENCE_MANIFEST_NAME',
    'ensure_reference_folder',
    'write_featureimp_reference_bundle',
]
