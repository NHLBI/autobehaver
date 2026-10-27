"""Reuse numerical prerank results independently of their presentation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autobehaver.atomic_io import atomic_write_json, sha256_file
from autobehaver.completion import code_digest


def _builtins(value):
    if isinstance(value, dict):
        return {str(k): _builtins(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtins(v) for v in value]
    if isinstance(value, np.ndarray):
        return _builtins(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def dependency_identity(gsea):
    """Identify the enrichment implementation used to validate cached numerical results."""
    sources = []
    module_file = getattr(gsea, '__file__', None)
    if module_file:
        directory = Path(module_file).parent
        sources = [directory / name for name in ('gsea.py', 'base.py', 'algorithm.py') if (directory / name).is_file()]
    return {'gseapy': str(getattr(gsea, '__version__', 'unknown')), 'numpy': np.__version__,
            'pandas': pd.__version__, 'source': code_digest(sources) if sources else 'unavailable'}


def cached_prerank(gsea, *, rnk, gene_sets, **kwargs):
    """Reuse checked numerical prerank tables and curves while reconstructing requested
    native plots.
    """
    directory = Path(kwargs['outdir']) / '.numeric_cache'
    directory.mkdir(parents=True, exist_ok=True)
    payload_path, receipt_path = directory / 'prerank.json', directory / 'COMPLETE.json'
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(rnk, index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(rnk.columns)).encode())
    numeric_settings = {k: v for k, v in kwargs.items() if k not in {'outdir', 'threads', 'verbose', 'no_plot', 'figsize', 'format', 'graph_num'}}
    request = {'groups': gene_sets, 'settings': numeric_settings, 'dependency': dependency_identity(gsea),
               'implementation': code_digest([Path(__file__)])}
    digest.update(json.dumps(_builtins(request), sort_keys=True, separators=(',', ':')).encode())
    fingerprint = digest.hexdigest()
    if receipt_path.is_file() and payload_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            receipt = {}
        if not isinstance(receipt, dict):
            receipt = {}
        if receipt.get('fingerprint') == fingerprint and receipt.get('payload_sha256') == sha256_file(payload_path):
            payload = json.loads(payload_path.read_text(encoding='utf-8'))
            if hasattr(gsea, 'Prerank') and {'results', 'ranking', 'res2d'} <= set(payload):
                result = gsea.Prerank(rnk=rnk.copy(), gene_sets=gene_sets, **kwargs)
                result.res2d = pd.DataFrame(payload['res2d']['data'], columns=payload['res2d']['columns'], index=payload['res2d']['index'])
                rank = payload['ranking']
                result.ranking = pd.Series(rank['data'], index=pd.Index(rank['index'],name=rank.get('index_name')), name=rank['name'], dtype=float)
                result._results = payload['results']
                if not kwargs.get('no_plot', False):
                    result._plotting({result.module: result.ranking})
                print(f'FSEA: reusing verified numerical results for {Path(kwargs["outdir"]).name}.')
                return result
        elif receipt.get('fingerprint') == fingerprint:
            print(f'FSEA: cached numerical payload changed; recomputing {Path(kwargs["outdir"]).name}.')
    from .execution import fsea_preflight, phase
    kwargs['threads'] = fsea_preflight(len(rnk), kwargs.get('permutation_num', 1000), kwargs.get('threads', 1))
    with phase('FSEA permutations'):
        result = gsea.prerank(rnk=rnk, gene_sets=gene_sets, **kwargs)
    # Keep only native result tables, ranking and enrichment curves. Never
    # persist dependency objects, logger state, or executable pickle payloads.
    if hasattr(result, '_results') and isinstance(getattr(result, 'ranking', None), pd.Series):
        payload = {'res2d': {'columns': list(result.res2d.columns), 'index': list(result.res2d.index),
                            'data': result.res2d.to_numpy().tolist()},
                   'ranking': {'index': list(result.ranking.index), 'data': result.ranking.tolist(), 'name': result.ranking.name, 'index_name': result.ranking.index.name},
                   'results': result._results}
        atomic_write_json(_builtins(payload), payload_path)
        atomic_write_json({'fingerprint': fingerprint, 'payload_sha256': sha256_file(payload_path)}, receipt_path)
    return result
