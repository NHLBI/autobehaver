"""Local GUI planning subprocess: outputs only inside its local prepared bundle."""
import json
import sys
from pathlib import Path
from .postinf.featureimp import core
from .postinf.featureimp.config import DEFAULT_SETTINGS
from .postinf.featureimp.execution import phase, worker_capacity


def main():
    target = Path(sys.argv[1]).resolve()
    settings = json.loads((target/'local_plan_settings.json').read_text(encoding='utf-8'))
    matrix = target/'featureimp_matrix.csv'
    folder = target/'analysis_seed'
    folder.mkdir()
    # Conservative CSV expansion/copies allowance, independent of scientific data.
    worker_capacity(1, max(512*1024**2, matrix.stat().st_size*24), 'Local planning')
    settings.update(matrix_path=str(matrix), matrix_dir=str(target), results_dir=str(target),
                    analysis_folder_path=str(folder), stage='plan')
    core.apply_settings({**DEFAULT_SETTINGS, **settings}, propagate=False, explicit_keys=set(settings))
    with phase('Normalize selected matrix'):
        normalized = core.normalize_feature_matrix(results_dir=str(target), raw_matrix_path=str(matrix),
                    analysis_folder_path=str(folder), skip_if_normalized_exists=False)
    with phase('Freeze folds, feature subsets and shard assignments'):
        core.plan_feature_importance_analysis(normalized_matrix_path=normalized, results_dir=str(target),
                    analysis_folder_path=str(folder), analysis_run_name=settings['analysis_run_name'],
                    control_group_tag=settings['control_group_tag'], training_group_a=settings['training_group_a'],
                    training_group_b=settings['training_group_b'],
                    include_feature_categories=settings['include_feature_categories'], sex_filter=settings['sex_filter'],
                    n_splits=settings['n_splits'], H_hits_per_feature=settings['H_hits_per_feature'],
                    n_features_per_model=settings['n_features_per_model'], num_shards=settings['num_shards'])
    print('[FeatureImp plan] Local plan complete.', flush=True)

if __name__ == '__main__':
    main()
