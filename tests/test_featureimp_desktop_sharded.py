import json
import os
import shlex
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from autobehaver import desktop_jobs as jobs

ROOT=Path(__file__).resolve().parents[1]

def test_real_local_plan_and_generated_stages(tmp_path):
    project=tmp_path/'data/experiment';project.mkdir(parents=True)
    (tmp_path/'bundles').mkdir()
    matrix=project/'matrix.csv'
    rng=np.random.default_rng(7)
    rows=[]
    for i in range(60):
        row=dict(individual=f'fly{i}',group='A' if i<30 else 'B',sex='MALE' if i%2 else 'FEMALE')
        row.update({f'Kinematic_feature{j}':float(rng.normal()+5*(i>=30)) for j in range(8)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(matrix,index=False)
    cfg=dict(kind='featureimp',run_name='fixture',dataset=str(project),data_local=str(tmp_path/'data'),
        data_remote='/data/test',jobs_local=str(tmp_path/'bundles'),jobs_remote='/data/bundles',
        python='/env/bin/python',resources='/resources',matrix=str(matrix),group_a='A',group_b='B',
        control_group='A',sex='all',pca=True,fsea=True,fi_sharded=True,fi_shards=2)
    record=jobs.build_job(cfg,ROOT,tmp_path/'local')
    target=Path(record['prepared']);plan=json.loads((target/'plan.json').read_text())
    assert plan['local_plan'] and plan['active_shards']==2
    assert plan['training_rows']==60 and plan['selected_features']==8
    assert record['job_counts']==dict(featureimp_train=2,featureimp_merge_shards=1,featureimp_pca=1,featureimp_fsea=1,collect=1)
    manifest=json.loads((target/'analysis_seed/_featureimp_plan/PLAN_MANIFEST.json').read_text())
    from autobehaver.postinf.featureimp.integrity import flatten_paths
    for path in flatten_paths(manifest['artifacts']):
        assert '\\' not in path and not Path(path).is_absolute()
        assert (target/'analysis_seed'/path).is_file()
    from autobehaver.postinf.featureimp.config import build_arg_parser
    for script in target.glob('fi_*.sbatch'):
        command=next(line for line in script.read_text().splitlines() if line.startswith('/usr/bin/time'))
        args=shlex.split(command);start=args.index('autobehaver.postinf.featureimp.cli')+1
        parsed=build_arg_parser().parse_args(args[start:args.index('--resource-monitor')])
        assert parsed.postanalysis_strict
    assert '--mem=32g' in (target/'fi_train_000.sbatch').read_text()
    assert '--mem=64g' in (target/'fi_fsea.sbatch').read_text()
    assert not Path(record['mounted']).exists()
    published=jobs.publish(record)
    jobs.verify_submission(published)
    (Path(record['mounted'])/'jobs.tsv').write_text('fi_train_000\t123\n')
    assert jobs.can_resume_submission(published)
    with pytest.raises(ValueError,match='accepted jobs'):jobs.verify_submission(published)
    assert jobs.verify_submission(published,resume=True).endswith(' --resume')

def test_memory_admission_and_fsea_preserves_permutations(monkeypatch):
    from autobehaver.postinf.featureimp import execution as ex
    monkeypatch.setattr(ex,'allowance',lambda:('test',16*ex.GIB,6*ex.GIB))
    monkeypatch.setattr(ex,'cpu_limit',lambda:12)
    assert ex.worker_capacity(8,2*ex.GIB,'fixture')==3
    with pytest.raises(MemoryError):ex.worker_capacity(1,7*ex.GIB,'fixture')
    with pytest.raises(MemoryError):ex.fsea_preflight(6000,100000,8)

def _square(x):return x*x

def test_measured_pool_completes_each_task_once():
    from autobehaver.postinf.featureimp.bounded_pool import BoundedProcessRunner
    with BoundedProcessRunner(num_workers=2,estimated_worker_bytes=1024**3) as pool:
        values=list(pool.run(_square,range(6),desc='fixture',stage_name='fixture',show_progress=False))
        assert sorted(values)==[0,1,4,9,16,25]
        assert pool.pilot_done and pool.observed_peak>0
        assert sorted(pool.run(_square,[6,7],desc='fixture',stage_name='fixture',show_progress=False))==[36,49]

@pytest.mark.parametrize('changed',[False,True])
def test_weighted_input_snapshot_rechecked_before_acceptance(tmp_path,monkeypatch,changed):
    from autobehaver.postinf.featureimp import final_merge as fm, integrity, shard_runner
    (tmp_path/'_featureimp_merge').mkdir();(tmp_path/'_featureimp_merge/WEIGHTS_DONE.json').write_text('{}')
    (tmp_path/'COMMITTEE_SUBSETS').mkdir();(tmp_path/'COMMITTEE_SUBSETS/SUBSET_WEIGHTS_REAL.csv').write_text('')
    monkeypatch.setattr(fm,'validate_receipt',lambda *a,**k:{})
    monkeypatch.setattr(fm,'load_subset_weights_mapping',lambda *a:{})
    monkeypatch.setattr(fm,'_load_task_manifest',lambda *a:[])
    monkeypatch.setattr(fm,'_reduce_one_shard',lambda args:(0,'fixture',str(tmp_path),{}))
    monkeypatch.setattr(shard_runner,'_get_runtime_num_workers',lambda:1)
    calls=[]
    def inputs(*args):
        calls.append(1)
        return {'fixture':'changed' if changed and len(calls)>1 else 'original'}
    monkeypatch.setattr(integrity,'stage_inputs',inputs)
    kwargs=dict(analysis_folder_path=str(tmp_path),manifest={},completed_training_shards=[(0,'fixture',str(tmp_path),{})],
        context={'df_for_training':[1],'df_for_shap_and_agg':[1],'selected_features':['feature']})
    if changed:
        with pytest.raises(ValueError,match='changed during'):fm._reduce_weighted_partials_for_completed_shards(**kwargs)
    else:assert len(fm._reduce_weighted_partials_for_completed_shards(**kwargs))==1
    assert len(calls)==2
