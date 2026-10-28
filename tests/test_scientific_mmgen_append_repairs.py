"""Append publication must bind source/calculation contents and recover atomically."""
from __future__ import annotations


def test_mmgen_append_content_identity_and_publication_recovery(tmp_path):
    import os
    import json
    import shutil
    from pathlib import Path
    from unittest.mock import patch
    import pandas as pd
    from autobehaver.postinf.mmgen import append_mode as app, ledger, integrity
    from autobehaver.postinf.mmgen.core import MMGenConfig
    from autobehaver.atomic_io import atomic_write_json, sha256_file

    OUT = tmp_path

    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)));assert condition,name
    def reject(name,fun):
        try:fun()
        except (RuntimeError,ValueError,FileNotFoundError,OSError):checks.append(dict(name=name,passed=True));return
        raise AssertionError(name)
    names=[f'GROUPA_FEMALE_demo_trial1_rpi-1_movie{i}_20250101120000PM_chamber5_positional_calcs_normalized_inference.csv' for i in range(3)]
    def pair(inf,pos,name,text='Pose\nWalking\n'):
        (inf/name).write_text(text);(pos/name.replace('_normalized_inference','')).write_text('centroid_x\n1\n')
    def cfgcase(label):
        folder=OUT/label
        if folder.exists():shutil.rmtree(folder)
        inf=folder/'inf';pos=folder/'pos';root=folder/'root'
        for p in (inf,pos,root):p.mkdir(parents=True,exist_ok=True)
        cfg=MMGenConfig(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(root),append_new_data=True,num_workers=1,emit_full_outputs=False,source_frame_offset=5)
        return inf,pos,root,cfg
    def generation(root,cfg,sources,values=None):
        root=Path(root);root.mkdir(exist_ok=True)
        current=app._build_current_inventory(inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag)
        led=current[current.source_file.isin(sources)].copy();led['n_frames']=40;led['animal_id']=led.row_identity;led['rpi']='rpi-1';led['chamber']='chamber5';led['trial']=1
        for p in ['source_exists','poscalc_exists']:led[p]=True
        matrix=pd.DataFrame([dict(individual=i,group='GROUPA',sex='FEMALE',Global_MeanSpeed_mm_s=(values or {}).get(i,1.)) for i in sorted(led.row_identity.unique())])
        matrix.to_csv(root/'MASTER_FEATURE_MATRIX.csv',index=False);led.to_csv(root/ledger.SOURCE_LEDGER_FILENAME,index=False)
        summary=ledger.build_row_identity_summary_df(source_ledger_df=led,master_matrix_df=matrix);summary.to_csv(root/ledger.ROW_IDENTITY_SUMMARY_FILENAME,index=False)
        (root/ledger.BATCH_MANIFEST_FILENAME).write_text(json.dumps({'processing':{'append_new_data':False}}))
        calculation=app._cfg_calculation(cfg);src=integrity.source_identities(sources,inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag,source_frame_offset=cfg.runtime_settings.source_frame_offset)
        ident=dict(calculation=calculation,calculation_sha256=integrity.content_hash(calculation),sources=src,sources_sha256=integrity.content_hash(src))
        atomic_write_json(ident,root/app.IDENTITY_RELATIVE_PATH)
        receipt=dict(calculation_sha256=ident['calculation_sha256'],sources_sha256=ident['sources_sha256'],artifacts={p:sha256_file(root/p) for p in app.CORE_ARTIFACTS})
        atomic_write_json(receipt,root/app.OUTPUT_RECEIPT_FILENAME)
        return matrix,led

    # Changed content must be detected even when size and modification time match.
    for column in ('inference','poscalc'):
        inf,pos,root,cfg=cfgcase('content_'+column);pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
        path=inf/names[0] if column=='inference' else pos/names[0].replace('_normalized_inference','')
        original=path.stat();text=path.read_text();path.write_text(text.replace('Walking','Standin') if column=='inference' else text.replace('1','2'))
        os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
        reject('Changed '+column+' identical size/mtime',lambda:app._create_append_plan(cfg))

    inf,pos,root,cfg=cfgcase('plan');pair(inf,pos,names[0]);generation(root,cfg,[names[0]]);pair(inf,pos,names[1])
    plan,selected,affected,batch=app._create_append_plan(cfg)
    check('Plan freezes bothinputhashes',set(plan['selected_sources'])==set(names[:2]))
    app._validate_plan_against_cfg(plan,cfg);check('Unchanged plan validation',True)
    worker_cfg=MMGenConfig(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(root),append_new_data=True,num_workers=8,emit_full_outputs=False,source_frame_offset=5)
    app._validate_plan_against_cfg(plan,worker_cfg);check('Harmless workercount does not invalidate',True)
    changed_cfg=MMGenConfig(inference_csv_dir=str(inf),poscalc_dir=str(pos),output_dir=str(root),append_new_data=True,num_workers=1,emit_full_outputs=False,source_frame_offset=4)
    reject('Changed scientificoffset',lambda:app._validate_plan_against_cfg(plan,changed_cfg))
    changed_code=app._cfg_calculation(cfg);changed_code['source_modules']=dict(changed_code['source_modules']);changed_code['source_modules']['fake_changed_math.py']='0'*64
    with patch.object(app,'_cfg_calculation',return_value=changed_code):reject('Changed codemath identity',lambda:app._validate_plan_against_cfg(plan,cfg))
    before=(batch/app.APPEND_SELECTED_FILES_FILENAME).read_bytes();(batch/app.APPEND_SELECTED_FILES_FILENAME).write_bytes(before+b'\n')
    reject('Changed planCSV content',lambda:app._load_append_plan(str(root),plan['append_batch_id']))
    (batch/app.APPEND_SELECTED_FILES_FILENAME).write_bytes(before)
    loaded=app._load_append_plan(str(root),plan['append_batch_id']);check('Restored plan readback',loaded[0]['plan_sha256']==plan['plan_sha256'])
    row=selected.row_identity.iloc[0];generation(batch,cfg,names[:2],{row:10.})
    app._apply_delta_batch_to_canonical(cfg=cfg,plan=plan,delta_output_dir=batch)
    check('Successful canonical replacement',pd.read_csv(root/'MASTER_FEATURE_MATRIX.csv').Global_MeanSpeed_mm_s.iloc[0]==10)
    check('Successful ledger includes bothsources',set(pd.read_csv(root/ledger.SOURCE_LEDGER_FILENAME).source_file)==set(names[:2]))
    app._verified_generation(root);check('Published matrixledgerreceipt coherent',True)
    reject('Stale plan cannot reapply after changedgeneration',lambda:app._validate_plan_against_cfg(plan,cfg))

    # Exercise native append planning/publication with a deterministic calculation.
    inf,pos,root,cfg=cfgcase('entry');pair(inf,pos,names[0]);generation(root,cfg,[names[0]]);pair(inf,pos,names[1])
    captured=[]
    def fake_run(inference,poscalc,delta,min_bout,edge_skip,inf_tag,**kwargs):
        captured.extend(kwargs['selected_files']);generation(Path(delta),cfg,kwargs['selected_files'],{app._build_current_inventory(inference_csv_dir=str(inf),poscalc_dir=str(pos),inf_tag=cfg.inf_tag).row_identity.iloc[0]:11.})
        (Path(delta)/'temp_store.h5').write_bytes(b'required temporary calculation fixture')
    with patch.object(app.pipeline,'run_post_inference_analysis',side_effect=fake_run):app.run_mmgen_append_new_data(cfg)
    check('Full append entrypoint selects whole affectedanimal',set(captured)==set(names[:2]))
    check('Full append entrypoint publishes coherentnewmatrix',pd.read_csv(root/'MASTER_FEATURE_MATRIX.csv').Global_MeanSpeed_mm_s.iloc[0]==11)

    # A changed unselected historical animal also invalidates append.
    inf,pos,root,cfg=cfgcase('outsidefilter');pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
    pair(inf,pos,names[1]);cfg.file_filter='movie1'
    old=(inf/names[0]).stat();(inf/names[0]).write_text('Pose\nStandin\n');os.utime(inf/names[0],ns=(old.st_atime_ns,old.st_mtime_ns))
    reject('Historical content outsidefilter rejected',lambda:app._create_append_plan(cfg))

    # Corruption/legacy outputs cannot be accepted.
    inf,pos,root,cfg=cfgcase('legacy');pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
    (root/app.OUTPUT_RECEIPT_FILENAME).unlink();reject('Legacy receipt missing',lambda:app._load_required_canonical_state(str(root)))
    generation(root,cfg,[names[0]]);(root/ledger.SOURCE_LEDGER_FILENAME).write_text((root/ledger.SOURCE_LEDGER_FILENAME).read_text()+'\n')
    reject('Ledgercontent tamper',lambda:app._verified_generation(root))
    generation(root,cfg,[names[0]]);(root/'MASTER_FEATURE_MATRIX.csv').write_text((root/'MASTER_FEATURE_MATRIX.csv').read_text().replace('1.0','9.0'))
    reject('Matrixcontent tamper',lambda:app._verified_generation(root))

    # Failure at any publication step must restore the previous generation.
    for failure_index,failed_name in enumerate((*app.CORE_ARTIFACTS,app.IDENTITY_RELATIVE_PATH,app.OUTPUT_RECEIPT_FILENAME)):
        inf,pos,root,cfg=cfgcase('failure_'+str(failure_index));pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
        stage=root/'stage';generation(stage,cfg,[names[0]],{app._build_current_inventory(inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag).row_identity.iloc[0]:99.})
        originals={relative:sha256_file(root/relative) for relative in (*app.CORE_ARTIFACTS,app.IDENTITY_RELATIVE_PATH,app.OUTPUT_RECEIPT_FILENAME)}
        real_replace=os.replace;triggered=[False]
        def replace_fail(src,dst):
            if Path(src).name.endswith('.publish') and Path(dst)==root/failed_name and not triggered[0]:
                triggered[0]=True;raise OSError('Injected publication failure')
            return real_replace(src,dst)
        with patch.object(app.os,'replace',side_effect=replace_fail):
            reject('Failure injection '+failed_name,lambda:app._publish_generation(root,stage,(*app.CORE_ARTIFACTS,app.IDENTITY_RELATIVE_PATH,app.OUTPUT_RECEIPT_FILENAME)))
        check('Fullgeneration rollback '+failed_name,all(sha256_file(root/r)==h for r,h in originals.items()))
        app._verified_generation(root);check('Rollback remainscertified '+failed_name,True)

    # An interrupted publication must roll back from its durable journal.
    inf,pos,root,cfg=cfgcase('restart');pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
    backup=root/'stage/original_generation';backup.mkdir(parents=True)
    originals={}
    for relative in (*app.CORE_ARTIFACTS,app.IDENTITY_RELATIVE_PATH,app.OUTPUT_RECEIPT_FILENAME):
        saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,saved);originals[relative]=sha256_file(saved)
    atomic_write_json(dict(status='publishing',backup_dir=str(backup),originals=originals),root/app.PUBLICATION_JOURNAL)
    (root/'MASTER_FEATURE_MATRIX.csv').write_text('broken partialpublication')
    app._load_required_canonical_state(str(root))
    check('Restart restores entire oldgeneration',all(sha256_file(root/r)==h for r,h in originals.items()))
    check('Restart journal removed onlyafterrecovery',not(root/app.PUBLICATION_JOURNAL).exists())

    # An active append lock prohibits another reader from performing rollback.
    atomic_write_json(dict(status='publishing',backup_dir=str(backup),originals=originals),root/app.PUBLICATION_JOURNAL)
    lock=root/'_mmgen_work/_mmgen_append.lock';lock.mkdir()
    reject('Active publication cannotbe rolledback byreader',lambda:app._verified_generation(root))
    check('Active journal preserved',(root/app.PUBLICATION_JOURNAL).exists())
    lock.rmdir();app._verified_generation(root);check('Unlocked interrupted generation recovers',not(root/app.PUBLICATION_JOURNAL).exists())

    # Delta receipt/content/source mismatch rejects before canonical publication.
    inf,pos,root,cfg=cfgcase('bad_delta');pair(inf,pos,names[0]);generation(root,cfg,[names[0]]);pair(inf,pos,names[1])
    plan,selected,_,batch=app._create_append_plan(cfg);generation(batch,cfg,names[:2])
    original_hash=sha256_file(root/'MASTER_FEATURE_MATRIX.csv');(batch/'MASTER_FEATURE_MATRIX.csv').write_text('bad delta')
    reject('Tampered delta artifact rejected',lambda:app._apply_delta_batch_to_canonical(cfg=cfg,plan=plan,delta_output_dir=batch))
    check('Tampered delta leaves canonicalunchanged',sha256_file(root/'MASTER_FEATURE_MATRIX.csv')==original_hash)

    # Reject a damaged recovery backup before modifying any target.
    inf,pos,root,cfg=cfgcase('badbackup');pair(inf,pos,names[0]);generation(root,cfg,[names[0]])
    backup=root/'stage/original_generation';backup.mkdir(parents=True)
    originals={}
    for relative in (*app.CORE_ARTIFACTS,app.IDENTITY_RELATIVE_PATH,app.OUTPUT_RECEIPT_FILENAME):
        saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,saved);originals[relative]=sha256_file(saved)
    atomic_write_json(dict(status='publishing',backup_dir=str(backup),originals=originals),root/app.PUBLICATION_JOURNAL)
    (backup/app.OUTPUT_RECEIPT_FILENAME).write_text('damagedbackup')
    (root/'MASTER_FEATURE_MATRIX.csv').write_text('mixedgeneration awaiting recovery')
    before=sha256_file(root/'MASTER_FEATURE_MATRIX.csv')
    reject('Damagedbackup prevents recovery',lambda:app._recover_publication(root))
    check('Recovery preflight does notpartiallyrestore',sha256_file(root/'MASTER_FEATURE_MATRIX.csv')==before)

    # Hash I/O exceptions cannot be turned into missing provenance.
    with patch.object(ledger,'sha256_file',side_effect=OSError('Injected hashreadfailure')):
        reject('Ledgerhashread failure propagates',lambda:ledger._safe_file_stats(str(inf/names[0])))

    # The ledger must verify committed input hashes and reject duplicate commits.
    inf,pos,root,cfg=cfgcase('ledger');pair(inf,pos,names[0]);inv=app._build_current_inventory(inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag)
    source=integrity.source_identities([names[0]],inference_csv_dir=cfg.inference_csv_dir,poscalc_dir=cfg.poscalc_dir,inf_tag=cfg.inf_tag,source_frame_offset=cfg.runtime_settings.source_frame_offset)
    lengths=pd.DataFrame([dict(source_file=names[0],n_frames=40,individual=inv.individual_id.iloc[0],group='GROUPA',sex='FEMALE',input_sha256=integrity.content_hash(source[names[0]]))])
    kwargs=dict(inference_csv_dir=str(inf),poscalc_dir=str(pos),animal_mapping={},file_to_animal={},inf_tag=cfg.inf_tag,source_frame_offset=cfg.runtime_settings.source_frame_offset)
    ledger.build_source_ledger_df(lengths_df=lengths,**kwargs);check('Commit hashes checked against actualinputs',True)
    reject('Duplicatecommits notsilentlycollapsed',lambda:ledger.build_source_ledger_df(lengths_df=pd.concat([lengths,lengths]),**kwargs))
    (pos/names[0].replace('_normalized_inference','')).write_text('centroid_x\n2\n')
    reject('Changed inputs beforeledgerpublication',lambda:ledger.build_source_ledger_df(lengths_df=lengths,**kwargs))
    assert len(checks) == 50
    assert all(check["passed"] for check in checks)
