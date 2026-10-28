"""Check frame identity, normalization and MMGen summaries against small references.

Fixtures use temporary files and the real package modules."""
from __future__ import annotations


def test_keypre_normalization_and_mmgen_scientific_references(tmp_path, monkeypatch):
    import importlib
    import ast
    import queue
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import h5py

    ROOT = Path(__file__).resolve().parents[1] / 'autobehaver'
    OUT = tmp_path

    def load(stage,name):return importlib.import_module('autobehaver.'+stage+'.'+name)
    cat=load('postinf.mmgen','categories');b=load('postinf.mmgen','worker_bouts');s=load('postinf.mmgen','worker_summaries');f=load('postinf.mmgen','feature_matrix');t=load('postinf.mmgen','time_budgets');tr=load('postinf.mmgen','transitions')
    norm=load('preinf.keyprenorm','normparams');na=load('preinf.keyprenorm','normalize');ni=load('preinf.keyprenorm','identity');inspect=load('preinf.keyprenorm','inspect_outputs');schema=load('preinf.keyprenorm','schema');dlc=load('preinf.keypre','dlc_io');crop=load('preinf.autocropper','detection');io=load('postinf.mmgen','worker_io')
    checks=[]
    def check(name,actual,expected):
        ok=bool(np.allclose(actual,expected,equal_nan=True,atol=1e-8,rtol=1e-10));checks.append(dict(name=name,passed=ok));assert ok,(name,actual,expected)
    def reject(name,fun):
        try:fun()
        except (ValueError,RuntimeError):checks.append(dict(name=name,passed=True));return
        raise AssertionError('Did not reject '+name)
    # Declared category sets, not cohort-observed states.
    check('Declared producer alphabets',[len(cat.POSE_LABELS),len(cat.SUMMARY_ORIENT_LABELS),len(cat.SUMMARY_COMBINED_LABELS)],[10,3,24])
    def summarize(poses,orients,ind='i',minimum=5,sleep=20):
        poses=np.array(poses,dtype=object);orients=None if orients is None else np.array(orients,dtype=object)
        vals=b.build_bouts_and_valid_arrays(masked_poses=poses,masked_orients=orients,has_orient=orients is not None,min_bout=minimum,num_frames=len(poses))
        counts,switches=cat.category_transition_tables(pose_valid=vals[5],orient_valid=vals[6],combined_valid=vals[7],source_file=ind+'.csv',individual=ind,group='G')
        sm,sl=s.compute_bout_and_sleep_summaries(pose_bouts=vals[0],pose_orient_bouts_for_boutsummary=vals[3],orient_bouts_for_boutsummary=vals[4],masked_orients=orients,has_orient=orients is not None,num_frames=len(poses),fps=40,sleep_threshold_frames=sleep,group='G',individual_id=ind,sex='MALE',source_file=ind+'.csv',masked_poses=poses,switch_counts=switches)
        return sm,sl,counts
    def event(counts,category,frm,to):
        x=counts[counts.category.eq(category)&counts.from_behavior.eq(frm)&counts.to_behavior.eq(to)]
        return x['count'].sum()
    def aggregate(summaries):
        df=pd.concat(summaries);df['animal_id']='A'
        from autobehaver.postinf.mmgen.summary_pooling import pool_bout_summaries
        return pool_bout_summaries(df)

    # Native adjacency and category independence.
    sm,sl,c=summarize(['Standing']*20+['Walking']*10+['Groom-Head']*20,['Floor']*50)
    check('Short valid Walking transitions',[event(c,'Pose+Orient','Standing + Floor','Walking + Floor'),event(c,'Pose+Orient','Walking + Floor','Groom-Head + Floor'),event(c,'Pose+Orient','Standing + Floor','Groom-Head + Floor')],[1,1,0])
    for middle in (['Walking']*3,[np.nan]*3):
        _,_,c=summarize(['Standing']*20+middle+['Groom-Head']*20,['Floor']*43)
        check('Missing/subthreshold interval does not bridge',event(c,'Pose','Standing','Groom-Head'),0)
    sm,_,c=summarize(['Walking']*40,['Floor']*20+['Ceiling']*20)
    check('Orientation change leaves Pose unchanged',[c[c.category.eq('Pose')]['count'].sum(),sm[sm.category.eq('Pose')].n_switches.sum()],[0,0])
    check('Physical impossible pair exclusion',event(c,'Pose+Orient','Walking + Floor','Walking + Ceiling'),0)
    sm,_,c=summarize(['Walking']*10+['Jump']+['Walking']*10,['Floor']*21)
    check('Bare Jump report retained',[event(c,'Pose+Orient','Walking + Floor','Jump'),event(c,'Pose+Orient','Jump','Walking + Floor')],[1,1])
    check('Special does not enter combined switchscope',sm[sm.category.eq('Pose+Orient')].n_switches.sum(),0)
    _,_,c=summarize(['Jump']+['Scramble']*5,[np.nan]*6)
    check('Jump Scramble counts once per nativecategory',[event(c,'Pose','Jump','Scramble'),event(c,'Pose+Orient','Jump','Scramble')],[1,1])
    reject('Unknown behavior class',lambda:summarize(['Unknown']*40,['Floor']*40))
    reject('Fully missing behavior',lambda:summarize([np.nan]*40,['Floor']*40))
    reject('Fully missing present orientation',lambda:summarize(['Walking']*40,[np.nan]*40))

    # Pool complete events and observed exposure, including zero-event recordings.
    a,_,_=summarize(['Standing']*10+['Walking']*20+['Standing']*10,['Floor']*40,'a')
    z,_,_=summarize(['Standing']*40,['Floor']*40,'z')
    agg=aggregate([a,z]);row=agg[agg.category.eq('Pose')&agg.bout_type.eq('Walking')].iloc[0]
    check('Zero-event recording included',[row.bout_frequency,row.initiation_prob,row.n_bouts,row.bout_duration_mean_s],[.5,1/60,1,.5])
    long,_,_=summarize(['Standing']*80,['Floor']*80,'long')
    row=aggregate([a,long]).query("category=='Pose' and bout_type=='Walking'").iloc[0]
    check('Unequal durations pool observed exposure',row.bout_frequency,1/3)
    row=z.query("category=='Pose' and bout_type=='Walking'").iloc[0]
    check('Absent bout duration undefined',[row.bout_frequency,row.bout_duration_mean_s,row.inter_bout_interval_mean_s],[0,np.nan,np.nan])
    aa,_,_=summarize(['Standing']*40,['Floor']*40,'r1');zz,_,_=summarize(['Walking']*40,['Floor']*40,'r2')
    check('No between recording switches',aggregate([aa,zz]).n_switches.sum(),0)
    _,zero,_=summarize(['Walking']*40,['Floor']*40)
    check('General noStanding sleep observedzero',zero.query("bout_type=='Standing'").n_sleep_bouts.iloc[0],0)
    check('NoStanding sleep duration remainsundefined',zero.query("bout_type=='Standing'").sleep_bout_duration_mean_s.iloc[0],np.nan)
    empty,empty_sleep=s.compute_bout_and_sleep_summaries(pose_bouts=[],pose_orient_bouts_for_boutsummary=[],orient_bouts_for_boutsummary=[],masked_orients=None,has_orient=False,num_frames=40,fps=40,sleep_threshold_frames=20,group='G',individual_id='missing',sex='MALE',source_file='missing.csv',masked_poses=np.full(40,np.nan))
    check('Unknown recording does not emit observed sleepzero',len(empty_sleep),0)
    check('Unknown recording does not emit observed eventzero',empty.bout_frequency.notna().sum(),0)
    _,sleep,_=summarize(['Walking']*5+['Standing']*40+['Walking']*5,['Floor']*30+['Ceiling']*20)
    check('Whole sleep dominant orientation conserved',[sleep.query("bout_type=='Standing'").sleep_total_length_frames.iloc[0],sleep.query("bout_type!='Standing'").sleep_total_length_frames.sum()],[40,40])

    # Fixed alphabet reference; table order and shard presence do not matter.
    for category,prefix in [('Pose','TimeBudget_Pose_'),('Orient','TimeBudget_Orient_'),('Pose+Orient','TimeBudget_PoseOrient_')]:
        allowed=cat.SUMMARY_LABELS[category];base=pd.DataFrame({prefix+allowed[0]:[.5],prefix+allowed[1]:[.5]})
        expected=np.log(2)/np.log(len(allowed))
        check('Fixed entropy '+category,cat.normalized_budget_entropy(base,prefix,allowed),[expected])
        extra=base.copy();extra[prefix+allowed[2]]=0
        check('Absentcolumn entropy invariant '+category,cat.normalized_budget_entropy(extra,prefix,allowed),[expected])
        uniform=pd.DataFrame({prefix+x:[1/len(allowed)] for x in allowed})
        check('Uniform entropy '+category,cat.normalized_budget_entropy(uniform,prefix,allowed),[1])
        single=pd.DataFrame({prefix+allowed[0]:[1,0,np.nan]})
        check('Single/empty/unknown entropy '+category,cat.normalized_budget_entropy(single,prefix,allowed),[0,np.nan,np.nan])
        reject('Unexpected entropy category '+category,lambda:cat.normalized_budget_entropy(pd.DataFrame({prefix+'Mystery':[1]}),prefix,allowed))

    # Actual budget/MASTER integration of zero rows and switches.
    allsm=pd.concat([a,z]);allsm['animal_id']=allsm.individual
    sheets=t.calculate_and_save_time_budgets(allsm,{'a':40,'z':40},str(OUT),save_outputs=False)
    check('Budget fullzero categories no conditionalassert',sheets['Pose (Overall)'].set_index('individual').loc['z','Walking'],0)
    check('Absent behavior conditionaldenominator unavailable',sheets['Orient given Pose (Overall)'].set_index('individual').loc['z','Floor | Walking'],np.nan)
    check('Valid behavior absentorientation event observedzero',sheets['Orient given Pose (Overall)'].set_index('individual').loc['a','Ceiling | Walking'],0)
    check('Absent orientation condition unavailable',sheets['Pose given Orient (Overall)'].set_index('individual').loc['a','Walking | Ceiling'],np.nan)
    single_a=t.calculate_and_save_time_budgets(allsm[allsm.individual.eq('a')],{'a':40},str(OUT),save_outputs=False)
    single_z=t.calculate_and_save_time_budgets(allsm[allsm.individual.eq('z')],{'z':40},str(OUT),save_outputs=False)
    for sheet in ('Pose given Orient (Overall)','Orient given Pose (Overall)',
                  'Pose given Orient (by Sex)','Orient given Pose (by Sex)'):
        merged=pd.concat([single_a[sheet],single_z[sheet]]).set_index('individual')
        cohort=sheets[sheet].set_index('individual')
        columns=[c for c in cohort if ' | ' in c]
        check('Conditional cohort/shard invariant '+sheet,
            cohort.loc[['a','z'],columns].to_numpy(float),merged.loc[['a','z'],columns].to_numpy(float))
    summary=aggregate([a,z]);dest=OUT/'matrix_reference';dest.mkdir(exist_ok=True)
    master=f.create_feature_matrix_from_sources(str(dest),f.MMGenFinalizationSources(bout_summary_df=summary),filter_features_by_bout_count=False)
    check('Native MASTER zero rate',master['BoutSummary_Pose_Walking_bout_frequency'].iloc[0],.5)
    check('No metadata leaked into matrix',sum('n_switches' in col for col in master),0)
    spatial=pd.DataFrame({'individual':['A']*80,'source_file':['r1']*40+['r2']*40,
        'frame_idx':list(range(40))*2,'r_norm':[.5]*80,'relative_angle':[0.]*80,
        'masked_pose':['Standing']*80,'masked_orient':['Floor']*80,'masked_pose_orient':['Standing + Floor']*80})
    budget_master=pd.DataFrame({'individual':['A'],'TimeBudget_Pose_Standing':[.5],'TimeBudget_Pose_Walking':[.5]})
    # Repertoire entropy belongs to complete time budgets, independent of spatial
    # coverage. Exercise the final matrix with the same fixed two-state reference.
    entropy_dest=OUT/'entropy_reference';entropy_dest.mkdir()
    pose_budget=pd.DataFrame({'individual':['A'],'group':['G'],'Standing':[.5],'Walking':[.5]})
    actual=f.create_feature_matrix_from_sources(str(entropy_dest),f.MMGenFinalizationSources(
        bout_summary_df=aggregate([aa,zz]),time_budget_sheets={'Pose (Overall)':pose_budget},
        spatial_df=spatial),filter_features_by_bout_count=False,video_fps=40)
    check('Actual matrix switch boundary fixed',actual.Transition_Repertoire_Pose_SwitchRate_per_min.iloc[0],0)
    check('Actual matrix fixed entropy',actual.Transition_Repertoire_Pose_EntropyNorm.iloc[0],np.log(2)/np.log(10))
    budget_master['TimeBudget_Orient_Floor']=1;budget_master['TimeBudget_Orient_Wall-Up']=0
    check('Zero unsupported displayed column does not expand entropy',cat.normalized_budget_entropy(budget_master,'TimeBudget_Orient_',cat.SUMMARY_ORIENT_LABELS),[0])
    # Full report consumer must not augment canonical Jump/Scramble from Pose.
    _,_,special_counts=summarize(['Jump']+['Scramble']*5,[np.nan]*6)
    capture={}
    original_report=tr._run_transition_analysis_for_category
    def capture_report(**kwargs):
        capture[kwargs['category_name']]=kwargs['df_category'].copy()
        return {'tidy_df':pd.DataFrame(),'sheets_to_save':{}}
    monkeypatch.setattr(tr, '_run_transition_analysis_for_category', capture_report)
    tr.calculate_and_save_transition_analysis(special_counts,str(OUT),run_transition_statistics=False,save_outputs=False)
    monkeypatch.setattr(tr, '_run_transition_analysis_for_category', original_report)
    check('Canonical special counts consumed once',capture['Pose+Orient']['count'].sum(),1)

    # Compare stored float32 values with the population-moment reference.
    monkeypatch.setattr(norm, '_STATS_CANONICAL_COLUMNS', ['x','y'])
    monkeypatch.setattr(norm, '_STATS_EDGE_SKIP', 0)
    for center,spread in [(300,.01),(10000,1),(1000000,1)]:
        vals=np.array([center-spread,center+spread]*20,dtype=np.float32)
        df=pd.DataFrame({'x':vals,'y':np.where(np.arange(len(vals))%3==0,np.nan,vals)})
        path=OUT/f'variance{center}.csv';df.to_csv(path,index=False)
        mean,m2,count=norm.process_stats_batch([str(path)])
        ref=df.to_numpy(dtype=np.float32).astype(float)
        check('Stable mean '+str(center),mean,np.nanmean(ref,axis=0))
        check('Stable populationvariance '+str(center),m2/count,np.nanvar(ref,axis=0))
        p1=OUT/'v1.csv';p2=OUT/'v2.csv';df.iloc[:11].to_csv(p1,index=False);df.iloc[11:].to_csv(p2,index=False)
        q1=norm.process_stats_batch([str(p1)]);q2=norm.process_stats_batch([str(p2)])
        merged=norm.merge_centered_moments(*q1[:3],*q2[:3])
        check('Stable batchpartition '+str(center),merged[1]/merged[2],np.nanvar(ref,axis=0))
    const=OUT/'constant.csv';pd.DataFrame({'x':[1,1,1],'y':[np.nan,7,np.nan]}).to_csv(const,index=False)
    cm,cv,cc=norm.process_stats_batch([str(const)])
    check('Constant and singleobservation populationSD',cv/cc,[0,0])

    # Frozen parameter application and identities/paired label integrity.
    data=OUT/'raw';labels=OUT/'labels';out=OUT/'normalized';lout=OUT/'labelout'
    for folder in (data,labels,out,lout):folder.mkdir(exist_ok=True)
    csv=data/'r_positional_calcs.csv';pd.DataFrame({'x':[0,1,2,3],'y':[np.nan,1,1,1]}).to_csv(csv,index=False)
    label=labels/'r_labels.csv';pd.DataFrame({'Frame':[0,1,2,3],'Behavior':['Walking']*4,'Orientation':['Floor']*4}).to_csv(label,index=False)
    monkeypatch.setattr(na, 'setup_worker_logging', lambda *args: None)
    na.init_normalize_worker(['x','y'],np.array([1.5,1]),np.array([1,0]),1,str(out),str(labels),str(lout),None,'INFO')
    res=na.normalize_one_csv(str(csv));assert res['status']=='success',res
    with h5py.File(out/'r_positional_calcs_normalized.h5') as hf:
        check('Frozen float32 application unchanged',hf['data'][:],[[-.5,0],[.5,0]])
        identity=hf.attrs['content_identity']
    check('Content identity accepted',inspect.inspect_normalized_h5(str(out/'r_positional_calcs_normalized.h5'),2,expected_rows=2,expected_identity=identity)[0],True)
    new_identity=ni.normalization_identity(csv,['x','y'],[2.5,1],[1,0],1)
    check('Changed params rejects reuse',inspect.inspect_normalized_h5(str(out/'r_positional_calcs_normalized.h5'),2,expected_rows=2,expected_identity=new_identity)[0],False)
    check('Changed trim rejects reuse',ni.normalization_identity(csv,['x','y'],[1.5,1],[1,0],0)==identity,False)
    check('Changed schema order rejects reuse',ni.normalization_identity(csv,['y','x'],[1.5,1],[1,0],1)==identity,False)
    pd.DataFrame({'Frame':[3,2,1,0],'Behavior':['Walking']*4}).to_csv(label,index=False)
    before=ni.file_hash(out/'r_positional_calcs_normalized.h5')
    bad=na.normalize_one_csv(str(csv));check('Bad labels fail before featurecommit',bad['status']=='failed',True)
    check('Bad labels preserve previously saved features',ni.file_hash(out/'r_positional_calcs_normalized.h5')==before,True)
    label_receipt=ni.label_identity(label,identity)
    check('Changed labels reject paired reuse',inspect.inspect_label_h5(str(lout/'r_labels.h5'),3,expected_identity=label_receipt)[0],False)
    duplicate=OUT/'duplicate.csv';duplicate.write_text('x,x\n1,2\n')
    check('Duplicate norm schema rejected',schema.read_csv_header_columns(str(duplicate))[0] is None,True)

    # Recomputed normalization parameters invalidate outputs using the prior scale.
    core=load('preinf.keyprenorm','core');cfgmod=load('preinf.keyprenorm','config')
    class Listener:
        def stop(self):pass
    monkeypatch.setattr(core, 'setup_logging_queue', lambda **kwargs:(queue.Queue(),Listener()))
    case=OUT/'orchestrator_case';inp=case/'input';noutput=case/'output';inp.mkdir(parents=True,exist_ok=True);noutput.mkdir(exist_ok=True)
    first=inp/'a_positional_calcs.csv';second=inp/'b_positional_calcs.csv'
    if second.exists():second.unlink()
    pd.DataFrame({'x':[0.,1.,2.,3.]}).to_csv(first,index=False)
    import argparse
    cfg=cfgmod.KeypreNormConfig(input_dir=str(inp),output_dir=str(noutput),calculate_norm=True,clobber=True,num_workers=1,edge_skip=0,log_level='ERROR')
    core.run_keyprenorm(cfg=cfg,args=argparse.Namespace())
    pd.DataFrame({'x':[10.,11.,12.,13.]}).to_csv(second,index=False)
    cfg=cfgmod.KeypreNormConfig(input_dir=str(inp),output_dir=str(noutput),calculate_norm=True,clobber=False,num_workers=1,edge_skip=0,log_level='ERROR')
    core.run_keyprenorm(cfg=cfg,args=argparse.Namespace())
    parameters=pd.read_csv(noutput/'norm_params.csv');mean=parameters.Mean.iloc[0];sd=parameters.StdDev.iloc[0]
    with h5py.File(noutput/'a_positional_calcs_normalized.h5') as hf:
        check('Actual orchestrator parametercache invalidation',hf['data'][:,0],((np.arange(4,dtype=np.float32)-np.float32(mean))/np.float32(sd)))

    # Check DLC frame/schema validation and MMGen's consecutive one-based offset.
    columns=pd.MultiIndex.from_product([['scorer'],['A'],['x','y','likelihood']])
    for indices in ([0,1,2],[1,2,3]):
        file=OUT/'dlc.csv';pd.DataFrame(np.ones((3,3)),columns=columns,index=indices).to_csv(file)
        check('Valid DLC framebase',len(dlc.load_dlc_csv_flat(str(file))),3)
    for indices in ([0,1,4],[0,1,1],[2,1,0]):
        pd.DataFrame(np.ones((3,3)),columns=columns,index=indices).to_csv(file)
        reject('Malformed DLC frameidentity',lambda:dlc.load_dlc_csv_flat(str(file)))
    pd.DataFrame(np.ones((3,2)),columns=columns[:2],index=[0,1,2]).to_csv(file)
    reject('Incomplete DLC triplet',lambda:dlc.load_dlc_csv_flat(str(file)))
    df=pd.DataFrame({'Frame Number':[1,2,3]});pos=pd.DataFrame({'x':range(15)})
    _,aligned=io._align_poscalc_to_inference(df,pos,edge_skip=5,source_name='test',source_frame_offset=5)
    check('Mmgen verifiedoffset5',aligned.x.to_numpy(),[5,6,7])
    reject('Mmgen sparse frames',lambda:io._align_poscalc_to_inference(pd.DataFrame({'Frame Number':[1,2,4]}),pos,edge_skip=5,source_name='test',source_frame_offset=5))

    # Calibrated bounded global chamber matching, no guessed tolerance.
    for width,height,scale in [(2016,1520,1),(2028,1520,1)]:
        centers=crop.calculate_theoretical_centers(width,height,3,2)
        circles=np.array([[x/2,y/2,100] for x,y in centers.values()])
        matched=crop.match_circles_to_chambers(circles,centers,2,max_center_distance_px=10)
        check('Calibrated chamberassignment scale'+str(scale),[v for v in matched.values()],[v for v in centers.values()])
        false=circles.copy();false[-1,:2]=[0,0]
        reject('False replacement circle rejected',lambda:crop.match_circles_to_chambers(false,centers,2,max_center_distance_px=10))
    reject('Uncalibrated layout rejected',lambda:crop.calculate_theoretical_centers(3840,2160,3,2))
    reject('No unbounded chambermatching',lambda:crop.match_circles_to_chambers(circles,centers,2))
    reject('Ambiguous chamber identity',lambda:crop.match_circles_to_chambers(np.array([[0,0,1],[0,0,1]]),{'a':(0,0),'b':(0,0)},1,max_center_distance_px=1))
    calibration={'image_width':2016,'image_height':1520,'centers':{k:[v[0]+3,v[1]-2] for k,v in centers.items()},'max_center_distance_px':10}
    check('Explicit cameraFOV calibration exactdims',len(crop.calculate_theoretical_centers(2016,1520,3,2,calibration)),6)
    reject('Wrong calibration dimension rejected',lambda:crop.calculate_theoretical_centers(2028,1520,3,2,calibration))

    # Every candidate module parses without generating bytecode.
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    checks.append(dict(name='All candidate modules parse',passed=True))
    kin=load('postinf.mmgen','kinematics')
    kinrows=pd.DataFrame([dict(individual='A',group='G',bout_type='Standing + Floor',
        speed_std_mean=0.,speed_mean_mean=1.),dict(individual='B',group='G',
        bout_type='Standing + Floor',speed_std_mean=2.,speed_mean_mean=1.)])
    cohort_kin=kin.calculate_and_save_kinematic_summary({'_inst':kinrows},str(OUT),save_outputs=False)
    zero_kin=kin.calculate_and_save_kinematic_summary({'_inst':kinrows.iloc[:1]},str(OUT),save_outputs=False)
    col='Standing+Floor_speed_std_mean'
    check('Measuredzero BB feature retained onzeroshard',zero_kin['Pose + Orient (inst)'].set_index('individual').loc['A',col],0)
    check('Measuredzero BB matchescohort summary',cohort_kin['Pose + Orient (inst)'].set_index('individual').loc['A',col],0)
    assert len(checks) == 90
    assert all(check["passed"] for check in checks)
