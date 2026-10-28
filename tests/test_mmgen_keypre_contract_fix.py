"""Regression tests against independently calculated quantities and invariances."""
import numpy as np
import pandas as pd
import pytest
from autobehaver.postinf.mmgen import kinematics as kin, occupancy as occ, postural
from autobehaver.postinf.mmgen.input_contract import validate_keypre_inputs, required_keypre_columns
from autobehaver.postinf.mmgen.worker_kinematics import compute_kinematics_outputs
from autobehaver.preinf.keypre import pose, egocentric, centroid_kinematics

def test_actual_keypre_signed_derivatives_feed_bb_without_alias():
    n=300;t=np.arange(n,dtype=float);x=.001*t**3
    df=pd.DataFrame({'centroid_x':x,'centroid_y':0.,'Front_x':x+1,'Front_y':0.,'Back_x':x-1,'Back_y':0.})
    signals={};centroid_kinematics.add_centroid_kinematics_features(df,signals)
    produced=pd.DataFrame(signals)
    expected_speed=(x[2:]-x[:-2])/2
    expected_acc=(expected_speed[2:]-expected_speed[:-2])/2
    expected_jerk=(expected_acc[2:]-expected_acc[:-2])/2
    np.testing.assert_allclose(produced.acceleration_speed_inst.iloc[2:-2],expected_acc)
    np.testing.assert_allclose(produced.jerk_speed_inst.iloc[3:-3],expected_jerk,atol=1e-12)
    assert 'acceleration_magnitude_inst' not in produced
    assert 'jerk_magnitude_inst' not in produced
    # Supply independent harmless values for the other required channels.
    extra={c:np.full(n,.99 if c.endswith('_likelihood') else 1.) for c in required_keypre_columns() if c not in produced}
    produced=pd.concat([produced,pd.DataFrame(extra)],axis=1)
    validate_keypre_inputs(produced)
    bout={'start':20,'end':279,'label':'Walking + Floor','bout_type':'Walking + Floor'}
    bb,real=compute_kinematics_outputs(aligned_poscalc_df=produced,pose_bouts=[],pose_orient_bouts=[bout],fps=40,mm_per_px=.02,group='G',individual_id='G_FEMALE_rpi-1_trial1_chamber1',sex='FEMALE',source_file='audit.csv')
    expected=np.percentile(expected_acc[18:278]*.02*40**2,95)
    assert bb['_inst'].iloc[0].accel_speed_max==pytest.approx(expected)
    assert np.isfinite(bb['_inst'].iloc[0].jerk_speed_max)
    np.testing.assert_allclose(real.acceleration_speed_inst,produced.acceleration_speed_inst*.02*40**2)
    np.testing.assert_allclose(real.jerk_speed_inst,produced.jerk_speed_inst*.02*40**3)
    assert not any('accel_mag' in c or 'jerk_mag' in c for c in bb['_inst'])

def test_missing_name_is_rejected_not_silently_filled():
    df=pd.DataFrame({c:[1.] for c in required_keypre_columns()})
    validate_keypre_inputs(df)
    df=df.rename(columns={'acceleration_speed_inst':'acceleration_magnitude_inst'})
    with pytest.raises(ValueError,match='acceleration_speed_inst'):validate_keypre_inputs(df)
    with pytest.raises(ValueError,match='missing columns'):validate_keypre_inputs(pd.DataFrame({'Ocelli_x_raw':[1.]}))
    # Disabled optional stages do not demand their unrelated coordinate columns.
    minimal=pd.DataFrame({c:[1.] for c in required_keypre_columns(include_postural=False,include_spatial=False,include_measurements=False)})
    validate_keypre_inputs(minimal,include_postural=False,include_spatial=False,include_measurements=False)

@pytest.mark.parametrize('heading',[0,90,180,270])
@pytest.mark.parametrize('origin',[(100.,200.),(400.,450.)])
def test_actual_rotated_tarsi_extension_preserves_distance_and_translation(heading,origin):
    center=np.array(origin);rad=np.deg2rad(heading-90);axis=np.array([np.cos(rad),np.sin(rad)])
    row={};parts=['Antennae','AbdomenTip']+list(postural_test_parts())
    for p,point in [('Front',center+axis),('Back',center-axis)]:
        row[p+'_x'],row[p+'_y']=point
    points={p:center+np.array([3.,4.]) for p in parts}
    points['Antennae']=center+2*axis;points['AbdomenTip']=center-2*axis
    for p,point in points.items():row[p+'_x'],row[p+'_y']=point;row[p+'_likelihood']=.99
    raw=pd.DataFrame([row]);pose.add_pose_features(raw,{})
    rotated=egocentric.add_centered_and_rotated_keypoints(raw,parts)
    out={}
    from autobehaver.preinf.keypre.export_keypoints import add_rotated_keypoints_and_likelihoods
    add_rotated_keypoints_and_likelihoods(rotated,out,parts)
    df=pd.DataFrame(out);df['centroid_x']=center[0];df['centroid_y']=center[1];df['stance_area']=4.
    for family,parts2 in [('lr',['Eyes','Thorax','Abdomen','Wings','FrontLegs','MidLegs','BackLegs']),('tb',['Head','Thorax','Abdomen'])]:
        for p in parts2:
            for a in ['x','y']:df[f'asym_{family}_{a}_{p}']=0.
    for orient in ['Floor','Ceiling']:
        result=postural._calculate_postural_features(df,np.array(['Standing']),np.array([orient]),'a.csv','G','G_FEMALE_rpi-1_trial1_chamber1',mm_per_px=.02)
        for leg in ['FrontLegs','MidLegs','BackLegs']:assert result[f'extension_raw_{leg}_avg'].iloc[0]==pytest.approx(5*.02)

def postural_test_parts():
    return tuple(f'{l}Tarsus{s}' for l in ['Front','Mid','Back'] for s in ['Left','Right'])

@pytest.mark.parametrize('bearing',[0.,90.,180.,270.])
@pytest.mark.parametrize('offset',[0.,90.,180.,-45.])
def test_radial_orientation_matches_keypre_at_cardinal_positions(bearing,offset):
    radial=np.deg2rad(bearing-90)
    df=pd.DataFrame({'centroid_x':[299.5+100*np.cos(radial)],'centroid_y':[299.5+100*np.sin(radial)],'angle':[(bearing+offset)%360],'individual':['G_FEMALE_rpi-1_trial1_chamber1']})
    reference=pose.calculate_polar_coordinates(df,600)
    expected=np.degrees(np.arctan2(reference['relative_angle_sin'],reference['relative_angle_cos']))
    transformed=occ._transform_positional_individual_df(df,fixed_arena_radius=265)[0]
    # Compare circular angle; +180 and -180 describe the same orientation.
    np.testing.assert_allclose(np.sin(np.radians(transformed.relative_angle)),np.sin(np.radians(expected)),atol=1e-12)
    np.testing.assert_allclose(np.cos(np.radians(transformed.relative_angle)),np.cos(np.radians(expected)),atol=1e-12)
    assert transformed.r_norm.iloc[0]==pytest.approx(100/265)

@pytest.mark.parametrize('bearing,vector',[(0,(0,-1)),(90,(1,0)),(180,(0,1)),(270,(-1,0))])
def test_quiver_points_in_actual_image_direction(bearing,vector):
    acc=occ._OccupancyAccumulator(quiver_grid_size=2)
    acc.add_quiver('masked_pose','Walking','G',np.array([100.]),np.array([100.]),np.array([bearing]))
    u,v,count=acc.get_quiver_arrays(('masked_pose','Walking','G'))
    assert u.sum()==pytest.approx(vector[0],abs=1e-7)
    assert v.sum()==pytest.approx(vector[1],abs=1e-7)
    assert count.sum()==1

def test_head_summary_does_not_modify_input_under_copy_on_write():
    df=pd.DataFrame({f'{prefix}_{p}_inst':[float(i+1)]*25 for p in kin.HEAD_KEYPOINTS for i,prefix in enumerate(['partvelocity','partaccel','partjerk'])})
    for i,p in enumerate(kin.HEAD_KEYPOINTS):df[p+'_likelihood']=.1+i*.1
    original=df.copy(deep=True)
    for func,args in [(kin.compute_head_movement_features,('_inst',)),(kin.compute_head_movement_features_minmax,('_inst',)),(kin.compute_head_movement_features_median_skewness,('_inst',1.,1.,2.,1.,3.,1.))]:
        values=func(df,*args);assert np.isfinite(values).all();pd.testing.assert_frame_equal(df,original)

def test_featureimp_groups_include_corrected_names_and_keep_legacy_names():
    from autobehaver.postinf.featureimp.fsea import _generate_kinematic_groups
    features=['KinematicBB_Pose+Orient_Walking+Floor_accel_speed_max_mean','KinematicBB_Pose+Orient_Walking+Floor_jerk_speed_max_mean','KinematicBB_Pose+Orient_Walking+Floor_accel_mag_max_mean','KinematicBB_Pose+Orient_Jump_accel_speed_mean_mean','KinematicBB_Pose+Orient_Scramble_jerk_speed_mean_mean']
    groups=_generate_kinematic_groups(features)
    assert features[0] in groups['Kinematic_Walking_Acceleration']
    assert features[1] in groups['Kinematic_Walking_Jerk']
    assert features[2] in groups['Kinematic_Walking_Acceleration']
    assert features[3] in groups['PoseStatsKinematic_Jump']
    assert features[4] in groups['PoseStatsKinematic_Scramble']

def test_sequential_hdf_transform_agrees_with_individual_transform(tmp_path):
    df=pd.DataFrame({'centroid_x':[399.5,299.5,199.5,299.5],'centroid_y':[299.5,399.5,299.5,199.5],'angle':[90.,180.,270.,0.],'individual':['G_FEMALE_rpi-1_trial1_chamber1']*4,'group':['G']*4,'masked_pose':['Walking']*4,'masked_orient':['Floor']*4,'masked_pose_orient':['Walking + Floor']*4,'frame_idx':np.arange(4)})
    for c in [c for c in df if pd.api.types.is_string_dtype(df[c].dtype)]:df[c]=df[c].astype(object)
    store=tmp_path/'positional.h5';df.to_hdf(store,key='positional',format='table',data_columns=['individual'])
    output=tmp_path/'transformed.csv'
    occ._stream_transform_write_and_accumulate(str(store),'positional',str(tmp_path),chamber_diameter_px=530,generate_quiver_plots=True,transformed_output_path=str(output))
    actual=pd.read_csv(output);reference=occ._transform_positional_individual_df(df,fixed_arena_radius=265)[0]
    np.testing.assert_allclose(actual.relative_angle,reference.relative_angle,atol=1e-12)
    np.testing.assert_allclose(actual.r_norm,reference.r_norm)

def test_per_recording_worker_reports_missing_contract_inputs(monkeypatch):
    from autobehaver.postinf.mmgen import worker
    df=pd.DataFrame({c:[1.] for c in required_keypre_columns()}).drop(columns=['acceleration_speed_inst'])
    monkeypatch.setattr(worker,'load_inference_and_poscalc',lambda *a,**kw:(pd.DataFrame({'Frame Number':[1]}),df,df))
    result=worker.process_video_data(('a.csv','unused',5,5,'',{}))
    assert result[-1]['error_stage']=='validate_keypre_inputs'
    assert 'acceleration_speed_inst' in result[-1]['error_message']

def test_signed_slowing_down_is_not_turned_into_magnitude():
    n=100;t=np.arange(n,dtype=float);x=100*t-.2*t*t
    df=pd.DataFrame({'centroid_x':x,'centroid_y':0.,'Front_x':x+1,'Front_y':0.,'Back_x':x-1,'Back_y':0.})
    signals={};centroid_kinematics.add_centroid_kinematics_features(df,signals)
    np.testing.assert_allclose(np.asarray(signals['acceleration_speed_inst'])[2:-2],-.4,atol=1e-12)

def test_quiver_renderer_uses_image_axes(monkeypatch,tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    original=Axes.quiver;captured=[]
    def record(self,*args,**kwargs):captured.append(kwargs.get('angles'));return original(self,*args,**kwargs)
    monkeypatch.setattr(Axes,'quiver',record)
    acc=occ._OccupancyAccumulator(quiver_grid_size=2)
    acc.add_quiver('masked_pose','Walking','G',np.full(30,100.),np.full(30,100.),np.zeros(30))
    occ._generate_orientation_quiver_plots_from_acc(acc,str(tmp_path))
    assert captured==['xy']
