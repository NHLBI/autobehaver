from pathlib import Path
import ast
import csv
import json
import types
import numpy as np
import pandas as pd
import pytest
import torch
from autobehaver.inference import smooth_bouts_vectorized
from autobehaver.checkpoint import validate_supervised_class_schema
from autobehaver.records import log_confusion_matrix
from autobehaver.losses import barlow_twins_loss
from autobehaver.registry import parse_genotype_sex_from_prefix
from autobehaver.resmooth import resmooth_file
from autobehaver.postinf.infmovie.frame_map import load_source_frames


def probabilities(labels, prefix='Pose'):
    return pd.DataFrame({f'{prefix}-{label}-Prob': np.ones(len(labels)) for label in set(labels) if label})


def test_jump_blank_and_current_bout_recheck():
    labels = ['Walking']*15+['Jump','Walking','Jump']+['Walking']*10
    out = smooth_bouts_vectorized(labels, probabilities(labels), 5, 'Pose', ['Jump'])
    assert list(out[15:18]) == ['Jump']*3
    labels = ['Standing']*5+[None]*2+['Walking']+[None]*2+['Standing']*5
    out = smooth_bouts_vectorized(labels, probabilities(labels), 5, 'Pose')
    assert out[7] == 'Walking' and all(pd.isna(out[i]) for i in (5,6,8,9))
    labels = [None]*5+['Walking']+['Standing']*5
    out = smooth_bouts_vectorized(labels, probabilities(labels)*0, 5, 'Pose')
    assert out[5] == 'Standing'
    labels = ['Standing']*5+['Walking','Groom-Head']+['Scramble']*5
    probs = probabilities(labels)*0
    probs.loc[5:6,'Pose-Standing-Prob'] = 1
    out = smooth_bouts_vectorized(labels, probs, 5, 'Pose')
    assert list(out[5:7]) == ['Standing','Standing']


@pytest.mark.parametrize('prefix', ['Pose','Orient'])
@pytest.mark.parametrize('length', [4,5,6])
def test_smoothing_threshold_edges(prefix, length):
    labels = ['A']*6+['B']*length+['A']*6
    out = smooth_bouts_vectorized(pd.Series(labels,index=range(100,100+len(labels))), probabilities(labels,prefix), 5,prefix)
    assert list(out[6:6+length]) == [('A' if length<5 else 'B')]*length
    assert len(smooth_bouts_vectorized([],pd.DataFrame(),5,prefix)) == 0
    assert all(pd.isna(x) for x in smooth_bouts_vectorized([None]*3,pd.DataFrame(index=range(3)),5,prefix))


def test_all_90_retained_jump_crops():
    evidence = Path(__file__).with_name('data')/'smoothing_raw_triggers.json'
    candidates = json.loads(evidence.read_text())['candidates']
    assert len(candidates)==90
    for candidate in candidates:
        labels = [bout['label'] for bout in candidate['bouts'] for _ in range(bout['length'])]
        out = smooth_bouts_vectorized(labels, probabilities(labels),5,'Pose',['Jump'])
        assert all(out[i]=='Jump' for i,label in enumerate(labels) if label=='Jump'), candidate['animal']


def test_resume_schema_and_confusion(tmp_path):
    saved = {'pose_dict':{'B':0,'A':1},'orient_dict':{'Floor':0,'Wall':1}}
    pose,orient = validate_supervised_class_schema(saved,{'A':0,'B':1},{'Wall':0,'Floor':1},pose_pred=True,orient_pred=True)
    assert list(pose.items())==[('B',0),('A',1)]
    for invalid in ({'A':0,'Z':1}, {}):
        with pytest.raises(ValueError):
            validate_supervised_class_schema(saved,invalid,orient,pose_pred=True,orient_pred=True)
    with pytest.raises(ValueError):
        validate_supervised_class_schema({},pose,{},pose_pred=True,orient_pred=False)
    path=tmp_path/'confusion.csv'
    log_confusion_matrix(np.diag([2,0,3]),['A','B','C'],1,str(path))
    rows=list(csv.reader(path.open()))
    assert len(rows[0])==len(rows[1])==10
    with pytest.raises(ValueError): log_confusion_matrix(np.ones((2,2)),['A','B','C'],2,str(path))
    with pytest.raises(ValueError): log_confusion_matrix(np.eye(3),['B','A','C'],2,str(path))


def test_barlow_and_lr_loss_contract():
    with pytest.raises(ValueError): barlow_twins_loss(torch.ones(3,2),torch.ones(3,2))
    z=torch.tensor([[1.,3.],[2.,1.],[4.,2.]],requires_grad=True)
    loss=barlow_twins_loss(z,z); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    source=Path(__file__).resolve().parents[1]/'autobehaver/lr_finder.py'
    tree=ast.parse(source.read_text())
    # Execute the actual loss block, independent of the LR sweep/plotting.
    code=source.read_text(); begin=code.index('            sample_indices ='); end=code.index('            if args.recon:',begin)
    block=code[begin:end]; block='\n'.join(line[12:] for line in block.splitlines())
    from torch.nn import functional as F
    logits=torch.tensor([[1.,2.],[2.,1.]])
    ns={'torch':torch,'F':F,'_idx':torch.tensor([3,1]),'device':torch.device('cpu'),
        'train_class_weights':torch.tensor([1.,.6,1.,2.]), 'pose_targets':torch.tensor([0,1]),
        'orient_targets':torch.tensor([-1,-1]),'pose_logits':logits,'orient_logits':None,
        'args':types.SimpleNamespace(pose_pred=True,orient_pred=False,pose_weight=1.),'total_loss':torch.tensor(0.)}
    exec(block,ns)
    expected=(F.cross_entropy(logits,ns['pose_targets'],reduction='none',label_smoothing=.1)*torch.tensor([2.,.6])).mean()
    assert torch.allclose(ns['total_loss'],expected)


def test_whole_sex_tokens():
    assert parse_genotype_sex_from_prefix('G_FEMALE_')==('G','FEMALE')
    assert parse_genotype_sex_from_prefix('G_MALE_')==('G','MALE')
    assert parse_genotype_sex_from_prefix('Genotype_female_')==('Genotype','FEMALE')
    assert parse_genotype_sex_from_prefix('G_NOTMALE_')==(None,None)


def test_native_prepare_data_restores_saved_class_meaning(tmp_path):
    import h5py
    from autobehaver._training_core import prepare_data
    path=tmp_path/'inputs.h5'; labels=tmp_path/'labels.h5'
    with h5py.File(path,'w') as handle:
        handle['data']=np.ones((7,2),dtype=np.float32);handle.attrs['columns']=np.array(['x','y'],dtype='S')
    with h5py.File(labels,'w') as handle:
        handle['Behavior']=np.array(['A','B','A','B','A','B','A'],dtype='S')
    checkpoint=tmp_path/'saved.pth'
    metadata=pd.DataFrame({'filename':['inputs.h5'],'labels':['labels.h5']})
    torch.save({'checkpoint_kind':'supervised','pose_dict':{'B':0,'A':1},
                'train_filenames':['inputs.h5'],'val_filenames':[]},checkpoint)
    result=prepare_data(str(tmp_path),metadata,3,pretrained_model_path=str(checkpoint),pose_pred=True,orient_pred=False,recon=False,max_class_weight=10.,defined_split=False,noval=True,num_workers=0)
    assert list(result[2].items())==[('B',0),('A',1)]
    instructions=result[0].dataset.dataset_instructions
    assert all(i['pose_index']==({'B':0,'A':1}[i['pose_label']]) for i in instructions)
    torch.save({'checkpoint_kind':'supervised','pose_dict':{'A':0,'Z':1},
                'train_filenames':['inputs.h5'],'val_filenames':[]},checkpoint)
    with pytest.raises(ValueError):
        prepare_data(str(tmp_path),metadata,3,pretrained_model_path=str(checkpoint),pose_pred=True,orient_pred=False,recon=False,max_class_weight=10.,defined_split=False,noval=True,num_workers=0)


def test_resmoothing_preserves_every_other_cell(tmp_path):
    source=tmp_path/'old.csv'; destination=tmp_path/'new.csv'
    labels=['Walking']*5+['Jump','Walking','Jump']+['Walking']*5
    table=pd.DataFrame({'Frame Number':range(1,len(labels)+1),'Predicted Pose Label':labels,
        'Pose-Walking-Prob':['0.800000']*len(labels),'Pose-Jump-Prob':['0.200000']*len(labels),
        'Other':['001']*len(labels)})
    table.to_csv(source,index=False)
    Path(str(source)+'.provenance.json').write_text(json.dumps({'source_frame_offset':5}))
    result=resmooth_file(source,destination)
    old=list(csv.DictReader(source.open())); new=list(csv.DictReader(destination.open()))
    assert all(all(n[k]==v for k,v in o.items()) for o,n in zip(old,new))
    assert new[5]['Smoothed Pose Label']==new[7]['Smoothed Pose Label']=='Jump'
    assert result['row_count']==len(labels) and Path(str(destination)+'.provenance.json').exists()
    assert load_source_frames(destination).tolist()==list(range(5,5+len(labels)))
    with pytest.raises(FileExistsError): resmooth_file(source,destination)


def test_resmoothing_cli_cohort_completion(tmp_path):
    from autobehaver.resmooth import main
    original=tmp_path/'original'; original.mkdir()
    destination=tmp_path/'corrected'
    path=original/'one_inference.csv'
    path.write_text('Frame Number,Predicted Pose Label,Pose-A-Prob\n1,A,1\n2,A,1\n')
    main(['--input-dir',str(original),'--output-dir',str(destination),'--source-frame-offset','5'])
    completion=json.loads((destination/'resmoothing.complete.json').read_text())
    assert completion['status']=='complete' and completion['file_count']==1
    assert completion['files'][0]['rows']==2
    with pytest.raises(FileExistsError): main(['--input-dir',str(original),'--output-dir',str(destination)])
