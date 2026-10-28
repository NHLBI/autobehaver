"""Check video and keypoint publication using a deterministic DLC test backend."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd


def test_native_dlc_regenerates_corruption_reuses_content_and_preserves_names(monkeypatch,tmp_path):
    import cv2
    from autobehaver.preinf.dlc import core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    from autobehaver.portable_paths import original_name, record_original_name
    source_dir=tmp_path/'movies';source_dir.mkdir()
    video=source_dir/'G_MALE_trial1_rpi-1_movie0_chamber1.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),40,(32,32))
    for i in range(6):writer.write(np.full((32,32,3),i*20,dtype=np.uint8))
    writer.release()
    record_original_name(video, 'Complete_G_MALE_trial1_rpi-1_movie0_chamber1.mp4')
    config=tmp_path/'config.yaml'
    config.write_text('bodyparts: [a, b]\nproject_path: '+tmp_path.as_posix()+'\n')
    output=tmp_path/'dlc';output.mkdir()
    calls=[]
    inspections=[]
    inspect_native=core.inspect_dlc_outputs
    def counted_inspect(*args, **kwargs):
        inspections.append(args)
        return inspect_native(*args, **kwargs)
    monkeypatch.setattr(core,'inspect_dlc_outputs',counted_inspect)
    def analyze(config,videos,**kwargs):
        directory=Path(kwargs['destfolder'])
        assert directory != output
        calls.extend(videos)
        columns=pd.MultiIndex.from_tuples([('scorer',part,coord) for part in ('a','b') for coord in ('x','y','likelihood')])
        table=pd.DataFrame(np.full((6,6),.8),columns=columns)
        for source in videos:
            stem=Path(source).stem+'DLC_very_long_original_scorer_name'
            table.to_csv(directory/(stem+'.csv'))
            table.to_hdf(directory/(stem+'.h5'),key='df_with_missing',format='table')
            (directory/(stem+'_meta.pickle')).write_bytes(b'test metadata')
    monkeypatch.setattr(core,'_require_deeplabcut',lambda:SimpleNamespace(analyze_videos=analyze))
    monkeypatch.setattr(core,'setup_gpu',lambda:None)
    monkeypatch.setattr(core,'model_identity',lambda *a:{'settings':{'bodyparts':['a','b']},'snapshot_files':['fixture']})
    cfg=DLCInferencerConfig(input_dir=str(source_dir),output_dir=str(output),dlc_config=str(config),max_retries=2)
    assert core.run_dlc(cfg)==0
    assert len(inspections)==2  # baseline plus one checked generated set
    csv=next(output.glob('*.csv'))
    assert original_name(csv).endswith('DLC_very_long_original_scorer_name.csv')
    before=csv.stat().st_mtime_ns
    assert core.run_dlc(cfg)==0
    assert len(inspections)==3  # validated reuse needs one content inspection
    assert len(calls)==1 and csv.stat().st_mtime_ns==before
    csv.write_text('broken\n')
    assert core.run_dlc(cfg)==0
    assert len(inspections)==5
    assert len(calls)==2 and list((output/'_old').glob('*.csv'))
    assert core.inspect_dlc_outputs(str(video),str(output))[0]
