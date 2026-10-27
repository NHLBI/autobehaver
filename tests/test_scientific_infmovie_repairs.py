from pathlib import Path
import csv
import types
import numpy as np
import pytest
from autobehaver.postinf.infmovie.frame_map import load_source_frames
from autobehaver.postinf.infmovie.io import find_movie_file
from autobehaver.postinf.infmovie.movies import render_overlay_movie, process_csv_and_movie
from autobehaver.postinf.infmovie.config import InfMoviePipelineConfig
from autobehaver.postinf.infmovie import pipeline


def write_csv(path, count=15, offset=5):
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['Frame Number','Predicted Pose Label','Predicted Orient Label','Pose-A-Prob','Pose-B-Prob'])
        for i in range(count):
            writer.writerow([i+1,f'original_frame_{i+offset}','Floor',.9 if i%2 else .1,.1 if i%2 else .9])


def test_movie_identity_and_frames(tmp_path):
    old='trial1_rpi1_movie0_20260101010000AM_chamber1.mp4'
    new='trial1_rpi1_movie0_20260102010000AM_chamber1.mp4'
    for name in (old,new): (tmp_path/name).touch()
    assert Path(find_movie_file(str(tmp_path),Path(new).stem)).name==new
    assert find_movie_file(str(tmp_path),'chamber10') is None
    with pytest.raises(ValueError): find_movie_file(str(tmp_path),'trial1_rpi1_movie0_chamber1')
    path=tmp_path/'frames.csv'; write_csv(path)
    assert load_source_frames(path,source_frame_offset=5,video_frames=20).tolist()==list(range(5,20))
    with pytest.raises(ValueError): load_source_frames(path)
    with pytest.raises(ValueError): load_source_frames(path,source_frame_offset=5,video_frames=19)
    path.write_text('Frame Number\n1\n3\n')
    with pytest.raises(ValueError): load_source_frames(path,source_frame_offset=0)


@pytest.mark.parametrize('offset,start,end,speed',[(5,0,None,1),(5,8,13,1),(0,3,10,2)])
@pytest.mark.parametrize('legacy',[False,True])
def test_both_renderer_frame_contract(tmp_path, monkeypatch, offset,start,end,speed,legacy):
    path=tmp_path/'trial1_rpi1_movie0_chamber1_inference.csv'; write_csv(path,15,offset)
    movie=tmp_path/'trial1_rpi1_movie0_chamber1.mp4'; movie.touch()
    written=[]; labels=[]; stamps=[]
    class Capture:
        def __init__(self,*args): self.i=0
        def isOpened(self): return True
        def get(self,prop): return 30 if prop==7 else 40
        def set(self,prop,value): self.i=int(value); return True
        def read(self):
            if self.i>=30:return False,None
            frame=np.full((100,120,3),self.i,dtype=np.uint8);self.i+=1;return True,frame
        def release(self):pass
    class Writer:
        def __init__(self,*args):self.path=Path(args[0]);self.path.touch()
        def write(self,image): written.append(int(image[0,0,0]))
        def release(self):pass
    def put_text(image,text,*args):
        if text.startswith('original_frame_'): labels.append(text)
        if text.startswith('t='):stamps.append(text)
        return image
    fake=types.SimpleNamespace(VideoCapture=Capture,VideoWriter=Writer,VideoWriter_fourcc=lambda *args:0,
        CAP_PROP_FRAME_COUNT=7,CAP_PROP_FPS=5,CAP_PROP_POS_FRAMES=1,FONT_HERSHEY_COMPLEX=3,LINE_AA=16,
        getTextSize=lambda *args:((20,5),0),putText=put_text,line=lambda *args:None,rectangle=lambda *args:None)
    monkeypatch.setitem(__import__('sys').modules,'cv2',fake)
    if legacy:
        process_csv_and_movie(str(path),str(tmp_path),str(tmp_path),skip_first_n_frames=0,
            source_frame_offset=offset,show_timestamp=True,speed_factor=speed,atomic_write=False)
    else:
        render_overlay_movie(str(path),str(movie),str(tmp_path/'output.mp4'),source_frame_offset=offset,
            start_frame=start,end_frame_exclusive=end,skip_first_n_frames=0,show_timestamp=True,
            timestamp_reference='movie',speed_factor=speed,atomic_write=False)
    assert written
    assert all(label==f'original_frame_{frame}' for label,frame in zip(labels,written))
    if not legacy:
        assert written[0]==max(offset,start)
        assert written[-1]<(end or offset+15)
        # The renderer draws a shadow and foreground copy of each timestamp.
        assert all(abs(float(text.split('=')[1].split('s')[0])-frame/40)<.051 for text,frame in zip(stamps[::2],written))


def test_all_plan_modes_use_source_coordinates(tmp_path,monkeypatch):
    path=tmp_path/'frames.csv';write_csv(path,60,5)
    movie=tmp_path/'movie.mp4';movie.touch()
    monkeypatch.setattr(pipeline,'_get_video_info',lambda *args,**kwargs:(10.,70))
    pairs=[{'recording_id':'one','inf_csv_path':str(path),'movie_path':str(movie)}]
    config=InfMoviePipelineConfig(recording_id='one',source_frame_offset=5,start_sec=.8,duration_sec=1,clip_duration_sec=1,
        interesting_min_unique_labels=1,interesting_min_changes=0,num_clips=1,qc_uncertainty_threshold=.95,
        qc_min_uncertainty_frames=1,skip_first_n_frames=0)
    manual=pipeline.build_plan_manual(config,pairs=pairs)[0]
    full=pipeline.build_plan_full(config,pairs=pairs)[0]
    assert manual['start_frame']==8 and manual['end_frame_exclusive']==18
    assert full['start_frame']==5 and full['end_frame_exclusive']==65 and full['start_sec']==.5
    for build in (pipeline.build_plan_interesting,pipeline.build_plan_qc):
        clips=build(config,pairs=pairs)
        assert clips and all(c['start_frame']>=5 and c['end_frame_exclusive']<=65 for c in clips)
        assert all(c['start_sec']==c['start_frame']/10 for c in clips)
