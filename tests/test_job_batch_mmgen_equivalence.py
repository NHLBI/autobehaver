"""Compare native one-job and batched MMGen on four 160-frame synthetic inputs."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from autobehaver.portable_paths import generated_path
from autobehaver.postinf.mmgen.core import MMGenConfig, run_mmgen
from autobehaver.postinf.mmgen.input_contract import required_keypre_columns


def test_batch_mmgen_preserves_single_job_matrix(tmp_path):
    inference, positional = tmp_path / "inf", tmp_path / "pos"
    inference.mkdir(); positional.mkdir()
    count = 160
    frame = np.arange(count, dtype=float)
    geometry = {"Antennae": (0,-12), "AbdomenTip": (0,15), "EyeLeft": (-3,-9), "EyeRight": (3,-9),
        "Ocelli": (0,-10), "Mouthparts": (0,-7), "ThoraxLeft": (-4,0), "ThoraxRight": (4,0),
        "ThoraxTop": (0,-2), "ThoraxBottom": (0,2), "AbdomenLeft": (-5,8), "AbdomenRight": (5,8),
        "AbdomenTop": (0,6), "AbdomenBottom": (0,10), "WingLeft": (-12,2), "WingRight": (12,2)}
    for leg, y in (("Front",-5),("Mid",0),("Back",5)):
        geometry[leg+"TarsusLeft"] = (-8,y)
        geometry[leg+"TarsusRight"] = (8,y)
    stems = []
    for animal in (1,2):
        for movie in (0,1):
            stem = f"G_FEMALE_exp_trial{animal}_rpi-1_movie{movie}_20261018120000PM_chamber1"
            stems.append(stem)
            values = {column: np.full(count, .99 if column.endswith("_likelihood") else 1.)
                      for column in required_keypre_columns()}
            for part, (x,y) in geometry.items():
                values[part+"_x"] = np.full(count,x,dtype=float)
                values[part+"_y"] = np.full(count,y,dtype=float)
            values.update(centroid_x=300+10*np.sin(frame/10), centroid_y=300+10*np.cos(frame/10),
                angle_sin=np.sin(frame/20), angle_cos=np.cos(frame/20),
                speed_inst=.5+.05*np.sin(frame/3), acceleration_speed_inst=.02*np.cos(frame/3),
                jerk_speed_inst=.005*np.sin(frame/3))
            pd.DataFrame(values).to_csv(generated_path(positional,stem+"_positional_calcs.csv"),index=False)
            pose = (["Standing"]*20+["Walking"]*20)*4 if animal == 1 else ["Standing"]*count
            indices = np.arange(25, count-25)
            pd.DataFrame({"Smoothed Pose Label":pose[25:count-25],
                          "Smoothed Orient Label":["Floor"]*len(indices),
                          "Source Frame Index":indices,"Frame Number":indices+1}).to_csv(
                generated_path(inference,stem+"_positional_calcs_normalized_inference.csv"),index=False)
    listing = tmp_path / "all.txt"; listing.write_text("\n".join(stems)+"\n")
    single, batched = tmp_path / "one", tmp_path / "split"
    cfg = MMGenConfig(inference_csv_dir=str(inference),poscalc_dir=str(positional),
        output_dir=str(single),num_workers=1,source_frame_offset=0,
        emit_master_matrix=True,emit_full_outputs=True,keep_temp_files=False,file_list=str(listing))
    run_mmgen(cfg)
    for index in range(2):
        shard = replace(cfg,output_dir=str(batched),num_shards=2,shard_index=index,
                        shard_by="row_identity",process_only=True,resume=True)
        if index == 0:
            with patch('autobehaver.postinf.mmgen.pipeline.ordered_worker_results',
                       side_effect=RuntimeError('temporary worker startup failure')):
                try:
                    run_mmgen(shard)
                except RuntimeError as exc:
                    assert 'temporary worker startup failure' in str(exc)
                else:
                    raise AssertionError('Injected startup failure did not reach the caller')
        run_mmgen(shard)
        # Repeating a committed shard must not add duplicate measurements.
        run_mmgen(shard)
    run_mmgen(replace(cfg,output_dir=str(batched),num_shards=2,shard_by="row_identity",merge_shards=True))
    a = pd.read_csv(single/"MASTER_FEATURE_MATRIX.csv").sort_values("individual").reset_index(drop=True)
    b = pd.read_csv(batched/"MASTER_FEATURE_MATRIX.csv").sort_values("individual").reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12,atol=1e-12)
    assert len(a) == 2
    assert not (single/"temp_store.h5").exists()
    assert not (batched/"temp_store.h5").exists()
    assert not list(batched.glob("_mmgen_shards/*/temp_store.h5"))
