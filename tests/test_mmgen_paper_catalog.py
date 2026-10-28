"""Native report generation preserves the declared feature catalog."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from autobehaver.postinf.mmgen import core
from autobehaver.postinf.mmgen.feature_schema import feature_columns
from tests.test_mmgen_repair_contracts import recording_name
from tests.test_mmgen_repair_workflows import native_config, write_native_pair


def test_full_output_reports_and_store_match_paper_catalog(tmp_path):
    cfg = native_config(tmp_path)
    write_native_pair(Path(cfg.inference_csv_dir), Path(cfg.poscalc_dir),
                      recording_name(), speed=2.)
    core.run_mmgen(replace(cfg, emit_full_outputs=True))
    out = Path(cfg.output_dir)
    frame = pd.read_csv(out / 'MASTER_FEATURE_MATRIX.csv')
    assert len(feature_columns()) == 9406
    assert frame.loc[0, 'KinematicBB_Pose+Orient_Standing+Floor_speed_mean_mean'] == 2.
    assert set(feature_columns()).issubset(frame.columns)
    receipt = json.loads((out / 'MMGEN_OUTPUT_RECEIPT.json').read_text())
    assert set(receipt['output_families']) == {
        'bout_summary', 'time_budgets', 'measurements', 'postural',
        'bb_kinematics', 'pose_transitions', 'combined_transitions', 'sleep', 'spatial',
    }
    assert receipt['output_families']['bb_kinematics'] == 'completed'
    sheet = pd.read_excel(out / '5_Kinematic_Features.xlsx', sheet_name='Pose + Orient (inst)')
    value = sheet.loc[sheet.individual == frame.individual.iloc[0], 'Standing+Floor_speed_mean_mean']
    assert value.iloc[0] == 2.
    with pd.HDFStore(out / 'temp_store.h5', 'r') as store:
        assert set(store.keys()) <= {
            '/bouts_inst', '/agg_inst_tmp', '/bout_summary_tmp', '/sleep_summary_tmp',
            '/transitions_tmp', '/video_lengths_tmp', '/recording_commits',
        }
