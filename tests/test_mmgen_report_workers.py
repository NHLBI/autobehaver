import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from autobehaver.postinf.mmgen import occupancy as occ
from autobehaver.postinf.mmgen.finalize_execution import recording_ranges, budgeted_results, GIB


class ReportWorkers(unittest.TestCase):
    def make_store(self, path, count=240):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(dict(individual=np.repeat(['c', 'a', 'b'], count),
            group=['g']*(count*3), source_file=np.repeat(['fc','fa','fb'],count),
            frame_idx=np.tile(np.arange(count),3),
            centroid_x=rng.normal(300,60,count*3), centroid_y=rng.normal(300,60,count*3),
            angle=rng.uniform(0,360,count*3), speed_mm_s=rng.uniform(0,2,count*3),
            masked_pose=['Walking']*(count*3), masked_orient=['Floor']*(count*3),
            masked_pose_orient=['Walking + Floor']*(count*3)))
        # Recording c occurs in multiple physical spans, crossing scan chunks.
        frame = pd.concat([frame.iloc[:count//2],frame.iloc[count:],frame.iloc[count//2:count]])
        frame.to_hdf(path,key='positional',format='table',data_columns=['individual'],index=False)
        return frame

    def test_ranges_cover_disjoint_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'s.h5'; frame=self.make_store(path)
            with pd.HDFStore(path,'r') as store:
                spans=recording_ranges(store,'positional',chunk_rows=71)
                for individual, ranges in spans.items():
                    actual=pd.concat([store.select('positional',start=a,stop=b) for a,b in ranges])
                    assert_frame_equal(actual,frame.loc[frame.individual.eq(individual)])

    def test_reports_match_legacy_and_resume(self):
        # Historical implementation is a test oracle, never used for production.
        baseline = Path(__file__).resolve().parents[3]/'MMGen efficiency investigation 20261022/before/occupancy.py'
        if not baseline.exists():
            self.skipTest('Local historical oracle is not distributed')
        spec=importlib.util.spec_from_file_location('autobehaver.postinf.mmgen.legacy_occ',baseline)
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'s.h5'; self.make_store(path,18000)
            legacy=root/'legacy'; legacy.mkdir()
            current=root/'new'; current.mkdir()
            expected=old._stream_transform_write_and_accumulate(str(path),'positional',str(legacy))
            actual=occ._stream_transform_write_and_accumulate(str(path),'positional',str(current),num_workers=2)
            self.assertEqual(expected.behaviors,actual.behaviors)
            for name in ('_heat_counts','_heat_n','_q_sum_u','_q_sum_v','_q_count'):
                self.assertEqual(set(getattr(expected,name)),set(getattr(actual,name)))
                for key in getattr(expected,name):
                    np.testing.assert_array_equal(getattr(expected,name)[key],getattr(actual,name)[key])
            for key in expected._kde:
                np.testing.assert_array_equal(expected._kde[key].keys,actual._kde[key].keys)
                np.testing.assert_array_equal(expected._kde[key].values,actual._kde[key].values)
            self.assertEqual((legacy/'temp_transformed_positional_data.csv').read_bytes(),
                             (current/'temp_transformed_positional_data.csv').read_bytes())
            receipts=list(current.rglob('*.pkl')); times={p:p.stat().st_mtime_ns for p in receipts}
            # A second execution reuses completed parts and reconstructs ordered output.
            again=occ._stream_transform_write_and_accumulate(str(path),'positional',str(current),num_workers=2)
            self.assertEqual(times,{p:p.stat().st_mtime_ns for p in receipts})
            for key in expected._kde:
                np.testing.assert_array_equal(expected._kde[key].values,again._kde[key].values)
            # Simulate one damaged/incomplete part after an interrupted run.
            receipts[0].write_bytes(b'incomplete')
            occ._stream_transform_write_and_accumulate(str(path),'positional',str(current),num_workers=2)
            self.assertNotEqual(times[receipts[0]],receipts[0].stat().st_mtime_ns)
            for path in receipts[1:]:
                self.assertEqual(times[path],path.stat().st_mtime_ns)
            self.assertEqual((legacy/'temp_transformed_positional_data.csv').read_bytes(),
                             (current/'temp_transformed_positional_data.csv').read_bytes())

    def test_memory_rejected_before_execution(self):
        with patch('autobehaver.postinf.mmgen.finalize_execution.memory_limit',return_value=(('test',4*GIB),[])):
            with self.assertRaisesRegex(MemoryError,'no recording was loaded'):
                list(budgeted_results(abs,[-1],[3*GIB],num_workers=8))

    def test_full_report_entry_uses_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'s.h5'; self.make_store(path)
            with patch.object(occ,'_generate_radial_distribution_plots_from_acc',return_value=[]), \
                 patch.object(occ,'_generate_2d_heatmaps_from_acc',return_value=[]), \
                 patch.object(occ,'_generate_orientation_quiver_plots_from_acc',return_value=[]):
                result=occ.calculate_and_save_occupancy_analysis(str(path),'positional',str(root),num_workers=2)
            self.assertTrue(Path(result).is_file())
            self.assertTrue((root/'MMGEN_OCCUPANCY_OUTPUTS.json').is_file())

if __name__=='__main__': unittest.main()
