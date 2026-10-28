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
