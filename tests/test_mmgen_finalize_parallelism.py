import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from autobehaver.postinf.mmgen.kinematics import calculate_and_save_timeweighted_kinematic_summary
from autobehaver.postinf.mmgen.naming import collapse_individual_id
from autobehaver.postinf.mmgen.occupancy import calculate_and_save_occupancy_analysis


class TestMMGenFinalizeParallelism(unittest.TestCase):
    def _write_store(self, path: str) -> None:
        kin_df = pd.DataFrame([
            {"group": "g1", "individual": "a1", "sex": "FEMALE", "source_file": "f1", "frame_idx": 0, "kin_context_valid": "Standing + Floor", "speed_inst": 1.0},
            {"group": "g1", "individual": "a1", "sex": "FEMALE", "source_file": "f1", "frame_idx": 1, "kin_context_valid": "Standing + Floor", "speed_inst": 2.0},
            {"group": "g1", "individual": "a2", "sex": "MALE",   "source_file": "f2", "frame_idx": 0, "kin_context_valid": "Standing + Floor", "speed_inst": 3.0},
            {"group": "g1", "individual": "a2", "sex": "MALE",   "source_file": "f2", "frame_idx": 1, "kin_context_valid": "Standing + Floor", "speed_inst": 4.0},
        ])
        positional_df = pd.DataFrame([
            {"group": "g1", "individual": "a1", "source_file": "f1", "frame_idx": 0, "masked_pose": "Standing", "masked_orient": "Floor", "masked_pose_orient": "Standing + Floor", "masked_pose_valid": "Standing", "masked_orient_valid": "Floor", "masked_pose_orient_valid": "Standing + Floor", "centroid_x": 300.0, "centroid_y": 300.0, "angle": 0.0, "speed_mm_s": 1.0, "anglefeature_Wings_L_R": 10.0, "anglefeature_Legs_FL_FR": 11.0, "anglefeature_Legs_ML_MR": 12.0, "anglefeature_Legs_BL_BR": 13.0},
            {"group": "g1", "individual": "a1", "source_file": "f1", "frame_idx": 1, "masked_pose": "Standing", "masked_orient": "Floor", "masked_pose_orient": "Standing + Floor", "masked_pose_valid": "Standing", "masked_orient_valid": "Floor", "masked_pose_orient_valid": "Standing + Floor", "centroid_x": 301.0, "centroid_y": 300.0, "angle": 1.0, "speed_mm_s": 1.1, "anglefeature_Wings_L_R": 10.5, "anglefeature_Legs_FL_FR": 11.5, "anglefeature_Legs_ML_MR": 12.5, "anglefeature_Legs_BL_BR": 13.5},
            {"group": "g1", "individual": "a2", "source_file": "f2", "frame_idx": 0, "masked_pose": "Standing", "masked_orient": "Floor", "masked_pose_orient": "Standing + Floor", "masked_pose_valid": "Standing", "masked_orient_valid": "Floor", "masked_pose_orient_valid": "Standing + Floor", "centroid_x": 302.0, "centroid_y": 300.0, "angle": 2.0, "speed_mm_s": 1.2, "anglefeature_Wings_L_R": 10.2, "anglefeature_Legs_FL_FR": 11.2, "anglefeature_Legs_ML_MR": 12.2, "anglefeature_Legs_BL_BR": 13.2},
            {"group": "g1", "individual": "a2", "source_file": "f2", "frame_idx": 1, "masked_pose": "Standing", "masked_orient": "Floor", "masked_pose_orient": "Standing + Floor", "masked_pose_valid": "Standing", "masked_orient_valid": "Floor", "masked_pose_orient_valid": "Standing + Floor", "centroid_x": 303.0, "centroid_y": 300.0, "angle": 3.0, "speed_mm_s": 1.3, "anglefeature_Wings_L_R": 10.3, "anglefeature_Legs_FL_FR": 11.3, "anglefeature_Legs_ML_MR": 12.3, "anglefeature_Legs_BL_BR": 13.3},
        ])
        with pd.HDFStore(path, mode="w") as store:
            store.put("/kin_frames", kin_df, format="table", data_columns=["individual"])
            store.put("/positional", positional_df, format="table", data_columns=["individual"])

    def _write_store_with_multi_movie_tw(self, path: str) -> tuple[str, str]:
        animal_a_movie0 = "g1_FEMALE_rpi-1_trial1_movie0_chamber5"
        animal_a_movie1 = "g1_FEMALE_rpi-1_trial1_movie1_chamber5"
        animal_b_movie0 = "g1_MALE_rpi-2_trial1_movie0_chamber6"
        kin_df = pd.DataFrame([
            {"group": "g1", "individual": animal_a_movie0, "sex": "FEMALE", "source_file": "a0", "frame_idx": 0, "kin_context_valid": "Standing + Floor", "speed_inst": 1.0},
            {"group": "g1", "individual": animal_a_movie0, "sex": "FEMALE", "source_file": "a0", "frame_idx": 1, "kin_context_valid": "Standing + Floor", "speed_inst": 3.0},
            {"group": "g1", "individual": animal_a_movie1, "sex": "FEMALE", "source_file": "a1", "frame_idx": 0, "kin_context_valid": "Standing + Floor", "speed_inst": 5.0},
            {"group": "g1", "individual": animal_a_movie1, "sex": "FEMALE", "source_file": "a1", "frame_idx": 1, "kin_context_valid": "Standing + Floor", "speed_inst": 7.0},
            {"group": "g1", "individual": animal_b_movie0, "sex": "MALE", "source_file": "b0", "frame_idx": 0, "kin_context_valid": "Standing + Floor", "speed_inst": 2.0},
            {"group": "g1", "individual": animal_b_movie0, "sex": "MALE", "source_file": "b0", "frame_idx": 1, "kin_context_valid": "Standing + Floor", "speed_inst": 4.0},
        ])
        with pd.HDFStore(path, mode="w") as store:
            store.put("/kin_frames", kin_df, format="table", data_columns=["individual"])
        return collapse_individual_id(animal_a_movie0), collapse_individual_id(animal_b_movie0)

    def test_tw_summary_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as td:
            store_path = os.path.join(td, "temp_store.h5")
            out_dir = os.path.join(td, "out")
            Path(out_dir).mkdir()
            self._write_store(store_path)

            serial = calculate_and_save_timeweighted_kinematic_summary(
                store_path,
                out_dir,
                min_context_frames=1,
                save_outputs=False,
                num_workers=1,
            )
            parallel = calculate_and_save_timeweighted_kinematic_summary(
                store_path,
                out_dir,
                min_context_frames=1,
                save_outputs=False,
                num_workers=2,
            )

            self.assertEqual(set(serial.keys()), set(parallel.keys()))
            for sheet_name in serial.keys():
                left = serial[sheet_name].sort_values(["individual", "group"]).reset_index(drop=True)
                right = parallel[sheet_name].sort_values(["individual", "group"]).reset_index(drop=True)
                right = right.reindex(columns=left.columns)
                assert_frame_equal(left, right, check_dtype=False)

    def test_tw_summary_collapses_movie_level_ids_before_parallel_merge(self):
        with tempfile.TemporaryDirectory() as td:
            store_path = os.path.join(td, "temp_store.h5")
            out_dir = os.path.join(td, "out")
            Path(out_dir).mkdir()
            collapsed_a, collapsed_b = self._write_store_with_multi_movie_tw(store_path)

            serial = calculate_and_save_timeweighted_kinematic_summary(
                store_path,
                out_dir,
                min_context_frames=1,
                save_outputs=False,
                num_workers=1,
            )
            parallel = calculate_and_save_timeweighted_kinematic_summary(
                store_path,
                out_dir,
                min_context_frames=1,
                save_outputs=False,
                num_workers=2,
            )

            pose_orient_serial = serial["Pose + Orient (inst)"].sort_values(["individual", "group"]).reset_index(drop=True)
            pose_orient_parallel = parallel["Pose + Orient (inst)"].sort_values(["individual", "group"]).reset_index(drop=True)
            assert_frame_equal(pose_orient_serial, pose_orient_parallel, check_dtype=False)

            self.assertEqual(set(pose_orient_serial["individual"].tolist()), {collapsed_a, collapsed_b})
            self.assertEqual(int((pose_orient_serial["individual"] == collapsed_a).sum()), 1)
            mean_a = float(
                pose_orient_serial.loc[
                    pose_orient_serial["individual"] == collapsed_a,
                    "Standing + Floor_speed_mean",
                ].iloc[0]
            )
            self.assertAlmostEqual(mean_a, 4.0)

    def test_occupancy_transform_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as td:
            store_path = os.path.join(td, "temp_store.h5")
            out_dir = os.path.join(td, "out")
            Path(out_dir).mkdir()
            self._write_store(store_path)

            serial_path = calculate_and_save_occupancy_analysis(
                store_path,
                "/positional",
                out_dir,
                save_outputs=False,
                num_workers=1,
                transformed_output_path=os.path.join(out_dir, "serial.csv"),
            )
            parallel_path = calculate_and_save_occupancy_analysis(
                store_path,
                "/positional",
                out_dir,
                save_outputs=False,
                num_workers=2,
                transformed_output_path=os.path.join(out_dir, "parallel.csv"),
            )

            left = pd.read_csv(serial_path)
            right = pd.read_csv(parallel_path)
            assert_frame_equal(left, right, check_dtype=False)

    def test_occupancy_transform_parallel_matches_serial_in_memory(self):
        with tempfile.TemporaryDirectory() as td:
            store_path = os.path.join(td, "temp_store.h5")
            out_dir = os.path.join(td, "out")
            Path(out_dir).mkdir()
            self._write_store(store_path)

            serial_df = calculate_and_save_occupancy_analysis(
                store_path,
                "/positional",
                out_dir,
                save_outputs=False,
                num_workers=1,
                return_transformed_df=True,
            )
            parallel_df = calculate_and_save_occupancy_analysis(
                store_path,
                "/positional",
                out_dir,
                save_outputs=False,
                num_workers=2,
                return_transformed_df=True,
            )

            left = serial_df.sort_values(["individual", "source_file", "frame_idx"]).reset_index(drop=True)
            right = parallel_df.sort_values(["individual", "source_file", "frame_idx"]).reset_index(drop=True)
            assert_frame_equal(left, right, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
