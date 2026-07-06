import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from autobehaver.postinf.mmgen.feature_matrix import (
    MMGenFinalizationSources,
    create_feature_matrix,
    create_feature_matrix_from_sources,
)
from autobehaver.postinf.mmgen.kinematics import calculate_and_save_timeweighted_kinematic_summary
from autobehaver.postinf.mmgen.naming import collapse_individual_id


class TestMMGenFeatureMatrixSources(unittest.TestCase):
    def _minimal_bout_summary(self) -> pd.DataFrame:
        common = {
            'individual': 'grp_trial1_fly1',
            'group': 'grp',
            'sex': 'FEMALE',
            'n_bouts': 6,
            'total_length_frames': 240,
            'bout_frequency': 0.10,
            'bout_duration_mean_s': 1.0,
            'inter_bout_interval_mean_s': 2.0,
            'inter_bout_interval_std_s': 0.5,
            'bout_duration_coeff_of_var': 0.25,
            'initiation_prob': 0.33,
        }
        rows = [
            {**common, 'category': 'Pose', 'bout_type': 'Walking'},
            {**common, 'category': 'Orient', 'bout_type': 'Floor'},
            {**common, 'category': 'Pose+Orient', 'bout_type': 'Walking + Floor'},
        ]
        return pd.DataFrame(rows)

    def _minimal_bout_summary_for_individual(self, individual: str, *, group: str = 'grp', sex: str = 'FEMALE') -> pd.DataFrame:
        common = {
            'individual': individual,
            'group': group,
            'sex': sex,
            'n_bouts': 6,
            'total_length_frames': 240,
            'bout_frequency': 0.10,
            'bout_duration_mean_s': 1.0,
            'inter_bout_interval_mean_s': 2.0,
            'inter_bout_interval_std_s': 0.5,
            'bout_duration_coeff_of_var': 0.25,
            'initiation_prob': 0.33,
        }
        rows = [
            {**common, 'category': 'Pose', 'bout_type': 'Standing'},
            {**common, 'category': 'Orient', 'bout_type': 'Floor'},
            {**common, 'category': 'Pose+Orient', 'bout_type': 'Standing + Floor'},
        ]
        return pd.DataFrame(rows)

    def test_direct_sources_match_directory_wrapper_for_minimal_inputs(self):
        bout_summary_df = self._minimal_bout_summary()

        with tempfile.TemporaryDirectory() as td_wrapper, tempfile.TemporaryDirectory() as td_direct:
            wrapper_dir = Path(td_wrapper)
            direct_dir = Path(td_direct)

            bout_summary_df.to_csv(wrapper_dir / '1_Individual_Bout_Summary.csv', index=False)
            create_feature_matrix(str(wrapper_dir))

            sources = MMGenFinalizationSources(bout_summary_df=bout_summary_df)
            create_feature_matrix_from_sources(str(direct_dir), sources)

            wrapper_df = pd.read_csv(wrapper_dir / 'MASTER_FEATURE_MATRIX.csv')
            direct_df = pd.read_csv(direct_dir / 'MASTER_FEATURE_MATRIX.csv')
            assert_frame_equal(wrapper_df, direct_df, check_dtype=False)

    def test_direct_sources_accept_raw_measurements_dataframe(self):
        bout_summary_df = self._minimal_bout_summary()
        measurements_df = pd.DataFrame(
            [
                {
                    'individual_id': 'grp_trial1_fly1',
                    'group_label': 'grp',
                    'sex': 'FEMALE',
                    'length_avg_mean': 1.2,
                    'length_avg_std': 0.1,
                    'length_avg_n_samples': 10,
                    'headwidth_avg_mean': 0.5,
                    'headwidth_avg_std': 0.05,
                    'headwidth_avg_n_samples': 10,
                    'thorax_width_avg_mean': 0.4,
                    'thorax_width_avg_std': 0.04,
                    'thorax_width_avg_n_samples': 10,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            sources = MMGenFinalizationSources(
                bout_summary_df=bout_summary_df,
                measurements_df=measurements_df,
            )
            matrix_df = create_feature_matrix_from_sources(str(out_dir), sources)
            self.assertIsNotNone(matrix_df)
            self.assertIn('StdMeasure_length_avg_mean', matrix_df.columns)
            self.assertIn('StdMeasure_headwidth_avg_mean', matrix_df.columns)
            self.assertNotIn('StdMeasure_length_avg_n_samples', matrix_df.columns)

    def test_tw_summary_output_with_collapsed_animal_ids_merges_into_master_matrix(self):
        raw_movie0 = 'grp_FEMALE_rpi-1_trial1_movie0_chamber5'
        raw_movie1 = 'grp_FEMALE_rpi-1_trial1_movie1_chamber5'
        collapsed = collapse_individual_id(raw_movie0)
        kin_df = pd.DataFrame([
            {'group': 'grp', 'individual': raw_movie0, 'sex': 'FEMALE', 'source_file': 'f0', 'frame_idx': 0, 'kin_context_valid': 'Standing + Floor', 'speed_inst': 1.0},
            {'group': 'grp', 'individual': raw_movie0, 'sex': 'FEMALE', 'source_file': 'f0', 'frame_idx': 1, 'kin_context_valid': 'Standing + Floor', 'speed_inst': 3.0},
            {'group': 'grp', 'individual': raw_movie1, 'sex': 'FEMALE', 'source_file': 'f1', 'frame_idx': 0, 'kin_context_valid': 'Standing + Floor', 'speed_inst': 5.0},
            {'group': 'grp', 'individual': raw_movie1, 'sex': 'FEMALE', 'source_file': 'f1', 'frame_idx': 1, 'kin_context_valid': 'Standing + Floor', 'speed_inst': 7.0},
        ])

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            store_path = out_dir / 'temp_store.h5'
            with pd.HDFStore(store_path, mode='w') as store:
                store.put('/kin_frames', kin_df, format='table', data_columns=['individual'])

            tw_sheets = calculate_and_save_timeweighted_kinematic_summary(
                str(store_path),
                str(out_dir),
                min_context_frames=1,
                save_outputs=False,
                num_workers=2,
            )
            sources = MMGenFinalizationSources(
                bout_summary_df=self._minimal_bout_summary_for_individual(collapsed),
                tw_kinematic_sheets=tw_sheets,
            )
            matrix_df = create_feature_matrix_from_sources(
                str(out_dir),
                sources,
                filter_features_by_bout_count=False,
            )

            self.assertIsNotNone(matrix_df)
            self.assertIn('KinematicTW_Pose+Orient_Standing+Floor_speed_mean', matrix_df.columns)
            self.assertEqual(matrix_df['individual'].tolist(), [collapsed])
            self.assertAlmostEqual(
                float(matrix_df.loc[0, 'KinematicTW_Pose+Orient_Standing+Floor_speed_mean']),
                4.0,
            )

    def test_uncollapsed_tw_sheet_raises_in_master_matrix_assembly(self):
        raw_movie0 = 'grp_FEMALE_rpi-1_trial1_movie0_chamber5'
        raw_movie1 = 'grp_FEMALE_rpi-1_trial1_movie1_chamber5'
        collapsed = collapse_individual_id(raw_movie0)
        bad_tw_sheets = {
            'Pose + Orient (inst)': pd.DataFrame(
                [
                    {'individual': raw_movie0, 'group': 'grp', 'Standing + Floor_speed_mean': 2.0},
                    {'individual': raw_movie1, 'group': 'grp', 'Standing + Floor_speed_mean': 6.0},
                ]
            )
        }

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            sources = MMGenFinalizationSources(
                bout_summary_df=self._minimal_bout_summary_for_individual(collapsed),
                tw_kinematic_sheets=bad_tw_sheets,
            )
            with self.assertRaisesRegex(ValueError, 'collapsed animal-level'):
                create_feature_matrix_from_sources(
                    str(out_dir),
                    sources,
                    filter_features_by_bout_count=False,
                )

    def test_direct_sources_spatial_dataframe_matches_spatial_csv_path(self):
        bout_summary_df = self._minimal_bout_summary()
        spatial_df = pd.DataFrame([
            {
                'individual': 'grp_trial1_fly1',
                'source_file': 'f1',
                'frame_idx': 0,
                'r_norm': 0.2,
                'relative_angle': 10.0,
                'speed_mm_s': 1.5,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
            {
                'individual': 'grp_trial1_fly1',
                'source_file': 'f1',
                'frame_idx': 1,
                'r_norm': 0.4,
                'relative_angle': -20.0,
                'speed_mm_s': 2.5,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
        ])

        with tempfile.TemporaryDirectory() as td_path, tempfile.TemporaryDirectory() as td_df:
            path_dir = Path(td_path)
            df_dir = Path(td_df)
            spatial_path = path_dir / 'temp_transformed_positional_data.csv'
            spatial_df.to_csv(spatial_path, index=False)

            path_sources = MMGenFinalizationSources(
                bout_summary_df=bout_summary_df,
                spatial_data_path=str(spatial_path),
            )
            df_sources = MMGenFinalizationSources(
                bout_summary_df=bout_summary_df,
                spatial_df=spatial_df,
            )

            path_matrix = create_feature_matrix_from_sources(str(path_dir), path_sources)
            df_matrix = create_feature_matrix_from_sources(str(df_dir), df_sources)

            self.assertIsNotNone(path_matrix)
            self.assertIsNotNone(df_matrix)
            assert_frame_equal(
                path_matrix.sort_index(axis=1),
                df_matrix.sort_index(axis=1),
                check_dtype=False,
            )

    def test_direct_sources_spatial_parallel_matches_serial(self):
        bout_summary_df = pd.concat([
            self._minimal_bout_summary(),
            self._minimal_bout_summary().assign(individual='grp_trial1_fly2', sex='MALE')
        ], ignore_index=True)
        spatial_df = pd.DataFrame([
            {
                'individual': 'grp_trial1_fly1',
                'source_file': 'f1',
                'frame_idx': 0,
                'r_norm': 0.2,
                'relative_angle': 10.0,
                'speed_mm_s': 1.5,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
            {
                'individual': 'grp_trial1_fly1',
                'source_file': 'f1',
                'frame_idx': 1,
                'r_norm': 0.4,
                'relative_angle': -20.0,
                'speed_mm_s': 2.5,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
            {
                'individual': 'grp_trial1_fly2',
                'source_file': 'f2',
                'frame_idx': 0,
                'r_norm': 0.3,
                'relative_angle': 5.0,
                'speed_mm_s': 1.0,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
            {
                'individual': 'grp_trial1_fly2',
                'source_file': 'f2',
                'frame_idx': 1,
                'r_norm': 0.6,
                'relative_angle': -15.0,
                'speed_mm_s': 3.0,
                'masked_pose_valid': 'Walking',
                'masked_orient_valid': 'Floor',
                'masked_pose_orient_valid': 'Walking + Floor',
                'masked_pose': 'Walking',
                'masked_orient': 'Floor',
                'masked_pose_orient': 'Walking + Floor',
            },
        ])

        with tempfile.TemporaryDirectory() as td_serial, tempfile.TemporaryDirectory() as td_parallel:
            serial_dir = Path(td_serial)
            parallel_dir = Path(td_parallel)

            serial_matrix = create_feature_matrix_from_sources(
                str(serial_dir),
                MMGenFinalizationSources(bout_summary_df=bout_summary_df, spatial_df=spatial_df),
                spatial_num_workers=1,
            )
            parallel_matrix = create_feature_matrix_from_sources(
                str(parallel_dir),
                MMGenFinalizationSources(bout_summary_df=bout_summary_df, spatial_df=spatial_df),
                spatial_num_workers=2,
            )

            self.assertIsNotNone(serial_matrix)
            self.assertIsNotNone(parallel_matrix)
            assert_frame_equal(
                serial_matrix.sort_index(axis=1),
                parallel_matrix.sort_index(axis=1),
                check_dtype=False,
            )


if __name__ == '__main__':
    unittest.main()
