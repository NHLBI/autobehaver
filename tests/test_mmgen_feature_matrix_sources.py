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
from autobehaver.postinf.mmgen.naming import collapse_individual_id


class TestMMGenFeatureMatrixSources(unittest.TestCase):
    def _minimal_bout_summary(self) -> pd.DataFrame:
        common = {
            'individual': 'grp_trial1_fly1',
            'group': 'grp',
            'sex': 'FEMALE',
            'n_bouts': 6,
            'n_switches': 0,
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
            'n_switches': 0,
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
                    'length_side_mean': 1.3,
                    'length_side_std': 0.1,
                    'length_side_n_samples': 10,
                    'headwidth_avg_mean': 0.5,
                    'headheight_side_mean': 0.6,
                    'headheight_side_std': 0.05,
                    'headheight_side_n_samples': 10,
                    'thorax_width_avg_mean': 0.4,
                    'thoraxheight_side_mean': 0.3,
                    'thoraxheight_side_std': 0.04,
                    'thoraxheight_side_n_samples': 10,
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
            self.assertAlmostEqual(matrix_df.loc[0, 'StdMeasure_length_avg_mean'], 1.2)
            self.assertAlmostEqual(matrix_df.loc[0, 'StdMeasure_length_side_std'], 0.1)
            self.assertNotIn('StdMeasure_length_side_n_samples', matrix_df.columns)



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
            self.assertIn('Spatial_Pose_Walking_mean_r_norm', path_matrix.columns)
            self.assertAlmostEqual(float(path_matrix.loc[0, 'Spatial_Pose_Walking_mean_r_norm']), 0.3)
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
            self.assertIn('Spatial_Pose_Walking_mean_r_norm', serial_matrix.columns)
            assert_frame_equal(
                serial_matrix.sort_index(axis=1),
                parallel_matrix.sort_index(axis=1),
                check_dtype=False,
            )


if __name__ == '__main__':
    unittest.main()
