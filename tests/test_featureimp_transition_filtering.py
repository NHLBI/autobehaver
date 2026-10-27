import json
from pathlib import Path

import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp import core
from autobehaver.postinf.featureimp.matrix import normalize_feature_matrix


def _build_transition_filter_fixture() -> pd.DataFrame:
    rows = []
    for i in range(20):
        group = "A" if i < 10 else "B"
        sex = "F" if i % 2 == 0 else "M"

        trans_control_zero = 0.0 if i >= 10 else (np.nan if i % 2 == 0 else 0.0)
        trans_low_prevalence = 0.25 if i == 0 else 0.0
        trans_keep = 0.15 if i in {1, 3, 12, 14} else 0.0

        rows.append(
            {
                "individual": f"fly{i:02d}",
                "group": group,
                "sex": sex,
                "TimeBudget_Pose_Walk": 0.55 if i % 2 == 0 else 0.45,
                "TimeBudget_Pose_Stop": 0.45 if i % 2 == 0 else 0.55,
                "StdMeasure_feat1": float(i + 1),
                "Transition_Pose_Walk_to_Stop": trans_control_zero,
                "Transition_Pose_Stop_to_Walk": trans_low_prevalence,
                "Transition_Pose_Jump_to_Scramble": trans_keep,
                "Transition_Repertoire_Pose_EntropyNorm": float(0.5 + (i * 0.1)),
                "metadata_note": f"note_{i}",
            }
        )
    return pd.DataFrame(rows)


def test_transition_filtering_drops_sparse_transition_probabilities_and_keeps_repertoire_scalars(tmp_path: Path):
    raw_df = _build_transition_filter_fixture()
    matrix_path = tmp_path / "MASTER_FEATURE_MATRIX.csv"
    raw_df.to_csv(matrix_path, index=False)

    normalized_path = normalize_feature_matrix(
        results_dir=str(tmp_path),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(tmp_path),
        control_group_tag="A",
        training_group_a="A",
        training_group_b="B",
        skip_if_normalized_exists=False,
    )
    assert normalized_path is not None

    normalized_df = pd.read_csv(normalized_path)
    params = json.loads((tmp_path / "NORMALIZATION_PARAMS.json").read_text())

    dropped_expected = {
        "Transition_Pose_Walk_to_Stop": "control_group_all_zero_or_nan",
        "Transition_Pose_Stop_to_Walk": "low_training_group_prevalence",
    }

    for feature in dropped_expected:
        assert feature not in normalized_df.columns
        assert feature not in params["numeric_feature_columns"]
        assert feature not in params["compositional_feature_columns"]

    transition_filtering = params["transition_filtering"]
    assert int(transition_filtering["dropped_transition_feature_count"]) == len(dropped_expected)
    assert set(transition_filtering["dropped_transition_features"]) == set(dropped_expected)

    kept_transition = "Transition_Pose_Jump_to_Scramble"
    repertoire_feature = "Transition_Repertoire_Pose_EntropyNorm"

    assert kept_transition in normalized_df.columns
    assert kept_transition in params["numeric_feature_columns"]
    assert kept_transition in params["compositional_feature_columns"]

    assert repertoire_feature in normalized_df.columns
    assert repertoire_feature in params["numeric_feature_columns"]
    assert repertoire_feature in params["scalar_feature_columns"]
    assert repertoire_feature not in params["compositional_feature_columns"]
    assert all(repertoire_feature not in cols for cols in params["compositional_sets"].values())
    assert float(normalized_df[repertoire_feature].abs().sum()) > 0.0

    transition_meta = params["transition_filtering"]
    assert transition_meta["dropped_transition_feature_count"] == 2
    assert set(transition_meta["dropped_transition_features"]) == set(dropped_expected)
    assert params["normalization_method"] == "clr_then_zscore_v4_canonical_rounding"


def test_normalization_does_not_amplify_roundoff_for_constant_clr_components(tmp_path: Path):
    rows = []
    for i in range(12):
        group = "CONTROL" if i < 6 else "MUTANT"
        # Every control has the same composition. Tiny representational noise
        # after CLR must not become an order-one Z-score.
        scale = float(i + 1)
        rows.append(
            {
                "individual": f"fly{i:02d}",
                "group": group,
                "sex": "F" if i % 2 == 0 else "M",
                "TimeBudget_Pose_Walk": 0.25 if group == "CONTROL" else 0.20 * scale,
                "TimeBudget_Pose_Stop": 0.75 if group == "CONTROL" else 0.80 * scale,
                "StdMeasure_signal": scale,
            }
        )

    matrix_path = tmp_path / "MASTER_FEATURE_MATRIX.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    normalized_path = normalize_feature_matrix(
        results_dir=str(tmp_path),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(tmp_path),
        control_group_tag="CONTROL",
        training_group_a="CONTROL",
        training_group_b="MUTANT",
    )

    normalized = pd.read_csv(normalized_path)
    params = json.loads((tmp_path / "NORMALIZATION_PARAMS.json").read_text())
    control = normalized[normalized["group"] == "CONTROL"]

    assert np.max(np.abs(control["TimeBudget_Pose_Walk"].to_numpy())) < 1e-12
    assert np.max(np.abs(control["TimeBudget_Pose_Stop"].to_numpy())) < 1e-12
    assert params["zscore_std_floor"] == 1e-12
    assert params["zscore_round_decimals"] == 12
    assert set(params["zscore_std_floor_features"]) >= {
        "TimeBudget_Pose_Walk",
        "TimeBudget_Pose_Stop",
    }


def test_core_normalize_feature_matrix_passes_training_groups_into_transition_filtering(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    results_dir = tmp_path / "results"
    analysis_dir = results_dir / "FEATURE_ANALYSIS_core_transition_manual"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(20):
        group = "A" if i < 10 else "B"
        rows.append(
            {
                "individual": f"fly{i:02d}",
                "group": group,
                "sex": "F" if i % 2 == 0 else "M",
                "StdMeasure_feat1": float(i + 1),
                "Transition_Pose_Walk_to_Stop": 0.2 if i == 0 else 0.0,
            }
        )
    matrix_path = matrix_dir / "MASTER_FEATURE_MATRIX.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)

    settings = {
        "analysis_run_name": "core_transition_manual",
        "matrix_dir": str(matrix_dir),
        "results_dir": str(results_dir),
        "matrix_path": str(matrix_path),
        "control_group_tag": "A",
        "training_group_a": "A",
        "training_group_b": "B",
    }
    core.apply_settings(settings)

    normalized_path = core.normalize_feature_matrix(
        results_dir=str(results_dir),
        raw_matrix_path=str(matrix_path),
        analysis_folder_path=str(analysis_dir),
        skip_if_normalized_exists=False,
    )
    assert normalized_path is not None

    normalized_df = pd.read_csv(normalized_path)
    params = json.loads((analysis_dir / "NORMALIZATION_PARAMS.json").read_text())

    assert "Transition_Pose_Walk_to_Stop" not in normalized_df.columns
    assert "Transition_Pose_Walk_to_Stop" not in params["numeric_feature_columns"]
    assert params["transition_filtering"]["training_group_a_used"] == "A"
    assert params["transition_filtering"]["training_group_b_used"] == "B"
