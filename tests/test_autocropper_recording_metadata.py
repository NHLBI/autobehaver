from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autobehaver.preinf.autocropper.recording_db import (
    find_latest_recording_database_csv,
    load_recording_database_index,
)


def test_session_database_normalizes_blank_subexperiment_and_empty_label(tmp_path: Path):
    csv_path = tmp_path / "recording_database_20260910_120000.csv"
    pd.DataFrame([{
        "experiment": "male_recovery_test",
        "subexperiment": None,
        "trial_number": 1,
        "camera_mappings": "rpi-1_chamber1:EMPTY::A_,rpi-1_chamber2:B_",
    }]).to_csv(csv_path, index=False)

    index = load_recording_database_index(csv_path)
    assert ("male_recovery_test", "", 1) in index.label_map
    assert index.get_label_prefix(
        experiment="male-recovery-test", subexperiment="", trial_number=1,
        rpi_number=1, chamber_number=1,
    ) == "A_"
    assert index.get_excluded_chambers(
        experiment="male_recovery_test", subexperiment="", trial_number=1, rpi_number=1,
    ) == {"chamber1"}


def test_detailed_occupancy_directory_uses_use_for_crop(tmp_path: Path):
    csv_path = tmp_path / "trial_0001" / "crop_occupancy_exp_trial0001_a.csv"
    csv_path.parent.mkdir()
    pd.DataFrame([
        {"experiment": "exp", "subexperiment": "sub", "trial_number": 1,
         "rpi_name": "rpi-2", "chamber_number": 1, "label": "A_",
         "occupied": 1, "use_for_crop": 1},
        {"experiment": "exp", "subexperiment": "sub", "trial_number": 1,
         "rpi_name": "rpi-2", "chamber_number": 2, "label": "B_",
         # Some recording exports retain a pre-recording status after a chamber
         # was marked empty; explicit occupancy/use_for_crop remain authoritative.
         "occupied": 0, "use_for_crop": 0, "recording_status": "occupied"},
    ]).to_csv(csv_path, index=False)

    index = load_recording_database_index(tmp_path)
    assert index.get_excluded_chambers(
        experiment="exp", subexperiment="sub", trial_number=1, rpi_number=2,
    ) == {"chamber2"}


def test_detailed_occupancy_prefers_legacy_label_and_publishes_one_separator(tmp_path: Path):
    from autobehaver.preinf.autocropper import pipeline
    csv_path = tmp_path / "crop_occupancy_exp.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "sub", "trial_number": 1,
        "rpi_name": "rpi-2", "chamber_number": 1, "label": "control",
        "legacy_label": "control_FEMALE_", "sex": "FEMALE",
        "occupied": 1, "use_for_crop": 1, "recording_status": "recorded",
    }]).to_csv(csv_path, index=False)

    old_index, old_args = pipeline.recording_db_index, pipeline.args
    try:
        pipeline.recording_db_index = load_recording_database_index(csv_path)
        pipeline.args = SimpleNamespace(skip_labeling=False)
        output = pipeline._expected_crop_output_path(
            Path("exp_sub_trial1_rpi-2_movie0_timestamp.mp4"), tmp_path, "chamber1"
        )
        from autobehaver.portable_paths import original_name
        assert original_name(output) == "control_FEMALE_exp_sub_trial1_rpi-2_movie0_timestamp_chamber1.mp4"
        assert "__" not in original_name(output)
    finally:
        pipeline.recording_db_index, pipeline.args = old_index, old_args


def test_detailed_occupancy_builds_sexed_label_when_legacy_label_is_absent(tmp_path: Path):
    csv_path = tmp_path / "crop_occupancy_exp.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "rpi_name": "2", "chamber_number": 1, "label": "control", "sex": "male",
        "occupied": 1, "use_for_crop": 1,
    }]).to_csv(csv_path, index=False)
    index = load_recording_database_index(csv_path)
    assert index.get_label_prefix(
        experiment="exp", subexperiment="", trial_number=1, rpi_number=2, chamber_number=1,
    ) == "control_MALE_"


def test_legacy_blank_subexperiment_does_not_produce_double_separator(tmp_path: Path):
    from autobehaver.preinf.autocropper import pipeline
    csv_path = tmp_path / "recording_database_local.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "camera_mappings": "rpi-1_chamber1:control__FEMALE_",
    }]).to_csv(csv_path, index=False)

    old_index, old_args = pipeline.recording_db_index, pipeline.args
    try:
        pipeline.recording_db_index = load_recording_database_index(csv_path)
        pipeline.args = SimpleNamespace(skip_labeling=False)
        output = pipeline._expected_crop_output_path(
            Path("exp__trial1_rpi-1_movie0_timestamp.mp4"), tmp_path, "chamber1"
        )
        from autobehaver.portable_paths import original_name
        assert original_name(output) == "control_FEMALE_exp_trial1_rpi-1_movie0_timestamp_chamber1.mp4"
        assert "__" not in original_name(output)
    finally:
        pipeline.recording_db_index, pipeline.args = old_index, old_args


def test_blank_use_for_crop_falls_back_to_occupied_instead_of_excluding(tmp_path: Path):
    csv_path = tmp_path / "crop_occupancy_exp.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "rpi_name": "rpi-1", "chamber_number": 1, "label": "A_",
        "occupied": 1, "use_for_crop": None,
    }]).to_csv(csv_path, index=False)
    index = load_recording_database_index(csv_path)
    assert not index.get_excluded_chambers(
        experiment="exp", subexperiment="", trial_number=1, rpi_number=1,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"use_for_crop": "perhaps"}, "use_for_crop must be one of"),
        ({"occupied": 0, "use_for_crop": 1}, "use_for_crop is true but occupied is false"),
        ({"chamber_number": 7}, "chamber_number must be 1..6"),
        ({"chamber_number": 1.5}, "chamber_number must be an integer"),
        ({"rpi_name": "camera-two"}, "rpi_name must look like"),
        ({"trial_number": 1.5}, "trial_number must be an integer"),
        ({"recording_status": "empty", "occupied": 1, "use_for_crop": 1},
         "recording_status is 'empty' but use_for_crop is true"),
    ],
)
def test_detailed_occupancy_rejects_invalid_or_conflicting_values(
    tmp_path: Path, updates: dict, message: str,
):
    row = {
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "rpi_name": "rpi-1", "chamber_number": 1, "label": "A_",
        "occupied": 1, "use_for_crop": 1, "recording_status": "recorded",
    }
    row.update(updates)
    csv_path = tmp_path / "crop_occupancy_invalid.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match=message):
        load_recording_database_index(csv_path)


def test_duplicate_detailed_chamber_rows_are_rejected_with_both_origins(tmp_path: Path):
    row = {
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "rpi_name": "rpi-1", "chamber_number": 1, "label": "A_",
        "occupied": 1, "use_for_crop": 1,
    }
    for name in ("crop_occupancy_a.csv", "crop_occupancy_b.csv"):
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)
    with pytest.raises(ValueError, match=r"duplicate chamber mapping.*first defined"):
        load_recording_database_index(tmp_path)


def test_duplicate_legacy_camera_mapping_is_rejected(tmp_path: Path):
    csv_path = tmp_path / "recording_database_local.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "camera_mappings": "rpi-1_chamber1:A_,rpi-1_chamber1:B_",
    }]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="duplicate camera mapping"):
        load_recording_database_index(csv_path)


def test_duplicate_legacy_session_rows_are_rejected_even_when_slots_differ(tmp_path: Path):
    csv_path = tmp_path / "recording_database_local.csv"
    pd.DataFrame([
        {"experiment": "exp", "subexperiment": "", "trial_number": 1,
         "camera_mappings": "rpi-1_chamber1:A_"},
        {"experiment": "exp", "subexperiment": "", "trial_number": 1,
         "camera_mappings": "rpi-1_chamber2:B_"},
    ]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="duplicate session"):
        load_recording_database_index(csv_path)


def test_malformed_legacy_mapping_is_rejected_instead_of_ignored(tmp_path: Path):
    csv_path = tmp_path / "recording_database_local.csv"
    pd.DataFrame([{
        "experiment": "exp", "subexperiment": "", "trial_number": 1,
        "camera_mappings": "rpi-1_chamber1:A_,broken-entry",
    }]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="malformed camera mapping entry"):
        load_recording_database_index(csv_path)


def test_normalized_session_name_collision_is_rejected(tmp_path: Path):
    csv_path = tmp_path / "recording_database_local.csv"
    pd.DataFrame([
        {"experiment": "exp_one", "subexperiment": "sub", "trial_number": 1,
         "camera_mappings": "rpi-1_chamber1:A_"},
        {"experiment": "exp-one", "subexperiment": "sub", "trial_number": 1,
         "camera_mappings": "rpi-1_chamber2:B_"},
    ]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="ambiguous experiment/subexperiment identities"):
        load_recording_database_index(csv_path)


def test_duplicate_csv_columns_are_rejected(tmp_path: Path):
    csv_path = tmp_path / "recording_database_local.csv"
    csv_path.write_text(
        "experiment,subexperiment,trial_number,camera_mappings,camera_mappings\n"
        "exp,,1,rpi-1_chamber1:A_,rpi-1_chamber2:B_\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        load_recording_database_index(csv_path)


def test_autodiscovery_uses_timestamped_database_name_deterministically(tmp_path: Path):
    older = tmp_path / "recording_database_20260910_120000.csv"
    newer = tmp_path / "recording_database_20260911_120000.csv"
    older.touch()
    newer.touch()
    assert find_latest_recording_database_csv(tmp_path) == newer.resolve()


def test_movie_parser_anchors_tokens_and_resolves_legacy_underscores(tmp_path: Path):
    from autobehaver.preinf.autocropper import pipeline
    csv_path = tmp_path / "recording_database_20260910_120000.csv"
    pd.DataFrame([{
        "experiment": "male_recovery_test", "subexperiment": None, "trial_number": 1,
        "camera_mappings": "rpi-3_chamber1:A_",
    }]).to_csv(csv_path, index=False)
    old_index = pipeline.recording_db_index
    try:
        pipeline.recording_db_index = load_recording_database_index(csv_path)
        identity = pipeline._parse_full_movie_identity(
            Path("male_recovery_test_trial1_rpi-3_movie0_20260910.mp4")
        )
        assert identity == ("male_recovery_test", "", 1, 3)
    finally:
        pipeline.recording_db_index = old_index
