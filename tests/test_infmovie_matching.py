import os

from autobehaver.postinf.infmovie.naming import extract_inf_csv_filename_part
from autobehaver.postinf.infmovie.io import find_movie_file


def test_infmovie_matches_movie_with_timestamp(tmp_path):
    movie_name = "SPD2_czmab_2day_trial1_rpi-6_movie0_20250819015233PM_chamber6.mp4"
    (tmp_path / movie_name).write_text("")

    csv_name = (
        "CONTROL-SPD2-czmab-2day_FEMALE_trial1_rpi-6_movie0_20250819015233PM_"
        "chamber6_positional_calcs_normalized_inference.csv"
    )

    filename_part = extract_inf_csv_filename_part(str(tmp_path / csv_name))
    assert filename_part == "trial1_rpi-6_movie0_20250819015233PM_chamber6"

    found = find_movie_file(str(tmp_path), filename_part)
    assert found is not None
    assert os.path.basename(found) == movie_name


def test_infmovie_matches_when_csv_lacks_timestamp(tmp_path):
    movie_name = "SPD2_czmab_2day_trial1_rpi-6_movie0_20250819015233PM_chamber6.mp4"
    (tmp_path / movie_name).write_text("")

    csv_name = "FEMALE_trial1_rpi-6_movie0_chamber6_positional_calcs_normalized_inference.csv"

    filename_part = extract_inf_csv_filename_part(str(tmp_path / csv_name))
    assert filename_part == "trial1_rpi-6_movie0_chamber6"

    found = find_movie_file(str(tmp_path), filename_part)
    assert found is not None
    assert os.path.basename(found) == movie_name


def test_infmovie_extracts_multi_digit_chamber_number(tmp_path):
    csv_name = (
        "FEMALE_trial1_rpi-6_movie0_20250819015233PM_chamber12_positional_calcs_"
        "normalized_inference.csv"
    )

    filename_part = extract_inf_csv_filename_part(str(tmp_path / csv_name))
    assert filename_part.endswith("_chamber12")


def test_discovery_ignores_reports_but_accepts_renamed_predictions(tmp_path):
    from autobehaver.postinf.infmovie.pipeline import discover_pairs_from_folders
    (tmp_path / "fly.mp4").touch()
    (tmp_path / "fly.csv").write_text("Frame Number,Predicted Pose Label\n1,Standing\n")
    (tmp_path / "report.csv").write_text("metric,value\naccuracy,0.9\n")
    (tmp_path / "fly_bouts.csv").write_text("start,end,label\n1,10,Standing\n")
    pairs = discover_pairs_from_folders(inference_csv_dir=tmp_path, movie_dir=tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["inf_csv_path"] == str((tmp_path / "fly.csv").resolve())


def test_discovery_still_reports_missing_prediction_movie(tmp_path):
    import pytest
    from autobehaver.postinf.infmovie.pipeline import discover_pairs_from_folders
    (tmp_path / "missing_inference.csv").write_text("malformed")
    with pytest.raises(FileNotFoundError, match="missing_inference.csv"):
        discover_pairs_from_folders(inference_csv_dir=tmp_path, movie_dir=tmp_path)
