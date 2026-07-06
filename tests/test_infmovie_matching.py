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
